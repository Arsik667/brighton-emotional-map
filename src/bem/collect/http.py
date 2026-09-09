"""
Вежливый HTTP-клиент для сбора данных.

Почему нельзя просто писать requests.get() в цикле:

1. Rate limits. Публичные API ограничивают частоту запросов. Если долбить
   без пауз - получишь 429 (Too Many Requests), а потом бан по IP.
2. Правила хорошего тона. OpenStreetMap прямо требует осмысленный
   User-Agent с контактом: чтобы админы могли написать тебе, а не банить вслепую.
3. Сеть ненадёжна. Таймауты и 503 случаются постоянно. Скрипт, который
   падает на 900-м запросе из 1000, - бесполезный скрипт.

Этот класс закрывает все три пункта: пауза между запросами, понятный
User-Agent и повторные попытки с экспоненциальной задержкой.
"""

from __future__ import annotations

import logging
import time

import requests

from bem import __version__
from bem.config import get_secret

logger = logging.getLogger(__name__)

# Коды, при которых имеет смысл повторить запрос.
#   429 - превысили лимит запросов
#   406 - Overpass отвечает так, когда у него нет свободного слота
#   5xx - проблемы на стороне сервера
RETRYABLE_STATUS = {406, 429, 500, 502, 503, 504}

# Исключения, которые тоже означают "попробуй ещё раз".
# SSLError сюда попал не случайно: перегруженный сервер часто просто
# обрывает TLS-соединение, и это выглядит как ошибка шифрования,
# хотя на деле это отказ в обслуживании.
RETRYABLE_EXCEPTIONS = (
    requests.Timeout,
    requests.ConnectionError,
    requests.exceptions.SSLError,
)


def build_user_agent() -> str:
    """
    Собрать User-Agent с контактом.

    OSM требует, чтобы по User-Agent можно было понять, что за приложение
    и к кому обращаться. Анонимные скрипты они блокируют.
    """
    contact = get_secret("OSM_CONTACT_EMAIL", "unknown@example.com")
    return f"brighton-emotional-map/{__version__} (portfolio project; {contact})"


class PoliteSession:
    """
    HTTP-сессия, которая сама выдерживает паузы и повторяет неудачные запросы.

    Параметры:
        min_interval - минимальная пауза между запросами, секунды.
        max_retries  - сколько раз повторить при временной ошибке.
        timeout      - сколько ждать ответа, прежде чем считать запрос провалившимся.
    """

    def __init__(
        self,
        min_interval: float = 1.0,
        max_retries: int = 4,
        timeout: float = 180.0,
        user_agent: str | None = None,
    ):
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout

        self._session = requests.Session()
        self._session.headers["User-Agent"] = user_agent or build_user_agent()

        # Момент последнего запроса. 0.0 = запросов ещё не было.
        self._last_request_at = 0.0

    def _wait_for_slot(self) -> None:
        """Досидеть паузу, если с прошлого запроса прошло слишком мало времени."""
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Сделать запрос с паузами и повторами. Возвращает успешный Response."""
        kwargs.setdefault("timeout", self.timeout)
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            self._wait_for_slot()

            try:
                response = self._session.request(method, url, **kwargs)
                self._last_request_at = time.monotonic()

                if response.status_code in RETRYABLE_STATUS:
                    # Сервер может сам подсказать, сколько ждать, - уважаем это.
                    wait = self._retry_delay(attempt, response)
                    logger.warning(
                        "HTTP %s от %s - попытка %d/%d, ждём %.1f с",
                        response.status_code, url, attempt, self.max_retries, wait,
                    )
                    time.sleep(wait)
                    continue

                if response.status_code >= 400:
                    # Без тела ответа отлаживать API невозможно: сервер
                    # обычно пишет там, что именно ему не понравилось.
                    raise requests.HTTPError(
                        f"HTTP {response.status_code} от {url}\n"
                        f"Ответ сервера: {response.text[:500]}",
                        response=response,
                    )
                return response

            except RETRYABLE_EXCEPTIONS as exc:
                self._last_request_at = time.monotonic()
                last_error = exc
                wait = self._retry_delay(attempt)
                logger.warning(
                    "Сетевая ошибка (%s) - попытка %d/%d, ждём %.1f с",
                    type(exc).__name__, attempt, self.max_retries, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Не удалось получить {url} за {self.max_retries} попыток. "
            f"Последняя ошибка: {last_error}"
        )

    def _retry_delay(self, attempt: int, response: requests.Response | None = None) -> float:
        """
        Сколько ждать перед следующей попыткой.

        Экспоненциальная задержка: 2, 4, 8, 16 секунд. Идея в том, чтобы
        не добивать перегруженный сервер запросами, а дать ему выдохнуть.
        """
        if response is not None and "Retry-After" in response.headers:
            try:
                return float(response.headers["Retry-After"])
            except ValueError:
                pass
        return float(2 ** attempt)

    def get(self, url: str, **kwargs) -> requests.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        return self.request("POST", url, **kwargs)
