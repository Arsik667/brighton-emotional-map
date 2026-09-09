"""
Сбор постов и комментариев из r/Brighton через PRAW.

Почему Reddit, а не отзывы Google:
  * контент публичный и API официальный - не парсинг, не серая зона;
  * тексты живые и привязаны ко времени, а нам нужна сезонность;
  * условия использования не запрещают исследовательский анализ.

Что важно про правила Reddit API:
  1. User-Agent обязателен и должен описывать приложение. Reddit прямо
     банит за дефолтный или враньё в User-Agent.
  2. Лимит - 100 запросов в минуту для OAuth-приложений. PRAW сам следит
     за заголовками X-Ratelimit-* и притормаживает; мы дополнительно
     ставим паузу между тредами.
  3. Search в Reddit отдаёт максимум ~250 результатов на запрос.
     Поэтому мы делаем МНОГО узких запросов, а не один широкий.

Что важно про приватность (это спросят на собеседовании):
  Имя автора - персональные данные. Мы его НЕ сохраняем: вместо ника
  пишем необратимый хэш. Так можно отличить одного автора от другого
  (например, чтобы выбросить спамера), но нельзя восстановить личность.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from bem.config import CONFIG_DIR, RAW_DIR, get_secret, require_secret

logger = logging.getLogger(__name__)

SUBREDDIT = "Brighton"
RAW_JSONL = RAW_DIR / "reddit_texts.jsonl"
KEYWORDS_CONFIG = CONFIG_DIR / "text_keywords.yaml"

# Соль для хэширования ников. Меняешь соль - меняются все хэши.
# В настоящем проекте её кладут в секреты; здесь она фиксирована,
# чтобы повторный запуск давал те же id.
_AUTHOR_SALT = "brighton-emotional-map"


def hash_author(name: str | None) -> str:
    """Превратить ник в необратимый идентификатор."""
    if not name:
        return "deleted"
    digest = hashlib.sha256((_AUTHOR_SALT + name.lower()).encode()).hexdigest()
    return digest[:12]


def build_reddit_client():
    """
    Создать клиент PRAW в режиме "только чтение".

    read_only=True означает, что мы не логинимся под пользователем и
    ничего не можем опубликовать. Для сбора данных этого достаточно,
    а риск случайно что-то запостить равен нулю.
    """
    import praw  # импорт внутри функции: praw нужен не всем, кто трогает модуль

    reddit = praw.Reddit(
        client_id=require_secret("REDDIT_CLIENT_ID"),
        client_secret=require_secret("REDDIT_CLIENT_SECRET"),
        user_agent=get_secret(
            "REDDIT_USER_AGENT",
            "python:brighton-emotional-map:v0.1 (portfolio project)",
        ),
        # PRAW сам подождёт, если упрётся в лимит, вместо того чтобы упасть.
        ratelimit_seconds=600,
    )
    reddit.read_only = True
    return reddit


def load_search_queries() -> list[str]:
    """
    Собрать список поисковых запросов из ключевых слов районов.

    Берём только достаточно специфичные слова: искать по "the beach"
    в сабреддите приморского города бессмысленно - вернётся всё подряд.
    """
    with open(KEYWORDS_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    queries = set()
    for district in cfg["districts"].values():
        for kw in district["keywords"]:
            if len(kw) >= 5 and kw not in {"the beach", "the arches", "the level"}:
                queries.add(kw)
    return sorted(queries)


def _submission_to_record(submission, district_hint: str | None) -> dict:
    """Превратить пост в словарь для сохранения."""
    return {
        "source": "reddit",
        "kind": "submission",
        "id": submission.id,
        "parent_id": None,
        "created_utc": datetime.fromtimestamp(
            submission.created_utc, tz=timezone.utc
        ).isoformat(),
        # title и selftext склеиваем: в заголовке часто вся суть поста
        "text": f"{submission.title}\n\n{submission.selftext or ''}".strip(),
        "score": submission.score,
        "author_hash": hash_author(str(submission.author) if submission.author else None),
        "permalink": submission.permalink,
        "query": district_hint,
    }


def _comment_to_record(comment, submission_id: str, district_hint: str | None) -> dict:
    return {
        "source": "reddit",
        "kind": "comment",
        "id": comment.id,
        "parent_id": submission_id,
        "created_utc": datetime.fromtimestamp(
            comment.created_utc, tz=timezone.utc
        ).isoformat(),
        "text": comment.body or "",
        "score": comment.score,
        "author_hash": hash_author(str(comment.author) if comment.author else None),
        "permalink": comment.permalink,
        "query": district_hint,
    }


def collect(
    limit_per_query: int = 60,
    comments_per_submission: int = 40,
    pause_seconds: float = 1.0,
    output: Path = RAW_JSONL,
) -> int:
    """
    Пройти по всем поисковым запросам и сохранить посты с комментариями.

    Формат вывода - JSONL: по одному JSON-объекту на строку. Он удобнее
    обычного JSON для дозаписи: не нужно держать в памяти весь файл,
    и прерванный сбор не портит уже собранное.

    Возвращает количество сохранённых записей.
    """
    reddit = build_reddit_client()
    subreddit = reddit.subreddit(SUBREDDIT)
    queries = load_search_queries()

    logger.info("Запросов к поиску: %d, сабреддит: r/%s", len(queries), SUBREDDIT)

    seen_ids: set[str] = set()
    written = 0
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:

        def write(record: dict) -> None:
            nonlocal written
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

        for i, query in enumerate(queries, 1):
            logger.info("[%d/%d] поиск: %r", i, len(queries), query)

            try:
                # sort="new" вместо "relevance": нам нужен разброс по времени,
                # а не самые заплюсованные посты всех времён.
                results = subreddit.search(
                    query, sort="new", time_filter="all", limit=limit_per_query
                )

                for submission in results:
                    if submission.id in seen_ids:
                        continue
                    seen_ids.add(submission.id)
                    write(_submission_to_record(submission, query))

                    # replace_more(limit=0) выбрасывает заглушки "показать
                    # ещё N комментариев" вместо того, чтобы догружать их.
                    # Каждая догрузка - отдельный запрос к API, а нам хватит
                    # верхнего уровня обсуждения.
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list()[:comments_per_submission]:
                        if comment.id in seen_ids:
                            continue
                        seen_ids.add(comment.id)
                        write(_comment_to_record(comment, submission.id, query))

            except Exception as exc:  # noqa: BLE001
                # Один сломавшийся запрос не должен убивать весь сбор:
                # мы уже потратили время на предыдущие.
                logger.warning("Запрос %r не отработал: %s", query, str(exc)[:200])

            time.sleep(pause_seconds)  # не давим на API между запросами

    logger.info("Сохранено записей: %d -> %s", written, output)
    return written
