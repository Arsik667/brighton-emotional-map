"""
Сбор данных о происшествиях из Police.uk API.

=== ПОЧЕМУ ЭТОТ ИСТОЧНИК ПОЯВИЛСЯ В ПРОЕКТЕ ===
В ноябре 2025 Reddit ввёл Responsible Builder Policy и закрыл
самостоятельную регистрацию API-приложений. Тексты для проекта стали
недоступны, и нужен был источник РЕАЛЬНЫХ данных о городе без ключей.

Police.uk подошёл идеально:
  * ключи не нужны вообще;
  * 36 месяцев истории - есть на чём смотреть сезонность;
  * у каждого инцидента есть координаты - работает наша привязка к районам;
  * лицензия Open Government Licence - данные можно публиковать;
  * категории прямо по теме: anti-social-behaviour, public-order,
    violent-crime. Это объективная мера того самого "эмоционального фона",
    который мы до этого измеряли только по текстам.

=== ЧЕГО ЭТОТ ИСТОЧНИК НЕ ДАЁТ, И ЭТО ВАЖНО ===
Это НЕ замена текстам. Здесь нет мнений, нет тональности, нет тем -
NLP-часть проекта к этим данным неприменима. Это другой угол зрения
на тот же вопрос, а не эквивалент.

=== ГЛАВНАЯ ОСТОРОЖНОСТЬ ПРИ ИНТЕРПРЕТАЦИИ ===
Число зарегистрированных происшествий - это НЕ уровень преступности.
Это уровень преступности, умноженный на готовность людей заявлять
и на плотность полицейского присутствия. В центре города, где много
камер и патрулей, зарегистрируют больше при том же реальном уровне.

И ещё: координаты в Police.uk НАМЕРЕННО округлены до ближайшей
"якорной точки" (обычно центр улицы) ради приватности пострадавших.
Поэтому инциденты кучкуются в одних и тех же точках, и на карте это
выглядит как "горячие точки", которых в реальности нет.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from bem.collect.http import PoliteSession
from bem.config import RAW_DIR, get_bbox

logger = logging.getLogger(__name__)

API_URL = "https://data.police.uk/api/crimes-street/all-crime"
DATES_URL = "https://data.police.uk/api/crimes-street-dates"
RAW_JSONL = RAW_DIR / "police_uk.jsonl"

# Категории, которые говорят о том, каково находиться в районе.
# Кражи из магазинов сюда не входят: они говорят о торговле, а не
# об атмосфере места.
ATMOSPHERE_CATEGORIES = {
    "anti-social-behaviour",
    "public-order",
    "violent-crime",
    "criminal-damage-arson",
    "drugs",
    "possession-of-weapons",
}


def city_polygon() -> str:
    """
    Bounding box проекта в формате, который понимает Police.uk:
    пары "широта,долгота", разделённые двоеточием.
    """
    b = get_bbox()
    return ":".join([
        f"{b.south},{b.west}",
        f"{b.south},{b.east}",
        f"{b.north},{b.east}",
        f"{b.north},{b.west}",
    ])


def available_months(session: PoliteSession) -> list[str]:
    """Какие месяцы вообще есть в API. Возвращает список вида ['2026-07', ...]."""
    response = session.get(DATES_URL)
    return [item["date"] for item in response.json()]


def fetch_month(session: PoliteSession, month: str, poly: str) -> list[dict]:
    """Забрать все происшествия за один месяц внутри полигона."""
    response = session.get(API_URL, params={"poly": poly, "date": month})
    return response.json()


def collect(months: list[str] | None = None, force: bool = False,
            cache_path: Path = RAW_JSONL) -> pd.DataFrame:
    """
    Собрать происшествия за указанные месяцы (по умолчанию - за все доступные).

    Один запрос на месяц: 36 запросов вместо 180, если бы мы ходили
    отдельно за каждым районом. Район определяем сами - у нас уже есть
    привязка точка-в-полигоне, и она работает на этих координатах.
    """
    if cache_path.exists() and not force:
        logger.info("Беру из кэша: %s (force=True чтобы перекачать)", cache_path)
        records = [json.loads(line) for line in open(cache_path, encoding="utf-8")]
        return _to_frame(records)

    session = PoliteSession(min_interval=1.2, timeout=120.0)
    poly = city_polygon()

    if months is None:
        months = available_months(session)
        logger.info("Доступно месяцев: %d (%s .. %s)", len(months), months[-1], months[0])

    records = []
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, "w", encoding="utf-8") as f:
        for i, month in enumerate(months, 1):
            try:
                items = fetch_month(session, month, poly)
            except Exception as exc:  # noqa: BLE001
                # Пропуск одного месяца не должен убивать весь сбор
                logger.warning("Месяц %s не забрался: %s", month, str(exc)[:150])
                continue

            for item in items:
                item["_month"] = month
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                records.append(item)

            logger.info("[%2d/%d] %s: %d происшествий", i, len(months), month, len(items))

    logger.info("Всего собрано: %d", len(records))
    return _to_frame(records)


def _to_frame(records: list[dict]) -> pd.DataFrame:
    """Разобрать сырой JSON в таблицу."""
    rows = []
    for item in records:
        location = item.get("location") or {}
        lat, lon = location.get("latitude"), location.get("longitude")
        if not lat or not lon:
            continue

        rows.append({
            "crime_id": item.get("persistent_id") or item.get("id"),
            "month": item.get("_month") or item.get("month"),
            "category": item.get("category"),
            "lat": float(lat),
            "lon": float(lon),
            "street": (location.get("street") or {}).get("name"),
            "outcome": (item.get("outcome_status") or {}).get("category"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["is_atmosphere"] = df["category"].isin(ATMOSPHERE_CATEGORIES)
    return df.reset_index(drop=True)
