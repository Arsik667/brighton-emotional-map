"""
Сбор заведений (POI) из OpenStreetMap через Overpass API.

Почему OSM, а не Google Places:

  * бесплатно и без биллинга - Places стоит ~$17 за 1000 запросов;
  * без API-ключа - можно клонировать репозиторий и сразу запустить;
  * лицензия ODbL - данные МОЖНО выложить в публичный репозиторий,
    в отличие от Google, где условия использования это запрещают.

Единственное, чего в OSM нет, - отзывов. Их мы возьмём из Reddit (Этап 3).
Такое разделение "гео из одного источника, тексты из другого" -
нормальная инженерная практика, а не костыль.

Правила приличия для Overpass (公共 инстанс работает на пожертвованиях):
  * один большой запрос лучше сотни мелких;
  * осмысленный User-Agent;
  * результат кэшируем на диск и не дёргаем API повторно без нужды.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from bem.config import CONFIG_DIR, RAW_DIR, get_bbox
from bem.collect.http import PoliteSession

logger = logging.getLogger(__name__)

# Overpass - это несколько независимых серверов с одними и теми же данными.
# Главный (overpass-api.de) часто перегружен и жёстко режет лимиты, поэтому
# держим список зеркал и перебираем их по очереди.
#
# Урок, полученный на практике при разработке этого проекта: если долбить
# один сервер запросами без пауз, он перестаёт отвечать - сначала кодом 406
# ("нет свободного слота"), потом просто обрывает соединение. Fallback на
# зеркала + паузы между запросами решают проблему.
OVERPASS_MIRRORS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
POI_CONFIG = CONFIG_DIR / "poi_categories.yaml"
RAW_JSON = RAW_DIR / "osm_places_raw.json"


def load_poi_config() -> dict:
    with open(POI_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_overpass_query(bbox_str: str, osm_tags: dict[str, list[str]]) -> str:
    """
    Собрать запрос на языке Overpass QL.

    Разбор синтаксиса на примере одной строки:

        nwr["amenity"~"^(cafe|bar)$"](50.8,-0.2,50.84,-0.1);

        nwr      - искать среди nodes, ways и relations сразу
                   (кафе может быть точкой, а может - контуром здания)
        ["amenity"~"..."]  - у объекта есть тег amenity, подходящий под regex
        ^(...)$  - якоря, чтобы "cafe" не совпало с "cafeteria"
        (...)    - bounding box: юг, запад, север, восток

    В конце `out center tags;`:
        center - для контуров зданий вернуть координату центра,
                 чтобы у каждого объекта была ровно одна точка
        tags   - вернуть теги, а не только id
    """
    clauses = []
    for tag, values in osm_tags.items():
        pattern = "|".join(values)
        clauses.append(f'  nwr["{tag}"~"^({pattern})$"]({bbox_str});')

    body = "\n".join(clauses)
    return f"[out:json][timeout:180];\n(\n{body}\n);\nout center tags;"


def fetch_raw(force: bool = False, cache_path: Path = RAW_JSON) -> dict:
    """
    Скачать сырой ответ Overpass (или взять из кэша).

    force=False означает: если файл уже есть - не ходим в сеть.
    Это экономит и твоё время, и ресурсы бесплатного публичного сервера.
    """
    if cache_path.exists() and not force:
        logger.info("Беру из кэша: %s (force=True чтобы перекачать)", cache_path)
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    cfg = load_poi_config()
    query = build_overpass_query(get_bbox().as_overpass(), cfg["osm_tags"])

    payload = _post_to_any_mirror(query)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    logger.info("Сохранено в %s (%d объектов)", cache_path, len(payload.get("elements", [])))
    return payload


def _post_to_any_mirror(query: str) -> dict:
    """
    Отправить запрос, перебирая зеркала, пока какое-нибудь не ответит.

    Каждое зеркало получает свои повторные попытки внутри PoliteSession;
    если оно всё равно молчит - переходим к следующему.
    """
    session = PoliteSession(min_interval=2.0, max_retries=3, timeout=240.0)
    errors = []

    for url in OVERPASS_MIRRORS:
        logger.info("Пробую зеркало %s (запрос может занять 10-60 секунд)...", url)
        try:
            response = session.post(url, data={"data": query})
            logger.info("Успех: ответ от %s", url)
            return response.json()
        except Exception as exc:  # noqa: BLE001 - нам важно перейти к следующему зеркалу
            logger.warning("Зеркало %s не ответило: %s", url, str(exc)[:200])
            errors.append(f"{url}: {str(exc)[:200]}")

    raise RuntimeError(
        "Ни одно зеркало Overpass не ответило.\n"
        + "\n".join(errors)
        + "\n\nПопробуй позже - публичные серверы Overpass бесплатные и часто перегружены."
    )


def _extract_coordinates(element: dict) -> tuple[float | None, float | None]:
    """
    Достать координаты объекта.

    У точки (node) координаты лежат прямо в lat/lon.
    У контура (way/relation) их нет - но благодаря `out center` есть
    словарь center с той же парой. Обрабатываем оба случая.
    """
    if "lat" in element and "lon" in element:
        return element["lat"], element["lon"]
    center = element.get("center")
    if center:
        return center.get("lat"), center.get("lon")
    return None, None


def parse_elements(payload: dict) -> pd.DataFrame:
    """Превратить сырой JSON Overpass в аккуратную таблицу."""
    cfg = load_poi_config()
    interesting_tags = cfg["osm_tags"]
    category_map = cfg["category_map"]

    rows = []
    for element in payload.get("elements", []):
        tags = element.get("tags", {})

        name = tags.get("name")
        if not name:
            # Объекты без названия нам не нужны: их нельзя сопоставить
            # с текстами и не о чем показывать пользователю.
            continue

        lat, lon = _extract_coordinates(element)
        if lat is None or lon is None:
            continue

        # Определяем "тип" объекта: первый из наших тегов, который у него есть.
        poi_type = None
        source_tag = None
        for tag, values in interesting_tags.items():
            value = tags.get(tag)
            if value in values:
                poi_type = value
                source_tag = tag
                break

        if poi_type is None:
            continue

        rows.append(
            {
                # Уникальный id: тип объекта + номер. Просто id не годится -
                # node/123 и way/123 это разные объекты с одинаковым номером.
                "place_id": f"{element['type']}/{element['id']}",
                "name": name,
                "lat": lat,
                "lon": lon,
                "osm_tag": source_tag,
                "osm_type": poi_type,
                "category": category_map.get(poi_type, "other"),
                "cuisine": tags.get("cuisine"),
                "street": tags.get("addr:street"),
                "housenumber": tags.get("addr:housenumber"),
                "website": tags.get("website") or tags.get("contact:website"),
                "opening_hours": tags.get("opening_hours"),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Один и тот же объект иногда есть в OSM и точкой, и контуром.
    # Дедуплицируем по имени и округлённым координатам (~11 м).
    df["_dedup_key"] = (
        df["name"].str.lower().str.strip()
        + "|" + df["lat"].round(4).astype(str)
        + "|" + df["lon"].round(4).astype(str)
    )
    before = len(df)
    df = df.drop_duplicates("_dedup_key").drop(columns="_dedup_key")
    logger.info("Дубликатов удалено: %d", before - len(df))

    return df.reset_index(drop=True)


def collect(force: bool = False) -> pd.DataFrame:
    """Полный цикл: скачать (или взять из кэша) и разобрать в DataFrame."""
    return parse_elements(fetch_raw(force=force))
