"""
Привязка текста к району Брайтона.

У заведения есть координаты - с ним всё просто. У поста на Reddit
координат нет, есть только слова. Значит, район надо извлечь из текста.

Два способа, в порядке убывания точности:

  1. В тексте упомянуто название заведения из нашей базы.
     "went to Fortune of War yesterday" -> паб -> The Lanes.
     Точно, но срабатывает редко.

  2. В тексте упомянут топоним района.
     "parking in Kemptown is a nightmare" -> Kemptown.
     Срабатывает чаще, но грубее.

Главная опасность здесь - ложные совпадения. В базе есть бар "Amsterdam"
и кафе "Terraces". Фраза "flew to Amsterdam last week" не имеет никакого
отношения к Брайтону, но наивный поиск подстроки её поймает. Поэтому:

  * названия короче 6 символов не используем вообще;
  * отбрасываем названия из стоп-листа (города, обычные слова);
  * ищем только по границам слов, чтобы "Terraces" не совпало
    внутри другого слова;
  * при нескольких совпадениях берём самое длинное - оно специфичнее.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import pandas as pd
import yaml

from bem.config import CONFIG_DIR, INTERIM_DIR
from bem.geo.districts import UNKNOWN

logger = logging.getLogger(__name__)

KEYWORDS_CONFIG = CONFIG_DIR / "text_keywords.yaml"

# Названия заведений, которые в обычном тексте почти наверняка значат
# что-то другое. Список пополняется по мере того, как ловишь ошибки -
# это нормальный итеративный процесс, а не признак плохого кода.
AMBIGUOUS_NAMES = {
    "amsterdam", "brighton", "hove", "the beach", "the office", "the library",
    "the deep end", "market", "the church", "the station", "the bridge",
    "the hub", "the pond", "the yard", "the lanes", "the level", "the arches",
    "victoria", "albert", "the gallery", "the shop", "food", "coffee",
    "the range", "the club", "the studio", "the bank", "the works",
}

MIN_NAME_LENGTH = 6


@lru_cache(maxsize=1)
def load_keywords() -> dict[str, list[str]]:
    """Ключевые слова районов из конфига."""
    with open(KEYWORDS_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {key: item["keywords"] for key, item in cfg["districts"].items()}


@lru_cache(maxsize=1)
def load_topic_keywords() -> dict[str, list[str]]:
    """Ключевые слова тем - пригодятся на Этапе 5 для именования тем."""
    with open(KEYWORDS_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["topics"]


def build_place_index(places: pd.DataFrame) -> dict[str, tuple[str, str, float, float]]:
    """
    Построить словарь {название в нижнем регистре: (place_id, район, lat, lon)}.

    Названия, которые встречаются больше чем в одном районе, выбрасываем:
    если в городе три "The Grand", то по упоминанию "The Grand" мы всё
    равно не поймём, о каком речь. Лучше не угадывать.
    """
    index: dict[str, tuple[str, str, float, float]] = {}
    seen_districts: dict[str, set[str]] = {}

    for row in places.itertuples(index=False):
        name = str(row.name).strip().lower()

        if len(name) < MIN_NAME_LENGTH or name in AMBIGUOUS_NAMES:
            continue
        if row.district == UNKNOWN:
            continue

        seen_districts.setdefault(name, set()).add(row.district)
        index[name] = (row.place_id, row.district, row.lat, row.lon)

    # --- Фильтр 1: название встречается в нескольких районах ---
    ambiguous = {n for n, districts in seen_districts.items() if len(districts) > 1}
    for name in ambiguous:
        index.pop(name, None)

    # --- Фильтр 2: название - часть другого названия из другого района ---
    #
    # Реальный баг, пойманный на демо-корпусе: в базе есть "Churros"
    # в The Lanes и "Donuts & Churros" в Kemptown. Текст про второе
    # заведение матчился на первое и уезжал не в тот район.
    #
    # Почему не спасал приоритет длинных названий: точное название
    # "Donuts & Churros" встречалось в двух районах и вылетело по
    # фильтру 1, а короткое "Churros" осталось и перехватило совпадение.
    #
    # Правило: если название N целиком входит (по границам слов) в название
    # другого заведения из ДРУГОГО района - по N привязываться нельзя.
    all_names = [
        (str(row.name).strip().lower(), row.district)
        for row in places.itertuples(index=False)
        if isinstance(row.name, str)
    ]

    substring_conflicts = set()
    for name, (_pid, district, _lat, _lon) in index.items():
        for other_name, other_district in all_names:
            if other_district == district or len(other_name) <= len(name):
                continue
            # быстрая проверка подстрокой, потом строгая по границам слов
            if name in other_name and re.search(rf"\b{re.escape(name)}\b", other_name):
                substring_conflicts.add(name)
                break

    for name in substring_conflicts:
        index.pop(name, None)

    logger.info(
        "Индекс заведений: %d названий "
        "(отброшено: неоднозначных %d, входящих в другие названия %d)",
        len(index), len(ambiguous), len(substring_conflicts),
    )
    return index


@lru_cache(maxsize=1)
def _keyword_patterns() -> list[tuple[str, str, re.Pattern]]:
    """
    Скомпилировать регулярки для ключевых слов районов.

    Сортируем по длине по убыванию: "st james's street" должно проверяться
    раньше, чем "hove", иначе более общее слово перехватит совпадение.
    """
    patterns = []
    for district, keywords in load_keywords().items():
        for kw in keywords:
            # \b - граница слова. Без неё "hove" совпадёт внутри "shove".
            pattern = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
            patterns.append((district, kw, pattern))

    return sorted(patterns, key=lambda x: -len(x[1]))


def link_text(
    text: str,
    place_index: dict[str, tuple[str, str, float, float]],
    place_patterns: list[tuple[str, re.Pattern]] | None = None,
) -> dict:
    """
    Определить район (и по возможности заведение) для одного текста.

    Возвращает словарь с районом, способом привязки и, если повезло,
    координатами конкретного заведения.
    """
    empty = {
        "district": UNKNOWN,
        "link_method": "none",
        "place_id": None,
        "place_name": None,
        "lat": None,
        "lon": None,
    }

    if not isinstance(text, str) or not text.strip():
        return empty

    lowered = text.lower()

    # --- Способ 1: упоминание заведения ---
    # Идём от самых длинных названий: "The Fortune of War" специфичнее,
    # чем "Fortune", и если совпало длинное - оно и правильное.
    best_name = None
    for name in sorted(place_index, key=len, reverse=True):
        if name in lowered:
            # Подстрока нашлась - теперь проверяем границы слов,
            # чтобы отсечь совпадения внутри других слов.
            if re.search(rf"\b{re.escape(name)}\b", lowered):
                best_name = name
                break

    if best_name:
        place_id, district, lat, lon = place_index[best_name]
        return {
            "district": district,
            "link_method": "place_name",
            "place_id": place_id,
            "place_name": best_name,
            "lat": lat,
            "lon": lon,
        }

    # --- Способ 2: топоним района ---
    for district, _keyword, pattern in _keyword_patterns():
        if pattern.search(text):
            return {
                "district": district,
                "link_method": "district_keyword",
                "place_id": None,
                "place_name": None,
                "lat": None,
                "lon": None,
            }

    return empty


def link_frame(texts: pd.DataFrame, places: pd.DataFrame,
               text_col: str = "text") -> pd.DataFrame:
    """Привязать целый DataFrame текстов. Добавляет колонки с районом и заведением."""
    place_index = build_place_index(places)

    links = [link_text(t, place_index) for t in texts[text_col]]
    link_df = pd.DataFrame(links, index=texts.index)

    result = pd.concat([texts.drop(columns=link_df.columns, errors="ignore"), link_df], axis=1)

    stats = result["link_method"].value_counts()
    logger.info("Привязка текстов: %s", stats.to_dict())
    return result
