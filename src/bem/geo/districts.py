"""
Привязка географических точек к районам Брайтона.

Главная задача модуля: по паре (широта, долгота) сказать, в каком районе
находится заведение. Внутри - классическая задача "точка в полигоне",
её за нас решает shapely.

Тонкость, о которой стоит помнить: наши районы ПЕРЕСЕКАЮТСЯ. Например,
пляжный бар на Kings Road попадает и в seafront, и в The Lanes. Поэтому
у каждого района есть priority, и мы берём район с наименьшим номером.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from shapely.geometry import Point, Polygon

from bem.config import load_districts_config

# Метка для точек, не попавших ни в один район.
# Не выбрасываем их и не ставим None - явная категория честнее и её видно в отчётах.
UNKNOWN = "other"


@dataclass(frozen=True)
class District:
    """Один район со своим полигоном."""

    key: str          # машинное имя: "north_laine"
    name: str         # человеческое: "North Laine"
    name_ru: str
    priority: int
    color: str
    polygon: Polygon

    @property
    def centroid_latlon(self) -> tuple[float, float]:
        """Центр района как (lat, lon) - например, чтобы поставить подпись на карте."""
        c = self.polygon.centroid
        return (c.y, c.x)  # shapely хранит x=долгота, y=широта


@lru_cache(maxsize=1)
def load_districts() -> list[District]:
    """
    Собрать список районов из YAML-конфига, отсортированный по priority.

    Сортировка здесь не косметика: она задаёт порядок проверки в assign_district.
    """
    cfg = load_districts_config()
    districts = []

    for key, item in cfg["districts"].items():
        # В YAML координаты лежат как [долгота, широта] - ровно то,
        # что ждёт shapely.Polygon. Менять местами не надо.
        polygon = Polygon(item["polygon"])

        if not polygon.is_valid:
            raise ValueError(
                f"Полигон района '{key}' невалидный (скорее всего, стороны "
                f"пересекаются). Проверь порядок точек в config/districts.yaml."
            )

        districts.append(
            District(
                key=key,
                name=item["name"],
                name_ru=item.get("name_ru", item["name"]),
                priority=item["priority"],
                color=item["color"],
                polygon=polygon,
            )
        )

    return sorted(districts, key=lambda d: d.priority)


def assign_district(lat: float, lon: float) -> str:
    """
    Определить район по координатам. Возвращает ключ района или "other".

    >>> assign_district(50.8265, -0.1400)   # North Laine
    'north_laine'
    """
    # Защита от битых данных: пустые координаты в датасетах - обычное дело.
    if lat is None or lon is None:
        return UNKNOWN

    point = Point(lon, lat)  # ВНИМАНИЕ: сначала долгота, потом широта!

    for district in load_districts():  # уже отсортированы по priority
        if district.polygon.contains(point):
            return district.key

    return UNKNOWN


def assign_districts_to_frame(df, lat_col: str = "lat", lon_col: str = "lon",
                              out_col: str = "district"):
    """
    То же самое, но для целого DataFrame. Добавляет колонку с районом.

    Работает построчно - для наших объёмов (тысячи строк) это доли секунды,
    и код читается гораздо проще, чем векторизованная версия.
    """
    df = df.copy()
    df[out_col] = [
        assign_district(lat, lon)
        for lat, lon in zip(df[lat_col], df[lon_col])
    ]
    return df


def district_names() -> dict[str, str]:
    """Словарь {ключ: человеческое имя} - для подписей на графиках."""
    names = {d.key: d.name for d in load_districts()}
    names[UNKNOWN] = "Other / за пределами районов"
    return names


def district_colors() -> dict[str, str]:
    """Словарь {ключ: цвет} - чтобы район был одного цвета на всех графиках."""
    colors = {d.key: d.color for d in load_districts()}
    colors[UNKNOWN] = "#999999"
    return colors
