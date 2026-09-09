"""
Тесты геопривязки.

Зачем они в pet-проекте: границы районов ты будешь править ещё не раз.
Без тестов легко сдвинуть полигон и не заметить, что половина заведений
переехала в другой район - а на карте это выглядит правдоподобно.
Наличие тестов в портфолио-репозитории - заметный плюс на собеседовании.

Запуск:  PYTHONPATH=src .venv/bin/pytest -v
"""

import pytest

from bem.geo.districts import (
    UNKNOWN,
    assign_district,
    district_colors,
    district_names,
    load_districts,
)

# Реальные координаты известных мест Брайтона и ожидаемый район.
LANDMARKS = [
    ("Brighton Palace Pier", 50.8161, -0.1367, "seafront"),
    ("i360", 50.8214, -0.1521, "seafront"),
    ("Royal Pavilion", 50.8225, -0.1372, "the_lanes"),
    ("Brighton Square", 50.8215, -0.1420, "the_lanes"),
    ("Trafalgar Street", 50.8272, -0.1400, "north_laine"),
    ("Brighton Station", 50.8290, -0.1410, "north_laine"),
    ("St James's Street", 50.8195, -0.1290, "kemptown"),
    ("Hove Town Hall", 50.8330, -0.1730, "hove"),
]


@pytest.mark.parametrize("label,lat,lon,expected", LANDMARKS)
def test_known_landmarks(label, lat, lon, expected):
    assert assign_district(lat, lon) == expected, f"{label} попал не в свой район"


def test_point_far_away_is_unknown():
    """Лондон не должен оказаться районом Брайтона."""
    assert assign_district(51.5074, -0.1278) == UNKNOWN


def test_missing_coordinates_do_not_crash():
    """Пустые координаты - обычное дело в сырых данных, падать нельзя."""
    assert assign_district(None, None) == UNKNOWN
    assert assign_district(50.82, None) == UNKNOWN


def test_lat_lon_not_swapped():
    """
    Страховка от самой частой ошибки в гео-коде: перепутанные lat и lon.
    Если поменять их местами, точка уедет в Индийский океан у берегов Африки.
    """
    correct = assign_district(50.8215, -0.1420)   # The Lanes
    swapped = assign_district(-0.1420, 50.8215)   # бессмыслица
    assert correct == "the_lanes"
    assert swapped == UNKNOWN


def test_all_polygons_are_valid():
    """Полигон невалиден, если его стороны пересекают сами себя."""
    for d in load_districts():
        assert d.polygon.is_valid, f"полигон {d.key} невалиден"
        assert d.polygon.area > 0, f"полигон {d.key} вырожден"


def test_districts_sorted_by_priority():
    """load_districts обязан отдавать районы по возрастанию priority."""
    priorities = [d.priority for d in load_districts()]
    assert priorities == sorted(priorities)


def test_names_and_colors_cover_all_districts():
    """Каждому району нужны имя и цвет, иначе графики поедут."""
    keys = {d.key for d in load_districts()} | {UNKNOWN}
    assert keys <= set(district_names())
    assert keys <= set(district_colors())
