"""
Единое место, где хранятся пути и настройки проекта.

Зачем отдельный модуль: чтобы нигде в коде не было строк вроде
"../../data/raw/places.csv". Такие пути ломаются, как только ты запускаешь
скрипт из другой папки или из ноутбука. Здесь пути вычисляются от
расположения самого файла - и работают всегда.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

# __file__ = .../src/bem/config.py
# parents[0] = .../src/bem, parents[1] = .../src, parents[2] = корень проекта
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"           # что скачали, как есть, не трогая
INTERIM_DIR = DATA_DIR / "interim"   # промежуточное: почищено, но ещё не итог
PROCESSED_DIR = DATA_DIR / "processed"  # финальные таблицы для дашборда

CONFIG_DIR = PROJECT_ROOT / "config"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DISTRICTS_YAML = CONFIG_DIR / "districts.yaml"

# Подхватываем .env, если он есть. override=False - реальные переменные
# окружения важнее файла (пригодится, когда будешь деплоить дашборд).
load_dotenv(PROJECT_ROOT / ".env", override=False)


def ensure_dirs() -> None:
    """Создать все рабочие папки, если их ещё нет. Вызывать в начале скриптов."""
    for directory in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BBox:
    """Прямоугольник на карте, которым ограничиваем сбор данных."""

    south: float
    west: float
    north: float
    east: float

    def as_overpass(self) -> str:
        """Строка в формате, который понимает Overpass API: (юг,запад,север,восток)."""
        return f"{self.south},{self.west},{self.north},{self.east}"

    def center(self) -> tuple[float, float]:
        """Центр прямоугольника как (lat, lon) - для центрирования карты."""
        return ((self.south + self.north) / 2, (self.west + self.east) / 2)


@lru_cache(maxsize=1)
def load_districts_config() -> dict:
    """
    Прочитать config/districts.yaml.

    lru_cache означает: файл читается с диска один раз за запуск программы,
    дальше возвращается уже разобранный словарь. Мелочь, но правильная привычка.
    """
    with open(DISTRICTS_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_bbox() -> BBox:
    """Bounding box проекта из конфига."""
    raw = load_districts_config()["bbox"]
    return BBox(south=raw["south"], west=raw["west"], north=raw["north"], east=raw["east"])


def get_secret(name: str, default: str | None = None) -> str | None:
    """
    Достать секрет из окружения.

    Никогда не пиши ключи прямо в коде - они утекут в git-историю,
    а вычистить их оттуда мучительно.
    """
    return os.environ.get(name, default)


def require_secret(name: str) -> str:
    """То же самое, но падает с понятным сообщением, если ключа нет."""
    value = get_secret(name)
    if not value:
        raise RuntimeError(
            f"Не найдена переменная окружения {name}. "
            f"Скопируй .env.example в .env и заполни её."
        )
    return value
