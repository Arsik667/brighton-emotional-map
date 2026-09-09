"""
Этап 2: собрать заведения Брайтона из OpenStreetMap и привязать к районам.

Запуск:
    PYTHONPATH=src .venv/bin/python scripts/01_collect_places.py
    PYTHONPATH=src .venv/bin/python scripts/01_collect_places.py --force  # перекачать

Результат:
    data/raw/osm_places_raw.json  - сырой ответ API (в git не попадает)
    data/interim/places.csv       - чистая таблица заведений с районами
"""

from __future__ import annotations

import argparse
import logging

from bem.collect.osm_places import collect
from bem.config import INTERIM_DIR, ensure_dirs
from bem.geo.districts import assign_districts_to_frame, district_names
from bem.logging_setup import setup_logging

logger = logging.getLogger("collect_places")

OUTPUT = INTERIM_DIR / "places.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="перекачать данные, даже если кэш уже есть",
    )
    args = parser.parse_args()

    setup_logging()
    ensure_dirs()

    df = collect(force=args.force)
    if df.empty:
        logger.error("Ничего не собралось. Проверь bbox в config/districts.yaml.")
        return

    df = assign_districts_to_frame(df)
    df.to_csv(OUTPUT, index=False)

    # --- краткий отчёт, чтобы сразу видеть, что получилось ---
    names = district_names()
    print(f"\nСобрано заведений: {len(df)}")
    print(f"Сохранено: {OUTPUT}\n")

    print("По районам:")
    counts = df["district"].value_counts()
    for key, count in counts.items():
        share = count / len(df) * 100
        print(f"  {names.get(key, key):<28} {count:>5}  ({share:4.1f}%)")

    print("\nПо категориям:")
    for cat, count in df["category"].value_counts().items():
        print(f"  {cat:<28} {count:>5}")

    unknown_share = (df["district"] == "other").mean()
    if unknown_share > 0.5:
        logger.warning(
            "Больше половины точек не попали ни в один район (%.0f%%). "
            "Скорее всего, полигоны в config/districts.yaml слишком узкие.",
            unknown_share * 100,
        )


if __name__ == "__main__":
    main()
