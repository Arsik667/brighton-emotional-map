"""
Реальные данные: происшествия из Police.uk (ключи не нужны).

Запуск:
    PYTHONPATH=src .venv/bin/python scripts/07_collect_police.py
    PYTHONPATH=src .venv/bin/python scripts/07_collect_police.py --force

Результат:
    data/raw/police_uk.jsonl            - сырое (в git не едет)
    data/processed/police_monthly.csv   - помесячно по районам
    data/processed/police_incidents.csv - инциденты с районами
"""

from __future__ import annotations

import argparse
import logging

from bem.collect.police_uk import ATMOSPHERE_CATEGORIES, collect
from bem.config import FIGURES_DIR, PROCESSED_DIR, ensure_dirs
from bem.geo.districts import assign_districts_to_frame, district_names
from bem.logging_setup import setup_logging
from bem.viz.labels import get_labels, get_months

logger = logging.getLogger("police")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--months", type=int, default=None,
                        help="взять только N последних месяцев (для быстрой проверки)")
    parser.add_argument("--lang", default="en", choices=["en", "ru"],
                        help="язык подписей на графике")
    args = parser.parse_args()

    setup_logging()
    ensure_dirs()

    months = None
    if args.months:
        from bem.collect.http import PoliteSession
        from bem.collect.police_uk import available_months
        months = available_months(PoliteSession())[: args.months]

    df = collect(months=months, force=args.force)
    if df.empty:
        logger.error("Ничего не собралось.")
        return

    df = assign_districts_to_frame(df)
    df.to_csv(PROCESSED_DIR / "police_incidents.csv", index=False)

    # Помесячная сводка по районам
    monthly = (
        df.groupby(["district", "month"])
        .agg(incidents=("crime_id", "size"),
             atmosphere=("is_atmosphere", "sum"))
        .reset_index()
    )
    monthly["atmosphere_share"] = monthly["atmosphere"] / monthly["incidents"]
    monthly.to_csv(PROCESSED_DIR / "police_monthly.csv", index=False)

    names = district_names()
    print(f"\nПроисшествий собрано: {len(df):,}".replace(",", " "))
    print(f"Период: {df['month'].min()} .. {df['month'].max()} "
          f"({df['month'].nunique()} месяцев)\n")

    print("По районам (всего за весь период):")
    for key, count in df["district"].value_counts().items():
        atmo = df[(df.district == key) & df.is_atmosphere].shape[0]
        print(f"  {names.get(key, key):<28} {count:>6}  из них 'атмосферных' {atmo:>5} "
              f"({atmo/count*100:4.1f}%)")

    print("\nТоп категорий:")
    for cat, count in df["category"].value_counts().head(10).items():
        mark = " *" if cat in ATMOSPHERE_CATEGORIES else ""
        print(f"  {cat:<30} {count:>6}{mark}")
    print("\n  * - категории, которые мы считаем говорящими об атмосфере района")

    plot_seasonality(df, FIGURES_DIR / "police_seasonality.png", lang=args.lang)
    print(f"\nСохранено в {PROCESSED_DIR}")
    print(f"График: {FIGURES_DIR / 'police_seasonality.png'}")



def plot_seasonality(df, path, lang: str = "en"):
    """
    Сезонность реальных происшествий по районам.

    Нормируем на среднемесячный уровень САМОГО района, а не сравниваем
    абсолютные числа. Иначе график покажет только то, что The Lanes
    больше Seafront, - а это мы и так знаем. Нас интересует форма
    года внутри каждого района.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from bem.geo.districts import district_colors, district_names

    L, months = get_labels(lang), get_months(lang)
    names, colors = district_names(), district_colors()

    atmo = df[(df["district"] != "other") & df["is_atmosphere"]].copy()
    atmo["month_num"] = pd.to_datetime(atmo["month"]).dt.month

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for district, group in atmo.groupby("district"):
        per_month = (group.groupby("month_num").size()
                     / group.groupby("month_num")["month"].nunique())
        normalised = per_month / per_month.mean()  # 1.0 = средний уровень района
        ax.plot(normalised.index, normalised.values, marker="o", linewidth=2,
                label=names.get(district, district), color=colors.get(district))

    ax.axvspan(7.6, 8.4, color="#ffd54f", alpha=0.30, zorder=0)
    ax.axvspan(4.6, 5.4, color="#90caf9", alpha=0.30, zorder=0)
    ax.text(8, ax.get_ylim()[1], L["pride"], va="top", fontsize=9, color="#856404")
    ax.text(5, ax.get_ylim()[1], L["fringe"], va="top", fontsize=9, color="#1565c0")

    ax.axhline(1.0, color="#999", linewidth=0.9, linestyle="--")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(months)
    ax.set_ylabel(L["police_ylabel"])
    # Числа берём из самих данных, а не хардкодим: если пересобрать
    # проект через год, в подписи окажется актуальное количество.
    ax.set_title(
        L["police_title"].format(
            months=df["month"].nunique(),
            n=f"{len(df):,}".replace(",", "\u00a0"),
        ),
        fontsize=12, pad=14,
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

if __name__ == "__main__":
    main()
