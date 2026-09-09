"""
Этап 6: динамика настроений по месяцам и сезонность.

Запуск:
    PYTHONPATH=src .venv/bin/python scripts/05_timeseries.py

Результат:
    data/processed/monthly_sentiment.csv   - помесячный ряд по районам
    data/processed/seasonal_profile.csv    - средний профиль года
    reports/figures/seasonality.png        - график сезонности
    reports/figures/monthly_trend.png      - динамика по районам
"""

from __future__ import annotations

import argparse
import logging
import warnings

import matplotlib
matplotlib.use("Agg")  # рисуем в файл, без окна - нужно для запуска на сервере
import matplotlib.pyplot as plt
import pandas as pd

from bem.analysis.timeseries import (
    MIN_OBSERVATIONS, add_rolling, city_wide_monthly, detect_anomalies,
    seasonal_profile, to_monthly,
)
from bem.config import FIGURES_DIR, INTERIM_DIR, PROCESSED_DIR, ensure_dirs
from bem.geo.districts import district_colors, district_names
from bem.logging_setup import setup_logging

warnings.filterwarnings("ignore")
logger = logging.getLogger("timeseries")

INPUT = INTERIM_DIR / "texts_topics.csv"
MONTHS = ["янв", "фев", "мар", "апр", "май", "июн",
          "июл", "авг", "сен", "окт", "ноя", "дек"]


def plot_seasonality(profile: pd.DataFrame, path) -> None:
    """Средний профиль года по районам."""
    names, colors = district_names(), district_colors()
    fig, ax = plt.subplots(figsize=(11, 5.5))

    for district, group in profile.groupby("district", observed=True):
        group = group.sort_values("month_num")
        ax.plot(group["month_num"], group["value"], marker="o", linewidth=2,
                label=names.get(district, district), color=colors.get(district))

    # Подсвечиваем месяцы фестивалей - без подписи график ни о чём не говорит
    ax.axvspan(7.6, 8.4, color="#ffd54f", alpha=0.30, zorder=0)
    ax.axvspan(4.6, 5.4, color="#90caf9", alpha=0.30, zorder=0)
    ax.text(8, ax.get_ylim()[1], " Pride", va="top", fontsize=9, color="#856404")
    ax.text(5, ax.get_ylim()[1], " Fringe", va="top", fontsize=9, color="#1565c0")

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTHS)
    ax.set_ylabel("Net sentiment (доля позитива − доля негатива)")
    ax.set_title("Сезонность настроений по районам Брайтона", fontsize=13, pad=14)
    ax.axhline(0, color="#999", linewidth=0.8, linestyle="--")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_monthly_trend(monthly: pd.DataFrame, path) -> None:
    """Динамика по месяцам с доверительными интервалами."""
    names, colors = district_names(), district_colors()
    districts = sorted(monthly["district"].unique())

    fig, axes = plt.subplots(len(districts), 1, figsize=(12, 2.2 * len(districts)),
                             sharex=True)

    for ax, district in zip(axes, districts):
        group = monthly[monthly["district"] == district].sort_values("month")
        x = group["month"].dt.to_timestamp()
        color = colors.get(district, "#333")

        # Доверительный интервал - самое важное на этом графике.
        # Без него ломаная выглядит убедительнее, чем данные позволяют.
        ax.fill_between(x, group["net_sentiment"] - group["ci95"],
                        group["net_sentiment"] + group["ci95"],
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(x, group["net_sentiment"], color=color, alpha=0.45, linewidth=1)
        ax.plot(x, group["net_sentiment_rolling"], color=color, linewidth=2.2)

        # Точки, где данных мало, помечаем крестиком
        weak = group[~group["reliable"]]
        if len(weak):
            ax.scatter(weak["month"].dt.to_timestamp(), weak["net_sentiment"],
                       marker="x", color="#d32f2f", s=28, zorder=5,
                       label=f"n < {MIN_OBSERVATIONS}")
            ax.legend(frameon=False, fontsize=8, loc="lower left")

        ax.axhline(0, color="#bbb", linewidth=0.8, linestyle="--")
        ax.set_ylabel(names.get(district, district), fontsize=9)
        ax.grid(alpha=0.2)

    axes[0].set_title(
        "Динамика настроений по месяцам\n"
        "тонкая линия - данные, толстая - скользящее среднее (3 мес), "
        "заливка - 95% доверительный интервал",
        fontsize=12, pad=14,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="distilbert", choices=["distilbert", "vader"])
    args = parser.parse_args()

    setup_logging()
    ensure_dirs()

    if not INPUT.exists():
        logger.error("Нет %s - сначала запусти scripts/04_topics.py", INPUT)
        return

    df = pd.read_csv(INPUT, parse_dates=["created_utc"])
    monthly = to_monthly(df, f"{args.model}_label", f"{args.model}_score")
    monthly = add_rolling(monthly)
    monthly = detect_anomalies(monthly)

    profile = seasonal_profile(monthly)
    city = city_wide_monthly(monthly)

    monthly.to_csv(PROCESSED_DIR / "monthly_sentiment.csv", index=False)
    profile.to_csv(PROCESSED_DIR / "seasonal_profile.csv", index=False)

    # ---------- отчёт ----------
    names = district_names()

    print(f"\n{'='*68}\nОбъём данных\n{'='*68}")
    print(f"месяцев в ряду: {monthly['month'].nunique()}, районов: {monthly['district'].nunique()}")
    print(f"текстов в ячейке (район x месяц): медиана {monthly['n'].median():.0f}, "
          f"минимум {monthly['n'].min():.0f}, максимум {monthly['n'].max():.0f}")
    print(f"ячеек с n < {MIN_OBSERVATIONS}: {(~monthly['reliable']).sum()} "
          f"из {len(monthly)} ({(~monthly['reliable']).mean()*100:.0f}%)")

    print(f"\n{'='*68}\nСезонность: средний net sentiment по месяцам (весь город)\n{'='*68}")
    city["month_num"] = city["month"].apply(lambda p: p.month)
    by_month = city.groupby("month_num").agg(value=("value", "mean"), n=("n", "sum"))
    lo, hi = by_month["value"].min(), by_month["value"].max()
    for m, row in by_month.iterrows():
        filled = int(round((row["value"] - lo) / (hi - lo) * 34)) if hi > lo else 0
        mark = ""
        if m == 8:
            mark = "  <- Pride"
        elif m == 5:
            mark = "  <- Fringe"
        print(f"  {MONTHS[m-1]}  {row['value']:+.3f}  n={int(row['n']):>4}  "
              f"{'█'*filled}{mark}")

    print(f"\n{'='*68}\nАномальные месяцы (|z| > 1.5)\n{'='*68}")
    anomalies = monthly[monthly["is_anomaly"]].sort_values("zscore", ascending=False)
    for _, r in anomalies.head(12).iterrows():
        direction = "выше" if r["zscore"] > 0 else "ниже"
        print(f"  {str(r['month']):<9} {names.get(r['district'], r['district']):<13} "
              f"z={r['zscore']:+.2f} ({direction} обычного), net={r['net_sentiment']:+.3f}, n={int(r['n'])}")

    # Проверяем, попал ли метод в заложенные события
    august = anomalies[(anomalies["month"].apply(lambda p: p.month) == 8) &
                       (anomalies["zscore"] > 0)]
    may = anomalies[(anomalies["month"].apply(lambda p: p.month) == 5) &
                    (anomalies["zscore"] > 0)]
    print(f"\n  из них августовских всплесков: {len(august)}, майских: {len(may)}")

    plot_seasonality(profile, FIGURES_DIR / "seasonality.png")
    plot_monthly_trend(monthly, FIGURES_DIR / "monthly_trend.png")
    logger.info("Графики: %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
