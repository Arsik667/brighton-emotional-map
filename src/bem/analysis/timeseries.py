"""
Динамика настроений по времени.

=== ГЛАВНАЯ ЛОВУШКА ЭТОГО ЭТАПА ===
У нас 7434 текста. Кажется, что много. Но как только мы режем их на
5 районов x 44 месяца = 220 ячеек, в каждой остаётся в среднем 34 текста.
А в феврале в Хоуве может оказаться и семь.

Средняя тональность по семи текстам - это почти шум. Если построить
такой график без доверительных интервалов, получится красивая ломаная,
в которой ты (и, что хуже, читатель) увидишь несуществующие "тренды".

Поэтому здесь всё сделано так:
  * рядом со средним ВСЕГДА показываем количество наблюдений;
  * считаем доверительный интервал и рисуем его на графике;
  * ячейки с n меньше порога помечаем как ненадёжные;
  * для сглаживания даём скользящее среднее, но не подменяем им данные.

Умение сказать "здесь мало данных, выводы делать рано" ценится
намного выше, чем красивый график.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Минимум наблюдений, ниже которого точке не стоит верить.
MIN_OBSERVATIONS = 20


def to_monthly(
    df: pd.DataFrame,
    label_col: str = "distilbert_label",
    score_col: str = "distilbert_score",
    date_col: str = "created_utc",
    district_col: str = "district",
) -> pd.DataFrame:
    """
    Свернуть тексты в помесячный ряд по каждому району.

    Возвращает таблицу с net_sentiment, средним, количеством и
    доверительным интервалом.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True, format="mixed")

    # to_period("M") превращает дату в месяц: 2026-08-17 -> 2026-08.
    # Это удобнее, чем группировать по строке "2026-08": периоды
    # правильно сортируются и умеют арифметику.
    df["month"] = df[date_col].dt.to_period("M")

    def aggregate(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        pos = (group[label_col] == "pos").sum()
        neg = (group[label_col] == "neg").sum()
        net = (pos - neg) / n

        mean_score = group[score_col].mean()
        std = group[score_col].std(ddof=1) if n > 1 else np.nan

        # Стандартная ошибка среднего = std / sqrt(n).
        # Отсюда видно, почему количество так важно: чтобы уменьшить
        # ошибку вдвое, наблюдений нужно вчетверо больше.
        stderr = std / np.sqrt(n) if n > 1 and not np.isnan(std) else np.nan
        ci95 = 1.96 * stderr if not np.isnan(stderr) else np.nan

        return pd.Series(
            {
                "n": n,
                "net_sentiment": net,
                "mean_score": mean_score,
                "ci95": ci95,
                "share_pos": pos / n,
                "share_neg": neg / n,
                "reliable": n >= MIN_OBSERVATIONS,
            }
        )

    result = (
        df.groupby([district_col, "month"], observed=True)
        .apply(aggregate, include_groups=False)
        .reset_index()
    )

    weak = (~result["reliable"]).mean()
    if weak > 0:
        logger.info(
            "Ячеек с малым числом наблюдений (n<%d): %.0f%% - на графике они "
            "помечены отдельно", MIN_OBSERVATIONS, weak * 100,
        )
    return result


def add_rolling(monthly: pd.DataFrame, window: int = 3,
                value_col: str = "net_sentiment") -> pd.DataFrame:
    """
    Добавить скользящее среднее.

    Зачем: помесячный ряд из ~30 наблюдений в точке сильно скачет, и
    за скачками не видно формы. Окно в 3 месяца усредняет соседей и
    показывает форму.

    Чем платим: сглаживание СДВИГАЕТ и ПРИТУПЛЯЕТ резкие всплески.
    Пик августовского Pride на сглаженной линии окажется ниже и шире,
    чем на самом деле. Поэтому сглаженную линию показываем ВМЕСТЕ с
    исходной, а не вместо неё.
    """
    out = []
    for district, group in monthly.groupby("district", observed=True):
        group = group.sort_values("month").copy()
        group[f"{value_col}_rolling"] = (
            group[value_col].rolling(window, center=True, min_periods=1).mean()
        )
        out.append(group)
    return pd.concat(out, ignore_index=True)


def seasonal_profile(monthly: pd.DataFrame, value_col: str = "net_sentiment",
                     district_col: str = "district") -> pd.DataFrame:
    """
    Средний профиль года: как выглядит типичный январь, февраль и так далее.

    Складываем все январи вместе, все феврали вместе и так далее.
    Это самый простой способ увидеть сезонность: если в августе среднее
    заметно выше, чем в остальные месяцы, - сезонный эффект есть.

    Ограничение, о котором надо знать: метод предполагает, что сезонность
    из года в год одинаковая. Если в 2024 Pride отменили, а в 2025 провели,
    усреднение это размажет. Для 3-4 лет данных приемлемо, но упомянуть
    об этом стоит.
    """
    monthly = monthly.copy()
    monthly["month_num"] = monthly["month"].apply(lambda p: p.month)

    profile = (
        monthly.groupby([district_col, "month_num"], observed=True)
        .agg(
            value=(value_col, "mean"),
            n_months=(value_col, "size"),
            total_texts=("n", "sum"),
        )
        .reset_index()
    )
    return profile


def detect_anomalies(monthly: pd.DataFrame, value_col: str = "net_sentiment",
                     z_threshold: float = 1.5) -> pd.DataFrame:
    """
    Найти месяцы, заметно выбивающиеся из обычного уровня района.

    Считаем z-оценку: на сколько стандартных отклонений точка отличается
    от среднего по району. |z| > 1.5 - уже заметное отклонение.

    Осторожно с интерпретацией: выброс - это НЕ доказательство события.
    Это повод пойти и проверить, что там происходило. В нашем случае
    мы знаем ответ заранее (Pride в августе), и это хорошая проверка
    того, что метод вообще что-то ловит.
    """
    out = []
    for district, group in monthly.groupby("district", observed=True):
        group = group.sort_values("month").copy()
        mean, std = group[value_col].mean(), group[value_col].std(ddof=1)

        group["zscore"] = (group[value_col] - mean) / std if std and std > 0 else 0.0
        group["is_anomaly"] = group["zscore"].abs() > z_threshold
        out.append(group)

    result = pd.concat(out, ignore_index=True)
    logger.info("Найдено аномальных месяцев: %d", int(result["is_anomaly"].sum()))
    return result


def city_wide_monthly(monthly: pd.DataFrame,
                      value_col: str = "net_sentiment") -> pd.DataFrame:
    """
    Общегородской ряд: усреднение по районам с весом по числу текстов.

    Именно ВЗВЕШЕННОЕ среднее, а не простое. Простое дало бы Хоуву с
    его двумя десятками февральских текстов тот же вес, что и Лейнам
    с тремя сотнями. Это исказило бы картину в пользу малых районов.
    """
    def weighted(group: pd.DataFrame) -> pd.Series:
        weights = group["n"]
        return pd.Series(
            {
                "value": np.average(group[value_col], weights=weights),
                "n": weights.sum(),
                "districts": len(group),
            }
        )

    return (
        monthly.groupby("month", observed=True)
        .apply(weighted, include_groups=False)
        .reset_index()
        .sort_values("month")
    )
