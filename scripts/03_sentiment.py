"""
Этап 4: sentiment-анализ. VADER против DistilBERT.

Запуск:
    PYTHONPATH=src .venv/bin/python scripts/03_sentiment.py
    PYTHONPATH=src .venv/bin/python scripts/03_sentiment.py --skip-transformer

Результат:
    data/interim/texts_sentiment.csv  - тексты с оценками обеих моделей
    reports/sentiment_comparison.md   - таблица сравнения для README
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings

import pandas as pd

from bem.config import INTERIM_DIR, REPORTS_DIR, ensure_dirs
from bem.geo.districts import district_names
from bem.logging_setup import setup_logging
from bem.nlp.sentiment import TransformerSentiment, VaderSentiment, evaluate, to_district_scores

warnings.filterwarnings("ignore")
logger = logging.getLogger("sentiment")

INPUT = INTERIM_DIR / "texts.csv"
OUTPUT = INTERIM_DIR / "texts_sentiment.csv"
REPORT = REPORTS_DIR / "sentiment_comparison.md"


def tune_neutral_band(scores, true_labels, bands) -> pd.DataFrame:
    """
    Подобрать порог уверенности, ниже которого считаем текст нейтральным.

    DistilBERT/SST-2 обучен на двух классах и всегда выбирает один из них.
    Нейтральный класс мы делаем сами: "модель не уверена -> нейтрально".
    Где провести границу - вопрос эмпирический, вот и проверим на данных.

    Это не подгонка под ответ: порог - гиперпараметр, и подбирать его
    на размеченных данных законно. Важно лишь честно сказать, что он
    подобран, а не взят из воздуха.
    """
    from sklearn.metrics import f1_score

    rows = []
    for band in bands:
        labels = [
            "neu" if abs(s) < band else ("pos" if s > 0 else "neg")
            for s in scores
        ]
        rows.append(
            {
                "band": band,
                "macro_f1": f1_score(true_labels, labels, average="macro",
                                     labels=["neg", "neu", "pos"], zero_division=0),
                "доля neu": sum(l == "neu" for l in labels) / len(labels),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-transformer", action="store_true",
                        help="только VADER (если нет torch или мало времени)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    setup_logging()
    ensure_dirs()

    if not INPUT.exists():
        logger.error("Нет %s - сначала запусти scripts/02_collect_texts.py", INPUT)
        return

    df = pd.read_csv(INPUT, parse_dates=["created_utc"])
    logger.info("Загружено текстов: %d", len(df))

    # --- VADER ---
    t0 = time.perf_counter()
    vader = VaderSentiment()
    df = pd.concat([df, vader.score_many(df["text"])], axis=1)
    vader_time = time.perf_counter() - t0
    logger.info("VADER: %.1f с (%.0f текстов/с)", vader_time, len(df) / vader_time)

    # --- DistilBERT ---
    transformer_time = None
    if not args.skip_transformer:
        t0 = time.perf_counter()
        model = TransformerSentiment(batch_size=args.batch_size)
        df = pd.concat([df, model.score_many(df["text"])], axis=1)
        transformer_time = time.perf_counter() - t0
        logger.info("DistilBERT: %.1f с (%.0f текстов/с)",
                    transformer_time, len(df) / transformer_time)

    df.to_csv(OUTPUT, index=False)
    logger.info("Сохранено: %s", OUTPUT)

    # --- сравнение (только если есть правильные ответы) ---
    lines = ["# Сравнение sentiment-моделей", ""]

    if "true_polarity" not in df.columns:
        print("\nВ данных нет колонки true_polarity - сравнить точность не с чем.")
        print("Это нормально для реальных данных с Reddit: там разметки нет.")
    else:
        # Смешанные тексты ("еда супер, но очередь ужас") выносим отдельно:
        # у них нет одного правильного ответа, и включать их в общую
        # точность нечестно по отношению к обеим моделям.
        clean = df[~df["is_mixed"].fillna(False)]
        mixed = df[df["is_mixed"].fillna(False)]

        print(f"\nОднозначных текстов: {len(clean)}, смешанных: {len(mixed)}")

        models = [("VADER", "vader_label", vader_time)]
        if not args.skip_transformer:
            models.append(("DistilBERT", "distilbert_label", transformer_time))

        lines.append("| модель | accuracy | macro-F1 | секунд | текстов/с |")
        lines.append("|---|---|---|---|---|")

        for label, col, elapsed in models:
            m = evaluate(clean[col], clean["true_polarity"], label)
            print(f"\n{'='*62}\n{label}\n{'='*62}")
            print(f"accuracy: {m['accuracy']:.3f}   macro-F1: {m['macro_f1']:.3f}")
            print(m["confusion_matrix"].to_string())
            print()
            print(m["report"])
            lines.append(
                f"| {label} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | "
                f"{elapsed:.1f} | {len(df)/elapsed:.0f} |"
            )

        # --- подбор порога нейтральности для DistilBERT ---
        if not args.skip_transformer:
            print(f"\n{'='*62}\nПодбор порога нейтральности для DistilBERT\n{'='*62}")
            bands = [0.5, 0.7, 0.85, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]
            tuning = tune_neutral_band(
                clean["distilbert_score"], clean["true_polarity"], bands
            )
            print(tuning.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            best = tuning.loc[tuning.macro_f1.idxmax()]
            print(f"\nЛучший порог: {best['band']} (macro-F1 {best['macro_f1']:.3f})")

            lines += ["", "## Подбор порога нейтральности (DistilBERT)", "",
                      "| порог | macro-F1 | доля neu |", "|---|---|---|"]
            for _, r in tuning.iterrows():
                lines.append(f"| {r['band']} | {r['macro_f1']:.3f} | {r['доля neu']:.3f} |")

        # --- как модели ведут себя на смешанных текстах ---
        if len(mixed):
            print(f"\n{'='*62}\nСмешанные тексты (n={len(mixed)})\n{'='*62}")
            print("У них нет одного верного ответа - смотрим, что модели выбирают:")
            for label, col, _ in models:
                dist = mixed[col].value_counts(normalize=True).reindex(
                    ["neg", "neu", "pos"]).fillna(0)
                print(f"  {label:<12} neg {dist['neg']:.0%}  "
                      f"neu {dist['neu']:.0%}  pos {dist['pos']:.0%}")

    # --- агрегация по районам ---
    model_prefix = "vader" if args.skip_transformer else "distilbert"
    label_col, score_col = f"{model_prefix}_label", f"{model_prefix}_score"
    print(f"\n{'='*62}\nНастроение по районам (модель: {model_prefix})\n{'='*62}")
    print("net_sentiment = доля позитива минус доля негатива, от -1 до +1")
    print("opinionated   = доля тех, кто высказался определённо\n")
    names = district_names()
    by_district = to_district_scores(df, label_col, score_col)
    by_district.index = [names.get(i, i) for i in by_district.index]
    print(by_district.to_string(float_format=lambda x: f"{x:.3f}"))

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Отчёт: %s", REPORT)


if __name__ == "__main__":
    main()
