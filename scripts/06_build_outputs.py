"""
Этап 7 (часть 1): собрать финальные датасеты и интерактивную карту.

Запуск:
    PYTHONPATH=src .venv/bin/python scripts/06_build_outputs.py

Результат (всё в data/processed - эти файлы МОЖНО коммитить в git):
    texts_final.csv        - тексты с тональностью, темой и районом
    district_summary.csv   - сводка по районам
    place_summary.csv      - сводка по заведениям
    monthly_sentiment.csv  - помесячный ряд (создан на Этапе 6)
    reports/figures/emotional_map.html - интерактивная карта
"""

from __future__ import annotations

import argparse
import logging
import warnings

import pandas as pd

from bem.config import FIGURES_DIR, INTERIM_DIR, PROCESSED_DIR, ensure_dirs
from bem.geo.districts import district_names
from bem.logging_setup import setup_logging
from bem.analysis.ranking import rank_places
from bem.nlp.sentiment import to_district_scores
from bem.viz.labels import get_labels
from bem.viz.maps import add_title, build_map, plot_static_map

warnings.filterwarnings("ignore")
logger = logging.getLogger("build_outputs")

INPUT = INTERIM_DIR / "texts_topics.csv"
PLACES = INTERIM_DIR / "places.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="distilbert", choices=["distilbert", "vader"])
    parser.add_argument("--lang", default="en", choices=["en", "ru"],
                        help="язык подписей на карте (README на GitHub английский)")
    parser.add_argument("--min-texts", type=int, default=3,
                        help="минимум упоминаний, чтобы заведение попало в рейтинг")
    parser.add_argument("--topic-col", default="keyword_topic",
                        help="какие темы показывать: keyword_topic / bertopic_topic_name")
    args = parser.parse_args()

    setup_logging()
    ensure_dirs()

    df = pd.read_csv(INPUT, parse_dates=["created_utc"])
    places = pd.read_csv(PLACES)

    label_col, score_col = f"{args.model}_label", f"{args.model}_score"

    # ---------- финальная таблица текстов ----------
    # Оставляем только нужные колонки: файл поедет в git и будет
    # читаться дашбордом при каждом запуске.
    keep = [
        "text_id", "source", "created_utc", "text", "district", "place_id",
        "place_name", "lat", "lon", "category", label_col, score_col,
        args.topic_col,
    ]
    keep = [c for c in keep if c in df.columns]
    texts = df[keep].rename(columns={
        label_col: "sentiment_label",
        score_col: "sentiment_score",
        args.topic_col: "topic",
    })
    texts.to_csv(PROCESSED_DIR / "texts_final.csv", index=False)

    # ---------- сводка по районам ----------
    district_summary = to_district_scores(df, label_col, score_col)
    district_summary["name"] = [district_names().get(i, i) for i in district_summary.index]
    district_summary.to_csv(PROCESSED_DIR / "district_summary.csv")

    # ---------- сводка по заведениям ----------
    # Заведения, о которых есть хотя бы несколько упоминаний. Порог нужен:
    # "средняя оценка" по одному тексту - это не оценка, а один текст.
    with_place = df[df["place_id"].notna()]
    place_stats = (
        with_place.groupby("place_id")
        .agg(
            n_texts=(label_col, "size"),
            share_pos=(label_col, lambda s: (s == "pos").mean()),
            share_neg=(label_col, lambda s: (s == "neg").mean()),
            mean_score=(score_col, "mean"),
        )
    )
    place_stats["net_sentiment"] = place_stats["share_pos"] - place_stats["share_neg"]

    # Сжимаем оценки к общегородскому среднему: без этого в топе окажутся
    # заведения с тремя случайными отзывами и идеальной оценкой 1.00.
    place_stats = rank_places(place_stats, min_count=args.min_texts)

    place_summary = places.merge(place_stats, on="place_id", how="inner")
    place_summary.to_csv(PROCESSED_DIR / "place_summary.csv", index=False)

    # ---------- карта ----------
    L = get_labels(args.lang)
    source_note = (
        L["source_demo"] if (texts["source"] == "demo").all() else L["source_real"]
    )

    # Две карты с разными цветовыми шкалами - см. пояснение в build_map.
    for mode, filename, note in [
        ("absolute", "emotional_map.html", L["map_note_absolute"]),
        ("relative", "emotional_map_relative.html", L["map_note_relative"]),
    ]:
        m = build_map(
            district_stats=district_summary,
            texts=texts,
            places=place_summary,
            value_col="net_sentiment",
            scale_mode=mode,
            lang=args.lang,
        )
        add_title(
            m, L["map_title"],
            L["map_subtitle"].format(n=len(texts), source=source_note) + f"<br>{note}",
        )
        m.save(FIGURES_DIR / filename)

    # Статичная версия для README - на GitHub интерактивная карта не покажется
    plot_static_map(district_summary, places, FIGURES_DIR / "emotional_map.png",
                    lang=args.lang)

    out = FIGURES_DIR / "emotional_map.html"

    # ---------- отчёт ----------
    print(f"\nТекстов:     {len(texts)}")
    print(f"Заведений с оценкой (>=3 упоминаний): {len(place_summary)}")
    print(f"\nСводка по районам:")
    cols = ["name", "n", "net_sentiment", "share_pos", "share_neg", "opinionated"]
    print(district_summary[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    cols = ["name", "district", "n_texts", "net_sentiment",
            "net_sentiment_shrunk", "confidence"]
    print("\nТоп-8 заведений (сжатая оценка - по ней и сортируем):")
    print(place_summary.nlargest(8, "net_sentiment_shrunk")[cols]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nХудшие 8:")
    print(place_summary.nsmallest(8, "net_sentiment_shrunk")[cols]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\nФайлы в {PROCESSED_DIR}")
    print(f"Карта: {out}")


if __name__ == "__main__":
    main()
