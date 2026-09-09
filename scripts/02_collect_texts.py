"""
Этап 3: собрать тексты о районах Брайтона и привязать их к районам.

Два источника:
  reddit - настоящие посты и комментарии из r/Brighton (нужны ключи в .env)
  demo   - синтетический корпус для разработки без ключей

Запуск:
    # сам решит: есть ключи -> reddit, нет -> demo
    PYTHONPATH=src .venv/bin/python scripts/02_collect_texts.py

    # явно
    PYTHONPATH=src .venv/bin/python scripts/02_collect_texts.py --source demo
    PYTHONPATH=src .venv/bin/python scripts/02_collect_texts.py --source reddit

Результат:
    data/raw/reddit_texts.jsonl  - сырые посты (только для reddit)
    data/interim/texts.csv       - чистая таблица текстов с районами
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from bem.collect.demo_corpus import generate
from bem.config import INTERIM_DIR, RAW_DIR, ensure_dirs, get_secret
from bem.geo.districts import district_names
from bem.logging_setup import setup_logging
from bem.nlp.cleaning import clean_frame
from bem.nlp.text_linking import link_frame

logger = logging.getLogger("collect_texts")

PLACES_CSV = INTERIM_DIR / "places.csv"
OUTPUT = INTERIM_DIR / "texts.csv"
REDDIT_JSONL = RAW_DIR / "reddit_texts.jsonl"


def have_reddit_keys() -> bool:
    return bool(get_secret("REDDIT_CLIENT_ID") and get_secret("REDDIT_CLIENT_SECRET"))


def collect_reddit(args, places: pd.DataFrame) -> pd.DataFrame:
    """Собрать с Reddit и привязать тексты к районам по словам."""
    from bem.collect.reddit_texts import collect

    if not REDDIT_JSONL.exists() or args.force:
        collect(limit_per_query=args.limit, comments_per_submission=args.comments)
    else:
        logger.info("Использую уже собранное: %s (--force чтобы перекачать)", REDDIT_JSONL)

    records = [json.loads(line) for line in open(REDDIT_JSONL, encoding="utf-8")]
    df = pd.DataFrame(records)
    logger.info("Загружено записей из Reddit: %d", len(df))

    df = df.rename(columns={"id": "text_id"})
    df = clean_frame(df)
    df = link_frame(df, places)

    # Категорию заведения подтягиваем там, где смогли определить место
    df = df.merge(
        places[["place_id", "category"]], on="place_id", how="left"
    )
    return df


def collect_demo(args, places: pd.DataFrame) -> pd.DataFrame:
    """Сгенерировать демо-корпус."""
    logger.warning(
        "Источник DEMO: тексты СИНТЕТИЧЕСКИЕ. Годятся для отладки пайплайна, "
        "но не для выводов о настоящем Брайтоне."
    )
    df = generate(n_texts=args.n_demo, seed=args.seed)
    df = clean_frame(df)
    df["link_method"] = "generated"
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["auto", "reddit", "demo"], default="auto")
    parser.add_argument("--force", action="store_true", help="перекачать Reddit заново")
    parser.add_argument("--limit", type=int, default=60, help="постов на один запрос")
    parser.add_argument("--comments", type=int, default=40, help="комментариев на пост")
    parser.add_argument("--n-demo", type=int, default=9000,
                        help="сколько демо-текстов сгенерировать до дедупликации")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    ensure_dirs()

    if not PLACES_CSV.exists():
        logger.error("Нет %s - сначала запусти scripts/01_collect_places.py", PLACES_CSV)
        return

    places = pd.read_csv(PLACES_CSV)

    source = args.source
    if source == "auto":
        source = "reddit" if have_reddit_keys() else "demo"
        logger.info("Источник выбран автоматически: %s", source)

    df = collect_reddit(args, places) if source == "reddit" else collect_demo(args, places)

    if df.empty:
        logger.error("Ничего не собралось.")
        return

    # Единый набор колонок независимо от источника - дальше по пайплайну
    # код не должен знать, откуда пришли данные.
    columns = [
        "text_id", "source", "kind", "created_utc", "text", "score",
        "author_hash", "district", "link_method", "place_id", "place_name",
        "lat", "lon", "category",
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = None
    extra = [c for c in ("true_polarity", "true_topic", "is_mixed") if c in df.columns]

    df = df[columns + extra]
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True, format="mixed")
    df = df.sort_values("created_utc").reset_index(drop=True)
    df.to_csv(OUTPUT, index=False)

    # --- отчёт ---
    names = district_names()
    print(f"\nСобрано текстов: {len(df)}")
    print(f"Источник: {source}")
    print(f"Период: {df.created_utc.min():%Y-%m-%d} .. {df.created_utc.max():%Y-%m-%d}")
    print(f"Сохранено: {OUTPUT}\n")

    print("По районам:")
    for key, count in df["district"].value_counts().items():
        print(f"  {names.get(key, key):<28} {count:>6}  ({count/len(df)*100:4.1f}%)")

    print("\nКак привязали к району:")
    for method, count in df["link_method"].value_counts().items():
        print(f"  {method:<28} {count:>6}  ({count/len(df)*100:4.1f}%)")

    linked = (df["district"] != "other").mean()
    if linked < 0.3:
        logger.warning(
            "Только %.0f%% текстов привязано к району. Стоит расширить "
            "ключевые слова в config/text_keywords.yaml", linked * 100
        )


if __name__ == "__main__":
    main()
