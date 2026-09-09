"""
Этап 5: topic modeling. LDA против BERTopic.

Запуск:
    PYTHONPATH=src .venv/bin/python scripts/04_topics.py
    PYTHONPATH=src .venv/bin/python scripts/04_topics.py --skip-bertopic
    PYTHONPATH=src .venv/bin/python scripts/04_topics.py --n-topics 12

Результат:
    data/interim/texts_topics.csv  - тексты с темами обеих моделей
    reports/topics_comparison.md   - сравнение для README
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
from bem.nlp.text_linking import load_topic_keywords
from bem.nlp.topics import (
    BertTopics, LdaTopics, keyword_topics, name_topics, topics_by_district,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger("topics")

INPUT = INTERIM_DIR / "texts_sentiment.csv"
OUTPUT = INTERIM_DIR / "texts_topics.csv"
REPORT = REPORTS_DIR / "topics_comparison.md"


def cluster_agreement(pred, truth) -> dict:
    """
    Насколько найденные темы совпадают с настоящими.

    Обычную accuracy тут применять нельзя: модель не знает наших названий
    тем и нумерует кластеры произвольно. Тема №3 у модели может полностью
    соответствовать нашей "noise", но по номерам они не совпадут.

    Поэтому берём метрики, которым имена не важны - они сравнивают
    РАЗБИЕНИЕ на группы, а не метки:

      ARI (Adjusted Rand Index): доля пар текстов, про которые модель
          и истина согласны ("вместе" или "порознь"). 0 = как случайно,
          1 = идеально.
      NMI (Normalized Mutual Information): сколько информации о настоящей
          теме содержится в предсказанной. От 0 до 1.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    return {
        "ARI": adjusted_rand_score(truth, pred),
        "NMI": normalized_mutual_info_score(truth, pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-topics", type=int, default=10, help="число тем для LDA")
    # 150 подобрано так, чтобы BERTopic давал ~10 тем, сопоставимых
    # с LDA. При 40 он дробит корпус на 40+ мелких тем-шаблонов.
    parser.add_argument("--min-topic-size", type=int, default=150)
    parser.add_argument("--skip-bertopic", action="store_true")
    args = parser.parse_args()

    setup_logging()
    ensure_dirs()

    if not INPUT.exists():
        logger.error("Нет %s - сначала запусти scripts/03_sentiment.py", INPUT)
        return

    df = pd.read_csv(INPUT, parse_dates=["created_utc"])
    texts = df["text"].fillna("").tolist()
    logger.info("Текстов: %d", len(texts))

    topic_keywords = load_topic_keywords()
    lines = [
        "# Сравнение моделей тем",
        "",
        "> **Важная оговорка о корректности сравнения.**",
        "> Если запуск сделан на демо-корпусе, baseline по ключевым словам",
        "> имеет НЕСПРАВЕДЛИВОЕ преимущество: шаблоны демо-текстов писались",
        "> той же лексикой, что лежит в `config/text_keywords.yaml`.",
        "> Сравнение получается частично круговым. Считать его доказательством",
        "> превосходства словарного метода нельзя - нужен прогон на настоящих",
        "> данных с Reddit, где такой связи нет.",
        "",
        "> Вторая оговорка: низкий ARI не означает, что модель бесполезна.",
        "> ARI считает совпадение по каждому документу, а продукту нужен",
        "> АГРЕГАТ по району. Профили районов BERTopic восстанавливает",
        "> заметно лучше, чем можно предположить по ARI.",
        "",
    ]
    results = []

    # ------------------------------------------ baseline по словарю ----
    # Всегда начинай с самого простого метода: если сложный его не
    # обгоняет, сложный не нужен.
    t0 = time.perf_counter()
    df["keyword_topic"] = keyword_topics(texts, topic_keywords)
    keyword_time = time.perf_counter() - t0

    print(f"\n{'='*70}\nBaseline по ключевым словам: {keyword_time:.2f} с\n{'='*70}")
    for topic, share in df["keyword_topic"].value_counts(normalize=True).items():
        print(f"  {topic:<16} {share:5.1%}")
    results.append(("Ключевые слова", keyword_time,
                    df["keyword_topic"].nunique(), "keyword_topic"))

    # ------------------------------------------------------------ LDA ----
    t0 = time.perf_counter()
    lda = LdaTopics(n_topics=args.n_topics)
    doc_topics = lda.fit_transform(texts)
    # argmax: у LDA каждый документ - смесь тем, берём самую весомую
    df["lda_topic"] = doc_topics.argmax(axis=1)
    lda_time = time.perf_counter() - t0

    lda_words = lda.top_words()
    lda_names = name_topics(lda_words, topic_keywords)
    df["lda_topic_name"] = df["lda_topic"].map(lda_names)

    print(f"\n{'='*70}\nLDA: {args.n_topics} тем за {lda_time:.1f} с\n{'='*70}")
    for topic_id, words in lda_words.items():
        share = (df["lda_topic"] == topic_id).mean()
        print(f"  [{topic_id:>2}] {lda_names[topic_id]:<14} {share:5.1%}  {', '.join(words[:8])}")

    results.append(("LDA", lda_time, args.n_topics, "lda_topic"))

    # ------------------------------------------------------- BERTopic ----
    if not args.skip_bertopic:
        t0 = time.perf_counter()
        bert = BertTopics(min_topic_size=args.min_topic_size)
        topics, _ = bert.fit_transform(texts)
        df["bertopic_topic"] = topics
        bert_time = time.perf_counter() - t0

        bert_words = bert.top_words()
        bert_names = name_topics(bert_words, topic_keywords)
        df["bertopic_topic_name"] = df["bertopic_topic"].map(bert_names)

        n_found = len([t for t in bert_words if t != -1])
        print(f"\n{'='*70}\nBERTopic: {n_found} тем за {bert_time:.1f} с\n{'='*70}")
        for topic_id in sorted(bert_words):
            share = (df["bertopic_topic"] == topic_id).mean()
            marker = "  (выбросы)" if topic_id == -1 else ""
            print(f"  [{topic_id:>2}] {bert_names[topic_id]:<14} {share:5.1%}  "
                  f"{', '.join(bert_words[topic_id][:8])}{marker}")

        results.append(("BERTopic", bert_time, n_found, "bertopic_topic"))

    # -------------------------------------------------- сравнение ----
    if "true_topic" in df.columns:
        print(f"\n{'='*70}\nСовпадение с настоящими темами\n{'='*70}")
        print("ARI и NMI: 0 = случайно, 1 = идеально\n")

        lines += ["| модель | тем | секунд | ARI | NMI |", "|---|---|---|---|---|"]
        for label, elapsed, n_topics, col in results:
            # Выбросы BERTopic исключаем: модель честно сказала "не знаю",
            # и наказывать её за это как за ошибку было бы некорректно.
            mask = (df[col] != -1) & (df[col] != "unknown")
            m = cluster_agreement(df.loc[mask, col], df.loc[mask, "true_topic"])
            covered = mask.mean()
            print(f"  {label:<10} тем={n_topics:>2}  ARI={m['ARI']:.3f}  "
                  f"NMI={m['NMI']:.3f}  покрытие={covered:.0%}  {elapsed:5.1f} с")
            lines.append(
                f"| {label} | {n_topics} | {elapsed:.1f} | {m['ARI']:.3f} | {m['NMI']:.3f} |"
            )

    # ------------------------------------------- темы по районам ----
    topic_col = "lda_topic_name" if args.skip_bertopic else "bertopic_topic_name"
    print(f"\n{'='*70}\nТемы по районам (по {topic_col})\n{'='*70}")

    shares = topics_by_district(df[df[topic_col] != "выбросы"], topic_col)
    shares.index = [district_names().get(i, i) for i in shares.index]
    print(shares.to_string(float_format=lambda x: f"{x:.2f}"))

    print("\nЧем каждый район ВЫДЕЛЯЕТСЯ (тема чаще, чем в среднем по городу):")
    overall = shares.mean(axis=0)
    for district in shares.index:
        diff = (shares.loc[district] - overall).sort_values(ascending=False)
        top = ", ".join(f"{t} +{v*100:.0f}%" for t, v in diff.head(3).items() if v > 0.01)
        print(f"  {district:<14} {top}")

    df.to_csv(OUTPUT, index=False)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Сохранено: %s", OUTPUT)


if __name__ == "__main__":
    main()
