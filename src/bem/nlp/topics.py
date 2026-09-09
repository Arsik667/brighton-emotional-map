"""
Topic modeling: о чём вообще пишут в каждом районе.

=== ЗАДАЧА ===
Sentiment говорит "в Кемптауне настроение хуже". Само по себе это
бесполезно: непонятно, что делать. Topic modeling отвечает на вопрос
"а из-за чего именно" - из-за шума, цен, грязи или чего-то ещё.
Вот эта связка "где + насколько плохо + почему" и делает проект
продуктом, а не просто раскрашенной картой.

=== ДВА ПОДХОДА ===

LDA (Latent Dirichlet Allocation), 2003 год:
  Статистическая модель. Считает, что каждый документ - смесь тем,
  а каждая тема - распределение вероятностей по словам. Работает
  с "мешком слов": порядок слов игнорируется полностью.
  + быстро, объяснимо, не нужна нейросеть
  + каждый документ получает доли ВСЕХ тем, а не одну метку
  - надо заранее назвать число тем, а откуда его знать?
  - плохо работает на коротких текстах - а у нас как раз такие
  - "banking" и "financial" для него разные несвязанные слова

BERTopic, 2022 год:
  Сначала превращает каждый текст в вектор через нейросеть (тексты,
  близкие по смыслу, получают близкие векторы), потом сжимает
  размерность (UMAP) и кластеризует (HDBSCAN), а имена темам даёт
  по словам, характерным для кластера (c-TF-IDF).
  + понимает синонимы и смысл, а не только совпадение слов
  + сам определяет число тем
  + честно отвечает "это шум" вместо того, чтобы впихнуть текст в тему
  - тяжелее и медленнее
  - результат зависит от случайности: два запуска дадут разные темы,
    если не зафиксировать random_state

Делаем оба и сравниваем - так же, как с sentiment.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Стоп-слова: служебные слова, которые есть в каждом тексте и потому
# ничего не различают.
#
# === НАЗВАНИЯ РАЙОНОВ ОБЯЗАТЕЛЬНО В СТОП-ЛИСТ ===
# Это не косметика, а исправление настоящей ошибки. Без них BERTopic
# нашёл "тему" под названием "north laine", в которую попал 71% текстов
# этого района. Модель честно сделала свою работу: слово "north laine"
# встречается в каждом тексте района и отлично их отделяет. Но это
# кластеризация ПО РАЙОНУ, а мы уже знаем район - нам нужно понять,
# О ЧЁМ там пишут.
#
# Это частный случай общей проблемы: если в тексте есть признак,
# однозначно определяющий группу, модель ухватится за него и не станет
# искать ничего более тонкого. В ML это называют утечкой (leakage).
DISTRICT_STOPWORDS = {
    "north", "laine", "lanes", "lane", "kemptown", "kemp", "seafront",
    "hove", "brighton", "town", "district", "area", "street", "road",
}

EXTRA_STOPWORDS = DISTRICT_STOPWORDS | {
    "brighton", "hove", "place", "just", "really", "got", "get", "going",
    "went", "like", "one", "would", "could", "also", "even", "still",
    "bit", "quite", "pretty", "lot", "thing", "things", "way", "time",
    "day", "week", "weekend", "year", "years", "people", "anyone",
    "anything", "something", "everything", "much", "many", "back",
    "around", "though", "actually", "honestly", "sure", "know", "think",
    "said", "say", "says", "im", "ive", "dont", "didnt", "cant", "isnt",
    # обрывки, которые остаются от вводных оборотов и сокращений
    "fair", "first", "visiting", "popped", "home", "mixed", "quick",
    "update", "asking", "been", "ve", "wasn", "wouldn", "couldn",
    "isn", "don", "didn", "doesn", "won", "hasn", "aren",
}


def build_stopwords() -> list[str]:
    """Английские стоп-слова плюс наши собственные."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return sorted(set(ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS)


# ----------------------------------------------------------------- LDA ----

class LdaTopics:
    """
    Классический LDA на sklearn.

    n_topics приходится задавать руками. Как выбрать - вопрос отдельный;
    в notebook мы переберём несколько значений и посмотрим на перплексию.
    """

    name = "lda"

    def __init__(self, n_topics: int = 10, max_features: int = 5000,
                 random_state: int = 42):
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer

        self.n_topics = n_topics

        # min_df=5: слово должно встретиться хотя бы в 5 текстах, иначе это
        #           опечатка или случайность, и тема из него не выйдет.
        # max_df=0.5: слово, которое есть в половине текстов, ничего не
        #           различает - выбрасываем, даже если оно не в стоп-листе.
        # ngram_range=(1,2): берём и отдельные слова, и пары. "car park"
        #           как пара осмысленнее, чем "car" и "park" по отдельности.
        self.vectorizer = CountVectorizer(
            stop_words=build_stopwords(),
            max_features=max_features,
            min_df=5,
            max_df=0.5,
            ngram_range=(1, 2),
        )

        self.model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            learning_method="batch",
            max_iter=20,
        )

    def fit_transform(self, texts) -> np.ndarray:
        """Обучиться и вернуть матрицу "документ x доля темы"."""
        counts = self.vectorizer.fit_transform(texts)
        logger.info("LDA: словарь %d слов, документов %d",
                    len(self.vectorizer.get_feature_names_out()), counts.shape[0])
        return self.model.fit_transform(counts)

    def top_words(self, n_words: int = 10) -> dict[int, list[str]]:
        """Самые характерные слова каждой темы."""
        vocab = self.vectorizer.get_feature_names_out()
        result = {}
        for topic_id, weights in enumerate(self.model.components_):
            # argsort сортирует по возрастанию, поэтому берём хвост и разворачиваем
            top_indices = weights.argsort()[: -n_words - 1 : -1]
            result[topic_id] = [vocab[i] for i in top_indices]
        return result

    def perplexity(self, texts) -> float:
        """
        Перплексия: насколько модель "удивлена" данными. Меньше - лучше.

        Осторожно: это техническая метрика, и она НЕ всегда согласуется
        с человеческой осмысленностью тем. Известный результат: модели с
        лучшей перплексией часто дают темы, которые человеку кажутся мусором.
        Поэтому смотрим и на неё, и на сами слова глазами.
        """
        return self.model.perplexity(self.vectorizer.transform(texts))


# ------------------------------------------------------------ BERTopic ----

class BertTopics:
    """
    BERTopic: эмбеддинги + кластеризация.

    min_topic_size - сколько текстов должно быть в кластере, чтобы он
    считался темой. Слишком маленькое значение даёт десятки крошечных
    "тем" из случайных совпадений; слишком большое схлопывает всё в
    два-три общих кластера.
    """

    name = "bertopic"

    def __init__(self, min_topic_size: int = 40, random_state: int = 42,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        from bertopic import BERTopic
        from bertopic.vectorizers import ClassTfidfTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP

        # UMAP по умолчанию недетерминирован: каждый запуск даёт другие темы.
        # random_state делает результат воспроизводимым. Без этого нельзя
        # ни отладить пайплайн, ни показать кому-то тот же результат.
        umap_model = UMAP(
            n_neighbors=15, n_components=5, min_dist=0.0,
            metric="cosine", random_state=random_state,
        )

        self.model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            vectorizer_model=CountVectorizer(
                stop_words=build_stopwords(), min_df=5, ngram_range=(1, 2)
            ),
            # reduce_frequent_words убирает слова, частые во всех темах:
            # без этого половина тем называется одинаковыми словами.
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
            min_topic_size=min_topic_size,
            calculate_probabilities=False,  # заметно быстрее
            verbose=False,
        )

    def fit_transform(self, texts) -> tuple[list[int], np.ndarray]:
        """
        Обучиться и разметить тексты.

        Тема -1 - это НЕ ошибка. Так BERTopic помечает выбросы: тексты,
        которые не легли ни в один кластер. Честное "не знаю" полезнее
        насильного отнесения к ближайшей теме.
        """
        topics, probs = self.model.fit_transform(list(texts))

        n_topics = len({t for t in topics if t != -1})
        outliers = sum(1 for t in topics if t == -1)
        logger.info(
            "BERTopic: найдено тем %d, выбросов %d (%.1f%%)",
            n_topics, outliers, outliers / len(topics) * 100,
        )
        return topics, probs

    def top_words(self, n_words: int = 10) -> dict[int, list[str]]:
        result = {}
        for topic_id in self.model.get_topics():
            words = self.model.get_topic(topic_id)
            result[topic_id] = [w for w, _score in words[:n_words]]
        return result


# ------------------------------------------------------- имена для тем ----

def name_topics(top_words: dict[int, list[str]],
                topic_keywords: dict[str, list[str]]) -> dict[int, str]:
    """
    Дать темам человеческие имена.

    Модель находит темы, но называет их списком слов вроде
    ['noise', 'loud', 'music', 'night']. На дашборде такое не покажешь.
    Сопоставляем список слов с нашими заранее заданными темами из
    config/text_keywords.yaml и берём ту, что пересекается сильнее всего.

    Это простой и прозрачный способ. Он не идеален, но у него есть
    важное достоинство: он объясним. Можно показать, ПОЧЕМУ тема
    названа именно так.
    """
    names = {}
    for topic_id, words in top_words.items():
        if topic_id == -1:
            names[topic_id] = "выбросы"
            continue

        word_set = {w.lower() for word in words for w in word.split()}

        best_name, best_overlap = None, 0
        for name, keywords in topic_keywords.items():
            overlap = len(word_set & {k.lower() for k in keywords})
            if overlap > best_overlap:
                best_name, best_overlap = name, overlap

        # Если ни с одной заготовленной темой не пересеклось - называем
        # первыми двумя словами самой темы. Честнее, чем врать.
        names[topic_id] = best_name or " / ".join(words[:2])

    return names


def topics_by_district(df: pd.DataFrame, topic_col: str,
                       district_col: str = "district") -> pd.DataFrame:
    """
    Доля каждой темы внутри каждого района.

    Считаем именно ДОЛЮ, а не количество. Районы разного размера, и в
    абсолютных числах самый большой район выиграет в любой теме -
    это скажет нам про размер выборки, а не про район.
    """
    counts = pd.crosstab(df[district_col], df[topic_col])
    return counts.div(counts.sum(axis=1), axis=0)


# ------------------------------------------------ простой baseline ----

def keyword_topics(texts, topic_keywords: dict[str, list[str]],
                   min_hits: int = 1) -> list[str]:
    """
    Определить тему по совпадению ключевых слов. Самый простой метод.

    === ЗАЧЕМ ОН НУЖЕН, ЕСЛИ ЕСТЬ BERTopic ===
    Правило, которое стоит усвоить рано: сначала делай самое тупое
    решение, потом сравнивай с ним умное. Иногда выясняется, что умное
    выигрывает совсем немного, а стоит в сто раз дороже и не объяснимо.

    У этого метода есть три свойства, которых нет у BERTopic:
      * он полностью предсказуем - одинаковый вход даёт одинаковый выход;
      * его решение можно объяснить: "тема noise, потому что в тексте
        встретились слова loud и music";
      * он не требует ни обучения, ни GPU, ни времени.

    Недостаток очевиден: он видит только те слова, которые мы написали
    в config/text_keywords.yaml. Синонима, о котором мы не подумали,
    для него не существует. Именно это и должен побеждать BERTopic -
    вопрос лишь в том, побеждает ли он на самом деле.
    """
    import re

    patterns = {
        topic: [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in keywords]
        for topic, keywords in topic_keywords.items()
    }

    result = []
    for text in texts:
        if not isinstance(text, str):
            result.append("unknown")
            continue

        # Считаем, сколько ключевых слов каждой темы встретилось
        hits = {
            topic: sum(1 for p in pats if p.search(text))
            for topic, pats in patterns.items()
        }
        best_topic = max(hits, key=hits.get)
        result.append(best_topic if hits[best_topic] >= min_hits else "unknown")

    return result
