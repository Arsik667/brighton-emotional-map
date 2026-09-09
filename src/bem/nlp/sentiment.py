"""
Sentiment-анализ: две модели и честное сравнение между ними.

=== ПОЧЕМУ ДВЕ МОДЕЛИ ===
На собеседовании вопрос "почему именно эта модель?" звучит почти всегда.
Ответ "она популярная" - плохой. Ответ "я сравнил две на своих данных,
вот метрики, вот компромисс" - хороший. Поэтому делаем обе.

=== VADER (baseline) ===
Словарный метод: у каждого слова есть заранее проставленная оценка
тональности, плюс правила для усилителей ("very good" сильнее "good"),
отрицаний ("not good" -> негатив), капса и восклицательных знаков.

  + мгновенный, не нужен GPU, не нужно ничего скачивать
  + полностью объяснимый: видно, какое слово дало какой вклад
  + отлично работает на коротком неформальном тексте - его для соцсетей и делали
  - не понимает контекст и сарказм
  - спотыкается на смешанных отзывах ("еда супер, но очередь ужас")

=== DistilBERT (fine-tuned на SST-2) ===
Трансформер, дообученный на разметке отзывов. Читает предложение целиком
и учитывает порядок слов.

  + заметно точнее на сложных фразах
  - медленнее в сотни раз, нужно скачать ~260 МБ весов
  - выдаёт только positive/negative, без нейтрального класса
  - "чёрный ящик": объяснить конкретное решение трудно

=== ГЛАВНАЯ ЛОВУШКА ===
DistilBERT/SST-2 НЕ УМЕЕТ говорить "нейтрально". Он всегда выберет
positive или negative, даже для фразы "the cafe is on the corner".
Наивное сравнение с VADER, у которого нейтральный класс есть, даст
несправедливые цифры. Поэтому мы вводим порог уверенности: если модель
не уверена - считаем нейтральным. Это важная деталь, о которой junior'ы
обычно забывают, а на собеседовании её любят.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Порог для VADER, рекомендованный авторами метода.
# compound лежит в [-1, 1]; всё между -0.05 и 0.05 считается нейтральным.
VADER_THRESHOLD = 0.05

# Порог уверенности для DistilBERT. Если модель дала positive с
# вероятностью 0.6 - это не уверенность, это "не знаю". Считаем нейтральным.
#
# Значение 0.995 не взято с потолка: оно подобрано перебором в
# scripts/03_sentiment.py. Разница огромна - macro-F1 растёт с 0.599
# (при интуитивном 0.85) до 0.726. Причина в том, что DistilBERT
# патологически самоуверен: обученный на двух классах, он выдаёт
# вероятности вида 0.97-0.99 даже для откровенно нейтральных фраз
# ("The cafe is on the corner"). Отсекать приходится очень высоко.
#
# ВАЖНО: порог подобран на демо-корпусе. На настоящих данных с Reddit
# его надо подобрать заново - там другое распределение текстов.
TRANSFORMER_NEUTRAL_BAND = 0.995

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


@dataclass
class SentimentResult:
    """Результат для одного текста: метка и непрерывная оценка."""

    label: str    # "pos" | "neu" | "neg"
    score: float  # от -1 (негатив) до +1 (позитив)


# ---------------------------------------------------------------- VADER ----

class VaderSentiment:
    """Словарный sentiment-анализ. Baseline, с которым сравниваем остальное."""

    name = "vader"

    def __init__(self, threshold: float = VADER_THRESHOLD):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self.analyzer = SentimentIntensityAnalyzer()
        self.threshold = threshold

    def score_one(self, text: str) -> SentimentResult:
        if not isinstance(text, str) or not text.strip():
            return SentimentResult("neu", 0.0)

        # compound - итоговая оценка, уже нормированная в [-1, 1]
        compound = self.analyzer.polarity_scores(text)["compound"]

        if compound >= self.threshold:
            label = "pos"
        elif compound <= -self.threshold:
            label = "neg"
        else:
            label = "neu"

        return SentimentResult(label, compound)

    def score_many(self, texts) -> pd.DataFrame:
        results = [self.score_one(t) for t in texts]
        return pd.DataFrame(
            {
                f"{self.name}_label": [r.label for r in results],
                f"{self.name}_score": [r.score for r in results],
            }
        )


# ----------------------------------------------------------- DistilBERT ----

class TransformerSentiment:
    """
    DistilBERT, дообученный на SST-2.

    Модель скачивается при первом запуске (~260 МБ) и кэшируется
    в ~/.cache/huggingface. Второй раз сеть уже не нужна.
    """

    name = "distilbert"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        neutral_band: float = TRANSFORMER_NEUTRAL_BAND,
        batch_size: int = 32,
        device: str | None = None,
    ):
        from transformers import pipeline
        import torch

        # На Mac с Apple Silicon есть MPS - это встроенный GPU.
        # Он ускоряет инференс в несколько раз по сравнению с CPU.
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        logger.info("Загружаю %s на устройство %s", model_name, device)

        self.pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device,
            truncation=True,   # длинные тексты обрезаем: у модели лимит 512 токенов
            max_length=512,
        )
        self.neutral_band = neutral_band
        self.batch_size = batch_size

    def score_many(self, texts) -> pd.DataFrame:
        """
        Обработать список текстов пачками.

        Батчинг важен: по одному тексту за раз модель работает в разы
        медленнее, потому что не использует параллелизм.
        """
        texts = [t if isinstance(t, str) and t.strip() else "n/a" for t in texts]

        raw = self.pipe(texts, batch_size=self.batch_size)

        labels, scores = [], []
        for item in raw:
            confidence = item["score"]          # вероятность выбранного класса
            is_positive = item["label"] == "POSITIVE"

            # Переводим в единую шкалу [-1, 1]
            signed = confidence if is_positive else -confidence

            # Модель не умеет говорить "нейтрально", поэтому решаем за неё:
            # низкая уверенность = нейтральный текст.
            if confidence < self.neutral_band:
                label = "neu"
            else:
                label = "pos" if is_positive else "neg"

            labels.append(label)
            scores.append(signed)

        return pd.DataFrame(
            {f"{self.name}_label": labels, f"{self.name}_score": scores}
        )


# ------------------------------------------------------------- метрики ----

def evaluate(pred_labels, true_labels, model_name: str = "model") -> dict:
    """
    Посчитать метрики качества.

    Почему не только accuracy: классы у нас несбалансированы (позитива
    заметно больше, чем негатива). Модель, которая всегда отвечает "pos",
    получит приличную accuracy и будет совершенно бесполезной.
    Macro-F1 усредняет F1 по классам и такой обман не пропускает.
    """
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score,
    )

    labels = ["neg", "neu", "pos"]

    return {
        "model": model_name,
        "accuracy": accuracy_score(true_labels, pred_labels),
        "macro_f1": f1_score(true_labels, pred_labels, average="macro", labels=labels, zero_division=0),
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(true_labels, pred_labels, labels=labels),
            index=[f"истина={l}" for l in labels],
            columns=[f"предсказано={l}" for l in labels],
        ),
        "report": classification_report(
            true_labels, pred_labels, labels=labels, zero_division=0, digits=3
        ),
    }


def to_district_scores(df: pd.DataFrame, label_col: str,
                       score_col: str | None = None,
                       district_col: str = "district") -> pd.DataFrame:
    """
    Агрегировать тональность по районам.

    === ЧТО ЗДЕСЬ ПРОИЗОШЛО (полезная история) ===
    Сначала эта функция считала mean, median и "поляризацию" как разницу
    между ними. Медиана оказалась бесполезной: у двухклассового
    классификатора оценки прижаты к краям (почти все около +1 или -1),
    распределение бимодальное, и медиана равнялась +0.96..+0.99 во ВСЕХ
    районах. Все районы выглядели одинаково, а "поляризация" измеряла
    не расхождение мнений, а форму распределения. Это была настоящая ошибка.

    Дальше я предположил, что и среднее плохо, и заменил его на
    Net Sentiment. Проверка на данных это НЕ подтвердила: на нашем
    корпусе среднее восстанавливает истинный порядок районов не хуже
    (корреляция Спирмена с заложенной истиной 0.80 против 0.50 у net).
    Поэтому считаем обе метрики и не выбрасываем среднее.

    Оговорка, без которой цифры выше нельзя воспринимать всерьёз:
    районов всего 5, и ни одно из этих различий не значимо (p > 0.1).
    На пяти точках выбрать лучшую метрику статистически невозможно.
    Это тот случай, когда честный ответ - "обе разумны, показываем обе".

    === ЧТО СЧИТАЕМ ===
    net_sentiment: доля позитива минус доля негатива, от -1 до +1.
        Считается по МЕТКАМ, легко объясняется нетехническому человеку:
        +0.3 = "позитивных высказываний на 30% больше, чем негативных".
    mean_score: среднее по непрерывным оценкам модели.
        Хуже интерпретируется, но чувствительнее к слабым различиям.
    opinionated: доля тех, кто высказался определённо (не нейтрально).
        Район, где спорят, и район, где пожимают плечами, - разные районы.
    """
    counts = (
        df.groupby([district_col, label_col]).size()
        .unstack(fill_value=0)
        .reindex(columns=["neg", "neu", "pos"], fill_value=0)
    )

    total = counts.sum(axis=1)
    result = pd.DataFrame(
        {
            "n": total,
            "share_pos": counts["pos"] / total,
            "share_neu": counts["neu"] / total,
            "share_neg": counts["neg"] / total,
        }
    )
    # Главная метрика: насколько позитива больше, чем негатива
    result["net_sentiment"] = result["share_pos"] - result["share_neg"]
    # Насколько люди вообще склонны высказываться определённо
    result["opinionated"] = result["share_pos"] + result["share_neg"]

    if score_col is not None:
        result["mean_score"] = df.groupby(district_col)[score_col].mean()

    return result.sort_values("net_sentiment", ascending=False)
