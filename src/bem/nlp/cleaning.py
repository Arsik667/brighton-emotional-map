"""
Очистка текстов перед анализом.

Что именно чистим и почему:

  * URL - ссылка не несёт тональности, но её слова ("best-deals-brighton")
    попадут в темы и всё испортят;
  * markdown-разметка Reddit (**жирный**, >цитата) - мусор для моделей;
  * заглушки [deleted] и [removed] - это не тексты, это следы удалённых;
  * слишком короткие тексты - по "lol" или "this" ничего не определить,
    но они раздувают выборку и портят статистику;
  * дубликаты - на Reddit один и тот же текст встречается в кросс-постах.

Чего НЕ делаем: не приводим к нижнему регистру, не убираем пунктуацию,
не лемматизируем. Это принципиально. VADER специально учитывает
ЗАГЛАВНЫЕ БУКВЫ как усиление эмоции, а восклицательный знак - как
её интенсивность. DistilBERT тоже обучался на нормальном тексте.
Агрессивная предобработка, привычная по классическим NLP-курсам,
современным моделям только вредит.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")  # [текст](ссылка) -> текст
QUOTE_RE = re.compile(r"^&gt;.*$|^>.*$", re.MULTILINE)   # цитаты
MULTISPACE_RE = re.compile(r"\s+")

PLACEHOLDERS = {"[deleted]", "[removed]", "", "nan"}

MIN_LENGTH = 25   # символов
MAX_LENGTH = 4000  # очень длинные посты обрезаем: у моделей есть лимит


def clean_text(text: str) -> str:
    """Очистить один текст. Возвращает пустую строку, если текст негодный."""
    if not isinstance(text, str):
        return ""

    if text.strip().lower() in PLACEHOLDERS:
        return ""

    text = MARKDOWN_LINK_RE.sub(r"\1", text)  # ссылку заменяем её текстом
    text = URL_RE.sub(" ", text)
    text = QUOTE_RE.sub(" ", text)

    # HTML-сущности, которые Reddit отдаёт в API
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&#39;", "'").replace("&quot;", '"')

    # Markdown-выделение: убираем символы, оставляем слова
    text = re.sub(r"[*_~`]+", "", text)

    text = MULTISPACE_RE.sub(" ", text).strip()

    return text[:MAX_LENGTH]


def clean_frame(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Очистить DataFrame: почистить тексты, выбросить негодные и дубликаты.

    Логируем, сколько потеряли на каждом шаге. Это не косметика:
    если после очистки осталось 3% данных, ты хочешь узнать об этом
    сразу, а не на этапе построения графиков.
    """
    before = len(df)
    df = df.copy()

    df[text_col] = df[text_col].map(clean_text)

    df = df[df[text_col].str.len() >= MIN_LENGTH]
    after_short = len(df)

    df = df.drop_duplicates(subset=[text_col])
    after_dupes = len(df)

    logger.info(
        "Очистка: %d -> %d (коротких/пустых: %d, дубликатов: %d)",
        before, after_dupes, before - after_short, after_short - after_dupes,
    )
    return df.reset_index(drop=True)
