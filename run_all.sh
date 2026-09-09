#!/usr/bin/env bash
# Полный пайплайн от сырых данных до дашборда.
# Запуск:  ./run_all.sh
set -euo pipefail

export PYTHONPATH=src
PY=.venv/bin/python

echo "==> 1/6  Сбор заведений из OpenStreetMap"
$PY scripts/01_collect_places.py

echo "==> 2/6  Сбор текстов"
$PY scripts/02_collect_texts.py

echo "==> 3/6  Sentiment-анализ (VADER + DistilBERT)"
$PY scripts/03_sentiment.py

echo "==> 4/6  Темы (ключевые слова + LDA + BERTopic)"
$PY scripts/04_topics.py

echo "==> 5/6  Временные ряды и финальные датасеты"
$PY scripts/05_timeseries.py
$PY scripts/06_build_outputs.py

echo "==> 6/6  Реальные данные: происшествия Police.uk (без ключей)"
$PY scripts/07_collect_police.py

echo
echo "Готово. Дашборд:  PYTHONPATH=src .venv/bin/streamlit run app/streamlit_app.py"
