"""
Подписи для графиков на двух языках.

Зачем отдельный модуль: репозиторий двуязычный (README.md английский,
README.ru.md русский), а графики - общие. Хардкодить подписи прямо
в коде рисования означает, что для смены языка придётся править
пять файлов и что-нибудь обязательно забудется.

По умолчанию английский: README на GitHub английский, и графики
в нём должны читаться теми же людьми.

Комментарии и вывод в консоль остаются русскими - это рабочий язык
автора, а не язык публикации.
"""

from __future__ import annotations

DEFAULT_LANG = "en"

MONTHS = {
    "en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "ru": ["янв", "фев", "мар", "апр", "май", "июн",
           "июл", "авг", "сен", "окт", "ноя", "дек"],
}

LABELS = {
    "en": {
        # общее
        "net_sentiment": "Net sentiment (share positive - share negative)",
        "pride": " Pride",
        "fringe": " Fringe",
        "venues_osm": "venues (OpenStreetMap)",
        "longitude": "longitude",
        "latitude": "latitude",

        # сезонность настроений
        "seasonality_title": "Seasonal sentiment by Brighton district",

        # помесячная динамика
        "trend_title": (
            "Monthly sentiment dynamics\n"
            "thin line - raw data, thick - 3-month rolling mean, "
            "shaded band - 95% confidence interval"
        ),
        "low_n": "n < {threshold}",

        # карта
        "map_title": "Emotional Map of Brighton",
        "scale_relative": (
            "relative scale {lo:+.2f}...{hi:+.2f} - red means "
            "\"lowest of the districts\", not \"bad\""
        ),
        "scale_absolute": "absolute scale, zero = neutral",
        "map_districts_layer": "Sentiment by district",
        "map_heat_layer": "Mention density",
        "map_places_layer": "Venues",
        "map_tooltip_district": "District",
        "map_tooltip_value": "Sentiment",
        "map_tooltip_n": "Texts",
        "map_tooltip_pos": "Share positive",
        "map_tooltip_neg": "Share negative",
        "map_popup_sentiment": "sentiment",
        "map_popup_mentions": "mentions",
        "map_subtitle": (
            "{n} texts - {source}<br>Layers can be toggled in the top right corner"
        ),
        "source_demo": "DEMO DATA (synthetic)",
        "source_real": "Reddit + OpenStreetMap data",
        "map_note_absolute": (
            "Absolute scale: zero = neutral. All districts are positive, "
            "so all are green - the differences between them are small."
        ),
        "map_note_relative": (
            "RELATIVE scale, stretched to the data range. "
            "Red means \"lowest of the five\", NOT \"bad\"."
        ),

        # police.uk
        "police_title": (
            "REAL data: seasonality of incidents affecting district atmosphere\n"
            "source: Police.uk, {months} months, {n} incidents"
        ),
        "police_ylabel": (
            "level relative to the district's own average\n(1.0 = a normal month)"
        ),
    },
    "ru": {
        "net_sentiment": "Net sentiment (доля позитива - доля негатива)",
        "pride": " Pride",
        "fringe": " Fringe",
        "venues_osm": "заведения (OpenStreetMap)",
        "longitude": "долгота",
        "latitude": "широта",

        "seasonality_title": "Сезонность настроений по районам Брайтона",

        "trend_title": (
            "Динамика настроений по месяцам\n"
            "тонкая линия - данные, толстая - скользящее среднее (3 мес), "
            "заливка - 95% доверительный интервал"
        ),
        "low_n": "n < {threshold}",

        "map_title": "Emotional Map of Brighton",
        "scale_relative": (
            "относительная шкала {lo:+.2f}...{hi:+.2f} - красный значит "
            "«худший из районов», а не «плохой»"
        ),
        "scale_absolute": "абсолютная шкала, ноль = нейтрально",
        "map_districts_layer": "Настроение по районам",
        "map_heat_layer": "Плотность упоминаний",
        "map_places_layer": "Заведения",
        "map_tooltip_district": "Район",
        "map_tooltip_value": "Настроение",
        "map_tooltip_n": "Текстов",
        "map_tooltip_pos": "Доля позитива",
        "map_tooltip_neg": "Доля негатива",
        "map_popup_sentiment": "настроение",
        "map_popup_mentions": "упоминаний",
        "map_subtitle": (
            "{n} текстов - {source}<br>Слои переключаются в правом верхнем углу"
        ),
        "source_demo": "ДЕМО-ДАННЫЕ (синтетические)",
        "source_real": "данные Reddit + OpenStreetMap",
        "map_note_absolute": (
            "Абсолютная шкала: ноль = нейтрально. Все районы в плюсе, "
            "поэтому все зелёные - различия между ними невелики."
        ),
        "map_note_relative": (
            "ОТНОСИТЕЛЬНАЯ шкала, растянута на диапазон данных. "
            "Красный означает «худший из пяти», а НЕ «плохой»."
        ),

        "police_title": (
            "РЕАЛЬНЫЕ данные: сезонность происшествий, влияющих на атмосферу района\n"
            "источник: Police.uk, {months} месяцев, {n} инцидентов"
        ),
        "police_ylabel": (
            "уровень относительно среднего по району\n(1.0 = обычный месяц)"
        ),
    },
}


def get_labels(lang: str = DEFAULT_LANG) -> dict:
    """Словарь подписей для языка. Неизвестный язык - откат на английский."""
    return LABELS.get(lang, LABELS[DEFAULT_LANG])


def get_months(lang: str = DEFAULT_LANG) -> list[str]:
    """Короткие названия месяцев."""
    return MONTHS.get(lang, MONTHS[DEFAULT_LANG])
