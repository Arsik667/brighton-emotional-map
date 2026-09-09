"""
Интерактивные карты на Folium.

=== ТРИ СЛОЯ И ЗАЧЕМ КАЖДЫЙ ===

1. Хороплет районов - районы залиты цветом по среднему настроению.
   Отвечает на вопрос "где в городе лучше, а где хуже". Это главный слой.

2. Тепловая карта - плотность упоминаний, без учёта тональности.
   Отвечает на другой вопрос: "о чём вообще говорят". Полезно понимать,
   что яркое пятно тут означает МНОГО разговоров, а не хороших.

3. Точки заведений - конкретные места с их оценками.
   Нужны, чтобы от общей картины можно было провалиться в детали.

Слои сделаны переключаемыми: пользователь сам решает, что смотреть.
Всё сразу на одной карте - типичная ошибка, из-за которой не видно ничего.

=== ПРО ВЫБОР ЦВЕТА ===
Для настроения берём РАСХОДЯЩУЮСЯ (diverging) палитру: красный -
жёлтый - зелёный, с нейтральным цветом ровно на нуле. Это принципиально.
Последовательная палитра (от светлого к тёмному) для величины со
знаком не годится: по ней невозможно на глаз найти границу
"позитив/негатив", а это ровно то, что читатель ищет в первую очередь.
"""

from __future__ import annotations

import logging

import folium
import pandas as pd
from branca.colormap import LinearColormap
from folium.plugins import HeatMap, MarkerCluster

from bem.config import get_bbox
from bem.geo.districts import district_names, load_districts

logger = logging.getLogger(__name__)

# Бесплатные тайлы без API-ключа.
# ВНИМАНИЕ: CartoDB с 2025 года требует ключ, поэтому используем OSM.
DEFAULT_TILES = "OpenStreetMap"

# Расходящаяся палитра: красный (плохо) -> жёлтый (нейтрально) -> зелёный (хорошо)
SENTIMENT_COLORS = ["#d32f2f", "#f57c00", "#fdd835", "#9ccc65", "#2e7d32"]


def districts_geojson(district_stats: pd.DataFrame | None = None) -> dict:
    """
    Превратить полигоны районов в GeoJSON.

    GeoJSON - стандартный формат для гео-данных, его понимают все
    картографические библиотеки. Структура: FeatureCollection из Feature,
    у каждого есть geometry (координаты) и properties (любые атрибуты).

    Если передать district_stats, статистика попадёт в properties и
    её можно будет показать во всплывающей подсказке.
    """
    names = district_names()
    features = []

    for district in load_districts():
        properties = {
            "key": district.key,
            "name": names.get(district.key, district.key),
            "color": district.color,
        }

        if district_stats is not None and district.key in district_stats.index:
            row = district_stats.loc[district.key]
            properties.update(
                {col: (None if pd.isna(row[col]) else float(row[col]))
                 for col in district_stats.columns
                 if pd.api.types.is_numeric_dtype(district_stats[col])}
            )

        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {
                    "type": "Polygon",
                    # GeoJSON требует замкнутый контур: последняя точка
                    # совпадает с первой. shapely это уже обеспечивает.
                    "coordinates": [list(district.polygon.exterior.coords)],
                },
            }
        )

    return {"type": "FeatureCollection", "features": features}


def sentiment_colormap(vmin: float = -0.5, vmax: float = 0.5) -> LinearColormap:
    """Цветовая шкала для настроения с нейтральным цветом на нуле."""
    colormap = LinearColormap(
        colors=SENTIMENT_COLORS, vmin=vmin, vmax=vmax,
        caption="Net sentiment: доля позитива − доля негатива",
    )
    return colormap


def build_map(
    district_stats: pd.DataFrame,
    texts: pd.DataFrame | None = None,
    places: pd.DataFrame | None = None,
    value_col: str = "net_sentiment",
    zoom: int = 14,
    scale_mode: str = "absolute",
) -> folium.Map:
    """
    Собрать карту со всеми слоями.

    district_stats - индекс это ключ района, колонки - метрики.
    texts - тексты с координатами (для тепловой карты).
    places - заведения (для точечного слоя).

    scale_mode задаёт цветовую шкалу и это НЕ косметическая настройка,
    а вопрос честности картинки:

      "absolute" - шкала симметрична вокруг нуля и покрывает весь
          осмысленный диапазон. Если все районы слегка позитивны,
          все они будут зелёными и почти неразличимыми. Выглядит скучно,
          но говорит правду: различия между районами маленькие.

      "relative" - шкала растянута ровно на диапазон наших данных.
          Различия становятся видны, НО появляется опасность: район
          с +0.15 покрасится в красный просто потому, что он худший
          из пяти хороших. Пользователь прочитает это как "здесь плохо",
          хотя там позитив.

    По умолчанию "absolute": лучше скучная правда, чем красивая ложь.
    Режим "relative" уместен, когда рядом честно написано, что шкала
    относительная, и подписаны реальные значения.
    """
    center = get_bbox().center()
    m = folium.Map(location=center, zoom_start=zoom, tiles=DEFAULT_TILES)

    values = district_stats[value_col].dropna()

    if scale_mode == "relative" and len(values) and values.max() > values.min():
        # Растягиваем шкалу на фактический диапазон. Подпись обязана
        # сообщить, что сравнение относительное - иначе картинка врёт.
        colormap = sentiment_colormap(float(values.min()), float(values.max()))
        colormap.caption = (
            f"ОТНОСИТЕЛЬНАЯ шкала: {values.min():+.2f} … {values.max():+.2f}. "
            f"Красный = худший из районов, а не «плохой»."
        )
    else:
        # Симметричная шкала вокруг нуля: нейтральный цвет ровно на нуле,
        # иначе слабый позитив можно принять за негатив.
        limit = max(abs(values.min()), abs(values.max()), 0.1) if len(values) else 0.5
        colormap = sentiment_colormap(-limit, limit)

    # ---------- слой 1: районы ----------
    geojson = districts_geojson(district_stats)

    def style_function(feature):
        value = feature["properties"].get(value_col)
        return {
            "fillColor": colormap(value) if value is not None else "#cccccc",
            "color": "#37474f",
            "weight": 2,
            "fillOpacity": 0.65,
        }

    tooltip_fields = ["name", value_col]
    tooltip_aliases = ["Район", "Настроение"]
    for extra, alias in [("n", "Текстов"), ("share_pos", "Доля позитива"),
                         ("share_neg", "Доля негатива")]:
        if extra in district_stats.columns:
            tooltip_fields.append(extra)
            tooltip_aliases.append(alias)

    folium.GeoJson(
        geojson,
        name="Настроение по районам",
        style_function=style_function,
        highlight_function=lambda f: {"weight": 4, "fillOpacity": 0.8},
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields, aliases=tooltip_aliases, localize=True,
            style="background:#fff;border:1px solid #999;border-radius:4px;padding:6px;",
        ),
    ).add_to(m)

    colormap.add_to(m)

    # ---------- слой 2: тепловая карта ----------
    if texts is not None:
        located = texts.dropna(subset=["lat", "lon"])
        if len(located):
            HeatMap(
                located[["lat", "lon"]].values.tolist(),
                name="Плотность упоминаний",
                radius=14, blur=20, min_opacity=0.25, show=False,
            ).add_to(m)
            logger.info("Тепловая карта: %d точек", len(located))

    # ---------- слой 3: заведения ----------
    if places is not None and len(places):
        # MarkerCluster схлопывает близкие маркеры в кружок с числом.
        # Без него две тысячи точек превращаются в нечитаемое месиво
        # и браузер начинает тормозить.
        cluster = MarkerCluster(name="Заведения", show=False)

        for row in places.itertuples(index=False):
            popup = f"<b>{row.name}</b><br>{row.osm_type}"
            if hasattr(row, "net_sentiment") and pd.notna(row.net_sentiment):
                popup += f"<br>настроение: {row.net_sentiment:+.2f}"
            if hasattr(row, "n_texts") and pd.notna(row.n_texts):
                popup += f"<br>упоминаний: {int(row.n_texts)}"

            folium.CircleMarker(
                location=(row.lat, row.lon),
                radius=4,
                color=(colormap(row.net_sentiment)
                       if hasattr(row, "net_sentiment") and pd.notna(row.net_sentiment)
                       else "#777"),
                fill=True, fill_opacity=0.85, weight=1,
                popup=folium.Popup(popup, max_width=260),
                tooltip=row.name,
            ).add_to(cluster)

        cluster.add_to(m)

    # Переключатель слоёв - обязателен, когда слоёв больше одного
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def add_title(m: folium.Map, title: str, subtitle: str = "") -> folium.Map:
    """Добавить заголовок поверх карты."""
    html = (
        '<div style="position:fixed;top:12px;left:60px;z-index:9999;'
        'background:rgba(255,255,255,.94);padding:10px 16px;border-radius:6px;'
        'border:1px solid #ccc;font-family:system-ui;max-width:420px">'
        f'<div style="font-size:15px;font-weight:600">{title}</div>'
        + (f'<div style="font-size:11px;color:#555;margin-top:4px">{subtitle}</div>'
           if subtitle else "")
        + "</div>"
    )
    m.get_root().html.add_child(folium.Element(html))
    return m


def plot_static_map(district_stats: pd.DataFrame, places: pd.DataFrame | None,
                    path, value_col: str = "net_sentiment",
                    relative: bool = True) -> None:
    """
    Статичная карта в PNG - для README и презентаций.

    Интерактивная карта хороша, но в README на GitHub её не вставишь:
    там показываются только картинки. Поэтому дублируем ключевой слой
    обычным графиком.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Polygon as MplPolygon

    names = district_names()
    values = district_stats[value_col].dropna()

    cmap = LinearSegmentedColormap.from_list("sentiment", SENTIMENT_COLORS)
    if relative and len(values) and values.max() > values.min():
        norm = Normalize(vmin=values.min(), vmax=values.max())
        scale_note = (f"относительная шкала {values.min():+.2f}…{values.max():+.2f} - "
                      f"красный значит «худший из районов», а не «плохой»")
    else:
        limit = max(abs(values.min()), abs(values.max()), 0.1)
        norm = Normalize(vmin=-limit, vmax=limit)
        scale_note = "абсолютная шкала, ноль = нейтрально"

    fig, ax = plt.subplots(figsize=(11, 7.5))

    # Фон: все заведения серыми точками - дают ощущение города
    if places is not None and len(places):
        ax.scatter(places["lon"], places["lat"], s=1.5, color="#b0bec5",
                   alpha=0.55, zorder=1, label="заведения (OSM)")

    for district in load_districts():
        value = district_stats.loc[district.key, value_col] \
            if district.key in district_stats.index else None
        color = cmap(norm(value)) if value is not None and pd.notna(value) else "#eceff1"

        ax.add_patch(MplPolygon(
            list(district.polygon.exterior.coords), closed=True,
            facecolor=color, edgecolor="#37474f", linewidth=1.6,
            alpha=0.78, zorder=2,
        ))

        lat, lon = district.centroid_latlon
        label = names.get(district.key, district.key)
        if value is not None and pd.notna(value):
            label += f"\n{value:+.3f}"
        ax.text(lon, lat, label, ha="center", va="center", fontsize=10,
                fontweight="bold", zorder=3,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          alpha=0.82, edgecolor="none"))

    bbox = get_bbox()
    ax.set_xlim(bbox.west, bbox.east)
    ax.set_ylim(bbox.south, bbox.north)
    # Широта и долгота имеют разный физический масштаб. На широте Брайтона
    # (~50.8°) один градус долготы примерно в 1.6 раза короче градуса
    # широты. Без поправки город растянется по горизонтали.
    ax.set_aspect(1 / 0.63)
    ax.set_xlabel("долгота")
    ax.set_ylabel("широта")
    ax.set_title(f"Emotional Map of Brighton\n{scale_note}", fontsize=13, pad=14)

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label="Net sentiment (доля позитива − доля негатива)",
                 fraction=0.035, pad=0.02)
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=145)
    plt.close(fig)
