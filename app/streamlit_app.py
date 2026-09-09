"""
Emotional Map of Brighton - интерактивный дашборд.

Запуск:
    PYTHONPATH=src .venv/bin/streamlit run app/streamlit_app.py

=== ПРИНЦИПЫ, ПО КОТОРЫМ СДЕЛАН ЭТОТ ДАШБОРД ===

1. Дашборд НИЧЕГО НЕ СЧИТАЕТ ТЯЖЁЛОГО.
   Он читает готовые файлы из data/processed. Все модели отработали
   заранее, в скриптах. Если запускать DistilBERT при каждом движении
   ползунка, интерфейс будет думать по полминуты, и им никто не станет
   пользоваться. Разделение "тяжёлый счёт офлайн, дашборд только
   показывает" - базовое правило продуктовой аналитики.

2. Кэширование обязательно.
   @st.cache_data означает: результат функции запоминается, и при
   следующем перерисовывании (а Streamlit перезапускает весь скрипт
   на КАЖДОЕ действие пользователя) файл с диска читается не заново.

3. Число наблюдений показывается всегда.
   Если после фильтров осталось 12 текстов, пользователь должен это
   видеть, а не гадать, почему график странный.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Чтобы работал импорт bem/ при запуске через streamlit run
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.components.v1 import html as st_html

from bem.config import PROCESSED_DIR
from bem.geo.districts import district_colors, district_names
from bem.viz.maps import add_title, build_map

st.set_page_config(page_title="Emotional Map of Brighton", page_icon="🌊", layout="wide")

# Язык дашборда. Интерфейс написан по-русски; здесь он задаётся один раз,
# чтобы карта внутри дашборда не разъезжалась с остальными подписями.
DASHBOARD_LANG = "ru"

MONTHS_RU = ["янв", "фев", "мар", "апр", "май", "июн",
             "июл", "авг", "сен", "окт", "ноя", "дек"]


# ------------------------------------------------------------ загрузка ----

@st.cache_data
def load_texts() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "texts_final.csv", parse_dates=["created_utc"])
    df["month"] = df["created_utc"].dt.to_period("M").dt.to_timestamp()
    df["district_name"] = df["district"].map(district_names())
    return df


@st.cache_data
def load_places() -> pd.DataFrame:
    path = PROCESSED_DIR / "place_summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_police() -> pd.DataFrame:
    """Реальные происшествия Police.uk. Файла может не быть - это не ошибка."""
    path = PROCESSED_DIR / "police_incidents.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["month_dt"] = pd.to_datetime(df["month"])
    df["month_num"] = df["month_dt"].dt.month
    df["district_name"] = df["district"].map(district_names())
    return df


def net_sentiment(labels: pd.Series) -> float:
    """Доля позитива минус доля негатива."""
    if len(labels) == 0:
        return 0.0
    return (labels == "pos").mean() - (labels == "neg").mean()


# ------------------------------------------------------------ интерфейс ----

texts = load_texts()
places = load_places()

st.title("🌊 Emotional Map of Brighton")

# ---------- фильтры ----------
with st.sidebar:
    st.header("Фильтры")

    min_date = texts["created_utc"].min().date()
    max_date = texts["created_utc"].max().date()
    date_range = st.slider(
        "Период", min_value=min_date, max_value=max_date,
        value=(min_date, max_date), format="MMM YYYY",
    )

    all_districts = sorted(texts["district_name"].dropna().unique())
    chosen_districts = st.multiselect("Районы", all_districts, default=all_districts)

    all_categories = sorted(texts["category"].dropna().unique())
    chosen_categories = st.multiselect(
        "Тип заведения", all_categories, default=all_categories,
        help="Пусто = без фильтра по типу. Тексты без привязки к заведению "
             "при выборе категорий отсеются.",
    )

    all_topics = sorted(texts["topic"].dropna().unique())
    chosen_topics = st.multiselect("Темы", all_topics, default=all_topics)

    st.divider()

    # Источники данных. Про синтетические тексты сказано здесь, а не
    # плашкой на главном экране: предупреждение нужно, но кричать им
    # на весь дашборд незачем.
    if (texts["source"] == "demo").all():
        st.caption(
            "Источники: OpenStreetMap (ODbL), Police.uk (OGL v3.0). "
            "Тексты - синтетические, сгенерированы "
            "`src/bem/collect/demo_corpus.py` для отладки пайплайна. "
            "Вкладка «Реальные данные» построена на настоящих данных."
        )
    else:
        st.caption(
            "Источники: OpenStreetMap (ODbL), Reddit, Police.uk (OGL v3.0)."
        )

    st.caption(
        "Все модели отработали заранее - дашборд только показывает результат."
    )

# ---------- применяем фильтры ----------
mask = (
    (texts["created_utc"].dt.date >= date_range[0])
    & (texts["created_utc"].dt.date <= date_range[1])
    & (texts["district_name"].isin(chosen_districts))
    & (texts["topic"].isin(chosen_topics))
)
if chosen_categories and len(chosen_categories) < len(all_categories):
    mask &= texts["category"].isin(chosen_categories)

view = texts[mask]

if view.empty:
    st.error("После фильтров не осталось ни одного текста. Ослабь условия слева.")
    st.stop()

# ---------- ключевые цифры ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Текстов", f"{len(view):,}".replace(",", "\u00a0"))
c2.metric("Net sentiment", f"{net_sentiment(view['sentiment_label']):+.3f}",
          help="Доля позитива минус доля негатива. От −1 до +1.")
c3.metric("Позитивных", f"{(view['sentiment_label'] == 'pos').mean():.0%}")
c4.metric("Районов в выборке", view["district"].nunique())

if len(view) < 100:
    st.info(
        f"В выборке всего {len(view)} текстов - для устойчивых выводов маловато. "
        "Графики ниже будут шумными.", icon="ℹ️",
    )

police = load_police()

tab_map, tab_time, tab_topics, tab_places, tab_texts, tab_police = st.tabs(
    ["🗺 Карта", "📈 Динамика", "🏷 Темы", "📍 Заведения", "💬 Тексты",
     "🚓 Реальные данные"]
)

# ---------- карта ----------
with tab_map:
    summary = (
        view.groupby("district")
        .agg(n=("sentiment_label", "size"),
             net_sentiment=("sentiment_label", net_sentiment),
             share_pos=("sentiment_label", lambda s: (s == "pos").mean()),
             share_neg=("sentiment_label", lambda s: (s == "neg").mean()))
    )
    summary["name"] = [district_names().get(i, i) for i in summary.index]

    # lang="ru": интерфейс дашборда русский, и карта внутри него должна
    # быть на том же языке. По умолчанию build_map рисует по-английски,
    # потому что те же карты сохраняются в файлы для английского README.
    m = build_map(summary, texts=view, places=places if len(places) else None,
                  lang=DASHBOARD_LANG)
    add_title(m, "Настроение по районам",
              f"{len(view)} текстов · слои переключаются справа сверху")
    st_html(m.get_root().render(), height=560)

    st.dataframe(
        summary[["name", "n", "net_sentiment", "share_pos", "share_neg"]]
        .rename(columns={"name": "Район", "n": "Текстов",
                         "net_sentiment": "Net sentiment",
                         "share_pos": "Доля позитива", "share_neg": "Доля негатива"})
        .sort_values("Net sentiment", ascending=False),
        use_container_width=True, hide_index=True,
    )

# ---------- динамика ----------
with tab_time:
    monthly = (
        view.groupby(["month", "district_name"])
        .agg(net=("sentiment_label", net_sentiment), n=("sentiment_label", "size"))
        .reset_index()
    )
    # Точки, посчитанные по горстке текстов, помечаем - иначе они
    # выглядят на графике так же уверенно, как надёжные.
    monthly["надёжность"] = monthly["n"].apply(lambda n: "n ≥ 20" if n >= 20 else "n < 20")

    fig = px.line(
        monthly, x="month", y="net", color="district_name", markers=True,
        color_discrete_map={district_names()[k]: v for k, v in district_colors().items()
                            if k in district_names()},
        labels={"month": "", "net": "Net sentiment", "district_name": "Район"},
        title="Настроение по месяцам",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#aaa")
    fig.update_layout(height=430, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    seasonal = view.copy()
    seasonal["month_num"] = seasonal["created_utc"].dt.month
    prof = (
        seasonal.groupby(["month_num", "district_name"])
        .agg(net=("sentiment_label", net_sentiment)).reset_index()
    )
    prof["Месяц"] = prof["month_num"].apply(lambda m: MONTHS_RU[m - 1])

    fig2 = px.line(
        prof.sort_values("month_num"), x="Месяц", y="net", color="district_name",
        markers=True, labels={"net": "Net sentiment", "district_name": "Район"},
        title="Средний профиль года (все годы вместе)",
    )
    fig2.add_vrect(x0=6.6, x1=7.4, fillcolor="#ffd54f", opacity=0.3, line_width=0,
                   annotation_text="Pride")
    fig2.add_vrect(x0=3.6, x1=4.4, fillcolor="#90caf9", opacity=0.3, line_width=0,
                   annotation_text="Fringe")
    fig2.add_hline(y=0, line_dash="dash", line_color="#aaa")
    fig2.update_layout(height=430)
    st.plotly_chart(fig2, use_container_width=True)

    low = monthly[monthly["n"] < 20]
    if len(low):
        st.caption(f"⚠️ Точек, посчитанных менее чем по 20 текстам: {len(low)} "
                   f"из {len(monthly)}. Им доверять не стоит.")

# ---------- темы ----------
with tab_topics:
    topic_district = (
        view.groupby(["district_name", "topic"])
        .size().reset_index(name="n")
    )
    topic_district["share"] = topic_district.groupby("district_name")["n"].transform(
        lambda s: s / s.sum()
    )

    fig = px.bar(
        topic_district, x="share", y="district_name", color="topic",
        orientation="h", labels={"share": "Доля текстов", "district_name": "",
                                 "topic": "Тема"},
        title="О чём пишут в каждом районе",
    )
    fig.update_layout(height=420, xaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    topic_sent = (
        view.groupby("topic")
        .agg(net=("sentiment_label", net_sentiment), n=("sentiment_label", "size"))
        .reset_index().sort_values("net")
    )
    fig2 = px.bar(
        topic_sent, x="net", y="topic", orientation="h", color="net",
        color_continuous_scale=["#d32f2f", "#fdd835", "#2e7d32"],
        color_continuous_midpoint=0, text="n",
        labels={"net": "Net sentiment", "topic": ""},
        title="Какие темы вызывают позитив, а какие раздражение (цифра = число текстов)",
    )
    fig2.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

# ---------- заведения ----------
with tab_places:
    if places.empty:
        st.info("Нет файла place_summary.csv - запусти scripts/06_build_outputs.py")
    else:
        st.markdown(
            "Оценки **сжаты к общегородскому среднему** пропорционально числу "
            "упоминаний. Без этого в топе оказывались бы места с тремя "
            "случайными отзывами и «идеальной» оценкой 1.00. "
            "`confidence` показывает, какая доля оценки взята из собственных "
            "данных места, а какая - из общего среднего."
        )
        show = places[places["district"].isin(
            [k for k, v in district_names().items() if v in chosen_districts]
        )]
        cols = ["name", "district", "category", "n_texts",
                "net_sentiment", "net_sentiment_shrunk", "confidence"]
        cols = [c for c in cols if c in show.columns]

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Лучшие")
            st.dataframe(show.nlargest(15, "net_sentiment_shrunk")[cols],
                         use_container_width=True, hide_index=True)
        with c2:
            st.subheader("Худшие")
            st.dataframe(show.nsmallest(15, "net_sentiment_shrunk")[cols],
                         use_container_width=True, hide_index=True)

# ---------- тексты ----------
with tab_texts:
    st.caption("Случайная выборка - чтобы можно было глазами проверить, "
               "что модель размечает адекватно. Это обязательный шаг: "
               "метрики врут тише, чем кажется.")
    polarity = st.radio("Показать", ["все", "pos", "neu", "neg"], horizontal=True)

    sample = view if polarity == "все" else view[view["sentiment_label"] == polarity]
    if sample.empty:
        st.info("Таких текстов в выборке нет.")
    else:
        st.dataframe(
            sample.sample(min(40, len(sample)), random_state=0)[
                ["created_utc", "district_name", "topic", "sentiment_label",
                 "sentiment_score", "text"]
            ].sort_values("created_utc"),
            use_container_width=True, hide_index=True,
        )

# ---------- реальные данные Police.uk ----------
with tab_police:
    if police.empty:
        st.info(
            "Нет файла police_incidents.csv - запусти "
            "`PYTHONPATH=src .venv/bin/python scripts/07_collect_police.py`. "
            "Ключи не нужны."
        )
    else:
        # Разделитель разрядов применяем ТОЛЬКО к числу.
        # Если сделать .replace(",", " ") на всей строке, оно съест
        # и обычные запятые в тексте - поймал на себе.
        total = f"{len(police):,}".replace(",", "\u00a0")
        # Согласование числительного: 1 происшествие, 2-4 происшествия,
        # 5-20 происшествий, дальше по последней цифре (81 171 -> ...ие).
        n = len(police)
        last, last_two = n % 10, n % 100
        if last == 1 and last_two != 11:
            word = "происшествие"
        elif last in (2, 3, 4) and last_two not in (12, 13, 14):
            word = "происшествия"
        else:
            word = "происшествий"
        st.success(
            f"**Это настоящие данные.** {total} {word} за "
            f"{police['month'].nunique()} месяцев из Police.uk API "
            f"(Open Government Licence). В отличие от вкладок выше, здесь "
            f"нет ничего синтетического.",
            icon="✅",
        )
        st.caption(
            "Осторожно с интерпретацией: число зарегистрированных происшествий - "
            "это не уровень преступности, а уровень, умноженный на готовность "
            "заявлять и на плотность полицейского присутствия. Координаты "
            "Police.uk намеренно округлены до центра улицы ради приватности."
        )

        only_atmosphere = st.checkbox(
            "Только категории, влияющие на атмосферу района",
            value=True,
            help="anti-social-behaviour, public-order, violent-crime, "
                 "criminal-damage-arson, drugs, possession-of-weapons. "
                 "Кражи из магазинов исключены: они говорят о торговле, "
                 "а не об атмосфере места.",
        )

        pol = police[police["district"] != "other"]
        if only_atmosphere:
            pol = pol[pol["is_atmosphere"]]

        # Нормируем на средний месяц САМОГО района - иначе график покажет
        # только то, что один район больше другого.
        per_month = (
            pol.groupby(["district_name", "month_num"])
            .agg(n=("crime_id", "size"), months=("month", "nunique"))
            .reset_index()
        )
        per_month["avg"] = per_month["n"] / per_month["months"]
        per_month["norm"] = per_month.groupby("district_name")["avg"].transform(
            lambda s: s / s.mean()
        )
        per_month["Месяц"] = per_month["month_num"].apply(lambda m: MONTHS_RU[m - 1])

        fig = px.line(
            per_month.sort_values("month_num"), x="Месяц", y="norm",
            color="district_name", markers=True,
            labels={"norm": "уровень относительно обычного месяца",
                    "district_name": "Район"},
            title="Сезонность происшествий (1.0 = обычный месяц для этого района)",
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="#aaa")
        fig.add_vrect(x0=6.6, x1=7.4, fillcolor="#ffd54f", opacity=0.3,
                      line_width=0, annotation_text="Pride")
        fig.add_vrect(x0=3.6, x1=4.4, fillcolor="#90caf9", opacity=0.3,
                      line_width=0, annotation_text="Fringe")
        fig.update_layout(height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "**Находка, опровергшая исходное предположение.** В демо-корпус "
            "август был заложен как главный пик года. Реальные данные "
            "показывают пик в **июле** у четырёх районов из пяти. "
            "Единственное исключение - **Seafront**, и это ровно та "
            "география, где проходит парад Brighton Pride."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Пик года по районам")
            peaks = (
                per_month.loc[per_month.groupby("district_name")["norm"].idxmax()]
                [["district_name", "Месяц", "norm"]]
                .rename(columns={"district_name": "Район",
                                 "norm": "уровень пика"})
                .sort_values("уровень пика", ascending=False)
            )
            st.dataframe(peaks, use_container_width=True, hide_index=True)
        with c2:
            st.subheader("Категории происшествий")
            cats = (
                police[police["district"] != "other"]["category"]
                .value_counts().head(10).reset_index()
            )
            cats.columns = ["Категория", "Количество"]
            st.dataframe(cats, use_container_width=True, hide_index=True)
