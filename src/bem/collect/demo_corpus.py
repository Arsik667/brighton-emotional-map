"""
Генератор ДЕМО-корпуса текстов о районах Брайтона.

=== ЧЕСТНОЕ ПРЕДУПРЕЖДЕНИЕ ===
Эти тексты СИНТЕТИЧЕСКИЕ. Их написал не человек, а вот этот файл.
Они нужны ровно для одного: чтобы можно было разрабатывать и отлаживать
NLP-пайплайн, пока нет доступа к настоящим данным.

В README и на собеседовании про это надо говорить прямо. Выдавать
сгенерированные данные за реальные - худшее, что можно сделать с
портфолио: это ловится первым же уточняющим вопросом.

=== ЗАЧЕМ ТАК ЗАМОРАЧИВАТЬСЯ ===
Можно было насыпать случайных слов. Но тогда sentiment-анализ и topic
modeling не нашли бы НИЧЕГО, и было бы непонятно: пайплайн сломан или
данные пустые. Поэтому в корпус намеренно заложен сигнал:

  * у каждого района свой профиль тем (в Kemptown чаще про nightlife,
    в Hove - про семьи и спокойствие);
  * у каждого месяца свой сдвиг настроения (лето лучше зимы);
  * события: Brighton Pride в августе, Fringe и Great Escape в мае -
    всплеск позитива, но одновременно рост жалоб на толпы и шум.

Когда на Этапе 6 мы построим динамику по месяцам и увидим август -
это будет проверкой того, что пайплайн действительно ловит сигнал.
Сигнал мы знаем заранее, потому что сами его положили.
"""

from __future__ import annotations

import hashlib
import logging
import random
from datetime import datetime, timedelta, timezone

import pandas as pd

from bem.config import INTERIM_DIR

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Профили районов: какие темы там обсуждают чаще и каков базовый настрой.
# base_sentiment сдвигает вероятность позитива: +0.10 = район приятнее среднего.
# --------------------------------------------------------------------------
DISTRICT_PROFILES = {
    "the_lanes": {
        "base_sentiment": 0.06,
        "topics": {"food": 3, "atmosphere": 3, "crowds": 3, "price": 3,
                   "service": 2, "nightlife": 1, "cleanliness": 1, "parking": 1},
    },
    "north_laine": {
        "base_sentiment": 0.12,
        "topics": {"atmosphere": 4, "food": 3, "shopping_vibe": 3, "price": 2,
                   "crowds": 2, "service": 2, "noise": 1, "cleanliness": 1},
    },
    "kemptown": {
        "base_sentiment": 0.02,
        "topics": {"nightlife": 4, "atmosphere": 3, "noise": 3, "food": 2,
                   "safety": 2, "price": 2, "cleanliness": 2, "parking": 1},
    },
    "seafront": {
        "base_sentiment": 0.04,
        "topics": {"atmosphere": 4, "crowds": 3, "weather": 3, "cleanliness": 3,
                   "price": 2, "food": 2, "safety": 1, "seagulls": 2},
    },
    "hove": {
        "base_sentiment": 0.14,
        "topics": {"atmosphere": 4, "food": 3, "price": 2, "parking": 3,
                   "cleanliness": 1, "service": 2, "crowds": 1},
    },
}

# --------------------------------------------------------------------------
# Сезонность: сдвиг настроения по месяцам и относительный объём разговоров.
# Зима в приморском городе - это серо, ветрено и пусто. Лето - наоборот.
# --------------------------------------------------------------------------
MONTH_SENTIMENT = {
    1: -0.12, 2: -0.10, 3: -0.04, 4: 0.04, 5: 0.10, 6: 0.12,
    7: 0.10, 8: 0.08, 9: 0.06, 10: -0.02, 11: -0.08, 12: -0.02,
}

MONTH_VOLUME = {
    1: 0.6, 2: 0.6, 3: 0.8, 4: 1.0, 5: 1.3, 6: 1.3,
    7: 1.5, 8: 1.8, 9: 1.1, 10: 0.8, 11: 0.7, 12: 0.9,
}

# События: месяц -> (районы, сдвиг настроения, доп. вес "толпных" тем)
EVENTS = {
    8: {  # Brighton Pride - первые выходные августа
        "name": "Brighton Pride",
        "districts": ["seafront", "kemptown", "the_lanes", "north_laine"],
        "sentiment_boost": 0.22,
        "crowd_boost": 3.0,
    },
    5: {  # Brighton Fringe + The Great Escape
        "name": "Brighton Fringe / Great Escape",
        "districts": ["north_laine", "the_lanes", "kemptown"],
        "sentiment_boost": 0.14,
        "crowd_boost": 2.0,
    },
}

# --------------------------------------------------------------------------
# Фразы по темам. Слоты в фигурных скобках заполняются на лету.
# --------------------------------------------------------------------------
PHRASES: dict[str, dict[str, list[str]]] = {
    "food": {
        "pos": [
            "the food at {place} was genuinely excellent, easily the best meal I've had this month",
            "had brunch at {place} and the portions were generous and properly tasty",
            "{place} does the best coffee and pastries in {district}, no contest",
            "really solid menu at {place}, everything came out fresh and well seasoned",
            "we shared a few small plates at {place} and every single one was delicious",
        ],
        "neg": [
            "the food at {place} was bland and lukewarm, sent half of it back",
            "paid a lot at {place} for a tiny portion that tasted of nothing",
            "ordered the special at {place} and it arrived overcooked and dry",
            "honestly the meal at {place} was forgettable, wouldn't order it again",
        ],
        "neu": [
            "went to {place} for lunch, food was fine, nothing remarkable either way",
            "standard menu at {place}, does the job if you're hungry",
        ],
    },
    "price": {
        "pos": [
            "surprisingly good value at {place} considering how central {district} is",
            "prices at {place} are reasonable for what you get, which is rare around here",
            "cheap and cheerful in {district}, you can still eat well without spending a fortune",
        ],
        "neg": [
            "{district} has got so expensive, {place} charged me nearly a tenner for a pint",
            "the prices at {place} are getting ridiculous, complete rip off for what it is",
            "everything in {district} feels overpriced now, priced out of my own neighbourhood",
            "paid tourist prices at {place} and it really wasn't worth it",
        ],
        "neu": [
            "prices at {place} are about what you'd expect for {district} these days",
        ],
    },
    "noise": {
        "pos": [
            "surprisingly quiet corner of {district}, you can actually have a conversation",
            "{place} manages to be lively without being deafening, which I appreciate",
        ],
        "neg": [
            "the noise in {district} at night is unbearable, shouting until 3am every weekend",
            "music from {place} rattles my windows, complained twice and nothing changed",
            "can't sleep with the racket coming off the street in {district}",
            "way too loud inside {place}, gave up trying to talk to anyone",
        ],
        "neu": [
            "{district} gets noisy at weekends, quiet enough midweek",
        ],
    },
    "atmosphere": {
        "pos": [
            "love the atmosphere in {district}, it has a character you don't get anywhere else",
            "{place} has such a warm, welcoming vibe, felt at home straight away",
            "{district} on a sunny afternoon is genuinely one of my favourite places to be",
            "there's a proper community feel around {place}, everyone knows each other",
            "wandered around {district} for hours, the independent shops and street art make it",
        ],
        "neg": [
            "{district} has lost its character, all the interesting places have closed",
            "{place} felt soulless and corporate, could be anywhere in the country",
            "the atmosphere in {district} has really gone downhill in the last couple of years",
        ],
        "neu": [
            "{district} is much like it always was, familiar in a comfortable way",
        ],
    },
    "service": {
        "pos": [
            "staff at {place} were friendly and attentive without hovering",
            "genuinely lovely service at {place}, they went out of their way to help",
        ],
        "neg": [
            "waited forty minutes at {place} before anyone even took our order",
            "the staff at {place} were rude and clearly couldn't be bothered",
            "service at {place} was chaotic, drinks arrived after we'd finished eating",
        ],
        "neu": [
            "service at {place} was ok, a bit slow but they were clearly short staffed",
        ],
    },
    "crowds": {
        "pos": [
            "{district} was buzzing today in the best way, great energy everywhere",
            "busy but manageable in {district}, nice to see the place alive again",
        ],
        "neg": [
            "{district} was absolutely rammed, couldn't move for tourists",
            "gave up on {place}, queue was out the door and down the street",
            "avoid {district} at weekends unless you enjoy being shoved along the pavement",
            "far too crowded in {district}, lost all the charm it has midweek",
        ],
        "neu": [
            "{district} was busy, about what you'd expect for the time of year",
        ],
    },
    "cleanliness": {
        "pos": [
            "{district} looked properly clean this week, someone's clearly been on it",
        ],
        "neg": [
            "the amount of rubbish left in {district} after the weekend is disgraceful",
            "bins overflowing all along the street in {district} again",
            "{district} smells of stale beer and worse by Sunday morning",
            "litter everywhere near {place}, nobody seems to collect it",
        ],
        "neu": [
            "{district} is about as clean as any city centre, which isn't saying much",
        ],
    },
    "safety": {
        "pos": [
            "always felt perfectly safe walking home through {district} at night",
        ],
        "neg": [
            "felt a bit dodgy walking through {district} late, lots of drunk people about",
            "had some antisocial behaviour outside {place}, police were called",
            "wouldn't walk alone through {district} after midnight these days",
        ],
        "neu": [
            "{district} is fine during the day, I'd be more careful late at night",
        ],
    },
    "nightlife": {
        "pos": [
            "brilliant night out in {district}, {place} was exactly what we needed",
            "the bars around {place} are consistently good, proper friendly crowd",
            "{district} still has the best nightlife in the city, nothing comes close",
        ],
        "neg": [
            "nightlife in {district} isn't what it was, half the venues have shut",
            "{place} was heaving with stag dos, ruined the whole evening",
        ],
        "neu": [
            "went out in {district}, decent enough night, nothing special",
        ],
    },
    "parking": {
        "pos": [
            "actually found parking near {place} without circling for an hour, small miracle",
        ],
        "neg": [
            "parking in {district} is a nightmare, drove around for 45 minutes",
            "got a ticket outside {place} within ten minutes, absolute joke",
            "the permit situation in {district} is impossible if you have visitors",
        ],
        "neu": [
            "parked in the multi storey for {district}, expensive but at least it exists",
        ],
    },
    "weather": {
        "pos": [
            "gorgeous day on the {district}, sea was calm and the light was perfect",
            "first proper warm weekend and {district} was glorious",
        ],
        "neg": [
            "wind on the {district} was brutal today, could barely stand up",
            "grey, wet and freezing on the {district}, place feels abandoned in winter",
            "rain came in sideways along the {district}, gave up and went home",
        ],
        "neu": [
            "typical changeable weather on the {district}, four seasons in an afternoon",
        ],
    },
    "seagulls": {
        "pos": [
            "watched the gulls over the {district} for ages, they're part of the character",
        ],
        "neg": [
            "a seagull took my chips right out of my hand on the {district}",
            "the gulls near {place} are genuinely aggressive now, someone will get hurt",
            "woken at 4am by seagulls again, the noise in {district} is relentless",
        ],
        "neu": [
            "seagulls doing seagull things on the {district}, as always",
        ],
    },
    "shopping_vibe": {
        "pos": [
            "the independent shops in {district} are why I still love this city",
            "found some great vintage bits around {place}, could browse there all day",
            "{district} has the kind of quirky little shops that are disappearing elsewhere",
        ],
        "neg": [
            "another independent shop in {district} has closed, rents are killing them",
            "{district} is turning into the same chains you get everywhere",
        ],
        "neu": [
            "did a bit of shopping around {place}, usual mix of stuff",
        ],
    },
}

# Фразы про события - добавляются поверх обычных в нужный месяц
EVENT_PHRASES = {
    "Brighton Pride": [
        "Pride weekend in {district} was absolutely joyful, best atmosphere of the year",
        "the Pride parade coming through {district} had everyone out on the street smiling",
        "so much colour and energy in {district} for Pride, genuinely moving to see",
        "Pride was brilliant but {district} was completely rammed, couldn't get near {place}",
        "loved Pride, hated the state {district} was left in the next morning",
    ],
    "Brighton Fringe / Great Escape": [
        "Fringe season in {district} is my favourite time of year, something on every corner",
        "caught three shows around {district} during Fringe, all of them worth it",
        "Great Escape weekend and {district} was full of bands and buzzing",
        "Fringe is great but every venue in {district} is packed out",
    ],
}


# Вводные обороты. Нужны не для красоты: без них шаблонов слишком мало,
# и генератор начинает выдавать дословные повторы. После дедупликации
# от корпуса оставалась бы треть.
OPENERS = [
    "", "", "", "",  # чаще всего - без вводного оборота
    "Honestly, ", "To be fair, ", "Went last weekend and ",
    "Not sure if it's just me but ", "Been going for years and ",
    "First time visiting and ", "Popped in on the way home, ",
    "Bit of a mixed one: ", "Quick update for anyone asking: ",
]

# Связки для второго предложения. "but" намеренно частая: смешанные отзывы
# ("еда отличная, но толпа кошмар") - самый частый и самый сложный
# для sentiment-моделей случай. Пусть он будет и в нашем корпусе.
CONNECTORS_SAME = [". ", ". Also, ", " and ", ". On top of that, "]
CONNECTORS_MIXED = [", but ", ". That said, ", " although ", ". Downside is "]


def _seasonal_sentiment(district: str, month: int, rng: random.Random) -> float:
    """
    Посчитать вероятность позитива для конкретного района и месяца.

    Складываем: базовый настрой района + сезонный сдвиг + бонус события
    + случайный шум. Шум обязателен: без него все тексты одного месяца
    получили бы одинаковую тональность, и это выглядело бы искусственно
    даже на графике.
    """
    score = 0.5
    score += DISTRICT_PROFILES[district]["base_sentiment"]
    score += MONTH_SENTIMENT[month]

    event = EVENTS.get(month)
    if event and district in event["districts"]:
        score += event["sentiment_boost"]

    score += rng.gauss(0, 0.18)  # индивидуальный разброс между авторами
    return min(max(score, 0.02), 0.98)


def _pick_topic(district: str, month: int, rng: random.Random) -> str:
    """Выбрать тему с учётом профиля района и событий месяца."""
    weights = dict(DISTRICT_PROFILES[district]["topics"])

    event = EVENTS.get(month)
    if event and district in event["districts"]:
        # Во время фестивалей люди заметно чаще пишут про толпы и шум
        for topic in ("crowds", "noise"):
            if topic in weights:
                weights[topic] *= event["crowd_boost"]

    topics = list(weights)
    return rng.choices(topics, weights=[weights[t] for t in topics], k=1)[0]


def _random_date(start: datetime, end: datetime, rng: random.Random) -> datetime:
    """
    Случайная дата в диапазоне, но с учётом MONTH_VOLUME.

    Летом о городе пишут больше, чем в феврале. Если генерировать даты
    равномерно, эта разница пропадёт, а она - часть того, что мы хотим
    показать на графике активности.
    """
    total_days = (end - start).days
    while True:
        candidate = start + timedelta(days=rng.randrange(total_days))
        volume = MONTH_VOLUME[candidate.month]
        if rng.random() < volume / max(MONTH_VOLUME.values()):
            return candidate + timedelta(
                hours=rng.randrange(24), minutes=rng.randrange(60)
            )


def _clean_place_name(name: str) -> str:
    """Убрать из названия то, что мешает вставить его в предложение."""
    return str(name).strip().replace("\n", " ")


def generate(
    n_texts: int = 6000,
    start: str = "2023-01-01",
    end: str = "2026-09-01",
    seed: int = 42,
    places_csv=None,
) -> pd.DataFrame:
    """
    Сгенерировать демо-корпус.

    Каждый текст привязан к РЕАЛЬНОМУ заведению из places.csv - значит,
    у него есть настоящие координаты, и геочасть пайплайна работает
    ровно так же, как работала бы на живых данных.

    seed фиксирован: при повторном запуске получится тот же корпус.
    Воспроизводимость - то, что отличает эксперимент от случайности.
    """
    rng = random.Random(seed)

    places_csv = places_csv or (INTERIM_DIR / "places.csv")
    places = pd.read_csv(places_csv)
    places = places[places["district"] != "other"].reset_index(drop=True)

    # Сколько разговоров приходится на район. Это НЕ пропорционально числу
    # заведений: на набережной их всего полсотни, но обсуждают её постоянно -
    # это главная общественная площадка города. Если сэмплировать заведения
    # равномерно, seafront получит 3% корпуса и на графиках будет пусто.
    DISTRICT_WEIGHTS = {
        "the_lanes": 22, "north_laine": 20, "seafront": 22,
        "kemptown": 18, "hove": 18,
    }
    places_by_district = {
        d: sub.reset_index(drop=True)
        for d, sub in places.groupby("district")
        if d in DISTRICT_WEIGHTS
    }
    district_keys = list(places_by_district)
    district_weights = [DISTRICT_WEIGHTS[d] for d in district_keys]

    if places.empty:
        raise RuntimeError(
            "Нет заведений с районом. Сначала запусти scripts/01_collect_places.py"
        )

    from bem.geo.districts import district_names
    names = district_names()

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

    rows = []
    for i in range(n_texts):
        district = rng.choices(district_keys, weights=district_weights, k=1)[0]
        district_places = places_by_district[district]
        place = district_places.iloc[rng.randrange(len(district_places))]
        created = _random_date(start_dt, end_dt, rng)

        positivity = _seasonal_sentiment(district, created.month, rng)
        topic = _pick_topic(district, created.month, rng)

        # Раскладываем вероятность позитива на три исхода.
        roll = rng.random()
        if roll < positivity - 0.15:
            polarity = "pos"
        elif roll > positivity + 0.15:
            polarity = "neg"
        else:
            polarity = "neu"

        slots = {
            "place": _clean_place_name(place["name"]),
            "district": names[district],
        }

        # Во время фестивалей часть текстов - прямо про событие
        event = EVENTS.get(created.month)
        use_event = (
            event
            and district in event["districts"]
            and rng.random() < 0.22
        )
        if use_event:
            sentence = rng.choice(EVENT_PHRASES[event["name"]]).format(**slots)
            topic = "events"
        else:
            sentence = rng.choice(PHRASES[topic][polarity]).format(**slots)

        # Больше половины текстов делаем из двух предложений - так корпус
        # ближе к реальным постам, где человек говорит больше одной вещи.
        is_mixed = False
        if rng.random() < 0.55:
            second_topic = _pick_topic(district, created.month, rng)

            # В 35% случаев второе предложение имеет ПРОТИВОПОЛОЖНУЮ
            # тональность. Это ключевой момент: "еда отличная, но было
            # не протолкнуться" - самый частый тип реального отзыва и
            # самый трудный для sentiment-моделей. На Этапе 4 мы увидим,
            # как VADER и DistilBERT справляются именно с такими.
            if polarity in ("pos", "neg") and rng.random() < 0.35:
                second_polarity = "neg" if polarity == "pos" else "pos"
                connector = rng.choice(CONNECTORS_MIXED)
                is_mixed = True
            else:
                second_polarity = polarity
                connector = rng.choice(CONNECTORS_SAME)

            second = rng.choice(PHRASES[second_topic][second_polarity]).format(**slots)
            if second != sentence:
                sentence = f"{sentence}{connector}{second}"

        sentence = rng.choice(OPENERS) + sentence
        text = sentence[0].upper() + sentence[1:]
        if not text.endswith("."):
            text += "."

        rows.append(
            {
                "text_id": f"demo_{i:06d}",
                "source": "demo",
                "kind": "synthetic",
                "created_utc": created.isoformat(),
                "text": text,
                "score": max(0, int(rng.gauss(6, 8))),
                # Стабильный псевдо-автор: тот же формат, что у Reddit,
                # чтобы дальше код не различал источники.
                "author_hash": hashlib.sha256(
                    f"demo{rng.randrange(400)}".encode()
                ).hexdigest()[:12],
                "place_id": place["place_id"],
                "place_name": place["name"],
                "lat": place["lat"],
                "lon": place["lon"],
                "category": place["category"],
                "district": district,
                # Эти две колонки - "правильные ответы", которые мы сами
                # заложили. На Этапе 4 по ним можно будет проверить,
                # насколько хорошо sentiment-модель угадывает тональность.
                "true_polarity": "pos" if use_event else polarity,
                "true_topic": topic,
                # Смешанные тексты стоит исключать из строгой проверки
                # точности: у них нет одного правильного ответа.
                "is_mixed": is_mixed,
            }
        )

    df = pd.DataFrame(rows).sort_values("created_utc").reset_index(drop=True)
    logger.info("Сгенерировано %d демо-текстов", len(df))
    return df
