"""Тесты привязки текста к району и очистки текстов."""

import pandas as pd
import pytest

from bem.geo.districts import UNKNOWN
from bem.nlp.cleaning import clean_frame, clean_text
from bem.nlp.text_linking import build_place_index, link_text


@pytest.fixture
def places():
    """Маленькая тестовая база заведений с намеренно каверзными случаями."""
    return pd.DataFrame([
        # обычное однозначное название
        {"place_id": "node/1", "name": "Fortune of War", "district": "the_lanes",
         "lat": 50.8204, "lon": -0.1446},
        # короткое название - должно отсеяться по длине
        {"place_id": "node/2", "name": "Pub", "district": "hove",
         "lat": 50.8330, "lon": -0.1730},
        # название-город из стоп-листа
        {"place_id": "node/3", "name": "Amsterdam", "district": "kemptown",
         "lat": 50.8201, "lon": -0.1349},
        # одно название в двух районах -> неоднозначное
        {"place_id": "node/4", "name": "The Grand Cafe", "district": "hove",
         "lat": 50.8330, "lon": -0.1730},
        {"place_id": "node/5", "name": "The Grand Cafe", "district": "kemptown",
         "lat": 50.8201, "lon": -0.1349},
        # подстрока чужого названия из другого района
        {"place_id": "node/6", "name": "Churros", "district": "the_lanes",
         "lat": 50.8215, "lon": -0.1420},
        {"place_id": "node/7", "name": "Donuts & Churros", "district": "kemptown",
         "lat": 50.8195, "lon": -0.1290},
    ])


@pytest.fixture
def index(places):
    return build_place_index(places)


def test_short_names_excluded(index):
    """'Pub' слишком короткое, чтобы по нему что-то привязывать."""
    assert "pub" not in index


def test_ambiguous_city_names_excluded(index):
    """'Amsterdam' - это чаще город, чем бар в Кемптауне."""
    assert "amsterdam" not in index


def test_name_in_two_districts_excluded(index):
    """Если название встречается в двух районах, угадывать нельзя."""
    assert "the grand cafe" not in index


def test_substring_of_other_place_excluded(index):
    """
    'Churros' входит в 'Donuts & Churros' из другого района - привязываться
    по нему нельзя. Настоящий баг, найденный на демо-корпусе.
    """
    assert "churros" not in index
    assert "donuts & churros" in index


def test_link_by_place_name(index):
    result = link_text("Had a great pint at Fortune of War last night", index)
    assert result["district"] == "the_lanes"
    assert result["link_method"] == "place_name"
    assert result["lat"] == pytest.approx(50.8204)


def test_link_by_district_keyword(index):
    result = link_text("Parking in Kemptown is an absolute nightmare", index)
    assert result["district"] == "kemptown"
    assert result["link_method"] == "district_keyword"
    # По топониму координат конкретного заведения нет - и это честно
    assert result["lat"] is None


def test_place_name_wins_over_keyword(index):
    """Название заведения точнее топонима, поэтому проверяется первым."""
    result = link_text("Fortune of War is my favourite pub in Hove honestly", index)
    assert result["district"] == "the_lanes"
    assert result["link_method"] == "place_name"


def test_word_boundaries_respected(index):
    """'hove' не должно совпадать внутри слова 'shove'."""
    result = link_text("Someone tried to shove past me in the queue", index)
    assert result["district"] == UNKNOWN


def test_unrelated_text_is_unknown(index):
    assert link_text("Does anyone know a good dentist?", index)["district"] == UNKNOWN
    assert link_text("", index)["district"] == UNKNOWN
    assert link_text(None, index)["district"] == UNKNOWN


# --- очистка текстов ---

def test_clean_removes_urls():
    assert "http" not in clean_text("Great place https://example.com/deals check it")


def test_clean_keeps_case_and_punctuation():
    """
    Важно: НЕ приводим к нижнему регистру и не режем пунктуацию.
    VADER читает капс и восклицательные знаки как усиление эмоции.
    """
    cleaned = clean_text("This place is AMAZING!!!")
    assert "AMAZING" in cleaned
    assert "!!!" in cleaned


def test_clean_drops_deleted_placeholders():
    assert clean_text("[deleted]") == ""
    assert clean_text("[removed]") == ""


def test_clean_unwraps_markdown_links():
    assert clean_text("see [this cafe](https://x.com) here") == "see this cafe here"


def test_clean_frame_drops_short_and_duplicates():
    df = pd.DataFrame({"text": [
        "This is a proper length review of a nice cafe here",
        "This is a proper length review of a nice cafe here",  # дубликат
        "lol",                                                  # слишком короткий
        "[deleted]",                                            # заглушка
    ]})
    assert len(clean_frame(df)) == 1
