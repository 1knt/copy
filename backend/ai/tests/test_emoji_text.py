from pathlib import Path

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from ai.emoji_text import LEGACY_PREPROCESSING_VERSION, PREPROCESSING_VERSION, find_emojis, model_input, text_for_model


BACKEND = Path(__file__).resolve().parents[2]


def test_training_and_serving_copies_are_identical():
    training = BACKEND / "ai" / "emoji_text.py"
    serving = BACKEND / "fastapi" / "app" / "emoji_text.py"
    assert training.read_bytes() == serving.read_bytes(), "Copy ai/emoji_text.py to fastapi/app/emoji_text.py after editing it."


@pytest.mark.parametrize("text", ["Are we still meeting at 6?", "Claim £100 now!!! www.example.com", "Nokia™ ©2011 ® :) <3", "OCR lost a character: �", "Tiếng Việt có dấu", ""])
def test_text_without_emoji_is_unchanged(text):
    assert text_for_model(text) == text
    assert find_emojis(text) == []


@pytest.mark.parametrize("text,names", [
    ("You won a 🎁", ["wrapped gift"]),
    ("👍🏽👍", ["thumbs up", "thumbs up"]),
    ("👨‍👩‍👧 dinner", ["family man woman girl"]),
    ("🇻🇳", ["Vietnam"]),
    ("Press 1️⃣ now", ["keycap 1"]),
    ("Brand™️", ["trade mark"]),
    ("🆓💰👉", ["FREE button", "money bag", "backhand index pointing right"]),
])
def test_find_emojis_reads_whole_sequences(text, names):
    found = find_emojis(text)
    assert [item["name"] for item in found] == names
    assert "".join(item["emoji"] for item in found) in text


def test_emoji_become_words_the_default_vectorizer_keeps():
    analyzer = TfidfVectorizer().build_analyzer()
    assert analyzer("🎁💰👉") == []
    assert analyzer(text_for_model("🎁💰👉")) == ["emoji", "wrapped", "gift", "emoji", "money", "bag", "emoji", "backhand", "index", "pointing", "right"]
    assert analyzer(text_for_model("free🎁gift")) == ["free", "emoji", "wrapped", "gift", "gift"]


def test_model_input_follows_the_release_preprocessing_version():
    assert model_input("🎁 gift", LEGACY_PREPROCESSING_VERSION) == "🎁 gift"
    assert model_input("🎁 gift", PREPROCESSING_VERSION) == text_for_model("🎁 gift")
    with pytest.raises(ValueError, match="Unknown preprocessing"):
        model_input("🎁", "tfidf-word-emoji-v99")
