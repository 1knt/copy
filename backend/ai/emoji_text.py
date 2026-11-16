"""Emoji detection and emoji-aware text normalization for the SMS models.

The TF-IDF token pattern (?u)\\b\\w\\w+\\b drops every emoji, so a model that
reads raw text cannot see them: "🎁💰👉" has no features at all, and
"You won a 🎁" scores exactly like "You won a". Before vectorizing, each emoji
is therefore replaced with the marker word "emoji" followed by its standard
English (Unicode CLDR) name, so "🎁" becomes " emoji wrapped gift ". The model
then reads an emoji through vocabulary it already knows (gift, free, money,
phone, ...), and a release trained on messages that contain emoji also learns
the marker itself. Skin-tone modifiers are dropped from the name because they
carry no spam signal.

The copyright, registered and trade mark signs are treated as emoji only when
written with the emoji variation selector (U+FE0F). Written plainly they are
typography in brand names ("Nokia™") and are left as they are.

Text without emoji is returned unchanged, so a release trained with this step
sees exactly the same input as before for every emoji-free message.

Training (backend/ai) and serving (backend/fastapi/app) keep byte-identical
copies of this file because the API is deployed without the ai package.
backend/ai/tests/test_emoji_text.py fails when the two copies differ.
"""

import re

import emoji


PREPROCESSING_VERSION = "tfidf-word-emoji-v1"
LEGACY_PREPROCESSING_VERSION = "tfidf-word-v1"
PREPROCESSING_VERSIONS = (LEGACY_PREPROCESSING_VERSION, PREPROCESSING_VERSION)
MARKER = "emoji"
TYPOGRAPHIC_SIGNS = frozenset("©®™")
SKIN_TONE = re.compile(r"(?:^|_)(?:light|medium-light|medium|medium-dark|dark)_skin_tone(?=_|$)")


def library_version():
    return emoji.__version__


def english_name(alias):
    """Turn an emoji library alias such as ':thumbs_up_medium_skin_tone:' into 'thumbs up'."""
    name = SKIN_TONE.sub("", alias.strip(":")).replace("_", " ").strip()
    return name or "skin tone"


def find_emojis(text):
    """Return every emoji in reading order as {"emoji": sequence, "name": English name}."""
    found = []
    for match in emoji.emoji_list(text):
        sequence = match["emoji"]
        if sequence in TYPOGRAPHIC_SIGNS:
            continue
        found.append({"emoji": sequence, "name": english_name(emoji.EMOJI_DATA[sequence]["en"])})
    return found


def _replacement(sequence, entry):
    if sequence in TYPOGRAPHIC_SIGNS:
        return sequence
    return f" {MARKER} {english_name(entry['en'])} "


def text_for_model(text):
    """Return the text the vectorizer reads: every emoji replaced by ' emoji <name> '."""
    return emoji.replace_emoji(text, replace=_replacement)


def model_input(text, preprocessing_version):
    """Apply the normalization a release was trained with, named by its preprocessing version."""
    if preprocessing_version == PREPROCESSING_VERSION:
        return text_for_model(text)
    if preprocessing_version == LEGACY_PREPROCESSING_VERSION:
        return text
    raise ValueError(f"Unknown preprocessing version {preprocessing_version!r}.")
