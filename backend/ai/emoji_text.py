"""Turn emoji into words before a message is scored.

The TF-IDF tokenizer keeps only letters and digits, so it drops emoji. Each emoji is
replaced by its English name from the emoji library, without the skin-tone part
("🎁" becomes " wrapped gift "), so the model reads it with vocabulary it already
learned. Plain ©, ® and ™ are brand typography and stay as they are. Text without
emoji is returned unchanged.

backend/ai/emoji_text.py and backend/fastapi/app/emoji_text.py must stay identical.
"""

import re

import emoji


SKIN_TONE = re.compile(r"(?:^|_)(?:light|medium-light|medium|medium-dark|dark)_skin_tone(?=_|$)")


def _as_words(sequence, data):
    if sequence in ("©", "®", "™"):
        return sequence
    return " " + SKIN_TONE.sub("", data["en"].strip(":")).replace("_", " ") + " "


def text_for_model(text):
    return emoji.replace_emoji(text, replace=_as_words)
