import json
import math
import os
import re
from http.client import HTTPSConnection

from .errors import ServiceError


HF_HOST = "router.huggingface.co"
TIMEOUT_SECONDS = 10
RESPONSE_LIMIT = 65536
MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")


def unavailable():
    return ServiceError("EXTERNAL_REFERENCE_UNAVAILABLE", "The external reference is unavailable.")


def validate_scores(value):
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not 1 <= len(value) <= 100:
        raise ValueError("Invalid classification response.")
    scores = []
    labels = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"label", "score"}:
            raise ValueError("Invalid classification response.")
        label, score = item["label"], item["score"]
        if not isinstance(label, str) or not label.strip() or len(label) > 200 or label in labels:
            raise ValueError("Invalid classification response.")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid classification response.")
        labels.add(label)
        scores.append({"label": label, "score": score})
    return scores


def classify_external_reference(text: str):
    token = os.environ.get("HF_TOKEN", "").strip()
    model_id = os.environ.get("HF_MODEL_ID", "").strip()
    if not token or not MODEL_ID.fullmatch(model_id) or ".." in model_id or "--" in model_id:
        raise unavailable()
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ServiceError("INVALID_TEXT", "Enter a message between 1 and 2000 characters.", 422)
    connection = None
    try:
        connection = HTTPSConnection(HF_HOST, timeout=TIMEOUT_SECONDS)
        connection.request(
            "POST", f"/hf-inference/models/{model_id}",
            body=json.dumps({"inputs": text}).encode("utf-8"),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("Provider unavailable.")
        body = response.read(RESPONSE_LIMIT + 1)
        if len(body) > RESPONSE_LIMIT:
            raise ValueError("Response too large.")
        scores = validate_scores(json.loads(body))
    except Exception:
        raise unavailable() from None
    finally:
        if connection is not None:
            connection.close()
    return {"provider": "huggingface", "model_id": model_id, "source": "external_reference", "scores": scores}
