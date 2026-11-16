import json
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ai.data import sha256_file
from ai.emoji_text import text_for_model
from ai.predict import predict


def build_release(root, tamper=False):
    texts = ["win a free prize now", "claim cash prize today", "see you at lunch", "are you home yet", "free entry to win", "call me later tonight"]
    labels = ["spam", "spam", "ham", "ham", "spam", "ham"]
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", LogisticRegression())]).fit(texts, labels)
    directory = root / "logistic_regression"
    directory.mkdir(parents=True)
    joblib.dump(pipeline, directory / "pipeline.joblib")
    digest = "0" * 64 if tamper else sha256_file(directory / "pipeline.joblib")
    (directory / "model-manifest.json").write_text(json.dumps({"pipeline_sha256": digest, "decision_threshold": 0.5}))
    (root / "release.json").write_text(json.dumps({"primary_model_id": "logistic_regression"}))


def test_predict_classifies_and_abstains(tmp_path):
    build_release(tmp_path)
    results = predict(["claim your free prize", "qqqq"], tmp_path)
    assert results[0]["status"] == "classified" and results[0]["label"] == "spam"
    assert results[1] == {"model_id": "logistic_regression", "label": None, "spam_score": None, "status": "unable_to_assess", "characters": 4}


def test_predict_refuses_tampered_pipeline(tmp_path):
    build_release(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="checksum"):
        predict(["hello"], tmp_path)


def test_predict_reads_emoji_as_words(tmp_path):
    build_release(tmp_path)
    result = predict(["🆓🆓"], tmp_path)[0]
    assert result["status"] == "classified" and result["label"] == "spam"
    assert text_for_model("Nokia™ costs £100 :)") == "Nokia™ costs £100 :)"
    backend = Path(__file__).resolve().parents[2]
    assert (backend / "ai" / "emoji_text.py").read_bytes() == (backend / "fastapi" / "app" / "emoji_text.py").read_bytes()
