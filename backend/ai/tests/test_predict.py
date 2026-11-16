import json

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ai.data import sha256_file
from ai.emoji_text import PREPROCESSING_VERSION, text_for_model
from ai.predict import predict


def build_release(root, tamper=False, preprocessing=None):
    texts = ["win a free gift card now", "claim cash prize today", "see you at lunch", "are you home yet", "free entry to win", "call me later tonight"]
    labels = ["spam", "spam", "ham", "ham", "spam", "ham"]
    if preprocessing == PREPROCESSING_VERSION:
        texts = [text_for_model(text) for text in texts]
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", LogisticRegression())]).fit(texts, labels)
    directory = root / "logistic_regression"
    directory.mkdir(parents=True)
    joblib.dump(pipeline, directory / "pipeline.joblib")
    digest = "0" * 64 if tamper else sha256_file(directory / "pipeline.joblib")
    manifest = {"pipeline_sha256": digest, "decision_threshold": 0.5}
    if preprocessing is not None:
        manifest["preprocessing_version"] = preprocessing
    (directory / "model-manifest.json").write_text(json.dumps(manifest))
    (root / "release.json").write_text(json.dumps({"primary_model_id": "logistic_regression"}))


def test_predict_classifies_and_abstains(tmp_path):
    build_release(tmp_path)
    results = predict(["claim your free prize", "qqqq"], tmp_path)
    assert results[0]["status"] == "classified" and results[0]["label"] == "spam"
    assert results[1] == {"model_id": "logistic_regression", "label": None, "spam_score": None, "status": "unable_to_assess", "characters": 4, "emojis": [], "emoji_handling": "ignored"}


def test_emoji_release_reads_emoji_that_a_legacy_release_ignores(tmp_path):
    legacy, aware = tmp_path / "legacy", tmp_path / "aware"
    build_release(legacy)
    build_release(aware, preprocessing=PREPROCESSING_VERSION)
    ignored = predict(["🎁🎁"], legacy)[0]
    assert ignored["status"] == "unable_to_assess" and ignored["emoji_handling"] == "ignored"
    read = predict(["🎁🎁"], aware)[0]
    assert read["status"] == "classified" and read["label"] == "spam" and read["emoji_handling"] == "read_as_words"
    assert read["emojis"] == [{"emoji": "🎁", "name": "wrapped gift"}, {"emoji": "🎁", "name": "wrapped gift"}]


def test_predict_refuses_unknown_preprocessing(tmp_path):
    build_release(tmp_path, preprocessing="tfidf-word-emoji-v99")
    with pytest.raises(ValueError, match="Unknown preprocessing"):
        predict(["hello"], tmp_path)


def test_predict_refuses_tampered_pipeline(tmp_path):
    build_release(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="checksum"):
        predict(["hello"], tmp_path)
