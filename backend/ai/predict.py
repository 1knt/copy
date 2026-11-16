import argparse
import json
import sys
from pathlib import Path

import joblib

from ai.data import sha256_file


ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = ("naive_bayes", "logistic_regression")


def load_model(release, model_id):
    release = Path(release)
    registry = json.loads((release / "release.json").read_text(encoding="utf-8"))
    if model_id == "primary":
        model_id = registry["primary_model_id"]
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unknown model id {model_id}.")
    manifest = json.loads((release / model_id / "model-manifest.json").read_text(encoding="utf-8"))
    pipeline_path = release / model_id / "pipeline.joblib"
    if sha256_file(pipeline_path) != manifest["pipeline_sha256"]:
        raise ValueError("Pipeline checksum differs from its manifest; refusing to load it.")
    return model_id, joblib.load(pipeline_path), manifest["decision_threshold"]


def predict(texts, release, model_id="primary"):
    model_id, pipeline, threshold = load_model(release, model_id)
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    features = vectorizer.transform(texts)
    scores = classifier.predict_proba(features)[:, list(classifier.classes_).index("spam")]
    results = []
    for text, score, recognized in zip(texts, scores, features.getnnz(axis=1) > 0):
        if not recognized:
            results.append({"model_id": model_id, "label": None, "spam_score": None, "status": "unable_to_assess", "characters": len(text)})
        else:
            results.append({"model_id": model_id, "label": "spam" if score >= threshold else "ham", "spam_score": round(float(score), 4), "threshold": threshold, "status": "classified", "characters": len(text)})
    return results


def main():
    parser = argparse.ArgumentParser(description="Classify SMS messages with a frozen release model.")
    parser.add_argument("messages", nargs="*", help="Messages to classify. Reads one message per line from standard input when omitted.")
    parser.add_argument("--release", type=Path, default=ROOT / "fastapi" / "artifacts" / "current")
    parser.add_argument("--model", default="primary", choices=("primary", *MODEL_IDS))
    arguments = parser.parse_args()
    messages = arguments.messages or [line.strip() for line in sys.stdin if line.strip()]
    if not messages:
        parser.error("Provide at least one message.")
    for result in predict(messages, arguments.release, arguments.model):
        print(json.dumps(result))


if __name__ == "__main__":
    main()
