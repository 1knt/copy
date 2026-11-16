import hashlib
import json
import platform
from dataclasses import replace
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import scipy
import sklearn
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from app.artifacts import InvalidRelease, checked_bytes, load_release
from app.main import create_app
from app.service import infer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return digest(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(matrix):
    (tn, fp), (fn, tp) = matrix
    pairs = {"accuracy": (tn + tp, tn + fp + fn + tp), "precision": (tp, tp + fp),
             "recall": (tp, tp + fn), "f1": (2 * tp, 2 * tp + fp + fn), "false_positive_rate": (fp, fp + tn)}
    return {key: numerator / denominator if denominator else None for key, (numerator, denominator) in pairs.items()}


@pytest.fixture
def release_dir(tmp_path):
    texts = ["meeting tomorrow lunch", "family dinner tonight", "free prize claim cash", "win cash prize now"]
    labels = ["ham", "ham", "spam", "spam"]
    summary = {"schema_version": 1, "dataset_id": "fixture", "split_id": "b" * 64, "models": {}, "rows": [
        {"sample_id": f"row-{i}", "source_id": "uci_sms", "split": "test", "label": label, "length": len(text), "scores": {}}
        for i, (text, label) in enumerate(zip(texts, labels))]}
    release = {"schema_version": 1, "release_id": "fixture-v1", "primary_model_id": "naive_bayes", "dataset_id": "fixture",
               "dataset_sha256": "a" * 64, "split_id": "b" * 64, "split_manifest_sha256": "b" * 64, "models": {}, "artifacts": {}}
    split_hash = write_json(tmp_path / "split-manifest.json", {"dataset_sha256": "a" * 64,
                            "assignments": [{"sample_id": row["sample_id"], "partition": row["split"], "source_id": row["source_id"]} for row in summary["rows"]]})
    release["split_manifest_sha256"] = split_hash
    features = {"analyzer": "word", "ngram_range": [1, 2], "min_df": 1, "max_features": 100,
                "lowercase": True, "token_pattern": r"(?u)\b\w\w+\b", "strip_accents": None, "norm": "l2", "sublinear_tf": False}
    for model_id, classifier in (("naive_bayes", MultinomialNB()), ("logistic_regression", LogisticRegression())):
        directory = tmp_path / model_id
        directory.mkdir()
        pipeline = Pipeline([("tfidf", TfidfVectorizer(**{**features, "ngram_range": (1, 2)})), ("classifier", classifier)]).fit(texts, labels)
        joblib.dump(pipeline, directory / "pipeline.joblib")
        scores = pipeline.predict_proba(texts)[:, list(pipeline.classes_).index("spam")]
        matrix = [[0, 0], [0, 0]]
        for i, score in enumerate(scores):
            summary["rows"][i]["scores"][model_id] = float(score)
            matrix[int(labels[i] == "spam")][int(score >= 0.5)] += 1
        values = metrics(matrix)
        metric_record = {"sample_count": 4, "support": {"ham": 2, "spam": 2},
                         "confusion_matrix": {"labels": ["ham", "spam"], "matrix": matrix}, "metrics": values,
                         "undefined_metrics": {key: "Zero denominator." for key, value in values.items() if value is None}}
        version = f"{model_id}-fixture"
        evaluation = {**metric_record, "model_id": model_id, "model_version": version, "dataset_sha256": "a" * 64,
                      "split_manifest_sha256": split_hash, "decision_threshold": 0.5, "partition": "test",
                      "operational": {**metric_record, "total_count": 4, "classified_count": 4, "abstained_count": 0,
                                      "coverage": 1.0, "abstention_support": {"ham": 0, "spam": 0}}}
        manifest = {"schema_version": 1, "model_id": model_id, "model_version": version, "class_labels": ["ham", "spam"],
                    "positive_label": "spam", "decision_threshold": 0.5, "dataset_sha256": "a" * 64,
                    "split_manifest_sha256": split_hash, "pipeline_sha256": digest(directory / "pipeline.joblib"),
                    "evaluation_sha256": write_json(directory / "evaluation.json", evaluation), "feature_configuration": features,
                    "preprocessing_version": "tfidf-v1", "library_versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                    "numpy": np.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__}}
        release["models"][model_id] = {"directory": model_id, "manifest_sha256": write_json(directory / "model-manifest.json", manifest)}
        summary["models"][model_id] = version
    clusters = {"clustering_version": "kmeans-fixture", "projection_version": "svd-fixture", "source_id": "uci_sms",
                "split_id": "b" * 64, "partition": "test", "sample_count": 4, "total_count": 4,
                "points": [{"sample_id": f"row-{i}", "x": float(i), "y": 0.0, "cluster_id": 0} for i in range(4)],
                "summaries": [{"cluster_id": 0, "count": 4}, {"cluster_id": 1, "count": 0}]}
    for key, filename, value in (("dataset_summary", "dataset-summary.json", summary), ("clusters", "cluster-visualization.json", clusters)):
        release["artifacts"][key] = {"file": filename, "sha256": write_json(tmp_path / filename, value)}
    (tmp_path / "clustering.joblib").write_bytes(b"not deserialized by serving")
    release["artifacts"]["clustering_pipeline"] = {"file": "clustering.joblib", "sha256": digest(tmp_path / "clustering.joblib")}
    write_json(tmp_path / "release.json", release)
    return tmp_path


def rehash_model(root, manifest, evaluation=None):
    directory = root / manifest["model_id"]
    if evaluation is not None:
        manifest["evaluation_sha256"] = write_json(directory / "evaluation.json", evaluation)
    release = read_json(root / "release.json")
    release["models"][manifest["model_id"]]["manifest_sha256"] = write_json(directory / "model-manifest.json", manifest)
    write_json(root / "release.json", release)


def test_real_pipeline_parity_and_abstention(release_dir):
    release = load_release(release_dir)
    with TestClient(create_app(release_dir)) as client:
        assert client.get("/api/v1/health/ready").status_code == 200
        response = client.post("/api/v1/predictions", json={"text": "free cash prize"})
        expected = release.models["naive_bayes"].pipeline.predict_proba(["free cash prize"])[0, 1]
        assert response.json()["spam_score"] == pytest.approx(expected)
        assert response.json()["label"] == "spam"
        comparison = client.post("/api/v1/comparisons", json={"text": "!!!"}).json()
        assert {row["model_id"] for row in comparison["results"]} == {"naive_bayes", "logistic_regression"}
        assert all(row["status"] == "unable_to_assess" and row["spam_score"] is None for row in comparison["results"])
        assert "free cash prize" not in response.text


def test_class_mapping_and_threshold_equality(release_dir):
    model = load_release(release_dir).models["naive_bayes"]
    fake = SimpleNamespace(classes_=np.array(["spam", "ham"]), predict_proba=lambda value: np.array([[0.5, 0.5]]))
    pipeline = SimpleNamespace(named_steps={"tfidf": model.pipeline.named_steps["tfidf"], "classifier": fake})
    prediction = infer(replace(model, pipeline=pipeline), "free prize", "test-id")
    assert prediction.spam_score == 0.5 and prediction.label == "spam"
    prediction = infer(replace(model, pipeline=pipeline, threshold=0.5001), "free prize", "test-id")
    assert prediction.label == "ham"


@pytest.mark.parametrize("relative", ["naive_bayes/pipeline.joblib", "logistic_regression/evaluation.json", "dataset-summary.json", "clustering.joblib", "split-manifest.json"])
def test_tampering_fails_before_deserialization(release_dir, relative, monkeypatch):
    path = release_dir / relative
    path.write_bytes(path.read_bytes() + b"tampered")
    calls = []
    monkeypatch.setattr(joblib, "load", lambda source: calls.append(source))
    with TestClient(create_app(release_dir)) as client:
        assert client.get("/api/v1/health/ready").status_code == 503
        assert client.post("/api/v1/predictions", json={"text": "valid"}).status_code == 503
    assert calls == []


@pytest.mark.parametrize("mutation", ["version", "threshold", "classes", "features", "library", "fpr", "support", "abstention", "matrix"])
def test_self_consistent_hashes_do_not_hide_invalid_metadata(release_dir, mutation):
    manifest = read_json(release_dir / "naive_bayes/model-manifest.json")
    evaluation = read_json(release_dir / "naive_bayes/evaluation.json")
    if mutation == "version":
        evaluation["model_version"] = "different"
    elif mutation == "threshold":
        manifest["decision_threshold"] = float("nan")
    elif mutation == "classes":
        manifest["class_labels"] = ["spam", "ham"]
    elif mutation == "features":
        manifest["feature_configuration"]["max_features"] = 99
    elif mutation == "library":
        manifest["library_versions"]["sklearn"] = "0.0.0"
    elif mutation == "fpr":
        evaluation["metrics"]["false_positive_rate"] = 0.9
    elif mutation == "support":
        evaluation["operational"]["support"]["ham"] = 99
    elif mutation == "abstention":
        evaluation["operational"]["abstention_support"]["ham"] = 3
    else:
        evaluation["confusion_matrix"]["matrix"] = [[0, 2], [2, 0]]
        evaluation["metrics"] = metrics([[0, 2], [2, 0]])
    rehash_model(release_dir, manifest, evaluation)
    with TestClient(create_app(release_dir)) as client:
        assert client.get("/api/v1/health/ready").status_code == 503


def test_paths_duplicate_keys_and_size_are_rejected(release_dir):
    release = read_json(release_dir / "release.json")
    release["models"]["naive_bayes"]["directory"] = "../outside"
    write_json(release_dir / "release.json", release)
    with pytest.raises(InvalidRelease):
        load_release(release_dir)
    with pytest.raises(InvalidRelease):
        checked_bytes(release_dir, "release.json", maximum=2)
    (release_dir / "release.json").write_text('{"schema_version":1,"schema_version":2}', encoding="utf-8")
    with pytest.raises(InvalidRelease):
        load_release(release_dir)


def test_symlink_model_is_not_loaded(release_dir, tmp_path):
    original = release_dir / "naive_bayes/pipeline.joblib"
    target = release_dir / "outside.joblib"
    original.rename(target)
    try:
        original.symlink_to(target)
    except OSError:
        pytest.skip("Symlink creation requires an OS privilege on this host.")
    with pytest.raises(InvalidRelease):
        load_release(release_dir)


def test_bounded_inference_and_sanitized_failures(release_dir, monkeypatch, caplog):
    with TestClient(create_app(release_dir, inference_concurrency=1)) as client:
        service = client.app.state.release_service
        service.slots.acquire()
        try:
            response = client.post("/api/v1/predictions", json={"text": "secret payload"})
            assert response.status_code == 503 and response.json()["error"]["code"] == "SERVICE_BUSY"
        finally:
            service.slots.release()
        vectorizer = service.resolve().pipeline.named_steps["tfidf"]
        def fail(texts):
            raise RuntimeError(texts[0])
        monkeypatch.setattr(vectorizer, "transform", fail)
        response = client.post("/api/v1/predictions", json={"text": "secret payload"})
        assert response.status_code == 500
        assert "secret payload" not in response.text and "secret payload" not in caplog.text
        assert service.slots.acquire(blocking=False)
        service.slots.release()


def test_unexpected_endpoint_exception_never_escapes_to_server_logs(release_dir, caplog):
    app = create_app(release_dir)
    @app.get("/explode")
    def explode():
        raise RuntimeError("private input must not escape")
    with TestClient(app, raise_server_exceptions=True) as client:
        response = client.get("/explode")
        assert response.status_code == 500
        assert "private input must not escape" not in response.text + caplog.text


def test_summary_filters_and_absent_sampled_cluster(release_dir):
    with TestClient(create_app(release_dir)) as client:
        summary = client.get("/api/v1/dataset-summary?actual_class=ham&split=test&model_id=logistic_regression").json()
        assert summary["sample_count"] == 2 and summary["excluded_count"] == 2
        assert sum(summary["score_histograms"]["all_counts"]) == 2
        assert sum(summary["length_bins"]["counts"]) == 2
        empty = client.get("/api/v1/dataset-summary?split=train").json()
        assert empty["sample_count"] == 0
        assert client.get("/api/v1/clusters?cluster_id=1").json()["points"] == []
        assert client.get("/api/v1/clusters?cluster_id=2").status_code == 422


def test_startup_loads_once_and_schema_documents_success(release_dir):
    with TestClient(create_app(release_dir)) as client:
        (release_dir / "naive_bayes/pipeline.joblib").write_bytes(b"changed after startup")
        assert client.post("/api/v1/predictions", json={"text": "free cash"}).status_code == 200
        paths = client.get("/openapi.json").json()["paths"]
        for path, methods in paths.items():
            for value in methods.values():
                assert value["responses"]["200"]["content"]["application/json"]["schema"]
    with TestClient(create_app(release_dir)) as client:
        assert client.get("/api/v1/health/ready").status_code == 503


@pytest.mark.parametrize("scores", [np.array([[float("nan"), 0.5]]), np.array([[0.8, 0.8]]), np.array([[-0.1, 1.1]]), np.array([[1.0]])])
def test_invalid_model_outputs_fail_safely(release_dir, monkeypatch, scores):
    with TestClient(create_app(release_dir)) as client:
        classifier = client.app.state.release_service.resolve().pipeline.named_steps["classifier"]
        monkeypatch.setattr(classifier, "predict_proba", lambda value: scores)
        response = client.post("/api/v1/predictions", json={"text": "free cash prize"})
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "INFERENCE_FAILED"


def test_histogram_edges_include_zero_one_and_overflow(release_dir):
    with TestClient(create_app(release_dir)) as client:
        service = client.app.state.release_service
        rows = tuple({**row, "length": length, "scores": {"naive_bayes": score, "logistic_regression": score}}
                     for row, length, score in zip(service.release.rows, (40, 2000, 2001, 1000000), (0, 0.1, 1.0, None)))
        service.release = replace(service.release, rows=rows)
        summary = client.get("/api/v1/dataset-summary").json()
        histogram = summary["score_histograms"]
        assert histogram["all_counts"] == [1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
        assert summary["abstained_count"] == 1
        assert summary["length_bins"]["counts"][1] == 1
        assert summary["length_bins"]["overflow_count"] == 3
        assert sum(summary["length_bins"]["counts"]) == 4


def test_projection_and_unknown_artifacts_are_verified(release_dir, monkeypatch):
    registry = read_json(release_dir / "release.json")
    (release_dir / "projection.joblib").write_bytes(b"projection")
    registry["artifacts"]["projection"] = {"file": "projection.joblib", "sha256": digest(release_dir / "projection.joblib")}
    write_json(release_dir / "release.json", registry)
    assert load_release(release_dir).release_id == "fixture-v1"
    (release_dir / "projection.joblib").write_bytes(b"tampered")
    calls = []
    monkeypatch.setattr(joblib, "load", lambda source: calls.append(source))
    with pytest.raises(InvalidRelease):
        load_release(release_dir)
    assert calls == []
    registry["artifacts"]["unrecognized"] = {"file": "arbitrary", "sha256": "a" * 64}
    write_json(release_dir / "release.json", registry)
    with pytest.raises(InvalidRelease):
        load_release(release_dir)


def test_selection_reference_requires_matching_file(release_dir):
    registry = read_json(release_dir / "release.json")
    registry["selection_decision_sha256"] = "a" * 64
    write_json(release_dir / "release.json", registry)
    with TestClient(create_app(release_dir)) as client:
        assert client.get("/api/v1/health/ready").status_code == 503
