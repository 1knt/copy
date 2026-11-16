import hashlib
import io
import json
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .emoji_text import PREPROCESSING_VERSION, PREPROCESSING_VERSIONS, library_version as emoji_version
from .schemas import EvaluationResponse


MODEL_IDS = ("naive_bayes", "logistic_regression")
SOURCE_IDS = ("uci_sms", "mendeley_smishing")
HASH = re.compile(r"[0-9a-f]{64}")


class InvalidRelease(ValueError):
    pass


def require(condition):
    if not condition:
        raise InvalidRelease("The release failed integrity validation.")


def finite(value, low=None, high=None):
    require(type(value) in (int, float) and np.isfinite(value))
    require(low is None or value >= low)
    require(high is None or value <= high)


def count(value):
    require(type(value) is int and value >= 0)


def identifier(value):
    require(isinstance(value, str) and 0 < len(value) <= 200)
    require(re.fullmatch(r"[a-zA-Z0-9_.:-]+", value) is not None)


def checked_bytes(root, relative, expected=None, maximum=32 * 1024 * 1024):
    path = root / relative
    require(not root.is_symlink())
    require(all(not part.is_symlink() for part in [path, *path.parents] if part != root.parent))
    require(path.resolve().is_relative_to(root.resolve()))
    require(path.is_file() and path.stat().st_size <= maximum)
    with path.open("rb") as source:
        value = source.read(maximum + 1)
    require(len(value) <= maximum)
    if expected is not None:
        require(isinstance(expected, str) and HASH.fullmatch(expected) is not None)
        require(hashlib.sha256(value).hexdigest() == expected)
    return value


def json_bytes(value):
    def pairs(items):
        result = {}
        for key, item in items:
            require(key not in result)
            result[key] = item
        return result
    def invalid_constant(value):
        raise InvalidRelease("Non-finite JSON is invalid.")
    return json.loads(value, object_pairs_hook=pairs, parse_constant=invalid_constant)


def validate_metrics(metrics, matrix):
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = tn + fp + fn + tp
    expected = {
        "accuracy": (tn + tp) / total if total else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
    }
    for key, actual in expected.items():
        value = metrics[key]
        if actual is None:
            require(value is None)
        else:
            finite(value, 0, 1)
            require(abs(value - actual) < 1e-8)


def validate_matrix(value, total):
    require(value["labels"] == ["ham", "spam"])
    matrix = value["matrix"]
    require(isinstance(matrix, list) and len(matrix) == 2)
    for row in matrix:
        require(isinstance(row, list) and len(row) == 2)
        for cell in row:
            count(cell)
    require(sum(map(sum, matrix)) == total)
    return matrix


def validate_evaluation(value, manifest):
    for key in ("model_id", "model_version", "dataset_sha256", "split_manifest_sha256", "decision_threshold"):
        require(value[key] == manifest[key])
    if "selection_decision_sha256" in manifest:
        require(value.get("selection_decision_sha256") == manifest["selection_decision_sha256"])
    require(value["partition"] == "test")
    count(value["sample_count"])
    require(value["sample_count"] > 0)
    matrix = validate_matrix(value["confusion_matrix"], value["sample_count"])
    require(set(value["support"]) == {"ham", "spam"})
    for index, label in enumerate(("ham", "spam")):
        count(value["support"][label])
        require(sum(matrix[index]) == value["support"][label])
    validate_metrics(value["metrics"], matrix)
    operational = value["operational"]
    for key in ("sample_count", "total_count", "classified_count", "abstained_count"):
        count(operational[key])
    require(operational["total_count"] == value["sample_count"] and operational["sample_count"] == operational["classified_count"])
    require(operational["classified_count"] + operational["abstained_count"] == value["sample_count"])
    finite(operational["coverage"], 0, 1)
    require(abs(operational["coverage"] - operational["classified_count"] / value["sample_count"]) < 1e-8)
    operational_matrix = validate_matrix(operational["confusion_matrix"], operational["classified_count"])
    for i in range(2):
        require(all(operational_matrix[i][j] <= matrix[i][j] for j in range(2)))
    validate_metrics(operational["metrics"], operational_matrix)
    for index, label in enumerate(("ham", "spam")):
        count(operational["support"][label])
        count(operational["abstention_support"][label])
        require(operational["support"][label] == sum(operational_matrix[index]))
        require(operational["support"][label] + operational["abstention_support"][label] == value["support"][label])
    for item in (value, operational):
        for key, metric in item["metrics"].items():
            if metric is None:
                require(isinstance(item["undefined_metrics"].get(key), str) and bool(item["undefined_metrics"][key]))
    return EvaluationResponse.model_validate({**value, "metric_notes": value.get("undefined_metrics", {})})


@dataclass(frozen=True)
class LoadedModel:
    model_id: str
    version: str
    threshold: float
    pipeline: Pipeline
    evaluation: EvaluationResponse
    preprocessing_version: str


@dataclass(frozen=True)
class Release:
    release_id: str
    primary_model_id: str
    dataset_id: str
    split_id: str
    models: object
    rows: tuple
    clusters: dict


def validate_manifest(value, model_id, release):
    require(value["schema_version"] == 1 and value["model_id"] == model_id)
    identifier(value["model_version"])
    require(value["class_labels"] == ["ham", "spam"] and value["positive_label"] == "spam")
    for key in ("dataset_sha256", "split_manifest_sha256"):
        require(value[key] == release[key])
    if "selection_decision_sha256" in release:
        require(value.get("selection_decision_sha256") == release["selection_decision_sha256"])
    finite(value["decision_threshold"], 0, 1)
    versions = value["library_versions"]
    for key, version in {"sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__}.items():
        require(versions[key] == version)
    require(versions["python"].split(".")[:2] == platform.python_version().split(".")[:2])
    require(isinstance(value["feature_configuration"], dict))
    require({"analyzer", "ngram_range", "min_df", "max_features", "lowercase", "token_pattern", "strip_accents", "norm", "sublinear_tf"}.issubset(value["feature_configuration"]))
    identifier(value["preprocessing_version"])
    require(value["preprocessing_version"] in PREPROCESSING_VERSIONS)
    if value["preprocessing_version"] == PREPROCESSING_VERSION:
        require(versions.get("emoji") == emoji_version())


def validate_pipeline(pipeline, model_id, manifest):
    require(type(pipeline) is Pipeline)
    require([name for name, _ in pipeline.steps] == ["tfidf", "classifier"])
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    require(type(vectorizer) is TfidfVectorizer)
    expected = MultinomialNB if model_id == "naive_bayes" else LogisticRegression
    require(type(classifier) is expected)
    require(list(classifier.classes_) == manifest["class_labels"])
    require(vectorizer.analyzer == "word" and vectorizer.preprocessor is None and vectorizer.tokenizer is None)
    require(vectorizer.vocabulary_ and np.isfinite(vectorizer.idf_).all())
    require(len(vectorizer.vocabulary_) == len(vectorizer.idf_))
    require(sorted(vectorizer.vocabulary_.values()) == list(range(len(vectorizer.vocabulary_))))
    parameters = vectorizer.get_params()
    for key, value in manifest["feature_configuration"].items():
        require(key in parameters)
        actual = parameters[key]
        if isinstance(actual, tuple):
            actual = list(actual)
        require(actual == value)
    arrays = ("feature_log_prob_", "class_log_prior_") if model_id == "naive_bayes" else ("coef_", "intercept_")
    for name in arrays:
        require(np.isfinite(getattr(classifier, name)).all())
    require(classifier.n_features_in_ == len(vectorizer.vocabulary_))
    sample = vectorizer.transform(["release startup validation"])
    probabilities = classifier.predict_proba(sample)
    require(probabilities.shape == (1, 2) and np.isfinite(probabilities).all())
    require((probabilities >= 0).all() and (probabilities <= 1).all() and np.allclose(probabilities.sum(axis=1), 1))


def validate_summary(summary, release, models):
    require(summary["schema_version"] == 1)
    require(summary["dataset_id"] == release["dataset_id"] and summary["split_id"] == release["split_id"])
    require(summary["models"] == {model_id: model.version for model_id, model in models.items()})
    rows = summary["rows"]
    require(isinstance(rows, list) and 0 < len(rows) <= 100000)
    seen = set()
    for row in rows:
        require(set(row) == {"sample_id", "source_id", "split", "label", "length", "scores"})
        identifier(row["sample_id"])
        require(row["sample_id"] not in seen)
        seen.add(row["sample_id"])
        require(row["source_id"] in SOURCE_IDS and row["split"] in {"train", "validation", "test"})
        require(row["label"] in {"ham", "spam"})
        count(row["length"])
        require(0 < row["length"] <= 1000000)
        require(set(row["scores"]) == set(MODEL_IDS))
        for value in row["scores"].values():
            if value is not None:
                finite(value, 0, 1)
    test = [row for row in rows if row["split"] == "test"]
    for model in models.values():
        require(len(test) == model.evaluation.sample_count)
        require({label: sum(row["label"] == label for row in test) for label in ("ham", "spam")} == model.evaluation.support)
        matrix = [[0, 0], [0, 0]]
        for row in test:
            score = row["scores"][model.model_id]
            if score is not None:
                matrix[int(row["label"] == "spam")][int(score >= model.threshold)] += 1
        require(matrix == model.evaluation.operational.confusion_matrix.matrix)
    return tuple(rows)


def validate_clusters(clusters, release, rows):
    for key in ("clustering_version", "projection_version"):
        identifier(clusters[key])
    require(clusters["source_id"] in {*SOURCE_IDS, "combined"} and clusters["split_id"] == release["split_id"])
    require(clusters["partition"] == "test")
    test_ids = {row["sample_id"] for row in rows if row["split"] == "test"}
    require(clusters["total_count"] == len(test_ids))
    points = clusters["points"]
    require(len(points) <= 1000 and clusters["sample_count"] == len(points))
    require(len(points) == min(1000, len(test_ids)))
    summaries = clusters["summaries"]
    ids = set()
    for summary in summaries:
        count(summary["cluster_id"])
        count(summary["count"])
        require(summary["cluster_id"] not in ids)
        ids.add(summary["cluster_id"])
    require(sum(summary["count"] for summary in summaries) == len(test_ids))
    seen = set()
    for point in points:
        require(set(point) == {"sample_id", "x", "y", "cluster_id"})
        require(point["sample_id"] in test_ids and point["sample_id"] not in seen)
        seen.add(point["sample_id"])
        finite(point["x"])
        finite(point["y"])
        require(point["cluster_id"] in ids)
    require([point["sample_id"] for point in points] == sorted(test_ids)[:1000])
    for summary in summaries:
        require(sum(point["cluster_id"] == summary["cluster_id"] for point in points) <= summary["count"])


def load_release(root: Path):
    release = json_bytes(checked_bytes(root, "release.json", maximum=65536))
    require(release["schema_version"] == 1)
    for key in ("release_id", "dataset_id", "split_id"):
        identifier(release[key])
    for key in ("dataset_sha256", "split_manifest_sha256"):
        require(HASH.fullmatch(release[key]) is not None)
    require(release["primary_model_id"] in MODEL_IDS and set(release["models"]) == set(MODEL_IDS))
    split_manifest = json_bytes(checked_bytes(root, "split-manifest.json", release["split_manifest_sha256"]))
    require(split_manifest["dataset_sha256"] == release["dataset_sha256"])
    selection = None
    if "selection_decision_sha256" in release:
        selection = json_bytes(checked_bytes(root, "selection-decision.json", release["selection_decision_sha256"]))
        for key in ("dataset_sha256", "split_manifest_sha256", "primary_model_id", "release_id"):
            require(selection[key] == release[key])
        require(selection["selection_partition"] == "validation")
    pending = []
    for model_id in MODEL_IDS:
        entry = release["models"][model_id]
        require(entry["directory"] == model_id)
        manifest = json_bytes(checked_bytes(root, f"{model_id}/model-manifest.json", entry["manifest_sha256"], 65536))
        validate_manifest(manifest, model_id, release)
        if selection is not None:
            for key in ("pipeline_sha256", "decision_threshold"):
                require(selection["models"][model_id][key] == manifest[key])
        evaluation = validate_evaluation(json_bytes(checked_bytes(root, f"{model_id}/evaluation.json", manifest["evaluation_sha256"], 262144)), manifest)
        pipeline_bytes = checked_bytes(root, f"{model_id}/pipeline.joblib", manifest["pipeline_sha256"])
        pending.append((model_id, manifest, evaluation, pipeline_bytes))
    artifacts = release["artifacts"]
    values = {}
    files = {"dataset_summary": "dataset-summary.json", "clusters": "cluster-visualization.json", "clustering_pipeline": "clustering.joblib"}
    if "projection" in artifacts:
        files["projection"] = "projection.joblib"
    require(set(artifacts) == set(files))
    if selection is not None:
        require("projection" in artifacts)
        require(selection["projection_sha256"] == artifacts["projection"]["sha256"])
        require(selection["clustering_pipeline_sha256"] == artifacts["clustering_pipeline"]["sha256"])
    for key, filename in files.items():
        require(artifacts[key]["file"] == filename)
        value = checked_bytes(root, filename, artifacts[key]["sha256"])
        if key in {"dataset_summary", "clusters"}:
            values[key] = json_bytes(value)
    models = {}
    for model_id, manifest, evaluation, pipeline_bytes in pending:
        pipeline = joblib.load(io.BytesIO(pipeline_bytes))
        validate_pipeline(pipeline, model_id, manifest)
        evaluation = evaluation.model_copy(update={"positive_label": manifest["positive_label"]})
        models[model_id] = LoadedModel(model_id, manifest["model_version"], manifest["decision_threshold"], pipeline, evaluation, manifest["preprocessing_version"])
    rows = validate_summary(values["dataset_summary"], release, models)
    assignments = split_manifest["assignments"]
    require(len(assignments) == len(rows))
    require({(entry["sample_id"], entry["partition"], entry["source_id"]) for entry in assignments}
            == {(row["sample_id"], row["split"], row["source_id"]) for row in rows})
    validate_clusters(values["clusters"], release, rows)
    return Release(release["release_id"], release["primary_model_id"], release["dataset_id"], release["split_id"], MappingProxyType(models), rows, values["clusters"])
