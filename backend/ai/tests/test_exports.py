import json

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from ai.clustering import export_clusters
from ai.data import write_json
from ai.train import train_release


def test_existing_output_is_never_overwritten(tmp_path):
    marker = tmp_path / "release.json"
    marker.write_text("frozen")
    with pytest.raises(ValueError, match="cannot be overwritten"):
        train_release(tmp_path, tmp_path / "unused")
    assert marker.read_text() == "frozen"


def test_json_rejects_nonfinite_numbers(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path / "invalid.json", {"score": float("nan")})
    assert not (tmp_path / "invalid.json").exists()


def test_artifact_json_uses_canonical_lf_bytes(tmp_path):
    path = tmp_path / "artifact.json"
    write_json(path, {"value": 1})
    assert b"\r\n" not in path.read_bytes()
    assert path.read_bytes().endswith(b"\n")


def test_cluster_export_is_bounded_deterministic_and_has_no_raw_messages():
    texts = ["meeting dinner normal", "normal dinner tomorrow", "cash free prize", "free prize winner", "project meeting tomorrow", "winner cash jackpot"]
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=2, random_state=42)
    normalizer = Normalizer()
    reduced = normalizer.fit_transform(svd.fit_transform(features))
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(reduced)
    pipeline = Pipeline([("tfidf", vectorizer), ("svd", svd), ("normalize", normalizer), ("clusterer", kmeans)])
    rows = [{"sample_id": f"sample-{index:04d}", "text": texts[index % len(texts)]} for index in range(1005, 0, -1)]
    result = export_clusters(pipeline, svd, rows, "test-release", "frozen-split", 42)
    assert result["sample_count"] == 1000
    assert result["total_count"] == 1005
    assert result["points"][0]["sample_id"] == "sample-0001"
    assert result["points"][-1]["sample_id"] == "sample-1000"
    assert sum(item["count"] for item in result["summaries"]) == 1005
    assert set(result["points"][0]) == {"sample_id", "x", "y", "cluster_id"}
    serialized = json.dumps(result)
    assert not any(text in serialized for text in texts)
    assert np.isfinite(result["points"][0]["x"])
