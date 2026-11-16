from bisect import bisect_right
from threading import BoundedSemaphore

import numpy as np

from .artifacts import MODEL_IDS, Release
from .emoji_text import text_for_model
from .errors import ServiceError
from .schemas import (
    ClustersResponse, ComparisonResponse, ComparisonResult, DatasetSummaryResponse,
    ModelResponse, PredictionResponse,
)


SCORE_EDGES = [index / 10 for index in range(11)]
LENGTH_EDGES = [0, 40, 80, 120, 160, 320, 640, 1280, 2000, 1000001]


def infer(model, text, request_id):
    vectorizer = model.pipeline.named_steps["tfidf"]
    classifier = model.pipeline.named_steps["classifier"]
    features = vectorizer.transform([text_for_model(text)])
    if features.nnz == 0:
        return PredictionResponse(request_id=request_id, status="unable_to_assess", label=None,
                                  spam_score=None, threshold=model.threshold, model_version=model.version,
                                  reason_code="NO_RECOGNIZED_FEATURES")
    scores = classifier.predict_proba(features)
    classes = list(classifier.classes_)
    if scores.shape != (1, len(classes)) or set(classes) != {"ham", "spam"}:
        raise ValueError("Invalid classifier output.")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any() or not np.allclose(scores.sum(axis=1), 1):
        raise ValueError("Invalid classifier output.")
    score = float(scores[0, classes.index("spam")])
    return PredictionResponse(request_id=request_id, status="classified", label="spam" if score >= model.threshold else "ham",
                              spam_score=score, threshold=model.threshold, model_version=model.version, reason_code=None)


class ReleaseService:
    def __init__(self, release: Release | None, concurrency: int):
        self.release = release
        self.slots = BoundedSemaphore(concurrency)

    def require_release(self, code="MODEL_UNAVAILABLE"):
        if self.release is None:
            raise ServiceError(code, "The verified release is unavailable.")
        return self.release

    def resolve(self, model_id="primary", code="MODEL_UNAVAILABLE"):
        release = self.require_release(code)
        resolved = release.primary_model_id if model_id == "primary" else model_id
        return release.models[resolved]

    def screen(self, text, request_id, comparison=False):
        release = self.require_release("COMPARISON_UNAVAILABLE" if comparison else "MODEL_UNAVAILABLE")
        if not self.slots.acquire(blocking=False):
            raise ServiceError("SERVICE_BUSY", "Analysis is busy. Try again shortly.")
        try:
            if comparison:
                results = [ComparisonResult(model_id=model_id, **infer(release.models[model_id], text, request_id).model_dump()) for model_id in MODEL_IDS]
                return ComparisonResponse(request_id=request_id, results=results)
            return infer(release.models[release.primary_model_id], text, request_id)
        except Exception:
            raise ServiceError("INFERENCE_FAILED", "The message could not be analyzed.", 500) from None
        finally:
            self.slots.release()

    def metadata(self, model_id):
        model = self.resolve(model_id)
        return ModelResponse(model_id=model.model_id, model_version=model.version, threshold=model.threshold,
                             positive_label=model.evaluation.positive_label,
                             release_id=self.release.release_id,
                             label_meanings={"ham": "Likely legitimate", "spam": "Likely spam"},
                             limitations=["Scores are model estimates, not verified safety probabilities.",
                                          "Evaluated on an historical English SMS corpus; other languages and domains are outside the validated scope.",
                                          "A ham classification is not a malware verdict.",
                                          "Messages with no recognized features receive no classification."])

    def summary(self, source, split, model_id, actual_class):
        release = self.require_release("DATA_UNAVAILABLE")
        model = self.resolve(model_id, "DATA_UNAVAILABLE")
        rows = [row for row in release.rows if (source == "all" or row["source_id"] == source)
                and (split == "all" or row["split"] == split)
                and (actual_class == "all" or row["label"] == actual_class)]
        lengths = [0] * (len(LENGTH_EDGES) - 1)
        by_class = {"ham": [0] * 10, "spam": [0] * 10}
        abstained = 0
        for row in rows:
            lengths[min(bisect_right(LENGTH_EDGES, row["length"]) - 1, len(lengths) - 1)] += 1
            score = row["scores"][model.model_id]
            if score is None:
                abstained += 1
            else:
                by_class[row["label"]][min(bisect_right(SCORE_EDGES, score) - 1, 9)] += 1
        return DatasetSummaryResponse(
            dataset_id=release.dataset_id, split_id=release.split_id, sample_count=len(rows),
            excluded_count=len(release.rows) - len(rows), abstained_count=abstained,
            source_counts={source: sum(row["source_id"] == source for row in rows) for source in sorted({row["source_id"] for row in rows})},
            class_counts={label: sum(row["label"] == label for row in rows) for label in ("ham", "spam")},
            length_bins={"bin_edges": LENGTH_EDGES, "counts": lengths, "overflow_count": lengths[-1]},
            score_histograms={"model_id": model.model_id, "model_version": model.version, "bin_edges": SCORE_EDGES,
                              "all_counts": [ham + spam for ham, spam in zip(by_class["ham"], by_class["spam"])],
                              "ham_counts": by_class["ham"], "spam_counts": by_class["spam"]},
            artifact_versions={"release": release.release_id, **{key: value.version for key, value in release.models.items()}},
        )

    def cluster_data(self, cluster_id):
        release = self.require_release("CLUSTERS_UNAVAILABLE")
        artifact = release.clusters
        summaries = artifact["summaries"]
        if cluster_id is not None and cluster_id not in {entry["cluster_id"] for entry in summaries}:
            raise ServiceError("INVALID_REQUEST", "The cluster filter is invalid.", 422)
        summaries = [entry for entry in summaries if cluster_id is None or entry["cluster_id"] == cluster_id]
        points = [point for point in artifact["points"] if cluster_id is None or point["cluster_id"] == cluster_id]
        return ClustersResponse(
            clustering_version=artifact["clustering_version"], projection_version=artifact["projection_version"],
            source_id=artifact["source_id"], split_id=artifact["split_id"], partition="test",
            sample_count=len(points), total_count=sum(entry["count"] for entry in summaries), points=points,
            clusters=[{"cluster_id": entry["cluster_id"], "size": entry["count"],
                       "description": f"Exploratory cluster {entry['cluster_id']}"} for entry in summaries],
        )
