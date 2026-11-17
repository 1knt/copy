import argparse
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.dummy import DummyClassifier

from ai.clustering import export_clusters, select_clustering
from ai.data import SEED, save_prepared_data, sha256_bytes, sha256_file, split_records, timestamp, write_json
from ai.experiments import select_classifiers
from ai.metrics import classification_metrics, evaluate_scores, score_model
from ai.sources import load_combined


ROOT = Path(__file__).resolve().parents[1]


def source_identity():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    sources = sorted((ROOT / "ai").glob("*.py"))
    inventory = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sources}
    source_hash = sha256_bytes(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode())
    return result.stdout.strip(), source_hash, inventory


def versions():
    return {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__}


def train_release(output, data_directory, seed=SEED):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Release output already exists. Choose a new directory; frozen releases cannot be overwritten.")
    if not output.name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in output.name):
        raise ValueError("Release directory name must contain only letters, digits, hyphens and underscores.")
    revision, source_hash, source_inventory = source_identity()
    output.mkdir(parents=True)
    started = perf_counter()
    records, mapping, quality, provenance = load_combined(Path(data_directory) / "raw")
    partitions = split_records(records, seed)
    prepared = Path(data_directory) / "processed" / f"combined-sms-seed-{seed}"
    split_manifest, split_hash = save_prepared_data(prepared, partitions, provenance, quality, mapping, seed)
    environment_hash = sha256_bytes(json.dumps(versions(), sort_keys=True).encode())
    release_id = f"sms-{source_hash[:10]}-{split_hash[:10]}-{environment_hash[:6]}"
    split_id = f"combined_sms_{split_hash[:16]}"
    write_json(output / "source-provenance.json", provenance)
    write_json(output / "quality-report.json", quality)
    write_json(output / "split-manifest.json", split_manifest)
    write_json(output / "training-source.json", {"source_revision": revision, "source_sha256": source_hash, "files": source_inventory, "note": "The source hash records exact training modules, including any changes not yet committed at execution time."})
    print(f"Prepared {len(records)} records; training {len(partitions['train'])}, validation {len(partitions['validation'])}, test {len(partitions['test'])}.", flush=True)
    selected, experiment_records, primary = select_classifiers(partitions, seed)
    print("Classifier selection complete; selecting clustering on validation data.", flush=True)
    clustering, projection, clustering_decision = select_clustering(partitions, seed)
    decision = {
        "schema_version": 1,
        "release_id": release_id,
        "created_at": timestamp(),
        "seed": seed,
        "dataset_sha256": provenance["dataset_sha256"],
        "split_manifest_sha256": split_hash,
        "source_revision": revision,
        "source_sha256": source_hash,
        "library_versions": versions(),
        "selection_partition": "validation",
        "primary_model_id": primary,
        "classification_selection_rule": "At default threshold maximize spam F1, then precision, then fewer vocabulary features, then lower parameter. Tune threshold only on the selected family candidate; break ties by distance to 0.5, then higher threshold. Primary uses tuned validation F1, precision, then naive_bayes.",
        "models": {model_id: {"configuration": model["configuration"], "decision_threshold": model["threshold"], "validation": model["validation"], "threshold_search": model["threshold_search"]} for model_id, model in selected.items()},
        "candidate_experiments": experiment_records,
        "clustering": clustering_decision,
        "test_evaluation_policy": "Candidates, thresholds, primary selection and clustering are frozen before test evaluation. No training refit or test-based selection occurs.",
    }
    for model_id, model in selected.items():
        directory = output / model_id
        directory.mkdir()
        joblib.dump(model["pipeline"], directory / "pipeline.joblib", compress=3)
        decision["models"][model_id]["pipeline_sha256"] = sha256_file(directory / "pipeline.joblib")
    joblib.dump(clustering, output / "clustering.joblib", compress=3)
    joblib.dump(projection, output / "projection.joblib", compress=3)
    decision["clustering_pipeline_sha256"] = sha256_file(output / "clustering.joblib")
    decision["projection_sha256"] = sha256_file(output / "projection.joblib")
    write_json(output / "selection-decision.json", decision)
    decision_hash = sha256_file(output / "selection-decision.json")
    print("Frozen selection-decision.json written; evaluating held-out test data once.", flush=True)
    test_text = [row["text"] for row in partitions["test"]]
    test_labels = [row["label"] for row in partitions["test"]]
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(partitions["train"]), 1)), [row["label"] for row in partitions["train"]])
    baseline = classification_metrics(test_labels, dummy.predict(np.zeros((len(test_labels), 1))))
    baseline["strategy"] = "most_frequent"
    write_json(output / "dummy-evaluation.json", baseline)
    model_registry = {}
    all_scores = {}
    flattened = [(name, row) for name, rows in partitions.items() for row in rows]
    for model_id, model in selected.items():
        directory = output / model_id
        pipeline = model["pipeline"]
        start = perf_counter()
        scores, classified = score_model(pipeline, test_text)
        seconds = perf_counter() - start
        model_version = f"{release_id}-{model_id}"
        evaluation = evaluate_scores(test_labels, scores, model["threshold"], classified)
        evaluation.update({"schema_version": 1, "model_id": model_id, "model_version": model_version, "dataset_sha256": provenance["dataset_sha256"], "split_manifest_sha256": split_hash, "decision_threshold": model["threshold"], "partition": "test", "evaluated_at": timestamp(), "selection_decision_sha256": decision_hash, "validation": model["validation"], "dummy_baseline": baseline, "inference_batch_seconds": seconds, "mean_batch_seconds_per_sample": seconds / len(test_labels), "latency_note": "Batch wall time divided by sample count; not an individual-request latency percentile."})
        write_json(directory / "evaluation.json", evaluation)
        manifest = {
            "schema_version": 1,
            "model_id": model_id,
            "model_version": model_version,
            "source_revision": revision,
            "source_sha256": source_hash,
            "pipeline_sha256": sha256_file(directory / "pipeline.joblib"),
            "dataset_sha256": provenance["dataset_sha256"],
            "split_manifest_sha256": split_hash,
            "selection_decision_sha256": decision_hash,
            "preprocessing_version": "tfidf-word-v1",
            "class_labels": [str(label) for label in pipeline.named_steps["classifier"].classes_],
            "positive_label": "spam",
            "decision_threshold": model["threshold"],
            "feature_configuration": model["configuration"]["feature_configuration"],
            "hyperparameters": model["configuration"]["hyperparameters"],
            "library_versions": versions(),
            "created_at": timestamp(),
            "evaluation_sha256": sha256_file(directory / "evaluation.json"),
            "score_interpretation": "Uncalibrated model score for spam; not a guarantee of correctness or safety.",
            "supported_domain": "English SMS-like messages, including smishing examples from the Mendeley SMS Phishing Dataset",
            "supported_language": "en",
            "limitations": ["Training data combines historical UCI SMS (2011) with the Mendeley SMS Phishing Dataset (2022); current messages and other languages may differ.", "Spam classification does not detect executable malware or verify links.", "Zero recognized vocabulary features cause abstention.", "Near-duplicate templates may remain after exact normalization deduplication."],
        }
        write_json(directory / "model-manifest.json", manifest)
        model_registry[model_id] = {"directory": model_id, "manifest_sha256": sha256_file(directory / "model-manifest.json")}
        summary_scores, summary_classified = score_model(pipeline, [row["text"] for _, row in flattened])
        all_scores[model_id] = [float(score) if recognized else None for score, recognized in zip(summary_scores, summary_classified, strict=True)]
    summary = {"schema_version": 1, "dataset_id": provenance["dataset_id"], "split_id": split_id, "models": {model_id: f"{release_id}-{model_id}" for model_id in selected}, "rows": [{"sample_id": row["sample_id"], "source_id": row["source_id"], "split": partition, "label": row["label"], "length": len(row["text"]), "scores": {model_id: all_scores[model_id][index] for model_id in selected}} for index, (partition, row) in enumerate(flattened)]}
    write_json(output / "dataset-summary.json", summary)
    write_json(output / "cluster-visualization.json", export_clusters(clustering, projection, partitions["test"], release_id, split_id, seed))
    artifacts = {name: {"file": filename, "sha256": sha256_file(output / filename)} for name, filename in (("dataset_summary", "dataset-summary.json"), ("clusters", "cluster-visualization.json"), ("clustering_pipeline", "clustering.joblib"), ("projection", "projection.joblib"))}
    release = {"schema_version": 1, "release_id": release_id, "primary_model_id": primary, "dataset_id": provenance["dataset_id"], "dataset_sha256": provenance["dataset_sha256"], "split_id": split_id, "split_manifest_sha256": split_hash, "models": model_registry, "artifacts": artifacts, "created_at": timestamp(), "training_seconds": perf_counter() - started, "selection_decision_sha256": decision_hash}
    write_json(output / "release.json", release)
    print(json.dumps({"release_id": release_id, "primary_model_id": primary, "release_sha256": sha256_file(output / "release.json"), "record_count": len(records), "training_seconds": release["training_seconds"]}), flush=True)
    return release


def main():
    parser = argparse.ArgumentParser(description="Fit, freeze and evaluate an immutable real SMS model release.")
    parser.add_argument("--output", type=Path, default=ROOT / "fastapi" / "artifacts" / "current")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "ai" / "data")
    parser.add_argument("--seed", type=int, default=SEED)
    arguments = parser.parse_args()
    train_release(arguments.output, arguments.data_dir, arguments.seed)


if __name__ == "__main__":
    main()
