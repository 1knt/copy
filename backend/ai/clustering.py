import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from ai.emoji_text import text_for_model


def select_clustering(partitions, seed):
    train_text = [text_for_model(row["text"]) for row in partitions["train"]]
    validation_text = [text_for_model(row["text"]) for row in partitions["validation"]]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    train_features = vectorizer.fit_transform(train_text)
    dimension = min(100, train_features.shape[0] - 1, train_features.shape[1] - 1)
    if dimension < 2:
        raise ValueError("Clustering requires at least three rows and three vocabulary features.")
    projection = TruncatedSVD(n_components=2, random_state=seed)
    projection.fit(train_features)
    reduction = TruncatedSVD(n_components=dimension, random_state=seed)
    normalizer = Normalizer()
    train_reduced = normalizer.fit_transform(reduction.fit_transform(train_features))
    validation_reduced = normalizer.transform(reduction.transform(vectorizer.transform(validation_text)))
    candidates = []
    for count in (2, 4, 6):
        if count >= len(train_text):
            continue
        models = [KMeans(n_clusters=count, n_init=10, max_iter=300, random_state=seed + offset).fit(train_reduced) for offset in range(3)]
        assignments = [model.predict(validation_reduced) for model in models]
        observed = np.unique(assignments[0])
        valid = len(observed) == count and len(observed) < len(validation_text)
        silhouette = float(silhouette_score(validation_reduced, assignments[0], metric="euclidean", sample_size=min(1000, len(validation_text)), random_state=seed)) if valid else None
        stability = float(np.mean([adjusted_rand_score(assignments[0], labels) for labels in assignments[1:]]))
        record = {"k": count, "validation_silhouette": silhouette, "silhouette_metric": "euclidean", "validation_cluster_counts": {str(cluster): int(np.sum(assignments[0] == cluster)) for cluster in range(count)}, "stability_adjusted_rand_index": stability, "stability_seeds": [seed, seed + 1, seed + 2], "train_inertia": float(models[0].inertia_), "valid": valid}
        candidates.append({"model": models[0], "record": record})
    valid_candidates = [item for item in candidates if item["record"]["valid"]]
    if not valid_candidates:
        raise ValueError("All prespecified cluster candidates are degenerate on validation data.")
    selected = max(valid_candidates, key=lambda item: (item["record"]["validation_silhouette"], item["record"]["stability_adjusted_rand_index"], -item["record"]["k"]))
    pipeline = Pipeline([("tfidf", vectorizer), ("svd", reduction), ("normalize", normalizer), ("clusterer", selected["model"])])
    return pipeline, projection, {"selected": selected["record"], "candidates": [item["record"] for item in candidates], "svd_dimensions": dimension, "selection_rule": "Reject missing validation clusters; maximize validation silhouette, then seed stability, then fewer clusters. Labels are unused."}


def export_clusters(pipeline, projection, rows, release_id, split_id, seed):
    texts = [text_for_model(row["text"]) for row in rows]
    features = pipeline.named_steps["tfidf"].transform(texts)
    reduced = pipeline.named_steps["normalize"].transform(pipeline.named_steps["svd"].transform(features))
    clusters = pipeline.named_steps["clusterer"].predict(reduced)
    coordinates = projection.transform(features)
    cluster_count = pipeline.named_steps["clusterer"].n_clusters
    silhouette = None
    if 1 < len(np.unique(clusters)) < len(rows):
        silhouette = float(silhouette_score(reduced, clusters, metric="euclidean", sample_size=min(1000, len(rows)), random_state=seed))
    indices = sorted(range(len(rows)), key=lambda index: rows[index]["sample_id"])[:1000]
    return {
        "schema_version": 1,
        "clustering_version": f"{release_id}-kmeans",
        "projection_version": f"{release_id}-svd2",
        "source_id": "combined",
        "split_id": split_id,
        "partition": "test",
        "sample_count": len(indices),
        "total_count": len(rows),
        "sampling": "First 1000 test rows in ascending SHA-derived sample ID order; all rows when fewer than 1000.",
        "points": [{"sample_id": rows[index]["sample_id"], "x": float(coordinates[index, 0]), "y": float(coordinates[index, 1]), "cluster_id": int(clusters[index])} for index in indices],
        "summaries": [{"cluster_id": cluster, "count": int(np.sum(clusters == cluster)), "description": f"Exploratory cluster {cluster}; not a spam label."} for cluster in range(cluster_count)],
        "test_silhouette": silhouette,
        "limitations": ["Projection coordinates are not probabilities.", "Cluster IDs have no intrinsic ordering or classification meaning.", "No raw messages or extracted terms are exposed."],
    }
