import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import adjusted_rand_score, auc, mean_absolute_error, precision_recall_curve, r2_score, roc_curve, silhouette_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from ai.analysis import COLORS, load_prepared, save
from ai.data import SEED, write_json
from ai.emoji_text import LEGACY_PREPROCESSING_VERSION, model_input
from ai.metrics import classification_metrics


ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = ["length", "word_count", "digit_ratio", "uppercase_ratio", "has_url", "has_email", "has_phone", "has_currency", "exclamation_count", "emoji_count"]


def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)


def binary_metrics(labels, predictions):
    result = classification_metrics(list(labels), list(predictions))
    return {"metrics": result["metrics"], "confusion_matrix": result["confusion_matrix"]["matrix"]}


def engineered(frame, scaler=None):
    values = frame[FEATURE_COLUMNS].astype(float).to_numpy()
    if scaler is None:
        scaler = StandardScaler().fit(values)
    return scaler.transform(values), scaler


def classifier_comparison(train, validation, seed):
    vectorizer = tfidf()
    train_text = vectorizer.fit_transform(train["model_text"])
    validation_text = vectorizer.transform(validation["model_text"])
    train_features, scaler = engineered(train)
    validation_features, _ = engineered(validation, scaler)
    train_both = hstack([train_text, csr_matrix(train_features)]).tocsr()
    validation_both = hstack([validation_text, csr_matrix(validation_features)]).tocsr()
    candidates = [
        ("Naive Bayes (TF-IDF)", MultinomialNB(alpha=0.5), "text"),
        ("Logistic regression (TF-IDF)", LogisticRegression(C=2.0, max_iter=2000, solver="liblinear", random_state=seed), "text"),
        ("Linear SVM (TF-IDF)", LinearSVC(C=1.0, random_state=seed), "text"),
        ("Random forest (TF-IDF)", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1), "text"),
        ("Decision tree (TF-IDF)", DecisionTreeClassifier(max_depth=30, random_state=seed), "text"),
        ("k-nearest neighbours (TF-IDF, k=5)", KNeighborsClassifier(n_neighbors=5, metric="cosine"), "text"),
        ("Logistic regression (engineered features only)", LogisticRegression(max_iter=2000, random_state=seed), "features"),
        ("Logistic regression (TF-IDF + engineered)", LogisticRegression(C=2.0, max_iter=2000, solver="liblinear", random_state=seed), "both"),
    ]
    inputs = {"text": (train_text, validation_text), "features": (train_features, validation_features), "both": (train_both, validation_both)}
    results = []
    for name, model, kind in candidates:
        start = perf_counter()
        model.fit(inputs[kind][0], train["label"])
        fit_seconds = perf_counter() - start
        predictions = model.predict(inputs[kind][1])
        results.append({"model": name, "input": kind, "fit_seconds": fit_seconds, **binary_metrics(validation["label"], predictions)})
    return results


def cross_source(train, validation, seed):
    uci_train = train[train["sources"] != "mendeley_smishing"]
    targets = {"Mendeley-only validation rows": validation[validation["sources"] == "mendeley_smishing"], "UCI validation rows": validation[validation["sources"] != "mendeley_smishing"]}
    results = []
    for training_name, training in (("UCI only", uci_train), ("UCI + Mendeley", train)):
        for model_name, model in (("Naive Bayes", MultinomialNB(alpha=0.5)), ("Logistic regression", LogisticRegression(C=2.0, max_iter=2000, solver="liblinear", random_state=seed))):
            vectorizer = tfidf()
            model.fit(vectorizer.fit_transform(training["model_text"]), training["label"])
            for target_name, target in targets.items():
                predictions = model.predict(vectorizer.transform(target["model_text"]))
                results.append({"training_data": training_name, "training_rows": int(len(training)), "model": model_name, "evaluated_on": target_name, "evaluated_rows": int(len(target)), **binary_metrics(target["label"], predictions)})
    return results


def subtype_task(train, validation, seed):
    keep = ["smishing", "spam"]
    fit = train[train["subtype"].isin(keep)]
    check = validation[validation["subtype"].isin(keep)]
    vectorizer = tfidf()
    train_x = vectorizer.fit_transform(fit["model_text"])
    check_x = vectorizer.transform(check["model_text"])
    results = []
    for name, model in (("Logistic regression", LogisticRegression(C=2.0, max_iter=2000, solver="liblinear", random_state=seed)), ("Random forest", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1))):
        model.fit(train_x, fit["subtype"])
        predictions = model.predict(check_x)
        renamed = ["spam" if value == "smishing" else "ham" for value in predictions]
        truth = ["spam" if value == "smishing" else "ham" for value in check["subtype"]]
        metrics = binary_metrics(truth, renamed)
        results.append({"model": name, "positive_class": "smishing", "negative_class": "generic spam", "training_rows": int(len(fit)), "validation_rows": int(len(check)), **metrics})
    return results


def length_regression(train, validation):
    columns = ["word_count", "digit_ratio", "uppercase_ratio", "has_url", "has_phone", "has_currency", "exclamation_count"]
    model = LinearRegression().fit(train[columns].astype(float), train["length"].astype(float))
    predictions = model.predict(validation[columns].astype(float))
    return {"target": "message length in characters", "features": columns, "mae": float(mean_absolute_error(validation["length"], predictions)), "r2": float(r2_score(validation["length"], predictions)), "coefficients": {name: float(value) for name, value in zip(columns, model.coef_)}, "purpose": "Sanity check that engineered counts are consistent; not used for spam decisions."}


def clustering(train, validation, seed, figures):
    vectorizer = tfidf()
    reducer = TruncatedSVD(n_components=100, random_state=seed)
    normalizer = Normalizer()
    train_x = normalizer.fit_transform(reducer.fit_transform(vectorizer.fit_transform(train["model_text"])))
    validation_x = normalizer.transform(reducer.transform(vectorizer.transform(validation["model_text"])))
    labels = (validation["label"] == "spam").astype(int).to_numpy()
    kmeans = []
    for k in range(2, 11):
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(train_x)
        assigned = model.predict(validation_x)
        kmeans.append({"k": k, "inertia": float(model.inertia_), "validation_silhouette": float(silhouette_score(validation_x, assigned)), "adjusted_rand_vs_label": float(adjusted_rand_score(labels, assigned)), "spam_purity": purity(assigned, labels)})
    figure, left = plt.subplots(figsize=(7, 4))
    left.plot([r["k"] for r in kmeans], [r["inertia"] for r in kmeans], marker="o", color="#4c72b0")
    left.set_xlabel("Number of clusters k")
    left.set_ylabel("Training inertia", color="#4c72b0")
    right = left.twinx()
    right.plot([r["k"] for r in kmeans], [r["validation_silhouette"] for r in kmeans], marker="s", color="#dd8452")
    right.set_ylabel("Validation silhouette", color="#dd8452")
    left.set_title("KMeans elbow and silhouette")
    save(figure, figures / "cluster-kmeans-elbow.png")
    distances = np.sort(NearestNeighbors(n_neighbors=5).fit(validation_x).kneighbors(validation_x)[0][:, -1])
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(distances, color="#55a868")
    axis.set_xlabel("Validation points sorted by distance")
    axis.set_ylabel("Distance to 5th nearest neighbour")
    axis.set_title("k-distance plot for choosing DBSCAN eps")
    save(figure, figures / "cluster-dbscan-k-distance.png")
    dbscan = []
    for eps in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1):
        assigned = DBSCAN(eps=eps, min_samples=5).fit_predict(validation_x)
        clusters = sorted(set(assigned) - {-1})
        noise = float(np.mean(assigned == -1))
        core = assigned != -1
        silhouette = float(silhouette_score(validation_x[core], assigned[core])) if len(clusters) > 1 and core.sum() > len(clusters) else None
        dbscan.append({"eps": eps, "min_samples": 5, "clusters": len(clusters), "noise_fraction": noise, "silhouette_without_noise": silhouette, "adjusted_rand_vs_label": float(adjusted_rand_score(labels, assigned))})
    return {"representation": "TF-IDF (1-2 grams) -> TruncatedSVD 100 -> L2 normalization, fitted on training text", "kmeans": kmeans, "dbscan": dbscan}


def purity(assigned, labels):
    total = 0
    for cluster in np.unique(assigned):
        members = labels[assigned == cluster]
        total += max(members.sum(), len(members) - members.sum())
    return float(total / len(labels))


def release_figures(release, test, figures):
    curves = {}
    roc_figure, roc_axis = plt.subplots(figsize=(5.5, 5))
    pr_figure, pr_axis = plt.subplots(figsize=(5.5, 5))
    matrix_figure, matrix_axes = plt.subplots(1, 2, figsize=(9, 4))
    truth = (test["label"] == "spam").astype(int).to_numpy()
    for index, (model_id, title) in enumerate((("naive_bayes", "Naive Bayes"), ("logistic_regression", "Logistic regression"))):
        pipeline = joblib.load(release / model_id / "pipeline.joblib")
        evaluation = json.loads((release / model_id / "evaluation.json").read_text(encoding="utf-8"))
        manifest = json.loads((release / model_id / "model-manifest.json").read_text(encoding="utf-8"))
        preprocessing = manifest.get("preprocessing_version", LEGACY_PREPROCESSING_VERSION)
        scores = pipeline.predict_proba([model_input(text, preprocessing) for text in test["text"]])[:, list(pipeline.classes_).index("spam")]
        fpr, tpr, _ = roc_curve(truth, scores)
        precision, recall, _ = precision_recall_curve(truth, scores)
        curves[model_id] = {"roc_auc": float(auc(fpr, tpr)), "pr_auc": float(auc(recall, precision)), "threshold": evaluation["decision_threshold"]}
        roc_axis.plot(fpr, tpr, label=f"{title} (AUC {curves[model_id]['roc_auc']:.3f})")
        pr_axis.plot(recall, precision, label=f"{title} (AUC {curves[model_id]['pr_auc']:.3f})")
        matrix = np.array(evaluation["confusion_matrix"]["matrix"])
        axis = matrix_axes[index]
        axis.imshow(matrix, cmap="Blues")
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, str(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
        axis.set_xticks([0, 1], ["ham", "spam"])
        axis.set_yticks([0, 1], ["ham", "spam"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_title(f"{title}, threshold {evaluation['decision_threshold']}")
    roc_axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
    roc_axis.set_xlabel("False positive rate")
    roc_axis.set_ylabel("True positive rate")
    roc_axis.set_title("ROC curves, held-out test partition")
    roc_axis.legend(loc="lower right")
    pr_axis.set_xlabel("Recall")
    pr_axis.set_ylabel("Precision")
    pr_axis.set_title("Precision-recall curves, test partition")
    pr_axis.legend(loc="lower left")
    save(roc_figure, figures / "eval-roc.png")
    save(pr_figure, figures / "eval-precision-recall.png")
    matrix_figure.suptitle("Confusion matrices on the held-out test partition")
    save(matrix_figure, figures / "eval-confusion-matrices.png")
    decision = json.loads((release / "selection-decision.json").read_text(encoding="utf-8"))
    figure, axis = plt.subplots(figsize=(7, 4))
    for model_id, title in (("naive_bayes", "Naive Bayes"), ("logistic_regression", "Logistic regression")):
        search = decision["models"][model_id]["threshold_search"]
        axis.plot([item["threshold"] for item in search], [item["evaluation"]["metrics"]["f1"] for item in search], marker=".", label=title)
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Validation spam F1")
    axis.set_title("Threshold search on the validation partition")
    axis.legend()
    save(figure, figures / "eval-threshold-search.png")
    return curves


def comparison_figure(results, figures):
    frame = pd.DataFrame([{"model": r["model"], **{k: r["metrics"][k] for k in ("precision", "recall", "f1")}} for r in results]).set_index("model")
    figure, axis = plt.subplots(figsize=(9, 5))
    frame.sort_values("f1").plot(kind="barh", ax=axis, color=["#4c72b0", "#dd8452", "#55a868"])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Validation score for the spam class")
    axis.set_title("Classifier comparison at the default decision rule")
    save(figure, figures / "exp-classifier-comparison.png")


def cross_source_figure(results, figures):
    frame = pd.DataFrame([{"setting": f"{r['model']}\ntrained on {r['training_data']}", "target": r["evaluated_on"], "recall": r["metrics"]["recall"]} for r in results])
    table = frame.pivot(index="setting", columns="target", values="recall")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    table.plot(kind="bar", ax=axis, color=[COLORS["smishing"], COLORS["ham"]])
    axis.set_ylim(0, 1.15)
    axis.legend(title="Evaluated on", loc="upper center", ncol=2)
    axis.set_xlabel("")
    axis.set_ylabel("Spam recall")
    axis.set_title("Effect of the additional dataset on recall")
    axis.tick_params(axis="x", rotation=0)
    save(figure, figures / "exp-cross-source-recall.png")


def run(prepared, release, output, seed=SEED):
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    data = load_prepared(prepared)
    train = data[data["partition"] == "train"].reset_index(drop=True)
    validation = data[data["partition"] == "validation"].reset_index(drop=True)
    test = data[data["partition"] == "test"].reset_index(drop=True)
    comparison = classifier_comparison(train, validation, seed)
    comparison_figure(comparison, figures)
    sources = cross_source(train, validation, seed)
    cross_source_figure(sources, figures)
    result = {
        "schema_version": 1,
        "seed": seed,
        "selection_partition": "validation",
        "note": "Exploratory experiments use the frozen training and validation partitions. Only the served release models are evaluated on the test partition.",
        "classifier_comparison": comparison,
        "cross_source_generalisation": sources,
        "smishing_subtype": subtype_task(train, validation, seed),
        "length_regression_check": length_regression(train, validation),
        "clustering": clustering(train, validation, seed, figures),
        "release_test_curves": release_figures(Path(release), test, figures),
    }
    write_json(output / "experiments.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Additional classification, cross-source, subtype and clustering experiments plus evaluation figures.")
    parser.add_argument("--prepared", type=Path, default=ROOT / "ai" / "data" / "processed" / f"combined-sms-seed-{SEED}")
    parser.add_argument("--release", type=Path, default=ROOT / "fastapi" / "artifacts" / "current")
    parser.add_argument("--output", type=Path, default=ROOT / "ai" / "reports")
    arguments = parser.parse_args()
    run(arguments.prepared, arguments.release, arguments.output)
    print(f"Experiments written to {arguments.output}")


if __name__ == "__main__":
    main()
