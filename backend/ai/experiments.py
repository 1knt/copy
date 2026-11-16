from time import perf_counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from ai.metrics import evaluate_scores, metric_rank, score_model


def features(ngram_range):
    return {"analyzer": "word", "ngram_range": list(ngram_range), "min_df": 2, "max_features": 20000, "lowercase": True, "token_pattern": r"(?u)\b\w\w+\b", "strip_accents": None, "norm": "l2", "sublinear_tf": False}


def select_classifiers(partitions, seed):
    train_text = [row["text"] for row in partitions["train"]]
    train_labels = [row["label"] for row in partitions["train"]]
    validation_text = [row["text"] for row in partitions["validation"]]
    validation_labels = [row["label"] for row in partitions["validation"]]
    selected = {}
    experiments = []
    for model_id, parameters in (("naive_bayes", (0.5, 1.0)), ("logistic_regression", (0.5, 1.0, 2.0))):
        candidates = []
        for ngrams in ((1, 1), (1, 2)):
            for value in parameters:
                classifier = MultinomialNB(alpha=value) if model_id == "naive_bayes" else LogisticRegression(C=value, max_iter=2000, random_state=seed, solver="liblinear")
                pipeline = Pipeline([("tfidf", TfidfVectorizer(ngram_range=ngrams, min_df=2, max_features=20000)), ("classifier", classifier)])
                start = perf_counter()
                pipeline.fit(train_text, train_labels)
                elapsed = perf_counter() - start
                scores, classified = score_model(pipeline, validation_text)
                evaluation = evaluate_scores(validation_labels, scores, 0.5, classified)
                configuration = {"model_id": model_id, "feature_configuration": features(ngrams), "hyperparameters": {"alpha": value} if model_id == "naive_bayes" else {"C": value, "max_iter": 2000, "solver": "liblinear", "random_state": seed}}
                record = {**configuration, "fit_seconds": elapsed, "validation_at_default_threshold": evaluation, "feature_count": len(pipeline.named_steps["tfidf"].vocabulary_)}
                experiments.append(record)
                candidates.append({"pipeline": pipeline, "record": record, "scores": scores, "classified": classified})
        winner = max(candidates, key=lambda item: (*metric_rank(item["record"]["validation_at_default_threshold"]), -item["record"]["feature_count"], -item["record"]["hyperparameters"].get("C", item["record"]["hyperparameters"].get("alpha", 0))))
        thresholds = []
        for threshold in np.round(np.arange(0.10, 0.901, 0.05), 2):
            evaluation = evaluate_scores(validation_labels, winner["scores"], float(threshold), winner["classified"])
            thresholds.append({"threshold": float(threshold), "evaluation": evaluation})
        operating = max(thresholds, key=lambda item: (*metric_rank(item["evaluation"]), -abs(item["threshold"] - 0.5), item["threshold"]))
        selected[model_id] = {"pipeline": winner["pipeline"], "configuration": winner["record"], "threshold": operating["threshold"], "validation": operating["evaluation"], "threshold_search": thresholds}
    primary = max(selected, key=lambda model_id: (*metric_rank(selected[model_id]["validation"]), model_id == "naive_bayes"))
    return selected, experiments, primary
