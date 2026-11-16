import numpy as np


def classification_metrics(labels, predictions):
    truth = np.asarray(labels)
    predicted = np.asarray(predictions)
    if len(truth) != len(predicted):
        raise ValueError("Metric inputs must have the same number of rows.")
    tn = int(np.sum((truth == "ham") & (predicted == "ham")))
    fp = int(np.sum((truth == "ham") & (predicted == "spam")))
    fn = int(np.sum((truth == "spam") & (predicted == "ham")))
    tp = int(np.sum((truth == "spam") & (predicted == "spam")))
    undefined = {}

    def ratio(name, numerator, denominator):
        if denominator == 0:
            undefined[name] = "The denominator is zero for this evaluated sample."
            return None
        return float(numerator / denominator)

    metrics = {
        "precision": ratio("precision", tp, tp + fp),
        "recall": ratio("recall", tp, tp + fn),
        "f1": ratio("f1", 2 * tp, 2 * tp + fp + fn),
        "accuracy": ratio("accuracy", tp + tn, len(truth)),
        "false_positive_rate": ratio("false_positive_rate", fp, fp + tn),
    }
    return {"sample_count": len(truth), "support": {"ham": tn + fp, "spam": fn + tp}, "confusion_matrix": {"labels": ["ham", "spam"], "matrix": [[tn, fp], [fn, tp]]}, "metrics": metrics, "undefined_metrics": undefined}


def score_model(pipeline, texts):
    transformed = pipeline.named_steps["tfidf"].transform(texts)
    classified = np.asarray(transformed.getnnz(axis=1) > 0)
    classifier = pipeline.named_steps["classifier"]
    class_order = list(classifier.classes_)
    scores = classifier.predict_proba(transformed)[:, class_order.index("spam")]
    return scores, classified


def evaluate_scores(labels, scores, threshold, classified=None):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    predicted = np.where(scores >= threshold, "spam", "ham")
    result = classification_metrics(labels, predicted)
    classified = np.ones(len(labels), dtype=bool) if classified is None else np.asarray(classified, dtype=bool)
    operational = classification_metrics(labels[classified], predicted[classified])
    operational.update({"total_count": len(labels), "classified_count": int(classified.sum()), "abstained_count": int((~classified).sum()), "coverage": float(classified.mean()) if len(labels) else None, "abstention_support": {label: int(np.sum((labels == label) & ~classified)) for label in ("ham", "spam")}})
    result["operational"] = operational
    return result


def metric_rank(evaluation):
    values = evaluation["metrics"]
    return tuple(-1.0 if values[key] is None else values[key] for key in ("f1", "precision"))
