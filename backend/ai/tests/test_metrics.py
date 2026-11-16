import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from ai.metrics import classification_metrics, evaluate_scores, score_model


def test_confusion_matrix_orientation_and_metrics():
    result = classification_metrics(["ham", "ham", "spam", "spam", "spam"], ["ham", "spam", "ham", "spam", "spam"])
    assert result["confusion_matrix"] == {"labels": ["ham", "spam"], "matrix": [[1, 1], [1, 2]]}
    assert result["support"] == {"ham": 2, "spam": 3}
    for metric in ("precision", "recall", "f1"):
        assert result["metrics"][metric] == pytest.approx(2 / 3)
    assert result["metrics"]["accuracy"] == 0.6
    assert result["metrics"]["false_positive_rate"] == 0.5


def test_undefined_ratios_are_null_with_explanations():
    result = classification_metrics(["ham"], ["ham"])
    for name in ("precision", "recall", "f1"):
        assert result["metrics"][name] is None
        assert name in result["undefined_metrics"]
    assert classification_metrics([], [])["metrics"]["accuracy"] is None


def test_threshold_equality_and_abstention_are_reported_separately():
    result = evaluate_scores(["spam", "ham", "spam"], [0.5, 0.2, 0.9], 0.5, [True, True, False])
    assert result["confusion_matrix"]["matrix"] == [[1, 0], [0, 2]]
    assert result["sample_count"] == 3
    assert result["operational"]["sample_count"] == 2
    assert result["operational"]["total_count"] == 3
    assert result["operational"]["abstained_count"] == 1
    assert result["operational"]["abstention_support"] == {"ham": 0, "spam": 1}
    assert result["operational"]["coverage"] == pytest.approx(2 / 3)


def test_zero_feature_detection_uses_fitted_vocabulary():
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", MultinomialNB())])
    pipeline.fit(["ordinary meeting", "win prize", "ordinary dinner", "free prize"], ["ham", "spam", "ham", "spam"])
    scores, recognized = score_model(pipeline, ["win prize", "zzunknownword", "!!!"])
    assert list(recognized) == [True, False, False]
    assert np.all((scores >= 0) & (scores <= 1))


def test_metric_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same number"):
        classification_metrics(["ham"], [])
