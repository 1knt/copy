from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ModelId = Literal["primary", "naive_bayes", "logistic_regression"]


class TextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=2000)

    @field_validator("text", mode="before")
    @classmethod
    def trim_text(cls, value):
        return value.strip() if isinstance(value, str) else value


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: Literal["spam-screening-api"] = "spam-screening-api"


class NumberLookupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    numbers: list[Annotated[str, Field(min_length=1, max_length=20)]] = Field(min_length=1, max_length=5)


class NumberLookupResult(BaseModel):
    input: str
    status: Literal["looked_up", "not_a_number", "unavailable"]
    valid: bool | None = None
    e164: str | None = None
    country_name: str | None = None
    country_code: str | None = None
    carrier: str | None = None
    line_type: str | None = None
    reason_code: str | None = None


class NumberLookupResponse(BaseModel):
    request_id: str
    provider: Literal["veriphone"] = "veriphone"
    results: list[NumberLookupResult]


class ExternalLabelScore(BaseModel):
    label: str
    score: float = Field(ge=0, le=1, allow_inf_nan=False)


class ExternalReferenceResponse(BaseModel):
    request_id: str
    provider: Literal["huggingface"] = "huggingface"
    model_id: str
    source: Literal["external_reference"] = "external_reference"
    scores: list[ExternalLabelScore]


class ReadinessResponse(BaseModel):
    status: Literal["ready"] = "ready"
    model_version: str
    release_id: str


class PredictionResponse(BaseModel):
    request_id: str
    status: Literal["classified", "unable_to_assess"]
    label: Literal["ham", "spam"] | None
    spam_score: float | None = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    model_version: str
    reason_code: Literal["NO_RECOGNIZED_FEATURES"] | None


class ComparisonResult(PredictionResponse):
    model_id: Literal["naive_bayes", "logistic_regression"]


class ComparisonResponse(BaseModel):
    request_id: str
    status: Literal["compared"] = "compared"
    results: list[ComparisonResult]


class ModelResponse(BaseModel):
    model_id: str
    model_version: str
    positive_label: str | None = None
    domain: str = "SMS messages"
    language: str = "English"
    threshold: float
    feature_family: str = "TF-IDF word features"
    label_meanings: dict[str, str]
    limitations: list[str]
    release_id: str
    source_name: str = "UCI SMS Spam Collection and Mendeley SMS Phishing Dataset"
    source_url: str = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
    source_license: str = "CC BY 4.0"
    source_citation: str = "Almeida, T. and Hidalgo, J. (2011). SMS Spam Collection. UCI Machine Learning Repository. DOI: 10.24432/C5CC84. Mishra, S. and Soni, D. (2022). SMS Phishing Dataset for Machine Learning and Pattern Recognition. Mendeley Data, V1. DOI: 10.17632/f45bkkt8pr.1."


class MetricValues(BaseModel):
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    false_positive_rate: float | None


class ConfusionMatrix(BaseModel):
    labels: list[str]
    matrix: list[list[int]]


class OperationalEvaluation(BaseModel):
    sample_count: int
    total_count: int
    classified_count: int
    abstained_count: int
    coverage: float | None
    support: dict[str, int]
    abstention_support: dict[str, int]
    undefined_metrics: dict[str, str] = Field(default_factory=dict)
    metrics: MetricValues
    confusion_matrix: ConfusionMatrix


class EvaluationResponse(BaseModel):
    model_id: str
    model_version: str
    positive_label: str | None = None
    dataset_sha256: str
    split_manifest_sha256: str
    decision_threshold: float
    partition: Literal["test"]
    sample_count: int
    support: dict[str, int]
    confusion_matrix: ConfusionMatrix
    metrics: MetricValues
    operational: OperationalEvaluation
    evaluated_at: str | None = None
    undefined_metrics: dict[str, str] = Field(default_factory=dict)
    metric_notes: dict[str, str] = Field(default_factory=dict)


class Histogram(BaseModel):
    bin_edges: list[float]
    counts: list[int]
    overflow_count: int = 0


class ScoreHistograms(BaseModel):
    model_id: str
    model_version: str
    bin_edges: list[float]
    all_counts: list[int]
    ham_counts: list[int]
    spam_counts: list[int]


class DatasetSummaryResponse(BaseModel):
    schema_version: int = 1
    dataset_id: str
    split_id: str
    source_counts: dict[str, int]
    class_counts: dict[str, int]
    sample_count: int
    excluded_count: int
    abstained_count: int
    length_bins: Histogram
    score_histograms: ScoreHistograms
    artifact_versions: dict[str, str]


class ClusterPoint(BaseModel):
    sample_id: str
    x: float
    y: float
    cluster_id: int


class ClusterSummary(BaseModel):
    cluster_id: int
    size: int
    description: str


class ClustersResponse(BaseModel):
    clustering_version: str
    projection_version: str
    source_id: str
    split_id: str
    partition: Literal["test"]
    sample_count: int
    total_count: int
    points: list[ClusterPoint]
    clusters: list[ClusterSummary]
    sampling: str = "Stable sample ID ordering, limited to 1000 test points before filtering."
