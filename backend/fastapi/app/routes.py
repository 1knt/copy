from typing import Annotated, Literal

from fastapi import APIRouter, Query, Request

from .errors import ErrorEnvelope
from .number_lookup import lookup_numbers
from .external_reference import classify_external_reference
from .schemas import (
    ClustersResponse, ComparisonResponse, DatasetSummaryResponse, EvaluationResponse,
    LivenessResponse, ModelId, ModelResponse, PredictionResponse, ReadinessResponse, TextRequest,
    NumberLookupRequest, NumberLookupResponse,
    ExternalReferenceResponse,
)


router = APIRouter(prefix="/api/v1", responses={
    422: {"model": ErrorEnvelope, "description": "Invalid request"},
    503: {"model": ErrorEnvelope, "description": "Release unavailable or inference busy"},
    500: {"model": ErrorEnvelope, "description": "Sanitized processing failure"},
})


@router.get("/health/live", response_model=LivenessResponse, tags=["Health"])
async def liveness():
    return LivenessResponse()


@router.get("/health/ready", response_model=ReadinessResponse, tags=["Health"])
async def readiness(request: Request):
    service = request.app.state.release_service
    release = service.require_release()
    return ReadinessResponse(model_version=service.resolve().version, release_id=release.release_id)


@router.post("/predictions", response_model=PredictionResponse, tags=["Screening"],
             responses={413: {"model": ErrorEnvelope}, 415: {"model": ErrorEnvelope}})
def predict(payload: TextRequest, request: Request):
    return request.app.state.release_service.screen(payload.text, request.state.request_id)


@router.post("/comparisons", response_model=ComparisonResponse, tags=["Screening"],
             responses={413: {"model": ErrorEnvelope}, 415: {"model": ErrorEnvelope}})
def compare(payload: TextRequest, request: Request):
    return request.app.state.release_service.screen(payload.text, request.state.request_id, comparison=True)


@router.post("/number-lookup", response_model=NumberLookupResponse, tags=["External metadata"],
             responses={502: {"model": ErrorEnvelope}, 413: {"model": ErrorEnvelope}, 415: {"model": ErrorEnvelope}})
def number_lookup(payload: NumberLookupRequest, request: Request):
    return NumberLookupResponse(request_id=request.state.request_id, results=lookup_numbers(payload.numbers))


@router.post("/external-reference", response_model=ExternalReferenceResponse, tags=["External reference"],
             responses={413: {"model": ErrorEnvelope}, 415: {"model": ErrorEnvelope}})
def external_reference(payload: TextRequest, request: Request):
    return ExternalReferenceResponse(request_id=request.state.request_id, **classify_external_reference(payload.text))


@router.get("/model", response_model=ModelResponse, tags=["Artifacts"])
def model(request: Request, model_id: ModelId = "primary"):
    return request.app.state.release_service.metadata(model_id)


@router.get("/evaluation", response_model=EvaluationResponse, tags=["Artifacts"])
def evaluation(request: Request, model_id: ModelId = "primary"):
    return request.app.state.release_service.resolve(model_id, "EVALUATION_UNAVAILABLE").evaluation


@router.get("/dataset-summary", response_model=DatasetSummaryResponse, tags=["Artifacts"])
def dataset_summary(request: Request, source: Literal["all", "uci_sms", "mendeley_smishing"] = "all",
                    split: Literal["all", "train", "validation", "test"] = "all",
                    model_id: ModelId = "primary", actual_class: Literal["all", "ham", "spam"] = "all"):
    return request.app.state.release_service.summary(source, split, model_id, actual_class)


@router.get("/clusters", response_model=ClustersResponse, tags=["Artifacts"])
def clusters(request: Request, cluster_id: Annotated[int | None, Query(ge=0)] = None):
    return request.app.state.release_service.cluster_data(cluster_id)
