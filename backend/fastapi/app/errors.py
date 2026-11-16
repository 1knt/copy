from fastapi import Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorEnvelope(BaseModel):
    error: ErrorDetail


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 503):
        self.code = code
        self.message = message
        self.status_code = status_code


def error_response(status_code: int, code: str, message: str, request_id: str):
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
        headers={"X-Request-ID": request_id, "Cache-Control": "no-store"},
    )


async def service_error_handler(request: Request, exc: ServiceError):
    return error_response(exc.status_code, exc.code, exc.message, request.state.request_id)


async def validation_error_handler(request: Request, exc: RequestValidationError):
    invalid_text = any(error["loc"] == ("body", "text") for error in exc.errors())
    code = "INVALID_TEXT" if invalid_text else "INVALID_REQUEST"
    message = (
        "Enter a message between 1 and 2000 characters."
        if invalid_text
        else "The request does not match the API schema."
    )
    return error_response(422, code, message, request.state.request_id)


async def http_error_handler(request: Request, exc: HTTPException):
    code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "INVALID_REQUEST")
    response = error_response(
        exc.status_code, code, "The requested operation is unavailable.", request.state.request_id
    )
    if exc.headers and "Allow" in exc.headers:
        response.headers["Allow"] = exc.headers["Allow"]
    return response


async def unexpected_error_handler(request: Request, exc: Exception):
    return error_response(
        500, "INFERENCE_FAILED", "The request could not be processed.", request.state.request_id
    )
