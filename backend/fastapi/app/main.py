import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

from .artifacts import load_release
from .errors import ServiceError, http_error_handler, service_error_handler, validation_error_handler
from .middleware import RequestBoundaryMiddleware
from .routes import router
from .service import ReleaseService


def create_app(artifact_dir: Path | None = None, inference_concurrency: int | None = None) -> FastAPI:
    directory = Path(artifact_dir) if artifact_dir is not None else Path(__file__).resolve().parents[1] / "artifacts" / "current"
    concurrency = inference_concurrency if inference_concurrency is not None else int(os.environ.get("INFERENCE_CONCURRENCY", "2"))
    if type(concurrency) is not int or not 1 <= concurrency <= 16:
        raise ValueError("INFERENCE_CONCURRENCY must be between 1 and 16.")

    @asynccontextmanager
    async def lifespan(application):
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False, interpolate=False)
        try:
            release = load_release(directory)
        except Exception:
            release = None
        application.state.release_service = ReleaseService(release, concurrency)
        yield

    application = FastAPI(title="HN1_8 Spam Screening API", version="0.2.0",
                          description="Frozen English SMS model serving with validated release artifacts and aggregate exploration.",
                          redirect_slashes=False, lifespan=lifespan)
    application.state.release_service = ReleaseService(None, concurrency)
    application.add_middleware(RequestBoundaryMiddleware)
    application.add_exception_handler(ServiceError, service_error_handler)
    application.add_exception_handler(RequestValidationError, validation_error_handler)
    application.add_exception_handler(HTTPException, http_error_handler)
    application.include_router(router)
    return application


app = create_app()
