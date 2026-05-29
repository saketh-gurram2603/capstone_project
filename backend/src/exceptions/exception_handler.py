"""
Central FastAPI exception handler.
Converts all IncidentKBException subclasses (and common FastAPI exceptions)
into a consistent JSON error response shape.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.exceptions.custom_exceptions import IncidentKBException
from src.handlers.logger import log_error, log_warning


def _error_response(status_code: int, error_type: str, message: str, details: dict) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error_type,
            "message": message,
            "details": details,
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the FastAPI app instance."""

    @app.exception_handler(IncidentKBException)
    async def handle_incident_kb_exception(
        request: Request, exc: IncidentKBException
    ) -> JSONResponse:
        log_error(
            "IncidentKBException | %s %s | %s | details=%s",
            request.method,
            request.url.path,
            exc.message,
            exc.details,
        )
        return _error_response(
            status_code=exc.status_code,
            error_type=exc.__class__.__name__,
            message=exc.message,
            details=exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        log_warning(
            "ValidationError | %s %s | %s",
            request.method,
            request.url.path,
            str(exc.errors()),
        )
        return _error_response(
            status_code=422,
            error_type="ValidationError",
            message="Request body failed schema validation.",
            details={"validation_errors": exc.errors()},
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        log_warning(
            "HTTPException | %s %s | %s",
            request.method,
            request.url.path,
            exc.detail,
        )
        return _error_response(
            status_code=exc.status_code,
            error_type="HTTPException",
            message=str(exc.detail),
            details={},
        )

    @app.exception_handler(Exception)
    async def handle_unhandled_exception(
        request: Request, exc: Exception
    ) -> JSONResponse:
        log_error(
            "UnhandledException | %s %s | %s",
            request.method,
            request.url.path,
            str(exc),
            exc_info=True,
        )
        return _error_response(
            status_code=500,
            error_type="InternalServerError",
            message="An unexpected error occurred. Please try again later.",
            details={},
        )
