"""Custom exception handlers for FastAPI.

Ensures that ALL error responses are JSON ``{error, ...}`` with no
Python tracebacks, file paths, line numbers, or internal module names
(VAL-BACKEND-API-002).
"""

from __future__ import annotations

import logging
import re

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

# Patterns that must NEVER appear in error responses to clients
_UNSAFE_PATTERNS = re.compile(
    r"Traceback|/Users/|/src/|\.py[\":]|File \"|line \d+|module named|ImportError|"
    r"AttributeError|KeyError|TypeError|ValueError|RuntimeError",
    re.IGNORECASE,
)


def _sanitize_message(msg: str) -> str:
    """Remove any Python-internal details from an error message."""
    if _UNSAFE_PATTERNS.search(msg):
        return "An internal error occurred. Please try again later."
    return msg


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle Starlette HTTPException — return safe JSON."""
    detail = _sanitize_message(str(exc.detail)) if exc.detail else None
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": _sanitize_message(str(exc.status_code)), "detail": detail},
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle any unhandled exception — return safe JSON, never leak traceback."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "An internal error occurred."},
    )


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle Pydantic/FastAPI validation errors — return safe JSON.

    FastAPI's default validation error handler returns detail about
    the validation failure, which is fine (it's about the request,
    not about our internals).  But we still format it consistently.
    """
    # FastAPI wraps Pydantic validation errors in RequestValidationError
    # which has a .body and .errors() method
    if hasattr(exc, "errors"):
        try:
            errors = exc.errors()
            # Sanitize each error message
            safe_errors = []
            for err in errors:
                msg = str(err.get("msg", "Validation error"))
                # Validation messages about input are safe to return
                # but strip any internal paths
                if _UNSAFE_PATTERNS.search(msg):
                    msg = "Invalid input"
                safe_err = {k: v for k, v in err.items() if k != "ctx"}
                safe_err["msg"] = msg
                safe_errors.append(safe_err)
            return JSONResponse(
                status_code=422,
                content={"error": "validation_error", "detail": safe_errors},
            )
        except Exception:
            logger.warning(
                "validation-error redaction failed; returning generic 422", exc_info=True
            )

    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "detail": "Invalid input"},
    )
