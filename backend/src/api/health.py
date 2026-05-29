"""
Health check endpoints.
  GET /health        — liveness  (is the process alive?)
  GET /health/ready  — readiness (can we accept traffic?)
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.handlers.logger import get_logger

logger = get_logger("api.health")

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Liveness probe")
async def liveness():
    """Kubernetes liveness probe — returns 200 if the process is running."""
    return {"status": "alive", "service": "incident-kb-assistant"}


@router.get("/health/ready", summary="Readiness probe")
async def readiness():
    """
    Readiness probe — checks Qdrant and Postgres connectivity.
    Returns 503 if Qdrant is unavailable (Postgres is optional; degraded is still serviceable).
    """
    checks: dict[str, bool] = {}

    # Qdrant — mandatory for search
    try:
        from src.integrations import _qdrant_store
        checks["qdrant"] = await _qdrant_store.health_check()
    except Exception:
        checks["qdrant"] = False

    # Postgres — optional; tickets fall back to in-memory if unavailable
    try:
        from src.integrations.database import health_check as pg_health
        checks["postgres"] = await pg_health()
    except Exception:
        checks["postgres"] = False

    # Service is ready as long as Qdrant is up; Postgres degraded is acceptable
    qdrant_ok = checks.get("qdrant", False)

    logger.info("Readiness check | %s", checks)

    return JSONResponse(
        status_code=200 if qdrant_ok else 503,
        content={
            "status": "ready" if qdrant_ok else "degraded",
            "checks": {k: "ok" if v else "fail" for k, v in checks.items()},
        },
    )
