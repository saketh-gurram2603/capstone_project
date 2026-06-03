"""
Evaluation API endpoints.

POST /evaluate  — run full eval pipeline (IR + LLM judge) over ground truth dataset
GET  /metrics   — return most recent evaluation run results
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from src.core.dependencies import require_api_key
from src.evaluation.runner import get_latest_eval_run, run_evaluation
from src.handlers.logger import get_logger, log_info, log_warning
from src.models.evaluation import EvalRequest, EvalResult, LatestMetricsResponse, MetricScore

logger = get_logger("api.evaluation")
router = APIRouter(tags=["Evaluation"])


@router.post(
    "/evaluate",
    response_model=EvalResult,
    summary="Run evaluation pipeline over ground truth dataset",
)
async def run_eval(request: Request, body: EvalRequest, _: None = Depends(require_api_key)) -> EvalResult:
    """
    Execute the full evaluation pipeline:

    - **IR metrics** (NDCG@10, MAP@10, Recall@10, Precision@10) against the
      built-in 30-case ground truth dataset.
    - **LLM-as-Judge** (Faithfulness, Answer Relevancy, Contextual Precision)
      using DeepEval + GPT-4o-mini. Set `run_llm_judge=false` to skip and
      save OpenAI costs.

    Results are persisted to Postgres and returned immediately.
    """
    # Evaluation runs against the LOCAL deduped store (set up in lifespan),
    # falling back to the live store only if the local one is unavailable.
    vector_store = (
        getattr(request.app.state, "eval_vector_store", None)
        or getattr(request.app.state, "vector_store", None)
    )
    app_config = getattr(request.app.state, "app_config", {})

    if vector_store is None:
        log_warning("Vector store not ready — cannot run evaluation")
        raise HTTPException(
            status_code=503,
            detail="Vector store is not ready. Please retry after ingestion is complete.",
        )

    from src.integrations.vector_db import QdrantLocalVectorStore
    _collection = app_config.get("QDRANT", {}).get("COLLECTION_NAME", "incidents")
    _backend = "local" if isinstance(vector_store, QdrantLocalVectorStore) else "cloud"
    try:
        _docs = await vector_store.count(_collection)
    except Exception:
        _docs = -1
    log_info(
        "POST /evaluate | backend=%s index_docs=%d ir=%s llm_judge=%s dataset_path=%s%s",
        _backend, _docs, body.run_ir_metrics, body.run_llm_judge,
        body.dataset_path or "built-in",
        "  ⚠ index NOT deduped (expected ~165) — IR scores will be unreliable"
        if _docs > 200 else "",
    )

    try:
        result = await run_evaluation(
            vector_store=vector_store,
            collection=app_config.get("QDRANT", {}).get("COLLECTION_NAME", "incidents"),
            app_config=app_config,
            dataset_path=body.dataset_path,
            run_llm_judge=body.run_llm_judge,
            run_ir_metrics=body.run_ir_metrics,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log_warning("Evaluation pipeline failed | error=%s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {exc}",
        )

    metrics = [MetricScore(**m) for m in result["metrics"]]

    return EvalResult(
        run_id=result["run_id"],
        metrics=metrics,
        overall_passed=result["overall_passed"],
        num_test_cases=result["num_test_cases"],
        latency_ms=result["latency_ms"],
        timestamp=result["timestamp"],
    )


@router.get(
    "/metrics",
    response_model=LatestMetricsResponse,
    summary="Get most recent evaluation run metrics",
)
async def get_metrics() -> LatestMetricsResponse:
    """
    Return the results of the most recent evaluation run from Postgres.

    Returns an empty response with a status message if no run has been
    executed yet.
    """
    log_info("GET /metrics")

    latest = await get_latest_eval_run()

    if latest is None:
        return LatestMetricsResponse(
            message="No evaluation runs found. POST /evaluate to run the first evaluation."
        )

    metrics = [MetricScore(**m) for m in latest.get("metrics", [])]

    return LatestMetricsResponse(
        run_id=latest["run_id"],
        metrics=metrics,
        overall_passed=latest.get("overall_passed"),
        timestamp=latest.get("timestamp"),
        message="",
    )
