"""Pydantic V2 schemas for Evaluation domain."""

from typing import Optional
from pydantic import BaseModel, Field


class MetricScore(BaseModel):
    """A single named metric result."""

    name: str = Field(..., description="Metric name e.g. ndcg_at_10")
    score: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., description="Minimum passing threshold")
    passed: bool = Field(..., description="True if score >= threshold")
    reason: Optional[str] = Field(
        None,
        description="Human-readable explanation of the score (retrieved counts, LLM judge rationale, etc.)",
    )


class EvalRequest(BaseModel):
    """Request body for POST /evaluate."""

    dataset_path: Optional[str] = Field(
        None,
        description="Path to ground truth dataset JSON. Defaults to built-in dataset.",
    )
    run_llm_judge: bool = Field(
        default=True,
        description="Run DeepEval LLM-as-Judge metrics (requires OpenAI calls)",
    )
    run_ir_metrics: bool = Field(
        default=True,
        description="Run classical IR metrics (NDCG, MAP, Recall, Precision)",
    )


class EvalResult(BaseModel):
    """Response body for POST /evaluate."""

    run_id: str = Field(..., description="Unique evaluation run identifier")
    metrics: list[MetricScore]
    overall_passed: bool = Field(..., description="True if ALL metrics passed their threshold")
    num_test_cases: int
    latency_ms: float = Field(..., description="Total evaluation run time in milliseconds")
    timestamp: str


class LatestMetricsResponse(BaseModel):
    """Response for GET /metrics — returns most recent eval run."""

    run_id: Optional[str] = None
    metrics: list[MetricScore] = Field(default_factory=list)
    overall_passed: Optional[bool] = None
    timestamp: Optional[str] = None
    message: str = Field(default="", description="Status message if no eval has been run yet")
