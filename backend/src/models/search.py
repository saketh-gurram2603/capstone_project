"""Pydantic V2 schemas for Search domain."""

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator

from src.models.incident import IncidentResponse, PriorityEnum, ImpactEnum


class SearchFilter(BaseModel):
    """Optional metadata filters applied before vector search."""

    priority: Optional[PriorityEnum] = Field(None, description="Filter by priority (P1-P4)")
    impact: Optional[ImpactEnum] = Field(None, description="Filter by impact level")
    category: Optional[str] = Field(None, description="Filter by incident category")
    state: Optional[str] = Field(None, description="Filter by incident state")


class SearchQuery(BaseModel):
    """Request body for POST /search."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language description of the incident to search for",
    )
    filters: Optional[SearchFilter] = Field(
        None,
        description="Optional metadata filters to narrow down results",
    )
    top_k: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Override number of results (default: adaptive-k)",
    )
    mode: Literal["hybrid", "semantic"] = Field(
        "hybrid",
        description="hybrid = BM25 + vector (RRF); semantic = vector-only",
    )

    @model_validator(mode="after")
    def query_not_blank(self) -> "SearchQuery":
        if not self.query.strip():
            raise ValueError("query cannot be blank.")
        return self


class ResolutionOption(BaseModel):
    """A unique resolution approach surfaced by the aggregator."""

    resolution_text: str = Field(..., description="The resolution notes text")
    occurrence_count: int = Field(..., description="How many past incidents used this fix")
    avg_similarity: float = Field(..., ge=0.0, le=1.0, description="Average similarity of cluster")
    source_incident_ids: list[str] = Field(
        default_factory=list,
        description="Incident IDs that share this resolution",
    )


class SearchResponse(BaseModel):
    """Response body for POST /search."""

    query: str
    total_found: int = Field(..., description="Total candidates before final trim")
    results: list[IncidentResponse] = Field(..., description="Ranked incident results")
    resolution_options: list[ResolutionOption] = Field(
        default_factory=list,
        description="Deduplicated unique resolution approaches with occurrence counts",
    )
    adaptive_k_used: int = Field(..., description="k value chosen by adaptive-k selector")
    retrieval_method: str = Field(
        default="hybrid",
        description="hybrid | bm25_only (fallback when Qdrant unavailable)",
    )
    cached: bool = Field(default=False, description="True if result was served from cache (always False — caching not active)")
    latency_ms: float = Field(..., description="Total retrieval latency in milliseconds")
