"""
Unit tests for all Pydantic V2 models.
Tests: valid inputs, invalid enums, missing required fields, edge cases.
No external services needed — pure in-memory validation.
"""

import pytest
from pydantic import ValidationError

from src.models.incident import (
    IncidentCreate,
    IncidentResponse,
    PriorityEnum,
    ImpactEnum,
    UrgencyEnum,
    IncidentStateEnum,
)
from src.models.search import SearchQuery, SearchFilter, SearchResponse, ResolutionOption
from src.models.triage import TriageRequest, TriageResult, EscalationTicket
from src.models.evaluation import EvalRequest, EvalResult, MetricScore


# ══════════════════════════════════════════════════════════════════════════════
# Incident Models
# ══════════════════════════════════════════════════════════════════════════════

class TestIncidentCreate:

    def test_valid_minimal(self):
        inc = IncidentCreate(number="INC001", description="Database timeout in production")
        assert inc.number == "INC001"
        assert inc.priority is None   # optional fields default to None

    def test_valid_full(self):
        inc = IncidentCreate(
            number="INC002",
            state=IncidentStateEnum.OPEN,
            impact=ImpactEnum.HIGH,
            urgency=UrgencyEnum.HIGH,
            priority=PriorityEnum.P1,
            category="Database",
            description="Connection pool exhausted under peak load",
            resolution_notes="Increased pool size from 10 to 50",
            assigned_to="DBA Team",
        )
        assert inc.priority == PriorityEnum.P1
        assert inc.impact == ImpactEnum.HIGH

    def test_description_stripped(self):
        inc = IncidentCreate(number="INC003", description="  memory leak detected  ")
        assert inc.description == "memory leak detected"

    def test_description_too_short_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            IncidentCreate(number="INC004", description="abc")
        assert "min_length" in str(exc_info.value).lower() or "least 5" in str(exc_info.value)

    def test_blank_description_raises(self):
        with pytest.raises(ValidationError):
            IncidentCreate(number="INC005", description="     ")

    def test_invalid_priority_enum_raises(self):
        with pytest.raises(ValidationError):
            IncidentCreate(number="INC006", description="valid desc", priority="P99")

    def test_invalid_impact_enum_raises(self):
        with pytest.raises(ValidationError):
            IncidentCreate(number="INC007", description="valid desc", impact="CRITICAL")

    def test_missing_number_raises(self):
        with pytest.raises(ValidationError):
            IncidentCreate(description="no number provided")

    def test_missing_description_raises(self):
        with pytest.raises(ValidationError):
            IncidentCreate(number="INC008")


class TestPriorityEnum:

    def test_all_values(self):
        assert PriorityEnum.P1 == "P1"
        assert PriorityEnum.P2 == "P2"
        assert PriorityEnum.P3 == "P3"
        assert PriorityEnum.P4 == "P4"

    def test_invalid_raises(self):
        with pytest.raises(ValidationError):
            IncidentCreate(number="X", description="valid desc here", priority="P5")


class TestIncidentResponse:

    def test_valid(self):
        resp = IncidentResponse(
            incident_id="INC001",
            description="DB timeout",
            similarity_score=0.92,
        )
        assert resp.similarity_score == 0.92
        assert resp.occurrence_count == 1  # default

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            IncidentResponse(
                incident_id="INC001",
                description="desc",
                similarity_score=1.5,  # > 1.0 — invalid
            )


# ══════════════════════════════════════════════════════════════════════════════
# Search Models
# ══════════════════════════════════════════════════════════════════════════════

class TestSearchQuery:

    def test_valid_minimal(self):
        q = SearchQuery(query="database timeout")
        assert q.filters is None
        assert q.top_k is None

    def test_valid_with_filters(self):
        q = SearchQuery(
            query="network latency spike",
            filters=SearchFilter(priority=PriorityEnum.P1, impact=ImpactEnum.HIGH),
            top_k=5,
        )
        assert q.filters.priority == PriorityEnum.P1
        assert q.top_k == 5

    def test_query_too_short_raises(self):
        with pytest.raises(ValidationError):
            SearchQuery(query="ab")  # min_length=3

    def test_blank_query_raises(self):
        with pytest.raises(ValidationError):
            SearchQuery(query="   ")

    def test_query_too_long_raises(self):
        with pytest.raises(ValidationError):
            SearchQuery(query="x" * 1001)  # max_length=1000

    def test_top_k_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            SearchQuery(query="valid query", top_k=0)  # ge=1

        with pytest.raises(ValidationError):
            SearchQuery(query="valid query", top_k=21)  # le=20

    def test_filter_invalid_priority_raises(self):
        with pytest.raises(ValidationError):
            SearchQuery(
                query="valid query",
                filters=SearchFilter(priority="INVALID"),
            )


class TestSearchFilter:

    def test_all_none(self):
        f = SearchFilter()
        assert f.priority is None
        assert f.impact is None
        assert f.category is None
        assert f.state is None

    def test_partial_filter(self):
        f = SearchFilter(priority=PriorityEnum.P2, category="Network")
        assert f.priority == PriorityEnum.P2
        assert f.impact is None


# ══════════════════════════════════════════════════════════════════════════════
# Triage Models
# ══════════════════════════════════════════════════════════════════════════════

class TestTriageRequest:

    def test_valid_minimal(self):
        t = TriageRequest(description="Service keeps throwing OOM errors every few minutes")
        assert t.impact is None
        assert t.urgency is None

    def test_valid_full(self):
        t = TriageRequest(
            description="Database connection pool exhausted under peak load causing 502s",
            impact=ImpactEnum.HIGH,
            urgency=UrgencyEnum.HIGH,
        )
        assert t.impact == ImpactEnum.HIGH

    def test_description_too_short_raises(self):
        with pytest.raises(ValidationError):
            TriageRequest(description="too short")  # min_length=10

    def test_description_too_long_raises(self):
        with pytest.raises(ValidationError):
            TriageRequest(description="x" * 2001)  # max_length=2000

    def test_invalid_impact_raises(self):
        with pytest.raises(ValidationError):
            TriageRequest(description="valid long enough description", impact="EXTREME")


class TestTriageResult:

    def test_valid_l1_result(self):
        result = TriageResult(
            escalation_level="L1",
            priority=PriorityEnum.P3,
            confidence=0.91,
            final_answer="Restart the affected service and monitor logs.",
            l1_summary="Found 8 similar incidents. Most were resolved by service restart.",
            model_used="gpt-4o-mini",
            fallback_used=False,
            latency_ms=320.5,
        )
        assert result.escalation_level == "L1"
        assert result.fallback_used is False

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            TriageResult(
                escalation_level="L1",
                confidence=1.5,   # > 1.0
                final_answer="answer",
                model_used="gpt-4o-mini",
                fallback_used=False,
                latency_ms=100.0,
            )

    def test_invalid_escalation_level_raises(self):
        with pytest.raises(ValidationError):
            TriageResult(
                escalation_level="L4",   # not in Literal["L1","L2","L3"]
                confidence=0.5,
                final_answer="answer",
                model_used="gpt-4o-mini",
                fallback_used=False,
                latency_ms=100.0,
            )


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation Models
# ══════════════════════════════════════════════════════════════════════════════

class TestEvalModels:

    def test_metric_score_passed(self):
        m = MetricScore(name="ndcg_at_10", score=0.87, threshold=0.80, passed=True)
        assert m.passed is True

    def test_metric_score_failed(self):
        m = MetricScore(name="faithfulness", score=0.55, threshold=0.70, passed=False)
        assert m.passed is False

    def test_metric_score_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            MetricScore(name="test", score=1.1, threshold=0.5, passed=True)

    def test_eval_request_defaults(self):
        req = EvalRequest()
        assert req.run_llm_judge is True
        assert req.run_ir_metrics is True
        assert req.dataset_path is None
