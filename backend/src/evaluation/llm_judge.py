"""
LLM-as-Judge evaluation using DeepEval.

Metrics computed:
  - Faithfulness          — does the answer stay grounded in the retrieved context?
  - Answer Relevancy      — does the answer address the question?
  - Contextual Precision  — are the retrieved docs actually relevant to the question?

All three metrics call OpenAI under the hood.  Results are returned as plain
dicts so the runner can serialise them to Postgres without a DeepEval dependency
in other modules.

If DeepEval or OpenAI is unavailable, each metric returns score=0.0 with
reason="unavailable" so the eval pipeline degrades gracefully.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

from src.handlers.logger import get_logger, log_info, log_warning

logger = get_logger("evaluation.llm_judge")

# ── DeepEval bootstrap ────────────────────────────────────────────────────────
# DeepEval 2.x writes a `.deepeval` JSON file (login state + telemetry) into
# `os.getcwd()` on EVERY metric init AND measure() call.  When the CWD is not
# writable (antivirus quarantine, file lock, restricted ACL — common on
# Windows) every judge call fails with "Permission denied: '.deepeval'"
# (EACCES, errno 13), silently scoring every case 0.0.
#
# Root cause: the write happens in FaithfulnessMetric.__init__() and inside
# measure(), NOT just during measure().  A fix that only chdirs around
# measure() is therefore insufficient — init still runs in the original CWD.
#
# Fix: consolidate ALL deepeval operations (import → init → measure) into a
# single synchronous function (_run_all_deepeval_sync) that runs under a
# threading lock with the CWD set to a guaranteed-writable temp directory.
# A single run_in_executor call invokes that function; the async wrapper
# awaits it and unpacks the results.
#
# IMPORTANT: these env vars must be set before `deepeval` is imported the
# first time, which is why they live at module-import scope.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT",        "YES")
os.environ.setdefault("IGNORE_DEEPEVAL_LOGIN_INVITATION",  "YES")
os.environ.setdefault("ERROR_REPORTING",                   "NO")

_DEEPEVAL_WORKDIR = Path(tempfile.gettempdir()) / "incident_kb_deepeval"
try:
    _DEEPEVAL_WORKDIR.mkdir(parents=True, exist_ok=True)
except OSError as _exc:
    log_warning("Could not create DeepEval workdir %s | error=%s", _DEEPEVAL_WORKDIR, _exc)

# Serialise ALL chdir operations so concurrent executor threads never race on
# the process-global CWD.
_deepeval_cwd_lock = threading.Lock()


def _run_all_deepeval_sync(
    query: str,
    actual_output: str,
    expected_output: str,
    retrieval_context: list,
    faithfulness_threshold: float,
    relevancy_threshold: float,
    contextual_precision_threshold: float,
):
    """
    Run ALL DeepEval operations — lazy imports, metric initialisation, and
    measure() — inside a guaranteed-writable scratch directory.

    This is the critical change vs. the previous implementation: metric
    __init__() calls also write `.deepeval` to CWD, so they must be covered
    by the same chdir context as measure().  Running everything in one locked
    block eliminates every EACCES write.

    Called via asyncio.run_in_executor so it executes on a thread-pool
    thread.  Returns (faith_metric, relevancy_metric, precision_metric).
    """
    with _deepeval_cwd_lock:
        previous = os.getcwd()
        try:
            os.chdir(_DEEPEVAL_WORKDIR)

            # Lazy imports — inside the workdir so any import-time file writes
            # also land in the writable temp directory.
            from deepeval.metrics import (          # noqa: PLC0415
                AnswerRelevancyMetric,
                ContextualPrecisionMetric,
                FaithfulnessMetric,
            )
            from deepeval.test_case import LLMTestCase  # noqa: PLC0415

            test_case = LLMTestCase(
                input=query,
                actual_output=actual_output,
                expected_output=expected_output,
                retrieval_context=retrieval_context,
            )

            # Metric __init__ writes .deepeval → must be inside the workdir.
            faith_metric = FaithfulnessMetric(
                threshold=faithfulness_threshold,
                model="gpt-4o-mini",
                include_reason=True,
            )
            relevancy_metric = AnswerRelevancyMetric(
                threshold=relevancy_threshold,
                model="gpt-4o-mini",
                include_reason=True,
            )
            precision_metric = ContextualPrecisionMetric(
                threshold=contextual_precision_threshold,
                model="gpt-4o-mini",
                include_reason=True,
            )

            # measure() also writes .deepeval → already covered by same chdir.
            faith_metric.measure(test_case)
            relevancy_metric.measure(test_case)
            precision_metric.measure(test_case)

            return faith_metric, relevancy_metric, precision_metric

        finally:
            os.chdir(previous)


def _get_openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "")


# ── Core evaluation function ──────────────────────────────────────────────────


async def evaluate_with_llm_judge(
    query: str,
    actual_output: str,
    expected_output: str,
    retrieval_context: list[str],
    faithfulness_threshold: float = 0.70,
    relevancy_threshold: float = 0.75,
    contextual_precision_threshold: float = 0.65,
) -> dict:
    """
    Run DeepEval LLM-as-Judge metrics for a single test case.

    Returns a dict:
    {
        "faithfulness":         {"score": float, "reason": str, "passed": bool},
        "answer_relevancy":     {"score": float, "reason": str, "passed": bool},
        "contextual_precision": {"score": float, "reason": str, "passed": bool},
    }
    """
    api_key = _get_openai_key()
    if not api_key:
        log_warning("OPENAI_API_KEY not set — skipping LLM judge")
        return _unavailable_result()

    # Quick import-availability check (does not trigger file writes).
    try:
        import deepeval  # noqa: F401
    except ImportError:
        log_warning("deepeval not installed — skipping LLM judge")
        return _unavailable_result()

    try:
        loop = asyncio.get_running_loop()

        # Run every DeepEval operation in a thread so the blocking OpenAI
        # calls don't stall the event loop, and so we can safely chdir.
        faith_metric, relevancy_metric, precision_metric = (
            await loop.run_in_executor(
                None,
                _run_all_deepeval_sync,
                query,
                actual_output,
                expected_output,
                retrieval_context,
                faithfulness_threshold,
                relevancy_threshold,
                contextual_precision_threshold,
            )
        )

        result = {
            "faithfulness": {
                "score": round(float(faith_metric.score or 0.0), 4),
                "reason": faith_metric.reason or "",
                "passed": bool(faith_metric.is_successful()),
            },
            "answer_relevancy": {
                "score": round(float(relevancy_metric.score or 0.0), 4),
                "reason": relevancy_metric.reason or "",
                "passed": bool(relevancy_metric.is_successful()),
            },
            "contextual_precision": {
                "score": round(float(precision_metric.score or 0.0), 4),
                "reason": precision_metric.reason or "",
                "passed": bool(precision_metric.is_successful()),
            },
        }

        log_info(
            "LLM judge | faith=%.2f relevancy=%.2f precision=%.2f",
            result["faithfulness"]["score"],
            result["answer_relevancy"]["score"],
            result["contextual_precision"]["score"],
        )
        return result

    except Exception as exc:
        log_warning("LLM judge evaluation failed | error=%s", exc)
        return _unavailable_result(reason=str(exc))


# ── Batch evaluation ──────────────────────────────────────────────────────────


async def batch_evaluate(
    test_cases: list[dict],
    faithfulness_threshold: float = 0.70,
    relevancy_threshold: float = 0.75,
    contextual_precision_threshold: float = 0.65,
) -> list[dict]:
    """
    Run LLM judge over a batch of test cases.

    Each test_case dict must have:
        query, actual_output, expected_output, retrieval_context (list[str])

    Returns list of result dicts (same order as input).
    """
    results = []
    for i, tc in enumerate(test_cases):
        log_info("LLM judge batch | case %d/%d", i + 1, len(test_cases))
        result = await evaluate_with_llm_judge(
            query=tc.get("query", ""),
            actual_output=tc.get("actual_output", ""),
            expected_output=tc.get("expected_output", ""),
            retrieval_context=tc.get("retrieval_context", []),
            faithfulness_threshold=faithfulness_threshold,
            relevancy_threshold=relevancy_threshold,
            contextual_precision_threshold=contextual_precision_threshold,
        )
        results.append(result)
    return results


def aggregate_llm_scores(judge_results: list[dict]) -> dict:
    """
    Average each LLM-judge metric across all test cases and collect a
    representative reason (first non-empty reason from the batch) for each.

    Returns:
        {
            "avg_faithfulness":            float,
            "avg_answer_relevancy":        float,
            "avg_contextual_precision":    float,
            "reason_faithfulness":         str,
            "reason_answer_relevancy":     str,
            "reason_contextual_precision": str,
        }
    """
    if not judge_results:
        return {
            "avg_faithfulness": 0.0,
            "avg_answer_relevancy": 0.0,
            "avg_contextual_precision": 0.0,
            "reason_faithfulness": "",
            "reason_answer_relevancy": "",
            "reason_contextual_precision": "",
        }

    def _avg_and_reason(metric_key: str) -> tuple[float, str]:
        scores  = [r[metric_key]["score"]  for r in judge_results if metric_key in r]
        reasons = [r[metric_key]["reason"] for r in judge_results if metric_key in r]
        avg    = round(sum(scores) / len(scores), 4) if scores else 0.0
        # Pick first non-empty, non-placeholder reason as representative
        reason = next(
            (rr for rr in reasons if rr and rr not in ("unavailable", "")),
            "",
        )
        return avg, reason

    faith_avg,     faith_reason     = _avg_and_reason("faithfulness")
    relevancy_avg, relevancy_reason = _avg_and_reason("answer_relevancy")
    precision_avg, precision_reason = _avg_and_reason("contextual_precision")

    return {
        "avg_faithfulness":            faith_avg,
        "avg_answer_relevancy":        relevancy_avg,
        "avg_contextual_precision":    precision_avg,
        "reason_faithfulness":         faith_reason,
        "reason_answer_relevancy":     relevancy_reason,
        "reason_contextual_precision": precision_reason,
    }


# ── Private helpers ───────────────────────────────────────────────────────────


def _unavailable_result(reason: str = "unavailable") -> dict:
    stub = {"score": 0.0, "reason": reason, "passed": False}
    return {
        "faithfulness": stub.copy(),
        "answer_relevancy": stub.copy(),
        "contextual_precision": stub.copy(),
    }
