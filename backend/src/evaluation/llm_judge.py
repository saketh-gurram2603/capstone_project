"""
LLM-as-Judge evaluation using DeepEval — Azure OpenAI backend.

Metrics computed:
  - Faithfulness          — does the answer stay grounded in the retrieved context?
  - Answer Relevancy      — does the answer address the question?
  - Contextual Precision  — are the retrieved docs actually relevant to the question?

DeepEval is configured to call our Azure deployment via _AzureJudgeModel, a thin
DeepEvalBaseLLM subclass that wraps the synchronous AzureOpenAI client.
Call init_llm_judge() once at startup before running evaluations.

Results are returned as plain dicts so the runner can serialise them to Postgres
without a DeepEval dependency in other modules.

If DeepEval or Azure is unavailable, each metric returns score=0.0 with
reason="unavailable" so the eval pipeline degrades gracefully.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

from src.handlers.logger import get_logger, log_info, log_warning

# ── Azure judge model state ───────────────────────────────────────────────────
# Populated by init_llm_judge() at startup.  The judge uses a *synchronous*
# AzureOpenAI client because DeepEval metrics run synchronously inside
# _run_all_deepeval_sync (which is already in a thread executor).
_azure_judge_client = None   # openai.AzureOpenAI — set at startup
_azure_judge_deployment: str = "synapt-dev-gpt-4o-mini"


def init_llm_judge(
    azure_api_key: str,
    azure_endpoint: str,
    azure_api_version: str,
    deployment: str = "synapt-dev-gpt-4o-mini",
) -> None:
    """
    Initialise the synchronous Azure OpenAI client used by DeepEval metrics.
    Call once from the FastAPI lifespan before any evaluation runs.
    """
    global _azure_judge_client, _azure_judge_deployment
    try:
        from openai import AzureOpenAI  # sync client — correct for DeepEval's sync path
        _azure_judge_client = AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
        )
        _azure_judge_deployment = deployment
        log_info(
            "Azure OpenAI judge client initialised | endpoint=%s deployment=%s",
            azure_endpoint, deployment,
        )
    except Exception as exc:
        log_warning("LLM judge init failed — evaluation will be unavailable | error=%s", exc)


class _AzureJudgeModel:
    """
    Thin DeepEvalBaseLLM subclass that routes metric calls to our Azure
    deployment instead of the default OpenAI endpoint.

    DeepEval calls generate() synchronously when running inside a thread
    (our run_in_executor pattern), so we only need the sync path.
    """

    def __init__(self, client: Any, deployment: str) -> None:
        self._client = client
        self._deployment = deployment

    # Required by DeepEvalBaseLLM -------------------------------------------------

    def load_model(self) -> Any:
        return self._client

    def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        # DeepEval may call this if it detects an async context; delegate to sync.
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self._deployment

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

    Metrics are configured with _AzureJudgeModel so every evaluation call
    goes through our Azure deployment instead of the default OpenAI endpoint.

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
            from deepeval.models.base_model import DeepEvalBaseLLM  # noqa: PLC0415
            from deepeval.test_case import LLMTestCase  # noqa: PLC0415

            # Build an Azure-backed judge model so DeepEval never calls the
            # raw OpenAI endpoint directly.  Falls back to the string name
            # "gpt-4o-mini" only when the Azure client was not initialised
            # (e.g. missing key at startup) — metrics will then error and be
            # caught by the outer try/except, returning score=0.0.
            if _azure_judge_client is not None:
                # Dynamically subclass DeepEvalBaseLLM so DeepEval accepts it
                # as a model argument (it checks isinstance at metric init).
                class _AzureModel(_AzureJudgeModel, DeepEvalBaseLLM):
                    pass
                judge_model: Any = _AzureModel(_azure_judge_client, _azure_judge_deployment)
            else:
                log_warning("Azure judge client not initialised — falling back to model name string")
                judge_model = _azure_judge_deployment   # type: ignore[assignment]

            test_case = LLMTestCase(
                input=query,
                actual_output=actual_output,
                expected_output=expected_output,
                retrieval_context=retrieval_context,
            )

            # Metric __init__ writes .deepeval → must be inside the workdir.
            faith_metric = FaithfulnessMetric(
                threshold=faithfulness_threshold,
                model=judge_model,
                include_reason=True,
            )
            relevancy_metric = AnswerRelevancyMetric(
                threshold=relevancy_threshold,
                model=judge_model,
                include_reason=True,
            )
            precision_metric = ContextualPrecisionMetric(
                threshold=contextual_precision_threshold,
                model=judge_model,
                include_reason=True,
            )

            # measure() also writes .deepeval → already covered by same chdir.
            faith_metric.measure(test_case)
            relevancy_metric.measure(test_case)
            precision_metric.measure(test_case)

            return faith_metric, relevancy_metric, precision_metric

        finally:
            os.chdir(previous)


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
    Run DeepEval LLM-as-Judge metrics for a single test case via Azure OpenAI.

    Returns a dict:
    {
        "faithfulness":         {"score": float, "reason": str, "passed": bool},
        "answer_relevancy":     {"score": float, "reason": str, "passed": bool},
        "contextual_precision": {"score": float, "reason": str, "passed": bool},
    }
    """
    if _azure_judge_client is None:
        log_warning("Azure judge client not initialised — skipping LLM judge (call init_llm_judge at startup)")
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
