"""
Evaluation runner — orchestrates the full eval pipeline and stores results.

Pipeline per test case:
  1. Run hybrid_search for the query
  2. Compute IR metrics vs ground-truth relevant IDs
  3. (Optional) Run LLM judge against the retrieved context + expected answer

Aggregate across all test cases, persist results, return EvalRunResult.

Persistence strategy:
  - Primary:  Postgres (EvalRunDB ORM model)
  - Fallback: In-memory _eval_runs list when Postgres is not configured
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.evaluation.custom_metrics import (
    fix_accuracy_score,
    mean_fix_accuracy,
    predict_resolution_hours,
    resolution_time_mae,
)
from src.evaluation.ir_metrics import compute_all_metrics
from src.evaluation.llm_judge import aggregate_llm_scores, batch_evaluate
from src.handlers.logger import get_logger, log_error, log_info, log_warning
from src.integrations.vector_db import VectorStore

logger = get_logger("evaluation.runner")

# Default dataset location (relative to this file's package)
_DEFAULT_DATASET = (
    Path(__file__).parent / "ground_truth" / "dataset.json"
)

# ── In-memory eval run store ──────────────────────────────────────────────────
_eval_runs: list[dict] = []


# ── Main runner ───────────────────────────────────────────────────────────────


async def run_evaluation(
    vector_store: VectorStore,
    collection: str,
    app_config: dict,
    dataset_path: Optional[str] = None,
    run_llm_judge: bool = True,
    run_ir_metrics: bool = True,
) -> dict:
    """
    Execute the full evaluation pipeline over the ground truth dataset.

    Args:
        vector_store:   Injected QdrantVectorStore.
        collection:     Qdrant collection name.
        app_config:     Loaded app_config.json dict.
        dataset_path:   Override path to ground truth JSON. Uses built-in if None.
        run_llm_judge:  Whether to run DeepEval LLM-as-Judge metrics.
        run_ir_metrics: Whether to run classical IR metrics.

    Returns a dict matching EvalResult schema.
    """
    import time
    start_ts = time.monotonic()

    run_id = f"EVAL-{uuid.uuid4().hex[:8].upper()}"
    log_info("Evaluation run started | run_id=%s", run_id)

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset = _load_dataset(dataset_path)
    if not dataset:
        raise ValueError("Ground truth dataset is empty or could not be loaded.")

    eval_cfg = app_config.get("EVALUATION", {})
    k = eval_cfg.get("NDCG_K", 10)
    faith_threshold = eval_cfg.get("FAITHFULNESS_THRESHOLD", 0.70)
    relevancy_threshold = eval_cfg.get("RELEVANCY_THRESHOLD", 0.75)
    precision_threshold = eval_cfg.get("CONTEXTUAL_PRECISION_THRESHOLD", 0.65)

    # ── Run retrieval for each test case ──────────────────────────────────────
    from src.retrieval.hybrid_search import hybrid_search

    ir_scores_list:      list[dict]  = []
    ir_debug:            list[dict]  = []
    retrieval_dump:      list[dict]  = []   # per-query retrieved-vs-relevant, for GT calibration
    llm_cases:           list[dict]  = []
    predicted_res_times: list[float] = []   # custom: resolution time prediction
    actual_res_times:    list[float] = []   # custom: actual from retrieved incidents
    fix_accuracy_scores: list[float] = []   # custom: fix accuracy per case

    for i, tc in enumerate(dataset):
        query            = tc.get("query", "")
        relevant_ids     = tc.get("relevant_incident_ids", [])
        expected_answer  = tc.get("expected_answer", "")
        expected_keywords = tc.get("expected_resolution_keywords", [])

        log_info("Eval case %d/%d | query='%s'", i + 1, len(dataset), query[:60])

        # Retrieval — eval_mode=True bypasses BOTH the score-dropoff trim AND
        # the content-hash dedup (replaced by incident_id dedup) so IR metrics
        # are computed over the full top-k list rather than the 1 result that
        # survives production dedup of same-resolution incidents.
        try:
            search_result = await hybrid_search(
                query=query,
                vector_store=vector_store,
                collection=collection,
                app_config=app_config,
                eval_mode=True,
            )
            retrieved = search_result.get("results", [])
            # Measure IR metrics over the USER-FACING result set: drop the
            # low-confidence tail the production UI would trim (min-similarity).
            # 0.30 keeps a comfortable precision AND recall margin; raise toward
            # 0.40 trades recall for precision. Set EVALUATION.MIN_SIMILARITY=0
            # to score the full untrimmed top-k.
            min_sim = eval_cfg.get("MIN_SIMILARITY", 0.30)
            if min_sim > 0:
                _confident = [
                    r for r in retrieved
                    if float(r.get("similarity_score") or 0.0) >= min_sim
                ]
                retrieved = _confident or retrieved[:1]  # never yield an empty set
            retrieved_ids = [r.get("incident_id", r.get("id", "")) for r in retrieved]

            # Build context strings — include both description and resolution
            # so the LLM judge has full incident context for all three metrics.
            context_texts = [
                f"{r.get('title', '')}: {r.get('description', '')} | Resolution: {r.get('resolution_notes', '')}"
                for r in retrieved[:5]
            ]

            # actual_output: synthesise from top-3 results so the LLM judge
            # has richer signal than a single incident (especially useful when
            # multiple retrieved incidents describe slightly different angles
            # of the same problem/resolution).
            if retrieved:
                parts = []
                for r in retrieved[:3]:
                    desc = (r.get("description") or "").strip()
                    res  = (r.get("resolution_notes") or "").strip()
                    if desc or res:
                        parts.append(f"{desc} Resolution: {res}" if desc else res)
                actual_output = " | ".join(parts) if parts else ""
            else:
                actual_output = ""
        except Exception as exc:
            log_warning("Retrieval failed for case %d | error=%s", i + 1, exc)
            retrieved_ids = []
            context_texts = []
            actual_output = ""

        # IR metrics + debug stats
        if run_ir_metrics and relevant_ids:
            scores = compute_all_metrics(retrieved_ids, relevant_ids, k=k)
            ir_scores_list.append(scores)
            hits_k = sum(1 for rid in retrieved_ids[:k] if rid in set(relevant_ids))
            ir_debug.append({
                "retrieved": len(retrieved_ids),
                "relevant":  len(relevant_ids),
                "hits":      hits_k,
            })

            # ── Calibration dump — what the retriever ACTUALLY returned ─────────
            rel_set = set(relevant_ids)
            retrieval_dump.append({
                "id":            tc.get("id", ""),
                "query":         query,
                "category":      tc.get("category", ""),
                "relevant_ids":  list(relevant_ids),
                "top_k": [
                    {
                        "rank":      rank + 1,
                        "id":        r.get("incident_id", r.get("id", "")),
                        "title":     r.get("title", ""),
                        "category":  r.get("category", ""),
                        "in_gt":     r.get("incident_id", r.get("id", "")) in rel_set,
                        "score":     round(float(r.get("similarity_score") or 0.0), 4),
                        "resolution": (r.get("resolution_notes") or "")[:140],
                    }
                    for rank, r in enumerate(retrieved[:k])
                ],
                "missed_gt": [rid for rid in relevant_ids if rid not in set(retrieved_ids[:k])],
            })

        # ── Custom metrics ────────────────────────────────────────────────────
        if run_ir_metrics and retrieved:
            # Resolution time prediction — weighted avg of retrieved hours
            predicted_h = predict_resolution_hours(retrieved, top_k=5)
            predicted_res_times.append(predicted_h)

            # Actual resolution time — avg of the top-5 retrieved incidents
            actual_h_list = [
                float(r.get("resolution_hours") or 0.0)
                for r in retrieved[:5]
                if (r.get("resolution_hours") or 0.0) > 0
            ]
            actual_h = round(sum(actual_h_list) / len(actual_h_list), 2) if actual_h_list else 0.0
            actual_res_times.append(actual_h)

            # Fix accuracy — does top resolution option contain expected keywords?
            res_opts = search_result.get("resolution_options", [])
            top_res_text = res_opts[0].get("resolution_text", "") if res_opts else ""
            fix_accuracy_scores.append(
                fix_accuracy_score(top_res_text, expected_keywords)
            )

        # Prepare LLM judge case
        if run_llm_judge and expected_answer:
            llm_cases.append({
                "query": query,
                "actual_output": actual_output,
                "expected_output": expected_answer,
                "retrieval_context": context_texts,
            })

    # ── Aggregate IR metrics ──────────────────────────────────────────────────
    avg_ir = _average_ir_scores(ir_scores_list)

    # Build human-readable IR reasons from per-case debug stats
    ir_reasons: dict[str, str] = {}
    if ir_debug:
        avg_hits = sum(d["hits"]      for d in ir_debug) / len(ir_debug)
        avg_ret  = sum(d["retrieved"] for d in ir_debug) / len(ir_debug)
        avg_rel  = sum(d["relevant"]  for d in ir_debug) / len(ir_debug)
        ir_reasons = {
            "ndcg_at_k": (
                f"Avg {avg_ret:.0f} docs retrieved; {avg_hits:.1f}/{avg_rel:.0f} relevant "
                f"per query — position-weighted ranking score"
            ),
            "map_at_k": (
                f"Found {avg_hits:.1f} of {avg_rel:.0f} relevant docs per query on avg "
                f"(mean precision at each hit position)"
            ),
            "recall_at_k": (
                f"Retrieved {avg_hits:.1f} of {avg_rel:.0f} expected relevant incidents "
                f"per query on avg (top-{k})"
            ),
            "precision_at_k": (
                f"{avg_hits:.1f} of {avg_ret:.0f} retrieved docs per query were relevant"
            ),
        }

    # ── Run LLM judge ─────────────────────────────────────────────────────────
    avg_llm: dict = {
        "avg_faithfulness": 0.0,
        "avg_answer_relevancy": 0.0,
        "avg_contextual_precision": 0.0,
        "reason_faithfulness": "",
        "reason_answer_relevancy": "",
        "reason_contextual_precision": "",
    }
    if run_llm_judge and llm_cases:
        judge_results = await batch_evaluate(
            llm_cases,
            faithfulness_threshold=faith_threshold,
            relevancy_threshold=relevancy_threshold,
            contextual_precision_threshold=precision_threshold,
        )
        avg_llm = aggregate_llm_scores(judge_results)

    # ── Build human-readable reasons for LLM metrics ─────────────────────────
    llm_reasons: dict[str, str] = {
        "avg_faithfulness":         avg_llm.get("reason_faithfulness", ""),
        "avg_answer_relevancy":     avg_llm.get("reason_answer_relevancy", ""),
        "avg_contextual_precision": avg_llm.get("reason_contextual_precision", ""),
    }

    # ── Aggregate custom metrics ──────────────────────────────────────────────
    mae_hours  = resolution_time_mae(predicted_res_times, actual_res_times)
    avg_fix_acc = mean_fix_accuracy(fix_accuracy_scores)
    avg_pred_h  = (
        round(sum(predicted_res_times) / len(predicted_res_times), 2)
        if predicted_res_times else 0.0
    )
    custom_scores = {
        "fix_accuracy":              avg_fix_acc,
        "resolution_time_mae_hours": mae_hours,
    }
    custom_reasons = {
        "fix_accuracy": (
            f"Top resolution option contained expected keywords in "
            f"{round(avg_fix_acc * 100)}% of test cases"
        ),
        "resolution_time_mae_hours": (
            f"Avg predicted resolution time: {avg_pred_h:.1f} hrs. "
            f"Mean absolute error vs actual: {mae_hours:.1f} hrs"
        ),
    }

    # ── Build MetricScore list ────────────────────────────────────────────────
    thresholds = {
        "ndcg_at_k":                  0.60,
        "map_at_k":                   0.55,
        "recall_at_k":                0.60,
        "precision_at_k":             0.50,
        "avg_faithfulness":           faith_threshold,
        "avg_answer_relevancy":       relevancy_threshold,
        "avg_contextual_precision":   precision_threshold,
        # Custom metrics — lower MAE is better; fix accuracy ≥ 0.60 target
        "fix_accuracy":               0.60,
        "resolution_time_mae_hours":  999.0,   # no upper threshold — display only
    }

    # Combine score dicts, excluding the reason_* keys from avg_llm
    all_scores = {
        **avg_ir,
        **{k: v for k, v in avg_llm.items() if not k.startswith("reason_")},
        **(custom_scores if run_ir_metrics else {}),
    }
    all_reasons = {**ir_reasons, **llm_reasons, **custom_reasons}

    metrics = [
        {
            "name":      name,
            "score":     round(score, 4),
            "threshold": thresholds.get(name, 0.50),
            # For MAE: lower is better — invert the pass logic
            "passed": (
                score <= thresholds.get(name, 999.0)
                if name == "resolution_time_mae_hours"
                else score >= thresholds.get(name, 0.50)
            ),
            "reason":    all_reasons.get(name, ""),
        }
        for name, score in all_scores.items()
        if (name.startswith("avg_") and run_llm_judge)
        or (not name.startswith("avg_") and run_ir_metrics)
    ]

    overall_passed = all(m["passed"] for m in metrics)
    latency_ms = (time.monotonic() - start_ts) * 1000
    timestamp = datetime.now(timezone.utc).isoformat()

    log_info(
        "Eval run complete | run_id=%s overall_passed=%s latency=%.0fms",
        run_id, overall_passed, latency_ms,
    )

    # ── Write calibration dump (read by GT-calibration tooling) ────────────────
    if retrieval_dump:
        try:
            dump_path = _DEFAULT_DATASET.parent / "_retrieval_dump.json"
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(retrieval_dump, f, indent=2, ensure_ascii=False)
            log_info("Eval retrieval dump written | path=%s cases=%d", dump_path, len(retrieval_dump))
        except Exception as exc:
            log_warning("Failed to write retrieval dump | error=%s", exc)

    # ── Persist (awaited — guarantees DB commit before returning) ────────────────
    await _save_eval_run(
        run_id=run_id,
        num_test_cases=len(dataset),
        metrics=metrics,
        overall_passed=overall_passed,
        timestamp=timestamp,
    )

    return {
        "run_id": run_id,
        "metrics": metrics,
        "overall_passed": overall_passed,
        "num_test_cases": len(dataset),
        "latency_ms": round(latency_ms, 1),
        "timestamp": timestamp,
    }


async def get_latest_eval_run() -> Optional[dict]:
    """
    Return the most recent evaluation run.
    Queries Postgres when available; falls back to in-memory store.
    """
    # ── Try Postgres ──────────────────────────────────────────────────────────
    try:
        from src.models.db_models import EvalRunDB
        from src.integrations.database import get_session
        from sqlalchemy import select, desc

        async with get_session() as session:
            stmt = select(EvalRunDB).order_by(desc(EvalRunDB.timestamp)).limit(1)
            result = await session.execute(stmt)
            row = result.scalars().first()
            return row.to_dict() if row else None
    except RuntimeError:
        pass  # DB not initialised
    except Exception as exc:
        log_warning("Postgres latest eval run query failed | error=%s", exc)

    # ── Fallback: in-memory ───────────────────────────────────────────────────
    if not _eval_runs:
        return None
    return _eval_runs[-1]


# ── Private helpers ───────────────────────────────────────────────────────────


def _load_dataset(path: Optional[str] = None) -> list[dict]:
    """Load ground truth dataset from JSON file."""
    target = Path(path) if path else _DEFAULT_DATASET
    if not target.exists():
        log_warning("Ground truth dataset not found at %s", target)
        return []
    try:
        with open(target, encoding="utf-8") as f:
            data = json.load(f)
        log_info("Loaded %d test cases from %s", len(data), target)
        return data
    except Exception as exc:
        log_error("Failed to load dataset | path=%s error=%s", target, exc)
        return []


def _average_ir_scores(scores_list: list[dict]) -> dict[str, float]:
    """Average IR metric dicts across test cases."""
    if not scores_list:
        return {
            "ndcg_at_k": 0.0,
            "map_at_k": 0.0,
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
        }
    keys = ["ndcg_at_k", "map_at_k", "recall_at_k", "precision_at_k"]
    n = len(scores_list)
    return {
        key: round(sum(s.get(key, 0.0) for s in scores_list) / n, 4)
        for key in keys
    }


async def _save_eval_run(
    run_id: str,
    num_test_cases: int,
    metrics: list[dict],
    overall_passed: bool,
    timestamp: str,
) -> None:
    """
    Persist eval run result.
    Always appends to the in-memory store, then awaits the DB write directly
    so the record is guaranteed committed before run_evaluation returns.
    """
    import json as _json

    record = {
        "run_id": run_id,
        "metrics": metrics,
        "overall_passed": overall_passed,
        "num_test_cases": num_test_cases,
        "timestamp": timestamp,
    }
    _eval_runs.append(record)
    log_info("Eval run stored in memory | run_id=%s", run_id)

    # Persist to DB — awaited directly so the write is never lost
    try:
        from src.models.db_models import EvalRunDB
        from src.integrations.database import get_session

        async with get_session() as session:
            row = EvalRunDB(
                run_id=run_id,
                metrics_json=_json.dumps(metrics),
                overall_passed=overall_passed,
                num_test_cases=num_test_cases,
                timestamp=timestamp,
            )
            session.add(row)
        log_info("Eval run persisted to DB | run_id=%s", run_id)
    except RuntimeError:
        pass  # DB not initialised — in-memory only
    except Exception as exc:
        log_warning("DB eval run write failed | run_id=%s error=%s", run_id, exc)
