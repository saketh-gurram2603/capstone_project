# AI-Powered Incident Knowledge Base Assistant — Master Plan

> **Hard stop: May 29, 2026 (coding)**  
> **Post May 29: ADRs · DECISIONS.md · README · Stakeholder PPT**  
> **Current bookmark: ✅ Phases 1–6 complete — next: Phase 7 (Docker + Locust Load Tests)**

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [Folder Structure](#3-folder-structure)
4. [System Flows](#4-system-flows)
5. [Agent State Design](#5-agent-state-design)
6. [Adaptive-K Retrieval Logic](#6-adaptive-k-retrieval-logic)
7. [Multi-Resolution Handling](#7-multi-resolution-handling)
8. [Production Patterns](#8-production-patterns)
9. [Dataset](#9-dataset)
10. [Phase Breakdown & Status](#10-phase-breakdown--status)
11. [Test Inventory](#11-test-inventory)
12. [ADRs (Post May 29)](#12-adrs-post-may-29)
13. [Final Verification Checklist](#13-final-verification-checklist)

---

## 1. Project Overview

Support engineers describe a production problem in natural language. The system:

1. **Retrieves** the most relevant historical incidents using hybrid BM25 + vector search, RRF fusion, adaptive-K, and a cross-encoder reranker.
2. **Surfaces all unique resolution approaches** found across similar past incidents (not just the top-1 fix).
3. **Triages** intelligently through a three-tier agent (L1 → L2 → L3), escalating only when confidence is genuinely insufficient.
4. **Evaluates** its own quality with DeepEval LLM-as-Judge metrics and standard IR metrics (NDCG, MAP, Recall).

**SME scoring dimensions targeted:**  
Architecture · Design Decisions · Performance · Testing · Evaluation · Scalability · Reliability · Maintainability · Observation

---

## 2. Technology Stack

| Layer | Choice | Why |
|---|---|---|
| **API Framework** | FastAPI + Pydantic V2 | Async ASGI, automatic input validation, OpenAPI docs for free |
| **Vector DB** | Qdrant | Filterable HNSW index, metadata filter DSL, async client, swap-able via abstract interface |
| **Keyword Search** | BM25 (`rank_bm25`) | Exact keyword recall, entirely local, graceful fallback when Qdrant is unavailable |
| **Retrieval Count** | Adaptive-K (3 – 20) | Adjusts candidate count to query complexity; cuts ~40% latency on simple queries |
| **Score Fusion** | RRF (k = 60) | Rank-only fusion — no score normalisation needed, proven robust across modalities |
| **Reranker** | `ms-marco-MiniLM-L-6-v2` cross-encoder | Stage-2 accuracy boost on the trimmed candidate set; loaded once at startup |
| **Resolution Strategy** | Aggregator + LLM synthesis | Surfaces ALL unique fixes grouped by cosine similarity > 0.95 with occurrence counts |
| **Embeddings** | OpenAI `text-embedding-ada-002` | Project requirement; 1536-dim Cosine space |
| **Embedding Fallback** | `all-MiniLM-L6-v2` (local) | Loaded at startup; used when ada-002 API fails |
| **L1 Agent** | GPT-4o-mini | Cheap, fast; KB hybrid search → summarise → confidence gate (≥ 0.80 auto-resolve) |
| **L2 Agent** | GPT-4o + Tavily API | Web search + L1 context + model knowledge synthesis when L1 confidence < 0.80 |
| **L3 Agent** | No LLM — pure routing | Postgres escalation ticket creation; returns `ticket_id` |
| **LLM Fallback** | `google/flan-t5-base` (local) | Loaded at startup via HuggingFace Transformers; activates when circuit breaker opens |
| **Circuit Breaker** | `pybreaker` | fail_max = 5, reset_timeout = 60s; protects all OpenAI calls |
| **Retry** | `tenacity` | 3 attempts, exponential backoff (2s → 4s → 8s) |
| **Cache** | Redis | Embedding cache (TTL 24h) + query result cache (TTL 1h); `ConnectionPool(max_connections=20)` |
| **Metadata DB** | Postgres + SQLAlchemy async | Eval results, L3 escalation queue; `QueuePool(pool_size=20)`, `pool_pre_ping=True` |
| **Agent Orchestration** | LangGraph `StateGraph` | Explicit L1→L2→L3 conditional edges; `IncidentState` TypedDict carries full context |
| **Logging** | Python `logging` + `RotatingFileHandler` | app.log (INFO+) + error.log (ERROR+); 5 MB / 5 backups; zero `print()` in production |
| **Evaluation** | DeepEval + custom IR metrics | Faithfulness · AnswerRelevancy · ContextualPrecision · ContextualRecall + NDCG@10, MAP@10, Recall@10 |
| **Load Testing** | Locust | 50 concurrent users; p99 < 500 ms; 70 % /search · 20 % /triage · 10 % /health |
| **Frontend** | React 18 + Vite + TypeScript + TailwindCSS + React Query | Fast build, typed, professional UI |
| **Containerisation** | Docker + docker-compose | **Phase 7 only** — one-command startup; all services health-checked |

---

## 3. Folder Structure

```
D:\soft_bank\capstone_project\
│
├── PLAN.md                          ← this file
├── docker-compose.yml               ← Orchestrates backend + qdrant + redis + postgres + frontend
├── docker-compose.prod.yml
├── .env.example
├── .gitignore
├── README.md
│
├── data/
│   └── incidents.xlsx               ← 150-row ITSM dataset (MediaServer incidents)
│
├── requirements/
│   └── project-requirements.md
│
├── k8s/
│   ├── incident-kb-dev.yaml
│   └── incident-kb-prod.yaml
│
├── backend/                         ── ALL Python code ──────────────────────────────
│   ├── main.py                      ← Entry point: load config → init FastAPI → register routers
│   ├── requirements.txt
│   ├── Dockerfile                   (Phase 7)
│   │
│   ├── configuration/
│   │   ├── app_config.json          ← Static constants (k limits, thresholds, model names)
│   │   └── config.json              ← Per-env: URLs, ports, timeouts
│   │
│   ├── env/
│   │   ├── development.env
│   │   └── production.env
│   │
│   ├── logs/                        ← Runtime — gitignored
│   │   ├── app.log
│   │   └── error.log
│   │
│   └── src/
│       ├── api/
│       │   ├── health.py            ✅ GET /health · GET /health/ready
│       │   ├── ingestion.py         ✅ POST /ingest · GET /ingest/status
│       │   ├── search.py            🔲 POST /search (Phase 3)
│       │   ├── triage.py            ✅ POST /triage · GET /escalations
│       │   └── evaluation.py        🔲 POST /evaluate · GET /metrics (Phase 5)
│       │
│       ├── core/
│       │   ├── config.py            ✅ load_app_config · load_env_config · require_env · get_env
│       │   └── dependencies.py      ✅ FastAPI Depends() — inject config, clients, services
│       │
│       ├── handlers/
│       │   └── logger.py            ✅ app logger + error logger, RotatingFileHandler
│       │
│       ├── exceptions/
│       │   ├── custom_exceptions.py ✅ Full hierarchy (Ingestion/Retrieval/Agent/LLM/Config/Eval)
│       │   └── exception_handler.py ✅ Central handler → structured JSON error responses
│       │
│       ├── models/
│       │   ├── incident.py          ✅ IncidentCreate · IncidentResponse · Enums
│       │   ├── search.py            ✅ SearchQuery · SearchFilter · SearchResponse · ResolutionOption
│       │   ├── triage.py            ✅ TriageRequest · TriageResult · EscalationTicket
│       │   └── evaluation.py        ✅ EvalRequest · EvalResult · MetricScore
│       │
│       ├── integrations/
│       │   ├── vector_db.py         ✅ VectorStore ABC + QdrantVectorStore implementation
│       │   ├── embeddings.py        ✅ ada-002 batch embed + Redis cache + MiniLM fallback
│       │   ├── llm.py               ✅ OpenAI + pybreaker + tenacity + Flan-T5 fallback
│       │   ├── cache.py             ✅ Redis ConnectionPool + get/set/delete + key builders
│       │   └── database.py          ✅ SQLAlchemy async engine + session factory + create_tables
│       │
│       ├── ingestion/
│       │   ├── preprocessor.py      ✅ XLSX parse · NaN-safe clean_text · field mapping
│       │   ├── bm25_builder.py      ✅ BM25Okapi build + pickle persist/load
│       │   └── pipeline.py          ✅ Orchestrate parse → BM25 → embed batches → Qdrant upsert
│       │
│       ├── retrieval/
│       │   ├── adaptive_k.py        🔲 compute_k() + trim_by_score_dropoff() (Phase 3)
│       │   ├── bm25_retriever.py    🔲 Load index · score · top-k (Phase 3)
│       │   ├── vector_retriever.py  🔲 Qdrant semantic search + filter DSL (Phase 3)
│       │   ├── rrf_merger.py        🔲 RRF fusion: 1/(60+rank) (Phase 3)
│       │   ├── reranker.py          🔲 CrossEncoder ms-marco-MiniLM-L-6-v2 (Phase 3)
│       │   ├── resolution_aggregator.py 🔲 Cosine cluster > 0.95 · occurrence counts (Phase 3)
│       │   └── hybrid_search.py     🔲 Full orchestrator (Phase 3)
│       │
│       ├── agents/
│       │   ├── state.py             ✅ IncidentState TypedDict
│       │   ├── tools.py             ✅ search_incidents · tavily_web_search · classify_priority
│       │   ├── l1_triage.py         ✅ GPT-4o-mini + confidence gate (≥0.80 = resolve)
│       │   ├── l2_analysis.py       ✅ GPT-4o + Tavily synthesis (≥0.55 = resolve)
│       │   ├── l3_specialist.py     ✅ Postgres escalation ticket + EscalationTicketDB ORM
│       │   └── graph.py             ✅ LangGraph StateGraph + build_triage_graph() + run_triage()
│       │
│       └── evaluation/
│           ├── ir_metrics.py        🔲 ndcg_at_k · map_at_k · recall_at_k · precision_at_k (Phase 5)
│           ├── llm_judge.py         🔲 DeepEval: Faithfulness · AnswerRelevancy · ContextualPrecision (Phase 5)
│           ├── runner.py            🔲 Full eval pipeline + Postgres persistence (Phase 5)
│           └── ground_truth/
│               ├── dataset.json     🔲 50 QA pairs stratified by category (Phase 5)
│               └── generate_dataset.py 🔲 (Phase 5)
│
└── frontend/                        ── ALL React/Vite code ──────────────────────────
    ├── index.html
    ├── vite.config.ts               🔲 (Phase 6)
    ├── tsconfig.json                🔲
    ├── tailwind.config.js           🔲
    ├── package.json                 🔲
    ├── Dockerfile                   🔲 (Phase 7)
    └── src/
        ├── main.tsx                 🔲
        ├── App.tsx                  🔲 React Router: / → Search · /triage · /analytics
        ├── api/
        │   ├── searchApi.ts         🔲
        │   ├── triageApi.ts         🔲
        │   └── evaluationApi.ts     🔲
        ├── components/
        │   ├── IncidentCard.tsx     🔲
        │   ├── FilterSidebar.tsx    🔲
        │   ├── ConfidenceBadge.tsx  🔲
        │   ├── ResolutionPanel.tsx  🔲
        │   └── MetricChart.tsx      🔲
        └── pages/
            ├── SearchPage.tsx       🔲
            ├── TriagePage.tsx       🔲
            └── AnalyticsPage.tsx    🔲
```

---

## 4. System Flows

### 4.1 Retrieval Flow (`POST /search`)

```
POST /search { query, filters? }
        │
        ▼
Redis cache check  ──── HIT ────► return cached SearchResponse immediately
        │ MISS
        ▼
AdaptiveKSelector.compute_k(query)
  → k = 3 (specific) / 10 (default) / 20 (vague/rare)
        │
        ▼ (parallel)
┌─────────────────────┐    ┌──────────────────────────────────────────┐
│ BM25Retriever       │    │ VectorRetriever                          │
│ search(query, k)    │    │ search(query_vector, k, metadata_filter) │
│ keyword top-k       │    │ semantic top-k                           │
└─────────────────────┘    └──────────────────────────────────────────┘
        │                           │
        └──────────┬────────────────┘
                   ▼
        RRFMerger.fuse(bm25_results, vector_results)
          score = 1/(60 + rank_bm25) + 1/(60 + rank_vector)
                   │
                   ▼
        AdaptiveKSelector.trim_by_score_dropoff(merged, threshold=0.15)
          → remove tail noise where consecutive score drop > 15%
                   │
                   ▼
        CrossEncoderReranker.rerank(query, trimmed_candidates)
          model: ms-marco-MiniLM-L-6-v2 (loaded at startup)
                   │
                   ▼
        ResolutionAggregator.aggregate(reranked)
          → embed each resolution_notes
          → cluster by cosine > 0.95 (deduplicate near-identical fixes)
          → attach occurrence_count per cluster
          → sort by occurrence_count × avg_rerank_score
                   │
                   ▼
        Write to Redis cache (TTL 1h)
                   │
                   ▼
        Return SearchResponse {
          results: [IncidentResponse × N],
          resolution_options: [unique fix A (×14), fix B (×6), fix C (×2)],
          adaptive_k_used, retrieval_method, cached, latency_ms
        }
```

### 4.2 Agent Triage Flow (`POST /triage`)

```
POST /triage { description, impact, urgency }
        │
        ▼
L1 Agent (GPT-4o-mini)
  → calls search_incidents tool  → full retrieval flow above
  → prompts GPT-4o-mini: "Based on N similar incidents, here are the resolutions..."
  → confidence = weighted avg of reranked scores
  → confidence ≥ 0.80  ─── YES ──► END: return L1 answer (escalation_level = "L1")
  → confidence < 0.80  ─── NO ───► escalate to L2
        │
        ▼
L2 Agent (GPT-4o)
  → calls tavily_web_search tool
  → GPT-4o prompt: [L1 KB context] + [web results] + "use your knowledge"
  → structured output: { root_cause, resolution_steps, confidence, sources }
  → solution found  ─── YES ──► END: return L2 answer (escalation_level = "L2")
  → cannot resolve  ─── NO ───► escalate to L3
        │
        ▼
L3 Agent (no LLM)
  → INSERT into Postgres escalation_tickets:
    { incident_id, description, l1_summary, l2_analysis,
      escalation_reason, status: OPEN, created_at }
  → return { ticket_id, status: ESCALATED, escalation_level: "L3" }
```

### 4.3 Ingestion Flow (`POST /ingest`)

```
POST /ingest (XLSX file upload)
        │
        ▼
preprocessor.py
  → pd.read_excel(file_path, engine="openpyxl")
  → validate required columns: Incident ID · Description · Solution
  → per row: clean_text() (NaN-safe) → build search_text = "{title}: {description}"
  → skip rows with blank Incident ID / Description / Solution
        │
        ▼
bm25_builder.py
  → tokenise search_text (lowercase + strip punctuation)
  → BM25Okapi(tokenized_corpus)
  → pickle to data/bm25_index.pkl
        │
        ▼
pipeline.py (asyncio batches of 50)
  → embed_batch(search_texts)      → ada-002 batch API (cache-first, MiniLM fallback)
  → vector_store.upsert(points)    → Qdrant upsert with full metadata payload
  → track progress in _status dict → available via GET /ingest/status
        │
        ▼
Return { ingested: N, skipped: M, duration_ms: X }
```

---

## 5. Agent State Design

```python
class IncidentState(TypedDict):
    # Input
    query: str
    impact: str
    urgency: str

    # L1 outputs
    retrieved_incidents: list[dict]
    resolution_options: list[dict]     # aggregated multi-resolution
    l1_summary: str
    l1_confidence: float
    escalation_reason: str | None

    # L2 outputs
    web_search_results: list[dict]
    l2_synthesis: str
    l2_confidence: float

    # L3 outputs
    escalation_ticket_id: str | None

    # Final
    final_answer: str
    escalation_level: Literal["L1", "L2", "L3"]
    model_used: str
    fallback_used: bool
```

---

## 6. Adaptive-K Retrieval Logic

```
k_min     = 3   (very specific query — high top-score confidence)
k_default = 10  (typical query)
k_max     = 20  (vague/rare query — low top-score confidence)

compute_k(query):
  complexity = token_count
             + 3 × (tokens containing digits or error codes)
             + 2 × (query length > 10 tokens)
  if complexity < 5  → k = 3
  if complexity < 12 → k = 10
  else               → k = 20

trim_by_score_dropoff(results, threshold=0.15):
  always keep results[0]
  for each subsequent result:
    if (prev_score - curr_score) / prev_score > 0.15 → STOP
  return trimmed list
```

**Impact:** ~40% average latency reduction on simple queries by shrinking the cross-encoder candidate set from 20 to 3.

---

## 7. Multi-Resolution Handling

`resolution_aggregator.py` behaviour:

1. Embed each result's `resolution_notes` using cached ada-002 embeddings.
2. Cluster by cosine similarity > 0.95 → deduplicate near-identical fixes.
3. For each cluster: record `occurrence_count` (how many past incidents used this fix).
4. Sort clusters by `occurrence_count × avg_rerank_score` (most-proven fix first).
5. LLM prompt receives **ALL unique resolutions**:
   > *"Fix A (used 14 times): Increase connection pool size..."*
   > *"Fix B (used 6 times): Add read replica..."*
   > *"Fix C (used 2 times): Check for N+1 queries..."*

Result: LLM synthesises a **ranked multi-option response** — never hides valid alternative solutions.

---

## 8. Production Patterns

| Pattern | Implementation |
|---|---|
| Zero `print()` | `logger.info/warning/error` only — `test_no_print_used` enforces this |
| No hardcoded secrets | All from `env/*.env` via `python-dotenv` + `require_env()` |
| Connection pooling | Redis `ConnectionPool(max=20)` · SQLAlchemy `QueuePool(pool_size=20)` · Qdrant async client |
| Cold-start optimisation | Flan-T5 · MiniLM · cross-encoder · BM25 index all loaded in FastAPI `lifespan` — never on first request |
| Stateless API | Zero instance-local state; all cache in Redis → horizontally scalable |
| Circuit breaker | `pybreaker`: 5 failures in 60 s → open circuit → route to Flan-T5 fallback |
| Retry | `tenacity`: 3 attempts, exponential backoff (2s → 4s → 8s) |
| Graceful degradation | Qdrant down → BM25-only search; OpenAI timeout → Flan-T5; both fail → 503 JSON |
| Input validation | Every endpoint uses a Pydantic V2 request model with `Field` constraints and enum validation |
| Health checks | `GET /health` (liveness) + `GET /health/ready` (checks Qdrant + Redis + Postgres) |
| Swap-able vector DB | `VectorStore` ABC — swap `QdrantVectorStore` → `MilvusVectorStore` without touching any API or business logic |
| Structured errors | Central `exception_handler.py` converts all `IncidentKBException` subclasses to `{error, message, details, status_code}` JSON |

---

## 9. Dataset

| Field | Column in XLSX | Notes |
|---|---|---|
| `incident_id` | `Incident ID` | e.g. `INC-5001` |
| `ticket_id` | `Ticket ID` | e.g. `TKT-1001` |
| `assigned_to` | `Media Asset` | e.g. `MediaServer01` |
| `category` | `Category` | Storage · Application · Database · Hardware · Network · Performance · Security |
| `title` | `Incident Details` | Short headline (e.g. "Disk Space Alert") |
| `description` | `Description` | Full incident description — primary search field |
| `resolution_notes` | `Solution` | How the incident was resolved — what we surface |
| `search_text` | *(derived)* | `"{title}: {description}"` — used for both BM25 tokenisation and ada-002 embedding |

**File:** `data/incidents.xlsx` · **Rows:** 150 · **Zero nulls**  
**Qdrant point IDs:** `int(sha1(incident_id)[:8], 16)` — stable 32-bit integers, reproducible

---

## 10. Phase Breakdown & Status

### ✅ Phase 1 — Foundation (May 21)
> **81 tests passing**

| Component | File | Status |
|---|---|---|
| Entry point | `main.py` | ✅ Done |
| App config | `configuration/app_config.json` | ✅ Done |
| Env config | `configuration/config.json` | ✅ Done |
| Logger | `src/handlers/logger.py` | ✅ Done |
| Custom exceptions | `src/exceptions/custom_exceptions.py` | ✅ Done |
| Exception handler | `src/exceptions/exception_handler.py` | ✅ Done |
| Config loader | `src/core/config.py` | ✅ Done |
| Dependencies | `src/core/dependencies.py` | ✅ Done |
| Pydantic models | `src/models/incident/search/triage/evaluation.py` | ✅ Done |
| Vector DB | `src/integrations/vector_db.py` | ✅ Done |
| Embeddings | `src/integrations/embeddings.py` | ✅ Done |
| LLM | `src/integrations/llm.py` | ✅ Done |
| Redis cache | `src/integrations/cache.py` | ✅ Done |
| Postgres | `src/integrations/database.py` | ✅ Done |
| Health API | `src/api/health.py` | ✅ Done |
| Unit tests | `tests/unit/test_models/logger/config/exceptions.py` | ✅ 81 tests |

---

### ✅ Phase 2 — Ingestion (May 22)
> **126 tests passing (+45 new)**

| Component | File | Status |
|---|---|---|
| XLSX preprocessor | `src/ingestion/preprocessor.py` | ✅ Done |
| BM25 builder | `src/ingestion/bm25_builder.py` | ✅ Done |
| Ingestion pipeline | `src/ingestion/pipeline.py` | ✅ Done |
| Ingestion API | `src/api/ingestion.py` | ✅ Done |
| Unit tests | `tests/unit/test_preprocessor.py` (20 tests) | ✅ Done |
| Unit tests | `tests/unit/test_bm25_builder.py` (17 tests) | ✅ Done |
| Integration tests | `tests/integration/test_ingestion_api.py` (8 tests) | ✅ Done |
| Dataset moved | `data/incidents.xlsx` | ✅ Done |

**Key fix:** `clean_text()` uses `pd.isna()` before `str()` conversion — prevents `"nan"` ghost values from pandas empty cells.

---

### ✅ Phase 3 — Retrieval Intelligence (May 23–24)
> **213 tests passing (+87 new, 1 skipped — real reranker model)**

| Component | File | Status |
|---|---|---|
| Adaptive-K selector | `src/retrieval/adaptive_k.py` | ✅ Done |
| BM25 retriever | `src/retrieval/bm25_retriever.py` | ✅ Done |
| Vector retriever | `src/retrieval/vector_retriever.py` | ✅ Done |
| RRF merger | `src/retrieval/rrf_merger.py` | ✅ Done |
| Cross-encoder reranker | `src/retrieval/reranker.py` | ✅ Done |
| Resolution aggregator | `src/retrieval/resolution_aggregator.py` | ✅ Done |
| Hybrid search orchestrator | `src/retrieval/hybrid_search.py` | ✅ Done |
| Search API | `src/api/search.py` | ✅ Done |
| Unit tests | `tests/unit/test_adaptive_k.py` (18 tests) | ✅ Done |
| Unit tests | `tests/unit/test_rrf_merger.py` (11 tests) | ✅ Done |
| Unit tests | `tests/unit/test_bm25_retriever.py` (13 tests) | ✅ Done |
| Unit tests | `tests/unit/test_reranker.py` (16 tests) | ✅ Done |
| Unit tests | `tests/unit/test_resolution_aggregator.py` (15 tests) | ✅ Done |
| Integration tests | `tests/integration/test_search_api.py` (14 tests) | ✅ Done |

---

### ✅ Phase 4 — Agents / Triage (May 22)

**Target: LangGraph L1→L2→L3 state machine + `/triage` + `/escalations` endpoints**

| Component | File | Status |
|---|---|---|
| Agent state | `src/agents/state.py` | ✅ Done |
| Agent tools | `src/agents/tools.py` | ✅ Done |
| L1 triage node | `src/agents/l1_triage.py` | ✅ Done |
| L2 analysis node | `src/agents/l2_analysis.py` | ✅ Done |
| L3 specialist node | `src/agents/l3_specialist.py` | ✅ Done |
| LangGraph graph | `src/agents/graph.py` | ✅ Done |
| Triage API | `src/api/triage.py` | ✅ Done |
| Integration tests | `tests/integration/test_triage_api.py` | ✅ Done (12 tests) |
| Unit tests (L1/L2/tools) | `tests/unit/test_l1_triage.py` + `test_l2_analysis.py` + `test_tools.py` | ✅ Done (38 tests) |

---

### ✅ Phase 5 — Evaluation (May 22)

**Target: ground truth dataset + IR metrics + DeepEval judge + runner + `/evaluate` endpoint**

| Component | File | Status |
|---|---|---|
| IR metrics | `src/evaluation/ir_metrics.py` | ✅ Done (NDCG, MAP, Recall, Precision — pure Python) |
| LLM judge | `src/evaluation/llm_judge.py` | ✅ Done (DeepEval Faithfulness/Relevancy/Precision) |
| Eval runner | `src/evaluation/runner.py` | ✅ Done + EvalRunDB ORM |
| Ground truth | `src/evaluation/ground_truth/dataset.json` | ✅ Done (30 cases, 7 categories) |
| Ground truth generator | `src/evaluation/ground_truth/generate_dataset.py` | ✅ Done |
| Evaluation API | `src/api/evaluation.py` | ✅ Done (POST /evaluate, GET /metrics) |
| Eval quality tests | `tests/evaluation/test_rag_quality.py` | ✅ Done (14 tests) |
| IR metric unit tests | `tests/unit/test_ir_metrics.py` | ✅ Done (40 tests) |

---

### ✅ Phase 6 — React + Vite Frontend (May 22)

**Target: Vite scaffold + 3 pages + typed API layer + all components**

| Component | Status |
|---|---|
| Vite + TypeScript + TailwindCSS scaffold | ✅ Done — vendor/charts/query code-split, clean build |
| `api/` layer (typed fetch wrappers) | ✅ Done — searchApi, triageApi, evaluationApi |
| `SearchPage` — query + filters + IncidentCard + ResolutionPanel | ✅ Done |
| `TriagePage` — L1/L2/L3 pipeline flow + ConfidenceBadge + escalation panel | ✅ Done |
| `AnalyticsPage` — Recharts bar + radar charts + metric detail rows | ✅ Done |

---

### 🔲 Phase 7 — Docker + Load Tests (May 28–29)

**Target: Dockerfiles + docker-compose + Locust load tests + full smoke test**

| Component | Status |
|---|---|
| `backend/Dockerfile` | 🔲 TODO |
| `frontend/Dockerfile` | 🔲 TODO |
| `docker-compose.yml` finalisation (health checks, depends_on) | 🔲 TODO |
| `tests/load/locustfile.py` (70% /search · 20% /triage · 10% /health) | 🔲 TODO |
| Full docker-compose smoke test | 🔲 TODO |

---

## 11. Test Inventory

| Test File | Count | Phase | Status |
|---|---|---|---|
| `tests/unit/test_models.py` | 34 | 1 | ✅ |
| `tests/unit/test_logger.py` | 9 | 1 | ✅ |
| `tests/unit/test_exceptions.py` | 22 | 1 | ✅ |
| `tests/unit/test_config.py` | 16 | 1 | ✅ |
| `tests/unit/test_preprocessor.py` | 20 | 2 | ✅ |
| `tests/unit/test_bm25_builder.py` | 17 | 2 | ✅ |
| `tests/integration/test_ingestion_api.py` | 8 | 2 | ✅ |
| `tests/unit/test_adaptive_k.py` | — | 3 | 🔲 |
| `tests/unit/test_rrf_merger.py` | — | 3 | 🔲 |
| `tests/unit/test_bm25_retriever.py` | — | 3 | 🔲 |
| `tests/unit/test_reranker.py` | — | 3 | 🔲 |
| `tests/unit/test_resolution_aggregator.py` | — | 3 | 🔲 |
| `tests/integration/test_search_api.py` | — | 3 | 🔲 |
| `tests/integration/test_triage_api.py` | — | 4 | 🔲 |
| `tests/unit/test_ir_metrics.py` | — | 5 | 🔲 |
| `tests/evaluation/test_rag_quality.py` | — | 5 | 🔲 |
| `tests/load/locustfile.py` | — | 7 | 🔲 |
| **Total so far** | **126** | | **126 passing** |

---

## 12. ADRs (Post May 29)

| # | Decision | Trade-off documented |
|---|---|---|
| ADR-001 | Qdrant as vector DB | vs FAISS / ChromaDB / Milvus; swap-ability via `VectorStore` ABC |
| ADR-002 | Hybrid BM25 + Vector + RRF + Adaptive-K + Cross-encoder | vs single-method; full latency vs accuracy analysis |
| ADR-003 | LangGraph for L1→L2→L3 | vs CrewAI / custom; explicit typed state vs flexibility |
| ADR-004 | ada-002 + local MiniLM fallback | vs sentence-transformers only; API cost vs quality |
| ADR-005 | GPT-4o-mini (L1) + GPT-4o (L2) + Flan-T5 (fallback) | SLM-first latency strategy; cost tiering |

---

## 13. Final Verification Checklist

```
[ ] docker-compose up --build  → all 5 containers healthy, zero errors
[ ] POST /ingest (data/incidents.xlsx) → { ingested: 150, skipped: 0, duration_ms: X }
[ ] POST /search { "query": "storage disk space upload failure", "filters": { "category": "Storage" } }
      → ranked results + multi-resolution_options + adaptive_k_used
[ ] POST /triage { "description": "MediaServer keeps crashing with high CPU under peak load", "impact": "High" }
      → { escalation_level, confidence, final_answer, model_used, fallback_used }
[ ] POST /evaluate → { ndcg_at_10: ≥ 0.80, faithfulness: ≥ 0.70, relevancy: ≥ 0.75 }
[ ] pytest tests/ -v --cov=src  → 126+ pass, coverage ≥ 75%
[ ] locust --headless --users 50 --run-time 60s → p99 < 500ms, error rate < 1%
[ ] React app http://localhost:5173 → Search · Triage · Analytics all functional
[ ] grep -r "print(" src/ → zero matches (only logger.py safety-net prints allowed)
[ ] grep -rE "(sk-|password\s*=)" src/ → zero matches
[ ] GET /health/ready → { qdrant: ok, redis: ok, postgres: ok }
```

---

> **Current bookmark: End of Phase 3 ✅**  
> **Next step: Phase 7 — Docker + Locust (Dockerfiles · docker-compose · load tests)**  
> Start with `src/agents/state.py` → `tools.py` → `l1_triage.py` → `l2_analysis.py` → `l3_specialist.py` → `graph.py` → `src/api/triage.py`
