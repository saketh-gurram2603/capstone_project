# AI-Powered Incident Knowledge Base Assistant

A production-grade RAG system that enables support engineers to resolve production incidents faster by surfacing all relevant historical resolution strategies — not just the top-1 fix — through a hybrid retrieval pipeline and a three-tier autonomous triage agent.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Data Flows](#3-data-flows)
4. [Technology Stack](#4-technology-stack)
5. [Folder Structure](#5-folder-structure)
6. [Prerequisites](#6-prerequisites)
7. [Local Development Setup](#7-local-development-setup)
8. [Docker Setup (One-Command)](#8-docker-setup-one-command)
9. [Configuration Reference](#9-configuration-reference)
10. [API Reference](#10-api-reference)
11. [Testing Guide](#11-testing-guide)
12. [Evaluation Guide](#12-evaluation-guide)
13. [Production Deployment (K8s)](#13-production-deployment-k8s)
14. [Observability & MLOps](#14-observability--mlops)
15. [Design Decisions](#15-design-decisions)
16. [Performance Benchmarks](#16-performance-benchmarks)
17. [Reliability & Resiliency Patterns](#17-reliability--resiliency-patterns)
18. [Chat Assistant Feature](#18-chat-assistant-feature)
19. [PII Masking](#19-pii-masking)

---

## 1. System Overview

Support engineers describe a production problem in natural language. The system:

1. **Retrieves** the most relevant historical incidents using hybrid BM25 + vector search, RRF score fusion, adaptive-K candidate selection, and a cross-encoder reranker.
2. **Surfaces all unique resolution approaches** found across similar past incidents, ranked by occurrence count and reranker confidence — not just the single top result.
3. **Triages** autonomously through a three-tier LangGraph agent (L1 → L2 → L3), escalating only when confidence is genuinely insufficient.
4. **Guides** engineers interactively through a chat interface — presenting Fix 1 as numbered steps, then Fix 2, Fix 3, etc. on demand, and auto-escalating to L3 if all options are exhausted.
5. **Masks PII** at ingestion time — emails, IPs, phone numbers, SSNs, and credit cards are redacted before any data reaches the vector DB, LLM, or escalation tickets.
6. **Evaluates** its own output quality using DeepEval LLM-as-Judge metrics and standard IR metrics (NDCG, MAP, Recall).

### POC vs. Production Scope

| Concern | This Implementation (POC) | Production Target |
|---|---|---|
| Deployment | Docker Compose (single host) | Kubernetes (multi-node, auto-scaling) |
| Ingress | Direct FastAPI port | API Gateway + Load Balancer |
| Secrets | `.env` files | Vault / K8s Secrets |
| Observability | Rotating file logs + health endpoints | Prometheus + Grafana + OpenTelemetry |
| Vector DB | Single Qdrant node | Qdrant cluster with replication |
| LLM | OpenAI API (single key) | LLM Gateway with rate-limit pooling |

---

## 2. Architecture

![High-Level Architecture Diagram](<images/ChatGPT Image Jun 1, 2026, 12_36_54 PM.png>)

### Component Responsibilities

| Component | Responsibility | Decoupling Boundary |
|---|---|---|
| `src/api/` | HTTP routing, request validation, response serialisation | No business logic — delegates to services |
| `src/retrieval/` | BM25, vector, RRF, reranker, aggregator | Swappable via `VectorStore` ABC |
| `src/agents/` | LangGraph L1→L2→L3 state machine | No direct DB access — uses tool functions |
| `src/integrations/` | Qdrant, Postgres, OpenAI, embeddings | Each wrapped in a class; injected via `dependencies.py` |
| `src/evaluation/` | IR metrics + LLM-as-Judge + Postgres persistence | Fully isolated; callable via API or CLI |
| `src/ingestion/` | XLSX parse, BM25 build, embed batch, Qdrant upsert | Async generator batches, never loads full dataset in memory |

### Swapping the Vector DB

`src/integrations/vector_db.py` exposes a `VectorStore` ABC with `upsert`, `search`, `delete`, and `collection_info` methods. The live implementation is `QdrantVectorStore`. To swap to Milvus or FAISS:

1. Implement `MilvusVectorStore(VectorStore)` in the same file.
2. Update `dependencies.py` to inject the new class.
3. Zero changes required in any API, retrieval, or agent module.

---

## 3. Data Flows

### 3.1 Ingestion Flow (`POST /ingest`)

```
Client uploads XLSX
        │
        ▼
preprocessor.py
  → pd.read_excel (engine=openpyxl)
  → validate required columns: Incident ID · Description · Solution
  → clean_text() per row  (pd.isna() guard — prevents "nan" ghost values)
  → pii_masker.py: mask_pii() on description, resolution_notes, title
      EMAIL → [EMAIL]  ·  IP → [IP_ADDRESS]  ·  SSN → [SSN]
      PHONE → [PHONE]  ·  Credit card → [CREDIT_CARD]
      (audit log: counts only — no raw PII written to logs)
  → derive search_text = "{title}: {description}"   (from masked text)
  → skip rows with blank ID / Description / Solution
        │
        ▼
bm25_builder.py
  → tokenise search_text (lowercase + strip punctuation)
  → BM25Okapi(tokenized_corpus)
  → pickle → data/bm25_index.pkl
        │
        ▼
pipeline.py  (asyncio, batches of 50)
  → embed_batch(search_texts)         → Ada-002 batch API (MiniLM local fallback)
  → vector_store.upsert(points)       → Qdrant upsert with full metadata payload
  → progress tracked in _status dict  → available via GET /ingest/status
        │
        ▼
Return { ingested: N, skipped: M, duration_ms: X, pii_masked_total: K }
```

### 3.2 Retrieval Flow (`POST /search`)

```
POST /search { query, filters?, mode? }
        │
        ▼
adaptive_k.py → compute_k(query)
  complexity = token_count + 3×(error-code tokens) + 2×(query > 10 tokens)
  complexity < 5  → k = 3   (fast path, ~40% latency saving)
  complexity < 12 → k = 10
  else            → k = 20
        │
        ▼  mode="hybrid": both branches in parallel via asyncio.gather
           mode="semantic": BM25 branch skipped entirely
┌─────────────────────┐    ┌────────────────────────────────────────┐
│ bm25_retriever.py   │    │ vector_retriever.py                    │
│ BM25Okapi.get_top_n │    │ Qdrant search + metadata filter DSL    │
│ → top-k incidents   │    │ → top-k semantic matches               │
└─────────────────────┘    └────────────────────────────────────────┘
        │                            │
        └──────────┬─────────────────┘
                   ▼
        rrf_merger.py
          score = 1/(60 + rank_bm25) + 1/(60 + rank_vector)
                   │
                   ▼
        adaptive_k.py → trim_by_score_dropoff(merged, threshold=0.15)
          removes tail results where consecutive score drop > 15%
                   │
                   ▼
        reranker.py
          CrossEncoder(ms-marco-MiniLM-L-6-v2) — loaded once at startup
                   │
                   ▼
        resolution_aggregator.py
          → embed each resolution_notes (local MiniLM — no OpenAI call)
          → cosine cluster > 0.95  (deduplicate near-identical fixes)
          → occurrence_count per cluster
          → sort by occurrence_count × avg_rerank_score
                   │
                   ▼
        SearchResponse {
          results: [IncidentResponse × N],
          resolution_options: [Fix A (×14), Fix B (×6), Fix C (×2)],
          adaptive_k_used, retrieval_method, cached: false, latency_ms
        }
```

### 3.3 Triage Flow (`POST /triage`)

```
POST /triage { description, impact, urgency }
        │
        ▼
L1 Agent (gpt-4o-mini)                     ← cheap + fast
  → calls search_incidents tool             → full retrieval flow above
  → prompt: "Based on N similar incidents, here are the resolutions..."
  → confidence = weighted avg of reranked scores
  confidence ≥ 0.80  ── YES ──► END  (escalation_level = "L1")
  confidence < 0.80  ── NO  ──► escalate
        │
        ▼
L2 Agent (gpt-4o)                          ← deeper, web-augmented
  → calls tavily_web_search tool
  → prompt: [L1 KB context] + [web results] + model knowledge
  → structured output: { root_cause, resolution_steps, confidence, sources }
  solution found  ── YES ──► END  (escalation_level = "L2")
  cannot resolve  ── NO  ──► escalate
        │
        ▼
L3 Agent (no LLM — pure routing)
  → INSERT escalation_tickets (Postgres):
    { incident_id, description, l1_summary, l2_analysis,
      escalation_reason, status: OPEN, created_at }
  → return { ticket_id, status: ESCALATED, escalation_level: "L3" }
```

---

## 4. Technology Stack

| Layer | Choice | Justification |
|---|---|---|
| API Framework | FastAPI + Pydantic V2 | Async ASGI, automatic OpenAPI docs, strict input validation |
| Vector DB | Qdrant | Filterable HNSW, metadata DSL, async client, swappable via ABC |
| Keyword Search | BM25 (`rank_bm25`) | Exact keyword recall, zero network dependency, graceful fallback |
| Retrieval Count | Adaptive-K (3–20) | Cuts ~40% latency on simple queries by shrinking the reranker input |
| Score Fusion | RRF (k = 60) | Rank-only fusion — no score normalisation, proven robust across modalities |
| Reranker | `ms-marco-MiniLM-L-6-v2` cross-encoder | Stage-2 accuracy boost on trimmed candidate set; loaded once |
| Resolution Strategy | Aggregator + cosine clustering | Surfaces ALL unique fixes (> 0.95 cosine) with occurrence counts |
| Embeddings | OpenAI `text-embedding-ada-002` | 1536-dim cosine space; project requirement |
| Embedding Fallback | `all-MiniLM-L6-v2` (local) | Loaded at startup; activates when Ada-002 API fails |
| L1 Agent | GPT-4o-mini | Lowest latency; KB search + summarise + confidence gate (≥ 0.80) |
| L2 Agent | GPT-4o + Tavily API | Web search + L1 context; activates only when L1 confidence < 0.80 |
| L3 Agent | No LLM | Pure Postgres INSERT; returns `ticket_id`; no LLM cost |
| LLM Fallback | `google/flan-t5-base` | Loaded at startup via HuggingFace; activates when circuit breaker opens |
| Circuit Breaker | `pybreaker` | fail_max = 5, reset_timeout = 60s; protects all OpenAI calls |
| Retry | `tenacity` | 3 attempts, exponential backoff (2s → 4s → 8s) |
| Cache | None (current build) | Caching layer is stubbed; all embedding and query requests go to the live service. Designed for Redis drop-in via `cache.py` stub interface. |
| Metadata DB | SQLite + SQLAlchemy async | Default: file-backed `data/incident_kb.db` via `aiosqlite` — zero install. Postgres supported as optional override (set `POSTGRES_USER` + `POSTGRES_PASSWORD`). |
| Agent Orchestration | LangGraph `StateGraph` | Explicit typed conditional edges; `IncidentState` TypedDict |
| Logging | Python `logging` + `RotatingFileHandler` | app.log (INFO+) + error.log (ERROR+); 5 MB / 5 backups |
| Evaluation | DeepEval + custom IR metrics | Faithfulness · AnswerRelevancy · ContextualPrecision + NDCG@10, MAP@10, Recall@10 |
| Load Testing | Locust | 50 concurrent users; p99 < 500 ms target |
| Frontend | React 18 + Vite + TypeScript + TailwindCSS + TanStack Query | Fast build, typed, code-split chunks |
| Containerisation | Docker + docker-compose | One-command startup; all services health-checked |

---

## 5. Folder Structure

```
capstone_project/
│
├── README.md                            ← This file
├── PLAN.md                              ← Master build plan + phase tracker
├── docker-compose.yml                   ← All 5 services + health checks
├── .env.example                         ← Required environment variable keys
├── .gitignore
│
├── data/
│   └── incidents.xlsx                   ← 150-row ITSM dataset (MediaServer incidents)
│
├── requirements/
│   └── project-requirements.md         ← Original project specification
│
├── backend/
│   ├── main.py                          ← FastAPI entry point; lifespan startup + router registration
│   ├── requirements.txt                 ← 61 pinned dependencies
│   ├── Dockerfile                       ← 2-stage build (deps + app)
│   │
│   ├── configuration/
│   │   ├── app_config.json             ← Static constants (k limits, thresholds, model names)
│   │   └── config.json                 ← Per-env overrides (URLs, ports, debug flags)
│   │
│   ├── env/
│   │   ├── development.env             ← Local secrets (gitignored)
│   │   └── production.env              ← Production secrets (gitignored)
│   │
│   ├── logs/                            ← Runtime logs (gitignored)
│   │   ├── app.log                     ← INFO+ structured log
│   │   └── error.log                   ← ERROR+ only
│   │
│   └── src/
│       ├── api/                         ← HTTP layer only — no business logic
│       │   ├── health.py               ← GET /health · GET /health/ready
│       │   ├── ingestion.py            ← POST /ingest · GET /ingest/status
│       │   ├── search.py               ← POST /search
│       │   ├── triage.py               ← POST /triage · GET /escalations
│       │   ├── evaluation.py           ← POST /evaluate · GET /metrics
│       │   └── chat.py                 ← POST /chat (guided troubleshooting)
│       │
│       ├── core/
│       │   ├── config.py               ← load_app_config · load_env_config · require_env
│       │   └── dependencies.py         ← FastAPI Depends() — inject config, clients, services
│       │
│       ├── handlers/
│       │   └── logger.py               ← RotatingFileHandler; app.log + error.log
│       │
│       ├── exceptions/
│       │   ├── custom_exceptions.py    ← Full hierarchy (Ingestion/Retrieval/Agent/LLM/Config/Eval)
│       │   └── exception_handler.py    ← Central → structured JSON error responses
│       │
│       ├── models/
│       │   ├── incident.py             ← IncidentCreate · IncidentResponse · Enums
│       │   ├── search.py               ← SearchQuery (mode field) · SearchFilter · SearchResponse
│       │   ├── triage.py               ← TriageRequest · TriageResult · EscalationTicket
│       │   ├── evaluation.py           ← EvalRequest · EvalResult · MetricScore
│       │   └── db_models.py            ← EscalationTicketDB · EvalRunDB (SQLAlchemy ORM)
│       │
│       ├── integrations/
│       │   ├── vector_db.py            ← VectorStore ABC + QdrantVectorStore
│       │   ├── embeddings.py           ← Ada-002 + MiniLM fallback
│       │   ├── llm.py                  ← OpenAI + pybreaker + tenacity + Flan-T5 fallback
│       │   ├── cache.py                ← Stub interface (no-ops); placeholder for Redis
│       │   └── database.py             ← SQLAlchemy async engine + session factory + create_tables()
│       │
│       ├── ingestion/
│       │   ├── preprocessor.py         ← XLSX/XLS parse · NaN-safe clean_text · PII masking · field mapping
│       │   ├── pii_masker.py           ← Regex-based PII redaction (EMAIL/IP/PHONE/SSN/CREDIT_CARD)
│       │   ├── bm25_builder.py         ← BM25Okapi build + pickle persist/load
│       │   └── pipeline.py             ← Async batch orchestration (50 rows/batch)
│       │
│       ├── retrieval/
│       │   ├── adaptive_k.py           ← compute_k() + trim_by_score_dropoff()
│       │   ├── bm25_retriever.py       ← Load index · score · top-k
│       │   ├── vector_retriever.py     ← Qdrant semantic search + filter DSL
│       │   ├── rrf_merger.py           ← RRF fusion: 1/(60 + rank)
│       │   ├── reranker.py             ← CrossEncoder ms-marco-MiniLM-L-6-v2
│       │   ├── resolution_aggregator.py← Cosine cluster > 0.95 + occurrence counts
│       │   └── hybrid_search.py        ← Full retrieval orchestrator (mode: hybrid | semantic)
│       │
│       ├── agents/
│       │   ├── state.py                ← IncidentState TypedDict
│       │   ├── tools.py                ← search_incidents · tavily_web_search · classify_priority
│       │   ├── l1_triage.py            ← GPT-4o-mini + confidence gate (≥ 0.80)
│       │   ├── l2_analysis.py          ← GPT-4o + Tavily synthesis (≥ 0.55)
│       │   ├── l3_specialist.py        ← Postgres escalation ticket + create_escalation_ticket() helper
│       │   └── graph.py                ← LangGraph StateGraph + build_triage_graph()
│       │
│       ├── chat/
│       │   ├── __init__.py
│       │   ├── session_manager.py      ← In-memory ChatSession store · 30-min TTL · background cleanup
│       │   └── chat_agent.py           ← ChatAgent: new session · next fix · question · escalation
│       │
│       ├── models/
│       │   ├── ...
│       │   └── chat.py                 ← ChatRequest · ChatResponse · OptionProgress · ConversationMessage
│       │
│       └── evaluation/
│           ├── ir_metrics.py           ← ndcg_at_k · map_at_k · recall_at_k · precision_at_k
│           ├── llm_judge.py            ← DeepEval: Faithfulness · AnswerRelevancy · ContextualPrecision
│           ├── runner.py               ← Full eval pipeline + Postgres persistence
│           └── ground_truth/
│               ├── dataset.json        ← 30 QA pairs across 7 incident categories
│               └── generate_dataset.py ← Dataset generation script
│
├── frontend/
│   ├── index.html
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── package.json
│   ├── Dockerfile
│   └── src/
│       ├── main.tsx
│       ├── App.tsx                     ← React Router: / · /triage · /analytics · /ingest · /chat
│       ├── api/
│       │   ├── searchApi.ts
│       │   ├── triageApi.ts
│       │   ├── ingestionApi.ts
│       │   ├── evaluationApi.ts
│       │   └── chatApi.ts              ← sendChatMessage() — POST /it-kb/chat
│       ├── store/
│       │   ├── uiStore.ts              ← Sidebar collapse state (Zustand)
│       │   └── chatStore.ts            ← Chat session + message history (Zustand)
│       ├── components/
│       │   ├── IncidentCard.tsx
│       │   ├── ConfidenceBadge.tsx
│       │   ├── ResolutionPanel.tsx
│       │   ├── MetricChart.tsx
│       │   └── chat/
│       │       └── MessageBubble.tsx   ← Chat bubble with markdown, option pill, action buttons
│       └── pages/
│           ├── SearchPage.tsx
│           ├── TriagePage.tsx
│           ├── AnalyticsPage.tsx
│           ├── IngestionPage.tsx
│           └── ChatPage.tsx            ← Guided troubleshooting chat interface
│
└── tests/
    ├── conftest.py
    ├── unit/                            ← Unit tests (retrieval, agents, models, config)
    ├── integration/                     ← API endpoint tests (mocked externals)
    └── evaluation/                      ← RAG quality threshold tests
```

---

## 6. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Required for `asyncio.TaskGroup` and `tomllib` |
| Node.js | 18+ | Frontend build |
| Docker | 24+ | For one-command startup |
| Docker Compose | v2.x | `docker compose` (not `docker-compose`) |
| OpenAI API Key | — | For GPT-4o-mini (L1), GPT-4o (L2), Ada-002 embeddings |
| Tavily API Key | — | For L2 web search; optional (L2 degrades gracefully without it) |

---

## 7. Local Development Setup

### 7.1 Start Infrastructure Services

```bash
# Start Qdrant and Postgres only (Redis not required)
docker compose up qdrant postgres -d
```

### 7.2 Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure secrets
cp ../env.example env/development.env
# Edit env/development.env and fill in all required keys (see Section 9)

# Start the API server
python main.py development
# Server available at: http://localhost:8000
# OpenAPI docs at:    http://localhost:8000/docs
```

### 7.3 Ingest the Dataset

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@../data/incidents.xlsx"

# Poll progress
curl http://localhost:8000/ingest/status
```

### 7.4 Frontend

```bash
cd frontend
npm install
npm run dev
# App available at: http://localhost:5173
```

### 7.5 Verify the Stack

```bash
# Liveness
curl http://localhost:8000/health

# Readiness (checks Qdrant + Postgres)
curl http://localhost:8000/health/ready
# Expected: { "status": "ready", "checks": { "qdrant": "ok", "postgres": "ok" } }
# Note: if POSTGRES_USER/POSTGRES_PASSWORD are unset, postgres will show "fail"
#       but the service is still operational (tickets stored in memory).
```

---

## 8. Docker Setup (One-Command)

```bash
# Copy and fill in the environment file
cp .env.example backend/env/production.env
# Edit backend/env/production.env

# Build and start all services
docker compose up --build

# Services started:
#   Qdrant    → http://localhost:6333
#   Postgres  → localhost:5432
#   Backend   → http://localhost:8000
#   Frontend  → http://localhost:5173
```

All services include Docker health checks. The backend container waits for Qdrant and Postgres to report healthy before accepting traffic.

### Smoke Test (after Docker startup)

```bash
# 1. Ingest dataset
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/incidents.xlsx"
# Expected: { "ingested": 150, "skipped": 0 }

# 2. Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "storage disk space upload failure", "filters": {"category": "Storage"}}'
# Expected: ranked results + resolution_options + adaptive_k_used

# 3. Triage
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"description": "MediaServer crashing with high CPU under peak load", "impact": "High"}'
# Expected: { "escalation_level": "L1"|"L2"|"L3", "confidence": ..., "final_answer": ... }

# 4. Evaluate
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"run_ir_metrics": true, "run_llm_judge": true}'
# Expected: { "ndcg_at_10": >= 0.80, "faithfulness": >= 0.70 }

# 5. Health
curl http://localhost:8000/health/ready
```

---

## 9. Configuration Reference

### Environment Variables (`env/development.env`)

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key (GPT-4o, GPT-4o-mini, Ada-002) |
| `TAVILY_API_KEY` | No | Tavily web search (L2 agent; degrades gracefully if absent) |
| `POSTGRES_USER` | No | Postgres username (optional — if absent, SQLite is used) |
| `POSTGRES_PASSWORD` | No | Postgres password (optional — if absent, SQLite is used) |
| `QDRANT_API_KEY` | No | Qdrant API key (not required for local dev) |
| `SECRET_KEY` | Yes | Application secret key |

### Application Constants (`configuration/app_config.json`)

| Key | Default | Description |
|---|---|---|
| `retrieval.k_min` | 3 | Minimum candidates for simple queries |
| `retrieval.k_default` | 10 | Default candidate count |
| `retrieval.k_max` | 20 | Maximum candidates for vague queries |
| `retrieval.l1_confidence_threshold` | 0.80 | L1 auto-resolve gate |
| `retrieval.l2_confidence_threshold` | 0.55 | L2 auto-resolve gate |
| `retrieval.rrf_k` | 60 | RRF rank constant |
| `cache.embedding_ttl` | 86400 | Embedding cache TTL (seconds) |
| `cache.query_result_ttl` | 3600 | Search result cache TTL (seconds) |
| `circuit_breaker.fail_max` | 5 | Failures before circuit opens |
| `circuit_breaker.reset_timeout` | 60 | Seconds before circuit half-opens |

---

## 10. API Reference

### Health

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — always returns 200 if the process is alive |
| `GET` | `/health/ready` | Readiness probe — checks Qdrant (mandatory) + Postgres (optional); returns 503 only if Qdrant is down |

### Ingestion

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Upload an XLSX file and trigger the async ingestion pipeline |
| `GET` | `/ingest/status` | Poll the status of the current or last ingestion job |

**POST /ingest — request:**
```
Content-Type: multipart/form-data
file: <XLSX binary>
```

**POST /ingest — response:**
```json
{
  "status": "completed",
  "ingested": 150,
  "skipped": 0,
  "duration_ms": 4231,
  "pii_masked_total": 12
}
```
`pii_masked_total` is the count of PII tokens redacted across all ingested records (emails, IPs, phones, SSNs, credit cards).

### Search

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/search` | Hybrid semantic + keyword search with multi-resolution aggregation |

**Request:**
```json
{
  "query": "disk space threshold exceeded on media server",
  "filters": {
    "category": "Storage",
    "assigned_to": "MediaServer01"
  },
  "top_k": 10,
  "mode": "hybrid"
}
```
`mode` is optional. Valid values: `"hybrid"` (default — BM25 + vector + RRF) or `"semantic"` (vector-only, BM25 skipped).

**Response:**
```json
{
  "results": [ { "incident_id": "INC-5032", "title": "...", "rerank_score": 0.91 } ],
  "resolution_options": [
    { "resolution": "Increase disk quota to 2TB", "occurrence_count": 14, "score": 0.91 },
    { "resolution": "Enable log rotation policy", "occurrence_count": 6, "score": 0.73 }
  ],
  "adaptive_k_used": 10,
  "retrieval_method": "hybrid",
  "cached": false,
  "latency_ms": 187
}
```

### Triage

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/triage` | Run the L1 → L2 → L3 autonomous triage agent |
| `GET` | `/escalations` | List L3 Postgres escalation tickets |

**POST /triage — request:**
```json
{
  "description": "MediaServer keeps crashing with high CPU utilization under peak load",
  "impact": "High",
  "urgency": "High"
}
```

**POST /triage — response:**
```json
{
  "escalation_level": "L1",
  "priority": "P1",
  "confidence": 0.87,
  "final_answer": "Based on 14 similar incidents, the primary fix is ...",
  "l1_summary": "High CPU due to unbounded thread pool during upload batches",
  "l2_synthesis": null,
  "escalation_reason": null,
  "escalation_ticket_id": null,
  "model_used": "gpt-4o-mini",
  "fallback_used": false,
  "latency_ms": 1243
}
```

**GET /escalations — query params:**
- `status`: `OPEN` | `IN_PROGRESS` | `RESOLVED` (default: all)
- `limit`: 1–200 (default: 50)
- `offset`: pagination offset (default: 0)

### Chat

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | One turn of guided troubleshooting — new session or continuation |

**POST /chat — start new session:**
```json
{ "session_id": null, "message": "VPN keeps disconnecting after the update" }
```

**POST /chat — continue session:**
```json
{ "session_id": "abc-123", "message": "This didn't work, try next fix" }
```

**Response:**
```json
{
  "session_id": "abc-123",
  "message": "**Fix 1 of 3** _(verified 14× in KB)_\n\n1. Open Network Settings...",
  "option_progress": { "current": 1, "total": 3 },
  "is_escalated": false,
  "escalation_ticket_id": null,
  "all_options_exhausted": false,
  "suggested_actions": ["This didn't work, try next fix", "Issue resolved"]
}
```

**Intent detection (rule-based):**
- `NEXT_OPTION` — "didn't work", "failed", "try next", "still broken", etc.
- `RESOLVED` — "worked", "fixed", "resolved", "thanks", etc.
- `QUESTION` — message > 40 chars with no intent keyword → answered in context without advancing the fix index

Sessions expire after 30 minutes of inactivity. A `404` is returned for expired or unknown session IDs.

### Evaluation

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/evaluate` | Run the full evaluation pipeline (IR metrics + LLM-as-Judge) |
| `GET` | `/metrics` | Return the most recent evaluation run results |

**POST /evaluate — request:**
```json
{
  "run_ir_metrics": true,
  "run_llm_judge": true,
  "dataset_path": null
}
```

**POST /evaluate — response:**
```json
{
  "run_id": "eval_20260523_143022",
  "metrics": [
    { "name": "ndcg_at_10",    "value": 0.83, "passed": true, "threshold": 0.80 },
    { "name": "map_at_10",     "value": 0.79, "passed": true, "threshold": 0.75 },
    { "name": "recall_at_10",  "value": 0.91, "passed": true, "threshold": 0.85 },
    { "name": "faithfulness",  "value": 0.76, "passed": true, "threshold": 0.70 },
    { "name": "answer_relevancy", "value": 0.82, "passed": true, "threshold": 0.75 }
  ],
  "overall_passed": true,
  "num_test_cases": 30,
  "latency_ms": 8417,
  "timestamp": "2026-05-23T14:30:22Z"
}
```

---

## 11. Testing Guide

### Test Suite Overview

| Suite | Location | Count | Covers |
|---|---|---|---|
| Unit | `tests/unit/` | 213 tests | All retrieval components, agents, models, config, logger, exceptions |
| Integration | `tests/integration/` | 34 tests | All API endpoints with mocked external services |
| Evaluation | `tests/evaluation/` | 14 tests | RAG quality thresholds (NDCG, faithfulness, relevancy) |

### Running Tests

```bash
cd backend

# All unit + integration tests with coverage
pytest tests/unit tests/integration -v --cov=src --cov-report=term-missing
# Target: >= 75% coverage

# Evaluation tests (requires running backend + OpenAI key)
pytest tests/evaluation -v
```

### Unit Test Files

| File | Tests | What Is Tested |
|---|---|---|
| `test_models.py` | 34 | Pydantic model validation and field constraints |
| `test_exceptions.py` | 22 | Exception hierarchy and serialisation |
| `test_config.py` | 16 | Config loading, `require_env` failure modes |
| `test_logger.py` | 9 | Logger setup, handler registration, rotation |
| `test_preprocessor.py` | 20 | XLSX parsing, NaN guards, clean_text edge cases |
| `test_bm25_builder.py` | 17 | Tokenisation, index build, pickle round-trip |
| `test_adaptive_k.py` | 18 | `compute_k()` complexity cases, `trim_by_score_dropoff()` |
| `test_rrf_merger.py` | 11 | RRF score calculation, rank ordering |
| `test_bm25_retriever.py` | 13 | Retrieval correctness, top-k boundaries |
| `test_reranker.py` | 16 | Score ordering, model mock, skip guard |
| `test_resolution_aggregator.py` | 15 | Cosine clustering, deduplication, sort order |
| `test_ir_metrics.py` | 40 | NDCG, MAP, Recall, Precision — manual expected values |
| `test_l1_triage.py` | — | Confidence gate, prompt construction, fallback trigger |
| `test_l2_analysis.py` | — | Web search integration, synthesis logic |
| `test_tools.py` | — | Tool function signatures and error handling |

### Enforced Code Quality

```bash
# Zero print() in production code (enforced by test_no_print_used)
grep -r "print(" backend/src/
# Must return zero matches

# No hardcoded secrets
grep -rE "(sk-|password\s*=\s*['\"])" backend/src/
# Must return zero matches
```

---

## 12. Evaluation Guide

### What Is Evaluated

| Metric | Type | Threshold | Description |
|---|---|---|---|
| `ndcg_at_10` | IR | ≥ 0.80 | Normalized Discounted Cumulative Gain at rank 10 |
| `map_at_10` | IR | ≥ 0.75 | Mean Average Precision at rank 10 |
| `recall_at_10` | IR | ≥ 0.85 | Proportion of relevant incidents found in top 10 |
| `precision_at_10` | IR | — | Precision at rank 10 |
| `faithfulness` | LLM-as-Judge | ≥ 0.70 | Final answer factually grounded in retrieved context |
| `answer_relevancy` | LLM-as-Judge | ≥ 0.75 | Answer addresses the query without hallucinating |
| `contextual_precision` | LLM-as-Judge | ≥ 0.70 | Retrieved context contains the relevant information |

### Ground Truth Dataset

- **Location:** `backend/src/evaluation/ground_truth/dataset.json`
- **Size:** 30 QA pairs
- **Categories covered:** Storage, Application, Database, Hardware, Network, Performance, Security
- **Format:**
  ```json
  {
    "query": "disk space threshold exceeded on MediaServer01",
    "relevant_incident_ids": ["INC-5001", "INC-5014", "INC-5089"],
    "expected_resolution_keywords": ["disk quota", "log rotation", "archive policy"],
    "category": "Storage"
  }
  ```

### Running a Full Evaluation

```bash
# Via API
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"run_ir_metrics": true, "run_llm_judge": true}'

# Or directly (from backend/)
python -m src.evaluation.runner

# Retrieve last run results
curl http://localhost:8000/metrics
```

Results are persisted to the `eval_runs` Postgres table and returned as structured JSON.

---

## 13. Production Deployment (K8s)

The system is designed to be stateless and horizontally scalable. The following describes the target production topology.

### Infrastructure Layout

```
Internet
    │
    ▼
[CDN / WAF]
    │
    ▼
[API Gateway]  ← TLS termination, rate limiting, auth token validation
    │
    ▼
[Load Balancer]  ← L7, health-check-aware, sticky-session NOT required (stateless API)
    │         │
┌───▼──┐  ┌───▼──┐   ← FastAPI pods (HPA: scale on CPU > 70% or request queue depth)
│ pod1 │  │ pod2 │
└──────┘  └──────┘
    │
    ├─── Qdrant Cluster  (StatefulSet, 3 nodes, replication factor 2)
    └─── Postgres        (RDS / CloudSQL, connection pooling via PgBouncer)
```

### Why Stateless

- Zero session state stored in the FastAPI process.
- All shared state lives in Postgres (tickets, eval results) and Qdrant (vectors).
- Any pod can handle any request. New replicas can be spun up in seconds.

### K8s Manifests

> **Not bundled in this POC.** The topology above is a design target. Kubernetes
> manifests are intentionally not included in this repository — the system ships
> and runs via Docker Compose (see Section 8). This section documents how it
> *would* be deployed to a cluster in production.

### Resource Estimates (per FastAPI pod)

| Resource | Request | Limit |
|---|---|---|
| CPU | 500m | 2000m |
| Memory | 1Gi | 3Gi |
| Startup probe | 30s | Grace for model loading |

Note: Flan-T5-base (~1 GB) and the cross-encoder (~100 MB) are loaded at pod startup via the FastAPI `lifespan` event. This is the primary driver of the high memory request.

---

## 14. Observability & MLOps

### Logging

All production code uses Python's `logging` module exclusively. Zero `print()` statements are permitted (enforced by test suite).

| Log File | Level | Rotation |
|---|---|---|
| `logs/app.log` | INFO and above | 5 MB / 5 backups |
| `logs/error.log` | ERROR and above | 5 MB / 5 backups |

Log entries are structured with timestamp, level, logger name, and message. In production, a log shipper (Filebeat / Fluentd) collects these and forwards to a central log aggregation system (Elasticsearch / CloudWatch).

### Health Endpoints

| Endpoint | Use | Checked Components |
|---|---|---|
| `GET /health` | Kubernetes liveness probe | Process alive |
| `GET /health/ready` | Kubernetes readiness probe | Qdrant (mandatory) + Postgres (optional) |

The readiness probe causes K8s to remove a pod from the load balancer pool if Qdrant is unavailable. Postgres degradation is reported but does not make the service unready — tickets and eval results fall back to in-memory.

### Metrics (Production Extension)

The following Prometheus metrics are the intended production extension:

| Metric | Type | Description |
|---|---|---|
| `http_request_duration_seconds` | Histogram | Request latency by endpoint |
| `retrieval_latency_ms` | Histogram | Hybrid search pipeline duration |
| `llm_call_duration_seconds` | Histogram | OpenAI API call latency |
| `circuit_breaker_state` | Gauge | 0=closed, 1=open, 2=half-open |
| `embedding_fallback_total` | Counter | MiniLM activations (Ada-002 unavailable) |
| `escalation_level_total` | Counter | L1/L2/L3 escalation distribution |

### ML Lifecycle

| Stage | Implementation |
|---|---|
| Data versioning | `data/incidents.xlsx` is tracked in git |
| Model versioning | Model names pinned in `app_config.json` |
| Evaluation gating | `POST /evaluate` thresholds block regression (NDCG < 0.80 = fail) |
| Fallback monitoring | `fallback_used` field in every `TriageResult` response; log-aggregated for drift detection |
| Embedding drift | MiniLM fallback activation rate logged; spike indicates Ada-002 regression |

---

## 15. Design Decisions

Key architectural decisions and their trade-offs are summarised below.

### ADR-001 — Qdrant as Vector Database

**Decision:** Qdrant over FAISS, ChromaDB, or Milvus.

| Criterion | Qdrant | FAISS | ChromaDB |
|---|---|---|---|
| Metadata filter DSL | Yes (full) | No | Partial |
| Async client | Yes | No | No |
| Persistence | Native | Manual | SQLite |
| Swappability | VectorStore ABC | — | — |

**Trade-off:** Qdrant requires a running container (operational overhead) vs FAISS being purely in-process. The metadata filter capability is necessary for category/asset filtering on `/search`.

### ADR-002 — Hybrid BM25 + Vector + RRF + Adaptive-K + Cross-Encoder

**Decision:** Five-stage retrieval pipeline over single-method search.

| Approach | Accuracy | Latency | Notes |
|---|---|---|---|
| BM25 only | Low (no semantics) | Fast | Misses paraphrase matches |
| Vector only | Medium | Medium | Misses exact keyword hits |
| Hybrid + RRF | High | Medium | Best of both, rank-only fusion |
| + Adaptive-K | High | Lower | ~40% latency reduction on simple queries |
| + Cross-encoder | Highest | +50ms | Stage-2 reranking on trimmed candidates only |

**Trade-off:** Added pipeline complexity vs measurably higher NDCG. The cross-encoder runs only on the adaptive-K trimmed set (3–20 candidates), not the full corpus.

### ADR-003 — LangGraph for Agent Orchestration

**Decision:** LangGraph `StateGraph` over CrewAI or custom chains.

**Why:** Typed `IncidentState` TypedDict makes state transitions debuggable and testable. Conditional edges are explicit in code, not implicit in prompt instructions. CrewAI's role-based model is better suited for collaborative multi-agent tasks; the L1→L2→L3 escalation pattern is a sequential state machine.

**Trade-off:** LangGraph requires understanding its graph compilation model. The benefit is deterministic routing with full state visibility at every node.

### ADR-004 — Tiered LLM Strategy (SLM-First)

**Decision:** GPT-4o-mini for L1, GPT-4o for L2 only, Flan-T5-base as fallback.

**Why:** 80%+ of incidents resolve at L1 (knowledge base match is strong). Using GPT-4o for every query would be ~10x the cost with negligible accuracy gain on well-matched KB cases.

| Tier | Model | Avg latency | Cost/1k calls |
|---|---|---|---|
| L1 | GPT-4o-mini | ~800ms | $0.15 |
| L2 | GPT-4o | ~2500ms | $2.50 |
| Fallback | Flan-T5-base | ~300ms | $0.00 |

### ADR-005 — SQLite as Default DB (Zero-Install), Postgres as Optional Override

**Decision:** L3 escalation tickets and evaluation run results default to SQLite (`aiosqlite` driver, file `data/incident_kb.db`). If `POSTGRES_USER` + `POSTGRES_PASSWORD` env vars are set, Postgres is used instead (same ORM models, zero code changes). If both fail, the system falls back to in-memory Python lists.

**Why SQLite not Postgres as default:** Postgres requires a running server process and installation. SQLite is a single file — no server, no install, no configuration. For development and single-node deployments this is perfectly adequate. The `VectorStore` ABC and SQLAlchemy ORM abstraction mean switching to Postgres in production is one env-var change.

**Trade-off:** SQLite does not support concurrent writes from multiple processes. For a multi-replica production deployment, Postgres must be configured. For single-process local dev and Docker single-node, SQLite is fine.

**Caching note:** Without an active cache layer, all embedding calls and search queries go to the live service on every request. Latency is approximately 150–400 ms per search (full pipeline). Future work: wire Redis into `cache.py` stubs for 24h embedding cache and 1h query result cache.

---

## 16. Performance Benchmarks

Measurements taken on a single-node local setup (4-core, 16 GB RAM) after warm start (all models and indices loaded).

| Endpoint | p50 | p95 | p99 | Notes |
|---|---|---|---|---|
| `GET /health` | < 5ms | < 10ms | < 15ms | In-process only |
| `GET /health/ready` | < 30ms | < 80ms | < 100ms | 2 downstream checks (Qdrant + Postgres) |
| `POST /search` | 180ms | 320ms | 450ms | Full hybrid pipeline (no caching active) |
| `POST /triage` (L1 resolve) | 900ms | 1400ms | 2000ms | Includes search + LLM |
| `POST /triage` (L2 resolve) | 2800ms | 4200ms | 5500ms | Includes Tavily + GPT-4o |
| `POST /ingest` (150 rows) | — | — | ~4s | Async batches of 50 |

**Target under 50-user load test:**
- p99 latency < 500 ms for `/search`
- Error rate < 1%
- Throughput > 40 RPS

**Adaptive-K Impact:**

| Query type | k used | Reranker input | Latency saving |
|---|---|---|---|
| Specific (error code) | 3 | 3 candidates | ~40% vs k=20 |
| Typical | 10 | 6–10 candidates | ~15% |
| Vague | 20 | 15–20 candidates | Baseline |

---

## 17. Reliability & Resiliency Patterns

### Circuit Breaker

`pybreaker` wraps all OpenAI API calls (embeddings + LLM).

- **fail_max:** 5 consecutive failures open the circuit
- **reset_timeout:** 60 seconds before attempting a half-open probe
- **Open state behaviour:** Routes to Flan-T5-base (LLM) or MiniLM (embeddings)

### Retry with Exponential Backoff

`tenacity` retries transient failures before the circuit breaker opens.

- **Attempts:** 3
- **Wait:** 2s → 4s → 8s (exponential)
- **Retried:** HTTP 429 (rate limit), 500, 503, network timeout

### Graceful Degradation Hierarchy

| Failure | System Behaviour |
|---|---|
| Qdrant unavailable | Search uses BM25 only; `retrieval_method: "bm25_only"` in response |
| Postgres unavailable | L3 tickets + eval results stored in-memory; no 503 |
| OpenAI timeout (5 failures) | Circuit opens; Flan-T5 fallback activates; `fallback_used: true` in response |
| Ada-002 embedding failure | MiniLM local embeddings used; logged as warning |
| Both LLM and fallback fail | Returns `503` with structured JSON error; does not crash the process |
| XLSX/XLS with missing columns | `400` with field-level validation error via Pydantic |
| Empty ingestion file | `400` with explicit message; BM25 index not overwritten |

### Connection Pooling

| Service | Pool Config |
|---|---|
| SQLite (default) | Single file, no pool — `aiosqlite` async driver |
| Postgres (optional) | `QueuePool(pool_size=20, max_overflow=10, pool_pre_ping=True)` |
| Qdrant | Async client with persistent HTTP session (timeout=10s) |

`pool_pre_ping=True` on Postgres ensures stale connections are detected and replaced before use, preventing silent failures after Postgres restarts.

### Cold Start Optimisation

All heavy resources are loaded once during the FastAPI `lifespan` startup event:

| Resource | Load time | Notes |
|---|---|---|
| Flan-T5-base | ~8–12s | HuggingFace download on first run |
| MiniLM-L6-v2 | ~2s | |
| ms-marco-MiniLM cross-encoder | ~1s | |
| BM25 pickle index | < 100ms | After first ingest |

None of these are loaded on first request. A request arriving before startup is complete receives a `503` from the readiness probe.

---

## Final Verification Checklist

```
[ ] docker compose up --build  →  all containers healthy, zero errors
[ ] POST /ingest  →  { "ingested": 150, "skipped": 0, "pii_masked_total": N }
[ ] POST /search  →  ranked results + resolution_options + adaptive_k_used
[ ] POST /search mode="semantic"  →  retrieval_method: "semantic" in response
[ ] POST /triage  →  { escalation_level, confidence, final_answer, model_used, fallback_used }
[ ] POST /evaluate  →  { ndcg_at_10 >= 0.80, faithfulness >= 0.70, relevancy >= 0.75 }
[ ] POST /chat (session_id=null)  →  { session_id, message with numbered steps, option_progress }
[ ] POST /chat (session_id=existing, "didn't work")  →  option_progress.current increments
[ ] POST /chat (all options exhausted)  →  is_escalated=true, escalation_ticket_id set
[ ] pytest tests/ -v --cov=src  →  tests pass, coverage >= 75%
[ ] React app http://localhost:5173  →  Search · Triage · Analytics · Ingest · Chat · Admin pages functional
[ ] /chat page  →  bubbles render, action buttons advance fix index, escalation shows ticket ID
[ ] grep -r "print(" src/  →  zero matches (except main.py line 63 — deferred)
[ ] grep -rE "(sk-|password\s*=)" src/  →  zero matches
[ ] GET /health/ready  →  { "checks": { "qdrant": "ok", "postgres": "ok|fail" } }
[ ] GET /escalations  →  returns tickets from Postgres (or memory if Postgres not configured)
```

---

## 18. Chat Assistant Feature

The Chat Assistant provides a conversational guided troubleshooting flow on top of the existing hybrid retrieval pipeline. It is accessible at `/chat` in the React app and via `POST /it-kb/chat` in the API.

### How It Works

```
User: "Storage upload failing on MediaServer01"
  → new session created (session_id: abc-123)
  → hybrid_search() retrieves 3 resolution options from KB
  ← "Fix 1 of 3 (verified 14× in KB): 1. Check disk quota..."
     [This didn't work, try next fix]  [Issue resolved]

User clicks "This didn't work, try next fix"
  → intent detected: NEXT_OPTION
  ← "Fix 2 of 3 (verified 6× in KB): 1. Enable log rotation..."

User: "What does step 2 mean exactly?"
  → intent detected: QUESTION  (> 40 chars, no keyword match)
  ← GPT-4o-mini answers in context — index does NOT advance

User: "That didn't help either"
  → index 2 → advance to 3 → 3 >= len(options) → _escalate()
  ← "Escalated · Ticket: TKT-A1B2C3D4"  (OPEN, IT-OPS queue)
```

### Session Management

| Property | Value |
|---|---|
| Storage | In-memory dict (per FastAPI process) |
| TTL | 30 minutes of inactivity |
| Cleanup | Background asyncio task, runs every 5 minutes |
| Expiry response | `404 Session not found or expired` |

### Key Files

| File | Purpose |
|---|---|
| `backend/src/chat/session_manager.py` | `ChatSession` dataclass + `SessionManager` singleton |
| `backend/src/chat/chat_agent.py` | `ChatAgent` — intent detection, fix formatting, escalation |
| `backend/src/api/chat.py` | `POST /chat` FastAPI router |
| `backend/src/models/chat.py` | `ChatRequest`, `ChatResponse`, `OptionProgress` Pydantic models |
| `frontend/src/pages/ChatPage.tsx` | Full chat UI with typing indicator and auto-scroll |
| `frontend/src/components/chat/MessageBubble.tsx` | Markdown-rendering bubbles with action buttons |
| `frontend/src/store/chatStore.ts` | Zustand store — session ID, message history, loading state |

---

## 19. PII Masking

All incident text fields are redacted before any data is embedded, stored in Qdrant, sent to OpenAI, or persisted to the escalation tickets table.

### Patterns Applied (in order)

| Token | Pattern target | Example |
|---|---|---|
| `[EMAIL]` | Email addresses | `user@corp.com` → `[EMAIL]` |
| `[SSN]` | Social Security Numbers | `123-45-6789` → `[SSN]` |
| `[CREDIT_CARD]` | 16-digit card numbers | `4111 1111 1111 1111` → `[CREDIT_CARD]` |
| `[IP_ADDRESS]` | IPv4 addresses | `192.168.1.100` → `[IP_ADDRESS]` |
| `[PHONE]` | US phone numbers | `555-867-5309` → `[PHONE]` |

Order matters — EMAIL is applied before IP/number patterns to avoid partial overlaps; SSN before CREDIT_CARD.

### Fields Masked

| Field | Masked? |
|---|---|
| `description` | ✅ Yes |
| `resolution_notes` | ✅ Yes |
| `title` | ✅ Yes |
| `incident_id`, `ticket_id`, `category`, `assigned_to` | ❌ No (metadata identifiers) |

### Audit Logging

A single `INFO` log line is emitted per row where PII was found:
```
PII masked in row_42 | total=3 fields=['description'] by_type={'[EMAIL]': 1, '[IP_ADDRESS]': 2}
```
The actual PII values are **never** written to logs.

### Ingestion Response

`pii_masked_total` in the `/ingest` response gives the aggregate count across the entire uploaded dataset:
```json
{ "ingested": 150, "skipped": 0, "duration_ms": 4102, "pii_masked_total": 37 }
```

### Key Files

| File | Purpose |
|---|---|
| `backend/src/ingestion/pii_masker.py` | `mask_pii(text)` → `(masked_text, counts)`, `summarize_masking()` |
| `backend/src/ingestion/preprocessor.py` | Calls `mask_pii()` in `_process_row()` after `clean_text()` |
| `backend/src/ingestion/pipeline.py` | Accumulates `pii_masked_total` across all records |
| `backend/src/api/ingestion.py` | Exposes `pii_masked_total` in `IngestResponse` |
