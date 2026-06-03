"""
main.py — Application entry point.
Mirrors Synapt-PersonalizedRAG-API structure:
  1. Parse CLI environment argument
  2. Load configs (app_config.json + config.json[env] + env secrets)
  3. Initialise all services in lifespan (models, DB, cache, vector store)
  4. Register routers and exception handlers
  5. Start Uvicorn

Usage:
  python main.py development
  python main.py docker
  python main.py production
"""

import os
import sys
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Config loading ────────────────────────────────────────────────────────────
from src.core.config import (
    load_app_config,
    load_env_config,
    load_environment,
    require_env,
    get_env,
)

# ── Logging ───────────────────────────────────────────────────────────────────
from src.handlers.logger import init_loggers, log_info, log_warning, log_error

# ── Exception handlers ────────────────────────────────────────────────────────
from src.exceptions.exception_handler import register_exception_handlers

# ── Integrations ──────────────────────────────────────────────────────────────
from src.integrations.vector_db import QdrantLocalVectorStore, QdrantVectorStore
from src.integrations.embeddings import init_embeddings
from src.integrations.llm import init_llm
from src.integrations.database import init_database, init_sqlite, create_tables
from src.retrieval.reranker import init_reranker
from src.retrieval.bm25_retriever import load_bm25_retriever, set_bm25_payload_map
import src.integrations as integrations_pkg

# ── Agent graph ───────────────────────────────────────────────────────────────
from src.agents.graph import build_triage_graph

# ── API routers ───────────────────────────────────────────────────────────────
from src.api.health import router as health_router
from src.api.ingestion import router as ingestion_router
from src.api.search import router as search_router
from src.api.triage import router as triage_router
from src.api.evaluation import router as evaluation_router
from src.api.chat import router as chat_router
from src.api.feedback import router as feedback_router
from src.chat.session_manager import session_manager as chat_session_manager

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
_VALID_ENVS = {"development", "docker", "production"}
environment = sys.argv[1] if len(sys.argv) > 1 else "development"
if environment not in _VALID_ENVS:
    print(f"[WARN] Unknown environment '{environment}' — defaulting to 'development'")
    environment = "development"

# Load configs immediately (before lifespan so they're available at import time)
app_config = load_app_config()
env_config = load_env_config(environment)
load_environment(environment, env_config)

# Initialise loggers as soon as env is loaded
_log_cfg = app_config["LOGGING"]
init_loggers(
    log_dir=_log_cfg["LOG_DIR"],
    max_bytes=_log_cfg["MAX_BYTES"],
    backup_count=_log_cfg["BACKUP_COUNT"],
    log_level=env_config.get("log_level", "INFO"),
)

log_info("=" * 60)
log_info("Starting %s | env=%s", app_config["APP_NAME"], environment)
log_info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — load ALL heavy resources at startup, never on first request
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise all services. Shutdown: clean up."""
    log_info("Lifespan startup — initialising services ...")

    # ── Vector Store — Cloud first, local embedded fallback ───────────────────
    #
    # Priority:
    #   1. Qdrant Cloud (QDRANT_URL) — tried with a 10-second timeout so a
    #      blocked/unreachable cloud endpoint fails fast during demos.
    #   2. Local embedded Qdrant (data/qdrant_local/) — zero network dependency.
    #      Pre-populate once via:  python setup_local_qdrant.py
    #
    # BM25 runs alongside whichever vector store is active.
    # ──────────────────────────────────────────────────────────────────────────
    import asyncio as _asyncio

    _collection   = app_config["QDRANT"]["COLLECTION_NAME"]
    _vector_size  = app_config["QDRANT"]["VECTOR_SIZE"]
    _qdrant_url   = get_env("QDRANT_URL") or env_config.get("qdrant_url", "")
    _qdrant_key   = get_env("QDRANT_API_KEY") or None

    vector_store = None

    # ── 1. Try Qdrant Cloud ───────────────────────────────────────────────────
    if _qdrant_url:
        try:
            _cloud = QdrantVectorStore(url=_qdrant_url, api_key=_qdrant_key)
            await _asyncio.wait_for(
                _cloud.create_collection(collection=_collection, vector_size=_vector_size),
                timeout=10.0,
            )
            vector_store = _cloud
            log_info("Qdrant Cloud connected | collection=%s", _collection)
        except Exception as _cloud_exc:
            log_warning(
                "Qdrant Cloud unreachable (timeout or network block) — "
                "switching to local embedded Qdrant | error=%s", _cloud_exc,
            )

    # ── 2. Local embedded fallback ────────────────────────────────────────────
    if vector_store is None:
        _local = QdrantLocalVectorStore(path="data/qdrant_local")
        try:
            await _local.create_collection(collection=_collection, vector_size=_vector_size)
            vector_store = _local
            log_info("Local embedded Qdrant active | collection=%s", _collection)
        except Exception as _local_exc:
            log_error(
                "Both Cloud and local Qdrant failed — vector search unavailable. "
                "Run setup_local_qdrant.py to initialise local store. | error=%s",
                _local_exc,
            )
            # Keep the local store object so the app starts in degraded mode
            vector_store = _local

    app.state.vector_store = vector_store
    integrations_pkg._qdrant_store = vector_store  # for health check

    # ── Evaluation store — ALWAYS the local deduped index ─────────────────────
    # DeepEval / IR metrics must be deterministic and run against the
    # dedup-consistent local store (data/qdrant_local), regardless of whether
    # the live app is on Qdrant Cloud. If the app is already on local, reuse
    # that instance to avoid a second embedded-client file lock.
    if isinstance(vector_store, QdrantLocalVectorStore):
        eval_vector_store = vector_store
    else:
        try:
            eval_vector_store = QdrantLocalVectorStore(path="data/qdrant_local")
            await eval_vector_store.create_collection(
                collection=_collection, vector_size=_vector_size
            )
        except Exception as _eval_exc:
            log_warning(
                "Eval local store unavailable — eval will fall back to the live store. "
                "Run setup_local_qdrant.py to build it. | error=%s", _eval_exc,
            )
            eval_vector_store = vector_store
    app.state.eval_vector_store = eval_vector_store

    try:
        _eval_docs = await eval_vector_store.count(_collection)
        _eval_backend = "local" if isinstance(eval_vector_store, QdrantLocalVectorStore) else "cloud"
        log_info(
            "Evaluation store ready | backend=%s docs=%d%s",
            _eval_backend, _eval_docs,
            "  ⚠ looks NOT deduped (expected ~165) — re-run setup_local_qdrant.py"
            if _eval_docs > 200 else "",
        )
    except Exception:
        pass

    # ── Database (SQLite default; Postgres if POSTGRES_USER is set) ──────────
    # Import ORM models first so Base.metadata is populated before create_tables()
    import src.models.db_models  # noqa: F401

    pg_user     = get_env("POSTGRES_USER")
    pg_password = get_env("POSTGRES_PASSWORD")

    if pg_user and pg_password:
        # ── Postgres (optional — for production / hosted environments) ────────
        try:
            init_database(
                host=env_config.get("postgres_host", "localhost"),
                port=int(env_config.get("postgres_port", 5432)),
                db=env_config.get("postgres_db", "incident_kb"),
                user=pg_user,
                password=pg_password,
                pool_size=app_config["DATABASE"]["POOL_SIZE"],
                max_overflow=app_config["DATABASE"]["MAX_OVERFLOW"],
                pool_recycle=app_config["DATABASE"]["POOL_RECYCLE_SECONDS"],
            )
            await create_tables()
            log_info("Postgres initialised and tables verified")
        except Exception as _pg_exc:
            log_error(
                "Postgres unavailable — falling back to SQLite. | error=%s", _pg_exc,
            )
            init_sqlite(db_path="data/incident_kb.db")
            await create_tables()
    else:
        # ── SQLite (default — zero install, file-backed) ──────────────────────
        try:
            init_sqlite(db_path="data/incident_kb.db")
            await create_tables()
            log_info("SQLite database initialised | path=data/incident_kb.db")
        except Exception as _db_exc:
            log_error(
                "SQLite init failed — ticket and eval persistence will use "
                "in-memory fallback. | error=%s", _db_exc,
            )

    # ── Seed sample feedback (demo data; only if the store is empty) ─────────
    try:
        from src.feedback.feedback_store import seed_sample_feedback
        await seed_sample_feedback()
    except Exception as _seed_exc:
        log_warning("Feedback seed skipped | error=%s", _seed_exc)

    # ── Azure credentials (shared across LLM + embeddings + judge) ───────────
    azure_api_key      = require_env("AZURE_OPENAI_API_KEY")
    azure_endpoint     = require_env("AZURE_OPENAI_ENDPOINT")
    azure_llm_version  = get_env("AZURE_OPENAI_API_VERSION") \
                         or app_config["LLM"]["AZURE_API_VERSION"]
    azure_emb_version  = get_env("AZURE_OPENAI_EMBEDDING_API_VERSION") \
                         or app_config["LLM"]["AZURE_EMBEDDING_API_VERSION"]

    # ── Embeddings ────────────────────────────────────────────────────────────
    init_embeddings(
        azure_api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_emb_version,
        embedding_model=app_config["LLM"]["EMBEDDING_MODEL"],
        fallback_model=app_config["LLM"]["EMBEDDING_FALLBACK_MODEL"],
        embedding_ttl=app_config["CACHE"]["EMBEDDING_TTL_SECONDS"],
    )

    # ── BM25 index + payload map ──────────────────────────────────────────────
    try:
        load_bm25_retriever(index_dir="data")
        log_info("BM25 index loaded successfully")

        # Scroll all Qdrant points to build incident_id → payload map
        # so BM25 keyword hits are enriched with full metadata
        _collection = app_config["QDRANT"]["COLLECTION_NAME"]
        all_points = await vector_store.scroll_all(_collection)
        payload_map = {
            p["payload"]["incident_id"]: p["payload"]
            for p in all_points
            if p.get("payload", {}).get("incident_id")
        }
        set_bm25_payload_map(payload_map)
    except Exception as _bm25_exc:
        log_error(
            "BM25 index not available at startup — keyword search disabled. "
            "Run ingest_data.py to build the index. | error=%s", _bm25_exc,
        )

    # ── Cross-encoder reranker ────────────────────────────────────────────────
    init_reranker(
        model_name=app_config["LLM"].get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    )

    # ── LLM (Azure OpenAI + Flan-T5 fallback) ────────────────────────────────
    init_llm(
        azure_api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_llm_version,
        l1_model=app_config["LLM"]["L1_MODEL"],
        l2_model=app_config["LLM"]["L2_MODEL"],
        fallback_model=app_config["LLM"]["FALLBACK_MODEL"],
        request_timeout=app_config["LLM"]["REQUEST_TIMEOUT_SECONDS"],
        max_retries=app_config["LLM"]["MAX_RETRIES"],
        retry_base_delay=app_config["LLM"]["RETRY_BASE_DELAY_SECONDS"],
    )

    # ── LLM Judge (Azure OpenAI — for DeepEval evaluation metrics) ───────────
    from src.evaluation.llm_judge import init_llm_judge
    init_llm_judge(
        azure_api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_llm_version,
        deployment=app_config["LLM"]["L1_MODEL"],
    )

    # ── Triage graph (LangGraph — compiled once, reused per request) ─────────
    app.state.triage_graph = build_triage_graph(
        vector_store=vector_store,
        collection=app_config["QDRANT"]["COLLECTION_NAME"],
        app_config=app_config,
    )

    # ── Chat session cleanup task ─────────────────────────────────────────────
    import asyncio as _asyncio
    _asyncio.create_task(chat_session_manager.start_cleanup_task())
    log_info("Chat session manager started | TTL=1800s")

    # ── Store configs on app.state for dependency injection ───────────────────
    app.state.app_config = app_config
    app.state.env_config = env_config

    log_info("All services initialised. Application ready.")

    yield  # Application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log_info("Lifespan shutdown — cleaning up ...")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=app_config["APP_NAME"],
    version=app_config["APP_VERSION"],
    description=app_config["DESCRIPTION"],
    docs_url=f"{app_config['API_PREFIX']}/docs",
    redoc_url=f"{app_config['API_PREFIX']}/redoc",
    openapi_url=f"{app_config['API_PREFIX']}/openapi.json",
    lifespan=lifespan,
)

# ── CORS (allow React dev server) ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Exception handlers ────────────────────────────────────────────────────────
register_exception_handlers(app)

# ── Routers ───────────────────────────────────────────────────────────────────
prefix = app_config["API_PREFIX"]
app.include_router(health_router)
app.include_router(ingestion_router, prefix=prefix)
app.include_router(search_router, prefix=prefix)
app.include_router(triage_router, prefix=prefix)
app.include_router(evaluation_router, prefix=prefix)
app.include_router(chat_router, prefix=prefix)
app.include_router(feedback_router, prefix=prefix)

log_info(
    "Routers registered | prefix=%s | docs=%s/docs",
    prefix,
    prefix,
)

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,
        reload=env_config.get("debug", False),
        log_level=env_config.get("log_level", "info").lower(),
    )
