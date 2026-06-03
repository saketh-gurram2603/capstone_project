"""
One-time setup: pre-populate the local embedded Qdrant store.

Run this ONCE while connected to the VPN (Azure OpenAI accessible).
The embedded store is saved to data/qdrant_local/ and persists across restarts.

Usage (from the backend/ directory):
    python setup_local_qdrant.py

After this runs, the main app will automatically use the local store
whenever Qdrant Cloud is unreachable.
"""

import asyncio
import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "env" / "development.env")

AZURE_API_KEY         = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_ENDPOINT        = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_EMB_VERSION     = os.environ.get("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-05-01-preview")
AZURE_EMB_DEPLOYMENT  = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
                                        "synapt-dev-text-embedding-ada-002")

XLSX_PATH    = Path(__file__).parent.parent / "data" / "incidents.xlsx"
LOCAL_PATH   = "data/qdrant_local"
COLLECTION   = "incidents"
VECTOR_SIZE  = 1536
BATCH_SIZE   = 50

if not AZURE_API_KEY:
    print("❌  AZURE_OPENAI_API_KEY not set — connect to VPN first"); sys.exit(1)
if not AZURE_ENDPOINT:
    print("❌  AZURE_OPENAI_ENDPOINT not set"); sys.exit(1)
if not XLSX_PATH.exists():
    print(f"❌  Dataset not found at {XLSX_PATH}"); sys.exit(1)


def incident_id_to_qdrant_id(incident_id: str) -> int:
    return int(hashlib.sha1(incident_id.encode()).hexdigest()[:8], 16)


def clean(val) -> str:
    import pandas as pd
    if val is None: return ""
    try:
        if pd.isna(val): return ""
    except (TypeError, ValueError): pass
    return str(val).strip()


def load_incidents() -> list[dict]:
    import pandas as pd
    df = pd.read_excel(XLSX_PATH, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]
    rows = []
    # (problem + solution) dedup — MUST match src/ingestion/preprocessor.py so
    # the local store is identical to the production-ingested collection.
    # Without this, the original 150 rows (≈7 archetypes repeated ~21x with
    # identical text) all get indexed, swamping search results with byte-
    # identical duplicates and capping eval Recall@10 at ~36%.
    seen_hashes: set[str] = set()
    duplicate_count = 0
    for _, row in df.iterrows():
        inc_id = clean(row.get("Incident ID", ""))
        desc   = clean(row.get("Description", ""))
        sol    = clean(row.get("Solution", ""))
        if not inc_id or not desc or not sol: continue
        title    = clean(row.get("Incident Details", ""))
        category = clean(row.get("Category", ""))

        content_hash = hashlib.md5(
            f"{title.lower().strip()}|{desc.lower().strip()}|{sol.lower().strip()}".encode("utf-8")
        ).hexdigest()
        if content_hash in seen_hashes:
            duplicate_count += 1
            continue
        seen_hashes.add(content_hash)
        opened_at  = clean(row.get("Opened At",  ""))
        resolved_at = clean(row.get("Resolved At", ""))
        try:
            res_hours = float(row.get("Resolution Hours", 0) or 0)
        except (ValueError, TypeError):
            res_hours = 0.0
        if res_hours <= 0:
            _RANGES = {
                "Storage": (2.0, 8.0), "Application": (1.0, 6.0),
                "Database": (1.0, 4.0), "Network": (0.5, 3.0),
                "Security": (2.0, 12.0), "Performance": (1.0, 4.0),
                "Hardware": (4.0, 24.0), "Authentication": (0.5, 2.0),
                "Monitoring": (0.5, 1.5), "Configuration": (0.5, 2.0),
            }
            lo, hi = _RANGES.get(category, (1.0, 6.0))
            res_hours = round((lo + hi) / 2.0, 2)

        rows.append({
            "incident_id":      inc_id,
            "ticket_id":        clean(row.get("Ticket ID", "")),
            "title":            title,
            "category":         category,
            "description":      desc,
            "resolution_notes": sol,
            "assigned_to":      clean(row.get("Media Asset", "")),
            "search_text":      f"{title}: {desc}" if title else desc,
            "opened_at":        opened_at,
            "resolved_at":      resolved_at,
            "resolution_hours": res_hours,
        })
    if duplicate_count:
        print(f"   ↳ skipped {duplicate_count} duplicate (problem+solution) rows — "
              f"{len(rows)} unique incidents will be indexed")
    return rows


async def embed_batch(texts: list[str]) -> list[list[float]]:
    from openai import AsyncAzureOpenAI
    client = AsyncAzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_EMB_VERSION,
    )
    resp = await asyncio.wait_for(
        client.embeddings.create(model=AZURE_EMB_DEPLOYMENT, input=texts),
        timeout=60.0,
    )
    return [item.embedding for item in resp.data]


async def run():
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    print("=" * 55)
    print("  Local Qdrant Setup")
    print(f"  Azure endpoint : {AZURE_ENDPOINT}")
    print(f"  Embedding      : {AZURE_EMB_DEPLOYMENT}")
    print(f"  Local path     : {LOCAL_PATH}")
    print("=" * 55 + "\n")

    # ── Load incidents ────────────────────────────────────────────────────────
    incidents = load_incidents()
    print(f"📂  Loaded {len(incidents)} incidents from {XLSX_PATH.name}\n")

    # ── Clean slate ───────────────────────────────────────────────────────────
    # delete_collection() on the embedded local store does NOT reliably purge
    # on-disk points — stale duplicates survive and the count never drops
    # (you'll re-ingest 165 but still see 315). Physically removing the store
    # directory guarantees a clean rebuild. We do this BEFORE opening any client
    # so no file lock blocks the wipe.
    import shutil
    from pathlib import Path as _Path

    store_dir = _Path(LOCAL_PATH)
    if store_dir.exists() and any(store_dir.iterdir()):
        ans = input(
            f"    Local store at '{LOCAL_PATH}' already exists. "
            f"Wipe it and re-ingest? [y/N]: "
        ).strip().lower()
        if ans != "y":
            print("Skipped — local store unchanged."); return
        shutil.rmtree(store_dir, ignore_errors=True)
        if store_dir.exists() and any(store_dir.iterdir()):
            print(
                "❌  Could not wipe the local store — it's likely still open by the "
                "running app. Stop the app and re-run this script."
            )
            return
        print(f"🧹  Wiped existing local store at '{LOCAL_PATH}'")

    os.makedirs(LOCAL_PATH, exist_ok=True)
    client = AsyncQdrantClient(path=LOCAL_PATH)
    await client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"✅  Collection '{COLLECTION}' created (clean)\n")

    # ── Embed + upsert in batches ─────────────────────────────────────────────
    batches = [incidents[i:i + BATCH_SIZE] for i in range(0, len(incidents), BATCH_SIZE)]
    total   = 0

    for i, batch in enumerate(batches, 1):
        print(f"  Embedding batch {i}/{len(batches)} ({len(batch)} incidents) …", end="", flush=True)
        vectors = await embed_batch([inc["search_text"] for inc in batch])

        points = [
            PointStruct(
                id=incident_id_to_qdrant_id(inc["incident_id"]),
                vector=vec,
                payload={k: inc[k] for k in inc if k != "search_text"},
            )
            for inc, vec in zip(batch, vectors)
        ]
        await client.upsert(collection_name=COLLECTION, points=points)
        total += len(batch)
        print(f" ✓  ({total}/{len(incidents)} done)")

    final = (await client.count(collection_name=COLLECTION, exact=True)).count
    print(f"\n✅  Local Qdrant ready — {final} vectors stored at '{LOCAL_PATH}'")

    # ── Build BM25 index ──────────────────────────────────────────────────────
    print("\n📑  Building BM25 index …")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from src.ingestion.bm25_builder import build_bm25_index, save_bm25_index
        bm25_index, corpus, ids = build_bm25_index(incidents)
        save_bm25_index(bm25_index, corpus, ids)
        print("✅  BM25 index saved to data/bm25_index.pkl")
    except Exception as e:
        print(f"⚠️   BM25 build failed (non-critical): {e}")

    print("\n🎉  Setup complete — both BM25 and local Qdrant are ready.")
    print("   The app will use this store automatically when Qdrant Cloud is unreachable.\n")


if __name__ == "__main__":
    asyncio.run(run())
