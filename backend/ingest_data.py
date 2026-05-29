"""
Standalone ingestion script — loads incidents.xlsx directly into Qdrant Cloud.

Usage (run from the backend/ directory):
    python ingest_data.py

No server, no Redis needed. Reads env from env/development.env directly.
"""

import asyncio
import hashlib
import os
import sys
from pathlib import Path

# ── Load env vars from development.env ────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / "env" / "development.env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
QDRANT_URL     = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
    print("❌  OPENAI_API_KEY not set in env/development.env")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
XLSX_PATH       = Path(__file__).parent.parent / "data" / "incidents.xlsx"
COLLECTION_NAME = "incidents"
VECTOR_SIZE     = 1536          # text-embedding-ada-002
EMBEDDING_MODEL = "text-embedding-ada-002"
BATCH_SIZE      = 50            # incidents per OpenAI embed call


# ── Helpers ───────────────────────────────────────────────────────────────────

def incident_id_to_qdrant_id(incident_id: str) -> int:
    """Stable 32-bit int from incident_id string via SHA-1."""
    return int(hashlib.sha1(incident_id.encode()).hexdigest()[:8], 16)


def clean(val) -> str:
    """Strip and normalise a cell value to a plain string."""
    import pandas as pd
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def load_xlsx(path: Path) -> list[dict]:
    """Parse the XLSX and return cleaned incident dicts."""
    import pandas as pd

    print(f"📂  Reading {path} …")
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]

    print(f"    Columns found: {list(df.columns)}")
    print(f"    Total rows:    {len(df)}")

    incidents = []
    skipped   = 0

    for _, row in df.iterrows():
        incident_id     = clean(row.get("Incident ID", ""))
        description     = clean(row.get("Description", ""))
        resolution      = clean(row.get("Solution", ""))

        if not incident_id or not description or not resolution:
            skipped += 1
            continue

        title    = clean(row.get("Incident Details", ""))
        category = clean(row.get("Category", ""))
        ticket   = clean(row.get("Ticket ID", ""))
        asset    = clean(row.get("Media Asset", ""))

        search_text = f"{title}: {description}" if title else description

        incidents.append({
            "incident_id":    incident_id,
            "ticket_id":      ticket,
            "title":          title,
            "category":       category,
            "description":    description,
            "resolution_notes": resolution,
            "assigned_to":    asset,
            "search_text":    search_text,
        })

    print(f"    Parsed:  {len(incidents)}")
    print(f"    Skipped: {skipped}")
    return incidents


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with OpenAI ada-002."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


async def ensure_collection(client) -> None:
    """Create Qdrant collection if it doesn't exist yet."""
    from qdrant_client.models import Distance, VectorParams

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        count = client.count(COLLECTION_NAME).count
        print(f"✅  Collection '{COLLECTION_NAME}' exists ({count} points already)")
    else:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"✅  Created collection '{COLLECTION_NAME}'")


async def ingest(incidents: list[dict]) -> None:
    """Embed + upsert all incidents into Qdrant in batches."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
    await ensure_collection(qdrant)

    batches = [
        incidents[i: i + BATCH_SIZE]
        for i in range(0, len(incidents), BATCH_SIZE)
    ]
    total_upserted = 0

    print(f"\n🚀  Embedding + upserting {len(incidents)} incidents in {len(batches)} batches …\n")

    for i, batch in enumerate(batches, 1):
        texts   = [inc["search_text"] for inc in batch]
        vectors = await embed_batch(texts)

        points = [
            PointStruct(
                id=incident_id_to_qdrant_id(inc["incident_id"]),
                vector=vec,
                payload={
                    "incident_id":     inc["incident_id"],
                    "ticket_id":       inc.get("ticket_id", ""),
                    "title":           inc.get("title", ""),
                    "category":        inc.get("category", ""),
                    "description":     inc["description"],
                    "resolution_notes": inc["resolution_notes"],
                    "assigned_to":     inc.get("assigned_to", ""),
                    "search_text":     inc["search_text"],
                },
            )
            for inc, vec in zip(batch, vectors)
        ]

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        total_upserted += len(batch)

        bar   = "█" * i + "░" * (len(batches) - i)
        pct   = round(total_upserted / len(incidents) * 100)
        print(f"  [{bar}] {pct:3d}%  batch {i}/{len(batches)}  ({total_upserted} upserted)", end="\r")

    print(f"\n\n✅  Done — {total_upserted} incidents in Qdrant collection '{COLLECTION_NAME}'")

    # Verify
    final_count = qdrant.count(COLLECTION_NAME).count
    print(f"📊  Collection now has {final_count} points")


# ── Also build + save BM25 index ─────────────────────────────────────────────

def build_bm25(incidents: list[dict]) -> None:
    """Build BM25 index and save it to data/ for the retrieval layer."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from src.ingestion.bm25_builder import build_bm25_index, save_bm25_index
        index, corpus, ids = build_bm25_index(incidents)
        save_bm25_index(index, corpus, ids)
        print("✅  BM25 index built and saved to data/bm25_index.pkl")
    except Exception as e:
        print(f"⚠️   BM25 index build skipped: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    print("=" * 55)
    print("  Incident KB — Data Ingestion")
    print(f"  Target: {QDRANT_URL}")
    print(f"  File:   {XLSX_PATH}")
    print("=" * 55 + "\n")

    if not XLSX_PATH.exists():
        print(f"❌  Dataset not found at {XLSX_PATH}")
        sys.exit(1)

    incidents = load_xlsx(XLSX_PATH)
    if not incidents:
        print("❌  No valid incidents parsed from the file.")
        sys.exit(1)

    build_bm25(incidents)
    await ingest(incidents)


if __name__ == "__main__":
    asyncio.run(main())
