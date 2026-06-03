"""
BM25 retriever — loads the persisted BM25Okapi index and scores queries.

Designed as a module-level singleton (loaded once at startup) so the
expensive pickle.load() never happens on the hot request path.

Usage:
    # At startup (main.py lifespan):
    load_bm25_retriever(index_dir="data")

    # At query time:
    results = bm25_search("disk space upload failure", top_k=10)
"""

from __future__ import annotations

from typing import Optional

from rank_bm25 import BM25Okapi

from src.exceptions.custom_exceptions import IndexNotFoundError
from src.handlers.logger import get_logger, log_info, log_warning
from src.ingestion.bm25_builder import load_bm25_index, save_bm25_index, tokenize

logger = get_logger("retrieval.bm25_retriever")


# ── Singleton state ───────────────────────────────────────────────────────────

class BM25Retriever:
    """Thin wrapper around a loaded BM25Okapi index."""

    def __init__(self) -> None:
        self._index: Optional[BM25Okapi] = None
        self._corpus: list[str] = []
        self._ids: list[str] = []

    def load(self, index_dir: Optional[str] = None) -> None:
        """Load (or reload) the pickled BM25 index from disk."""
        self._index, self._corpus, self._ids = load_bm25_index(index_dir=index_dir)
        log_info("BM25Retriever loaded | docs=%d", len(self._ids))

    def search(self, query: str, top_k: int) -> list[dict]:
        """
        Score all documents for ``query`` and return the top-k with
        positive BM25 scores.

        Returns
        -------
        list[dict]  Each item: {id, score, rank, source="bm25"}
        """
        if self._index is None:
            raise IndexNotFoundError()

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._index.get_scores(tokens)

        # Rank by score descending, keep only positive scores
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = []
        for rank, (idx, score) in enumerate(ranked):
            if score <= 0.0:
                break
            results.append(
                {
                    "id": self._ids[idx],
                    "score": float(score),
                    "rank": rank,
                    "payload": {},      # payload populated later from Qdrant or cache
                    "source": "bm25",
                }
            )

        logger.debug("BM25 search | query='%s' top_k=%d returned=%d", query, top_k, len(results))
        return results

    @property
    def is_loaded(self) -> bool:
        return self._index is not None

    @property
    def doc_count(self) -> int:
        return len(self._ids)


# ── Module-level singleton + convenience wrappers ─────────────────────────────

_retriever = BM25Retriever()
_payload_map: dict[str, dict] = {}   # incident_id → full Qdrant payload


def load_bm25_retriever(index_dir: Optional[str] = None) -> None:
    """Load the BM25 index into the module-level singleton. Call at startup."""
    _retriever.load(index_dir=index_dir)


def set_bm25_payload_map(payload_map: dict[str, dict]) -> None:
    """
    Register an incident_id → payload lookup built from Qdrant scroll.
    Called once at startup after both BM25 and Qdrant are initialised.
    """
    global _payload_map
    _payload_map = payload_map
    log_info("BM25 payload map populated | entries=%d", len(payload_map))


def bm25_search(query: str, top_k: int) -> list[dict]:
    """Search using the module-level BM25 singleton."""
    results = _retriever.search(query, top_k)
    # Enrich each hit with its Qdrant payload via incident_id lookup
    for r in results:
        if not r.get("payload") and r["id"] in _payload_map:
            r["payload"] = _payload_map[r["id"]]
    return results


def is_bm25_loaded() -> bool:
    """Return True if the BM25 index has been loaded."""
    return _retriever.is_loaded


def add_document_to_bm25(
    incident_id: str,
    search_text: str,
    payload: dict,
    index_dir: str = "data",
) -> None:
    """
    Add a single new document to the live BM25 index without restarting.

    Steps:
      1. Append the raw search_text and incident_id to the in-memory corpus.
      2. Rebuild BM25Okapi from the full updated corpus (fast for < 1000 docs).
      3. Register the payload in _payload_map so BM25 hits get enriched.
      4. Save the updated pickle to disk so the next server restart picks it up.

    Called after an IT team member resolves an escalation ticket and provides
    resolution steps — those steps are immediately searchable.
    """
    if not _retriever.is_loaded:
        log_warning("BM25 index not loaded — skipping live update for %s", incident_id)
        return

    # 1. Append to in-memory corpus
    _retriever._corpus.append(search_text)
    _retriever._ids.append(incident_id)

    # 2. Rebuild index
    tokenized = [tokenize(t) for t in _retriever._corpus]
    _retriever._index = BM25Okapi(tokenized)

    # 3. Register payload
    _payload_map[incident_id] = payload

    # 4. Persist to disk
    try:
        save_bm25_index(
            _retriever._index,
            _retriever._corpus,
            _retriever._ids,
            index_dir=index_dir,
        )
        log_info(
            "BM25 updated live | new_doc=%s total_docs=%d",
            incident_id, len(_retriever._ids),
        )
    except Exception as exc:
        log_warning("BM25 pickle save failed after live update | error=%s", exc)
