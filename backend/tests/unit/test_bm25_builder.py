"""
Unit tests for the BM25 index builder.
All I/O uses tmp_path — no network calls, no shared state.
"""

import os
import pytest

from src.ingestion.bm25_builder import (
    build_bm25_index,
    save_bm25_index,
    load_bm25_index,
    bm25_index_exists,
    tokenize,
)
from src.exceptions.custom_exceptions import IndexNotFoundError


# ── Fixtures ──────────────────────────────────────────────────────────────────


SAMPLE_INCIDENTS = [
    {
        "incident_id": "INC-001",
        "search_text": "disk space storage threshold upload failures",
    },
    {
        "incident_id": "INC-002",
        "search_text": "memory leak cpu high usage processing service",
    },
    {
        "incident_id": "INC-003",
        "search_text": "database connection pool exhausted timeout error",
    },
    {
        "incident_id": "INC-004",
        "search_text": "network latency spike packet loss routing issue",
    },
    {
        "incident_id": "INC-005",
        "search_text": "unauthorized access security breach login failed",
    },
]


# ── tokenize ──────────────────────────────────────────────────────────────────


class TestTokenize:

    def test_lowercases_text(self):
        tokens = tokenize("DISK SPACE Alert")
        assert all(t == t.lower() for t in tokens)

    def test_splits_on_whitespace(self):
        tokens = tokenize("a b c")
        assert tokens == ["a", "b", "c"]

    def test_strips_punctuation(self):
        tokens = tokenize("failed, again! error.")
        assert "failed" in tokens
        assert "again" in tokens
        assert "error" in tokens
        # punctuation-only tokens should not appear
        assert "," not in tokens

    def test_keeps_numeric_tokens(self):
        tokens = tokenize("502 error http timeout")
        assert "502" in tokens

    def test_empty_string(self):
        assert tokenize("") == []


# ── build_bm25_index ──────────────────────────────────────────────────────────


class TestBuildBm25Index:

    def test_builds_without_error(self):
        index, corpus, ids = build_bm25_index(SAMPLE_INCIDENTS)
        assert index is not None
        assert len(corpus) == len(SAMPLE_INCIDENTS)
        assert len(ids) == len(SAMPLE_INCIDENTS)

    def test_ids_match_input(self):
        _, _, ids = build_bm25_index(SAMPLE_INCIDENTS)
        assert ids == [inc["incident_id"] for inc in SAMPLE_INCIDENTS]

    def test_corpus_matches_search_text(self):
        _, corpus, _ = build_bm25_index(SAMPLE_INCIDENTS)
        expected = [inc["search_text"] for inc in SAMPLE_INCIDENTS]
        assert corpus == expected

    def test_relevant_doc_scores_higher(self):
        """'database connection pool' query should rank INC-003 highest."""
        from rank_bm25 import BM25Okapi
        from src.ingestion.bm25_builder import tokenize as tok

        index, corpus, ids = build_bm25_index(SAMPLE_INCIDENTS)
        query_tokens = tok("database connection pool")
        scores = index.get_scores(query_tokens)

        best_idx = int(scores.argmax())
        assert ids[best_idx] == "INC-003"

    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError):
            build_bm25_index([])

    def test_uses_search_text_field(self):
        """Should use 'search_text'; falls back to 'description' if absent."""
        docs = [
            {"incident_id": "INC-A", "description": "fallback text used"},
        ]
        index, corpus, ids = build_bm25_index(docs)
        assert corpus[0] == "fallback text used"


# ── save / load / exists ──────────────────────────────────────────────────────


class TestPersistence:

    def test_save_creates_file(self, tmp_path):
        index, corpus, ids = build_bm25_index(SAMPLE_INCIDENTS)
        path = save_bm25_index(index, corpus, ids, index_dir=str(tmp_path))
        assert os.path.exists(path)

    def test_load_roundtrip(self, tmp_path):
        index, corpus, ids = build_bm25_index(SAMPLE_INCIDENTS)
        save_bm25_index(index, corpus, ids, index_dir=str(tmp_path))

        loaded_index, loaded_corpus, loaded_ids = load_bm25_index(index_dir=str(tmp_path))
        assert loaded_ids == ids
        assert loaded_corpus == corpus
        assert loaded_index is not None

    def test_loaded_index_scores_same(self, tmp_path):
        """Loaded index should produce identical BM25 scores as the original."""
        from src.ingestion.bm25_builder import tokenize as tok

        index, corpus, ids = build_bm25_index(SAMPLE_INCIDENTS)
        save_bm25_index(index, corpus, ids, index_dir=str(tmp_path))
        loaded, _, _ = load_bm25_index(index_dir=str(tmp_path))

        query = tok("disk storage upload")
        original_scores = list(index.get_scores(query))
        loaded_scores = list(loaded.get_scores(query))
        assert original_scores == loaded_scores

    def test_load_raises_when_missing(self, tmp_path):
        with pytest.raises(IndexNotFoundError):
            load_bm25_index(index_dir=str(tmp_path / "nonexistent"))

    def test_bm25_index_exists_true(self, tmp_path):
        index, corpus, ids = build_bm25_index(SAMPLE_INCIDENTS)
        save_bm25_index(index, corpus, ids, index_dir=str(tmp_path))
        assert bm25_index_exists(index_dir=str(tmp_path)) is True

    def test_bm25_index_exists_false(self, tmp_path):
        assert bm25_index_exists(index_dir=str(tmp_path / "empty")) is False
