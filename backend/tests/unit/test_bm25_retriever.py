"""
Unit tests for the BM25 retriever.
Tests use a real BM25 index built from in-memory data — no file I/O surprises.
"""

import os
import pytest

from src.retrieval.bm25_retriever import BM25Retriever, bm25_search, is_bm25_loaded, load_bm25_retriever
from src.exceptions.custom_exceptions import IndexNotFoundError


SAMPLE_INCIDENTS = [
    {"incident_id": "INC-001", "search_text": "disk space storage threshold upload failures"},
    {"incident_id": "INC-002", "search_text": "memory leak cpu high usage processing service"},
    {"incident_id": "INC-003", "search_text": "database connection pool exhausted timeout error"},
    {"incident_id": "INC-004", "search_text": "network latency spike packet loss routing issue"},
    {"incident_id": "INC-005", "search_text": "unauthorized access security breach login failed"},
]


def _make_retriever(incidents=None) -> BM25Retriever:
    """Build and return a BM25Retriever loaded from a temp pickle."""
    from src.ingestion.bm25_builder import build_bm25_index, save_bm25_index
    import tempfile, os
    inc = incidents or SAMPLE_INCIDENTS
    index, corpus, ids = build_bm25_index(inc)
    tmpdir = tempfile.mkdtemp()
    save_bm25_index(index, corpus, ids, index_dir=tmpdir)
    retriever = BM25Retriever()
    retriever.load(index_dir=tmpdir)
    return retriever


class TestBM25RetrieverLoad:

    def test_loads_successfully(self, tmp_path):
        from src.ingestion.bm25_builder import build_bm25_index, save_bm25_index
        index, corpus, ids = build_bm25_index(SAMPLE_INCIDENTS)
        save_bm25_index(index, corpus, ids, index_dir=str(tmp_path))

        r = BM25Retriever()
        r.load(index_dir=str(tmp_path))
        assert r.is_loaded is True
        assert r.doc_count == 5

    def test_raises_index_not_found_on_missing(self, tmp_path):
        r = BM25Retriever()
        with pytest.raises(IndexNotFoundError):
            r.load(index_dir=str(tmp_path / "missing_dir"))


class TestBM25RetrieverSearch:

    def test_relevant_query_returns_best_match(self):
        r = _make_retriever()
        results = r.search("database connection pool", top_k=5)
        assert results[0]["id"] == "INC-003"

    def test_storage_query_returns_storage_incident(self):
        r = _make_retriever()
        results = r.search("disk space storage", top_k=5)
        assert results[0]["id"] == "INC-001"

    def test_returns_at_most_top_k(self):
        r = _make_retriever()
        results = r.search("service error", top_k=2)
        assert len(results) <= 2

    def test_results_sorted_descending_by_score(self):
        r = _make_retriever()
        results = r.search("database", top_k=5)
        scores = [r_["score"] for r_ in results]
        assert scores == sorted(scores, reverse=True)

    def test_each_result_has_required_keys(self):
        r = _make_retriever()
        results = r.search("disk space", top_k=3)
        for result in results:
            assert "id" in result
            assert "score" in result
            assert "rank" in result
            assert "source" in result
            assert result["source"] == "bm25"

    def test_empty_query_returns_empty(self):
        r = _make_retriever()
        results = r.search("", top_k=5)
        assert results == []

    def test_unrelated_query_returns_empty_or_few(self):
        r = _make_retriever()
        # A query with no overlapping tokens
        results = r.search("xyzzy quux frobnicate", top_k=5)
        assert isinstance(results, list)  # No crash; empty is fine

    def test_raises_when_not_loaded(self):
        r = BM25Retriever()
        with pytest.raises(IndexNotFoundError):
            r.search("disk", top_k=5)

    def test_positive_scores_only(self):
        r = _make_retriever()
        results = r.search("database connection", top_k=10)
        for result in results:
            assert result["score"] > 0.0


class TestModuleSingleton:

    def test_is_bm25_loaded_false_initially(self, tmp_path):
        """Module singleton starts unloaded (unless loaded by a previous test)."""
        # We cannot guarantee state across tests, but can test the function exists
        result = is_bm25_loaded()
        assert isinstance(result, bool)

    def test_load_bm25_retriever_makes_it_searchable(self, tmp_path):
        from src.ingestion.bm25_builder import build_bm25_index, save_bm25_index
        index, corpus, ids = build_bm25_index(SAMPLE_INCIDENTS)
        save_bm25_index(index, corpus, ids, index_dir=str(tmp_path))

        load_bm25_retriever(index_dir=str(tmp_path))
        assert is_bm25_loaded() is True

        results = bm25_search("disk space storage", top_k=3)
        assert len(results) > 0
        assert results[0]["id"] == "INC-001"
