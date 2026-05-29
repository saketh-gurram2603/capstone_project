"""
BM25 index builder and persistence layer.

The BM25Okapi index is built over the ``search_text`` field of each incident
(title + description combined).  The index is pickled to disk so the retrieval
layer can load it at startup without re-ingesting.

Saved artefact layout (pickle dict):
  {
    "index"  : BM25Okapi,           # rank-bm25 model
    "corpus" : list[str],           # raw search_text per doc (for debug / reranking)
    "ids"    : list[str],           # incident_id per doc — positional alignment
  }
"""

from __future__ import annotations

import os
import pickle
import re
from typing import Optional

from rank_bm25 import BM25Okapi

from src.exceptions.custom_exceptions import IndexNotFoundError
from src.handlers.logger import get_logger, log_info, log_warning

logger = get_logger("ingestion.bm25_builder")

# Default persistence path — override via BM25_INDEX_DIR env var
_DEFAULT_INDEX_DIR = os.environ.get("BM25_INDEX_DIR", "data")
_INDEX_FILENAME = "bm25_index.pkl"


# ── Public API ─────────────────────────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokeniser.
    Lower-cases text, strips punctuation, splits on whitespace.
    Keeps numeric tokens (e.g. "502", "cpu_100") intact.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # replace punctuation with space
    return text.split()


def build_bm25_index(
    incidents: list[dict],
) -> tuple[BM25Okapi, list[str], list[str]]:
    """
    Build a BM25Okapi index from preprocessed incident records.

    Parameters
    ----------
    incidents : list[dict]
        Each dict must have ``incident_id`` and ``search_text`` keys.

    Returns
    -------
    index  : BM25Okapi
    corpus : list[str]   raw search_text strings (positional)
    ids    : list[str]   incident_id strings (positional)
    """
    if not incidents:
        raise ValueError("Cannot build BM25 index from empty incident list.")

    ids: list[str] = []
    corpus: list[str] = []
    tokenized: list[list[str]] = []

    for inc in incidents:
        ids.append(inc["incident_id"])
        text = inc.get("search_text") or inc.get("description", "")
        corpus.append(text)
        tokenized.append(tokenize(text))

    index = BM25Okapi(tokenized)
    log_info("BM25 index built | docs=%d", len(ids))
    return index, corpus, ids


def save_bm25_index(
    index: BM25Okapi,
    corpus: list[str],
    ids: list[str],
    index_dir: Optional[str] = None,
) -> str:
    """
    Pickle the BM25 artefact to disk.

    Returns
    -------
    str  Absolute path of the saved file.
    """
    dir_path = index_dir or _DEFAULT_INDEX_DIR
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, _INDEX_FILENAME)

    payload = {"index": index, "corpus": corpus, "ids": ids}
    with open(file_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    log_info("BM25 index saved | path=%s | docs=%d", file_path, len(ids))
    return file_path


def load_bm25_index(
    index_dir: Optional[str] = None,
) -> tuple[BM25Okapi, list[str], list[str]]:
    """
    Load the pickled BM25 artefact from disk.

    Raises
    ------
    IndexNotFoundError  if the pickle file does not exist.
    """
    dir_path = index_dir or _DEFAULT_INDEX_DIR
    file_path = os.path.join(dir_path, _INDEX_FILENAME)

    if not os.path.exists(file_path):
        raise IndexNotFoundError()

    with open(file_path, "rb") as fh:
        data = pickle.load(fh)

    index: BM25Okapi = data["index"]
    corpus: list[str] = data["corpus"]
    ids: list[str] = data["ids"]

    log_info("BM25 index loaded | path=%s | docs=%d", file_path, len(ids))
    return index, corpus, ids


def bm25_index_exists(index_dir: Optional[str] = None) -> bool:
    """Return True if the BM25 pickle file exists on disk."""
    dir_path = index_dir or _DEFAULT_INDEX_DIR
    return os.path.exists(os.path.join(dir_path, _INDEX_FILENAME))
