"""
Ingestion preprocessor — XLSX parsing and field normalisation.

Column mapping from ITSM dataset:
  Incident ID      → incident_id  (e.g. "INC-5001")
  Ticket ID        → ticket_id    (e.g. "TKT-1001")
  Media Asset      → asset        (e.g. "MediaServer01")
  Category         → category     (e.g. "Storage", "Database", ...)
  Incident Details → title        (short description headline)
  Description      → description  (full incident description)
  Solution         → resolution_notes

The combined search text (used for both BM25 and vector embedding) is:
  "{title}: {description}"
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

import pandas as pd

from src.exceptions.custom_exceptions import EmptyDatasetError, InvalidFileFormatError
from src.handlers.logger import get_logger, log_info, log_warning
from src.ingestion.pii_masker import mask_pii, summarize_masking

logger = get_logger("ingestion.preprocessor")

# ── Column name constants (match the XLSX headers exactly) ────────────────────

_COL_INCIDENT_ID = "Incident ID"
_COL_TICKET_ID = "Ticket ID"
_COL_ASSET = "Media Asset"
_COL_CATEGORY = "Category"
_COL_TITLE = "Incident Details"
_COL_DESCRIPTION = "Description"
_COL_SOLUTION = "Solution"

_REQUIRED_COLUMNS: set[str] = {
    _COL_INCIDENT_ID,
    _COL_DESCRIPTION,
    _COL_SOLUTION,
}


# ── Public API ─────────────────────────────────────────────────────────────────


def preprocess_xlsx(
    file_path: str,
) -> tuple[list[dict], list[str]]:
    """
    Parse an XLSX file and return cleaned incident records.

    Returns
    -------
    incidents : list[dict]
        Cleaned incident records ready for embedding and Qdrant upsert.
        Each dict carries all payload fields plus a ``search_text`` key.
    skipped : list[str]
        Row identifiers (or indices) that were skipped due to missing/blank
        required fields.
    """
    if not str(file_path).lower().endswith((".xlsx", ".xls")):
        raise InvalidFileFormatError(str(file_path))

    try:
        df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as exc:
        raise InvalidFileFormatError(str(file_path)) from exc

    if df.empty:
        raise EmptyDatasetError()

    _validate_columns(df, file_path)

    # Normalise column names: strip whitespace
    df.columns = [c.strip() for c in df.columns]

    incidents: list[dict] = []
    skipped: list[str] = []
    seen_hashes: set[str] = set()
    duplicate_count = 0

    for idx, row in df.iterrows():
        label = f"row_{idx}"
        incident = _process_row(row, label, skipped)
        if incident is None:
            continue

        # ── (problem, solution) dedup ────────────────────────────────────────
        # A row is a true duplicate only when BOTH the problem (title +
        # description) AND the solution (resolution_notes) match a previously
        # seen row.  Same problem with a different fix is NOT a duplicate —
        # those rows are valuable because they give the resolution_aggregator
        # multiple proven solutions to surface for one symptom.
        content_hash = _content_hash(
            incident["title"],
            incident["description"],
            incident["resolution_notes"],
        )
        if content_hash in seen_hashes:
            log_warning(
                "Skipping %s — duplicate (problem + solution) (id=%s, title=%s)",
                label, incident["incident_id"], incident["title"][:60],
            )
            skipped.append(label)
            duplicate_count += 1
            continue
        seen_hashes.add(content_hash)

        incidents.append(incident)

    if not incidents:
        raise EmptyDatasetError()

    log_info(
        "Preprocessor completed | parsed=%d skipped=%d (incl. %d duplicates) file=%s",
        len(incidents),
        len(skipped),
        duplicate_count,
        file_path,
    )
    return incidents, skipped


def clean_text(text) -> str:
    """
    Normalise text: strip outer whitespace, collapse internal runs to single space.
    Returns empty string for None, NaN, or non-string input (e.g. pandas NaN floats).
    """
    if text is None:
        return ""
    # Treat pandas NaN (float) as empty
    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_search_text(title: str, description: str) -> str:
    """Combine title and description into a single search-optimised string."""
    parts = [p for p in (title, description) if p]
    return ": ".join(parts) if len(parts) > 1 else (parts[0] if parts else "")


# ── Severity enrichment ─────────────────────────────────────────────────────
# The MediaServer dataset carries no impact/urgency/priority columns, so the
# /search priority & impact filters would match nothing. We derive these fields
# at ingest from the incident text + category using an explainable heuristic, so
# metadata filtering (priority/impact/category) is genuinely functional.

_HIGH_IMPACT_CATEGORIES = {"Security", "Database", "Network"}
_MED_IMPACT_CATEGORIES = {"Storage", "Application", "Hardware", "Performance"}

_HIGH_SEVERITY_KEYWORDS = (
    "crash", "outage", "down", "unavailable", "fail", "failure", "cannot",
    "unable", "breach", "corrupt", "data loss", "exceeded", "threshold",
    "freeze", "frozen", "unresponsive", "denied", "critical", "not working",
    "timeout", "timed out", "lost",
)
_MED_SEVERITY_KEYWORDS = (
    "slow", "slowly", "delay", "delayed", "intermittent", "degrad", "latency",
    "partial", "lag", "spike", "high cpu", "high memory", "retry", "warning",
)


def derive_severity(category: str, title: str, description: str) -> tuple[str, str, str]:
    """
    Derive (impact, urgency, priority) for an incident.

    urgency : from symptom keywords in the title + description.
    impact  : from the category, escalated to High when high-severity symptoms
              are present.
    priority: impact × urgency matrix (same mapping as agents.classify_priority).
    """
    text = f"{title} {description}".lower()

    if any(k in text for k in _HIGH_SEVERITY_KEYWORDS):
        urgency = "High"
    elif any(k in text for k in _MED_SEVERITY_KEYWORDS):
        urgency = "Medium"
    else:
        urgency = "Low"

    cat = (category or "").strip()
    if urgency == "High" or cat in _HIGH_IMPACT_CATEGORIES:
        impact = "High"
    elif cat in _MED_IMPACT_CATEGORIES:
        impact = "Medium"
    else:
        impact = "Low"

    priority = _priority_from(impact, urgency)
    return impact, urgency, priority


def _priority_from(impact: str, urgency: str) -> str:
    """Map impact × urgency to P1–P4 (additive matrix)."""
    score = {"High": 3, "Medium": 2, "Low": 1}
    total = score.get(impact, 2) + score.get(urgency, 2)
    if total >= 6:
        return "P1"
    if total >= 5:
        return "P2"
    if total >= 3:
        return "P3"
    return "P4"


def _content_hash(title: str, description: str, resolution: str) -> str:
    """
    Fingerprint an incident by its (problem, solution) pair:
      hash = md5(title + description + resolution_notes)

    Two incidents are treated as duplicates ONLY when BOTH the problem and
    the solution match.  This deliberately preserves:
      • Same problem, different solutions  → kept as separate records, so the
        resolution_aggregator can surface multiple proven fixes downstream.
      • Different problems, same solution  → kept as separate records.
      • Same problem, same solution        → collapsed (true duplicate row).
    """
    blob = (
        f"{(title or '').lower().strip()}|"
        f"{(description or '').lower().strip()}|"
        f"{(resolution or '').lower().strip()}"
    )
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


# ── Private helpers ───────────────────────────────────────────────────────────


def _validate_columns(df: pd.DataFrame, file_path: str) -> None:
    """Raise InvalidFileFormatError if any required column is absent."""
    actual = set(c.strip() for c in df.columns)
    missing = _REQUIRED_COLUMNS - actual
    if missing:
        raise InvalidFileFormatError(
            f"{file_path} (missing columns: {sorted(missing)})"
        )


def _process_row(
    row: "pd.Series",
    label: str,
    skipped: list[str],
) -> Optional[dict]:
    """
    Map a DataFrame row to a cleaned incident dict.
    Returns None (and appends to skipped) if required fields are blank.
    """
    incident_id = clean_text(row.get(_COL_INCIDENT_ID, ""))
    description = clean_text(row.get(_COL_DESCRIPTION, ""))
    resolution_notes = clean_text(row.get(_COL_SOLUTION, ""))

    # Required fields must be non-empty
    if not incident_id or not description or not resolution_notes:
        log_warning("Skipping %s — missing required field(s)", label)
        skipped.append(label)
        return None

    title = clean_text(row.get(_COL_TITLE, ""))
    ticket_id = clean_text(row.get(_COL_TICKET_ID, ""))
    asset = clean_text(row.get(_COL_ASSET, ""))
    category = clean_text(row.get(_COL_CATEGORY, ""))

    # ── PII masking ───────────────────────────────────────────────────────────
    description, desc_counts = mask_pii(description)
    resolution_notes, res_counts = mask_pii(resolution_notes)
    title, title_counts = mask_pii(title)

    pii_summary = summarize_masking({
        "description": desc_counts,
        "resolution_notes": res_counts,
        "title": title_counts,
    })
    pii_masked_count = pii_summary["total_masked"]
    if pii_masked_count > 0:
        log_info(
            "PII masked in %s | total=%d fields=%s by_type=%s",
            label,
            pii_masked_count,
            pii_summary["fields_affected"],
            pii_summary["by_type"],
        )

    search_text = build_search_text(title, description)

    # Derive impact/urgency/priority so metadata filtering is functional
    # (the source dataset has no such columns).
    impact, urgency, priority = derive_severity(category, title, description)

    return {
        "incident_id": incident_id,
        "ticket_id": ticket_id,
        "title": title,
        "category": category,
        "impact": impact,
        "urgency": urgency,
        "priority": priority,
        "description": description,
        "resolution_notes": resolution_notes,
        "assigned_to": asset,
        "search_text": search_text,
        "pii_masked_count": pii_masked_count,
    }
