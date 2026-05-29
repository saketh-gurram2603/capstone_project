"""
PII masking utility for incident text fields.

Applies regex-based redaction before data is embedded or stored.
Order of patterns matters — EMAIL must precede IP/generic numbers,
SSN must precede CREDIT_CARD.
"""

from __future__ import annotations

import re

# ── Ordered masking patterns ──────────────────────────────────────────────────
# Each tuple: (replacement_token, compiled_pattern)
_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "[EMAIL]",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "[SSN]",
        re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"),
    ),
    (
        "[CREDIT_CARD]",
        re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"),
    ),
    (
        "[IP_ADDRESS]",
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ),
    (
        "[PHONE]",
        re.compile(r"\b(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"),
    ),
]


# ── Public API ────────────────────────────────────────────────────────────────


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """
    Scan *text* for PII patterns and replace each match with a placeholder.

    Returns
    -------
    masked_text : str
        The text with all PII occurrences replaced.
    counts : dict[str, int]
        Per-token replacement count, e.g. {"EMAIL": 2, "IP_ADDRESS": 1, ...}.
        Zero-count tokens are included for a complete audit trail.
    """
    if not text:
        return text, {token: 0 for token, _ in _PATTERNS}

    counts: dict[str, int] = {}
    for token, pattern in _PATTERNS:
        matches = pattern.findall(text)
        counts[token] = len(matches)
        if matches:
            text = pattern.sub(token, text)

    return text, counts


def summarize_masking(field_summaries: dict[str, dict[str, int]]) -> dict:
    """
    Aggregate per-field PII counts into a top-level summary.

    Parameters
    ----------
    field_summaries : dict mapping field_name → counts_dict from mask_pii()

    Returns
    -------
    {
      "total_masked": int,
      "fields_affected": list[str],   # only fields where total > 0
      "by_type": {"EMAIL": int, ...}
    }
    """
    by_type: dict[str, int] = {}
    fields_affected: list[str] = []

    for field, counts in field_summaries.items():
        field_total = sum(counts.values())
        if field_total > 0:
            fields_affected.append(field)
        for token, count in counts.items():
            by_type[token] = by_type.get(token, 0) + count

    return {
        "total_masked": sum(by_type.values()),
        "fields_affected": fields_affected,
        "by_type": by_type,
    }
