"""
Unit tests for the XLSX ingestion preprocessor.
All tests use in-memory DataFrames written to tmp_path — no network calls.
"""

import os
import pytest
import pandas as pd

from src.ingestion.preprocessor import (
    preprocess_xlsx,
    clean_text,
    build_search_text,
)
from src.exceptions.custom_exceptions import (
    EmptyDatasetError,
    InvalidFileFormatError,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_xlsx(tmp_path, rows: list[dict], filename: str = "incidents.xlsx") -> str:
    """Write a list-of-dicts to an XLSX file and return its path."""
    df = pd.DataFrame(rows)
    path = str(tmp_path / filename)
    df.to_excel(path, index=False, engine="openpyxl")
    return path


def _valid_row(**overrides) -> dict:
    base = {
        "Incident ID": "INC-5001",
        "Ticket ID": "TKT-1001",
        "Media Asset": "MediaServer01",
        "Category": "Storage",
        "Incident Details": "Disk Space Alert",
        "Description": "Storage exceeded threshold causing upload failures",
        "Solution": "Archive old media files and expand storage volume",
    }
    base.update(overrides)
    return base


# ── clean_text ────────────────────────────────────────────────────────────────


class TestCleanText:

    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_collapses_internal_spaces(self):
        assert clean_text("too  many   spaces") == "too many spaces"

    def test_returns_empty_for_none(self):
        assert clean_text(None) == ""  # type: ignore[arg-type]

    def test_handles_nan(self):
        import math
        assert clean_text(math.nan) == ""  # type: ignore[arg-type]

    def test_plain_string_unchanged(self):
        assert clean_text("disk space alert") == "disk space alert"


# ── build_search_text ─────────────────────────────────────────────────────────


class TestBuildSearchText:

    def test_combines_title_and_desc(self):
        text = build_search_text("Disk Alert", "Storage exceeded threshold")
        assert text == "Disk Alert: Storage exceeded threshold"

    def test_title_only(self):
        assert build_search_text("Disk Alert", "") == "Disk Alert"

    def test_desc_only(self):
        assert build_search_text("", "Storage exceeded") == "Storage exceeded"

    def test_both_empty(self):
        assert build_search_text("", "") == ""


# ── preprocess_xlsx ───────────────────────────────────────────────────────────


class TestPreprocessXlsx:

    def test_loads_valid_file(self, tmp_path):
        path = _make_xlsx(tmp_path, [_valid_row()])
        incidents, skipped = preprocess_xlsx(path)
        assert len(incidents) == 1
        assert len(skipped) == 0

    def test_field_mapping(self, tmp_path):
        path = _make_xlsx(tmp_path, [_valid_row()])
        incidents, _ = preprocess_xlsx(path)
        inc = incidents[0]
        assert inc["incident_id"] == "INC-5001"
        assert inc["ticket_id"] == "TKT-1001"
        assert inc["assigned_to"] == "MediaServer01"
        assert inc["category"] == "Storage"
        assert inc["title"] == "Disk Space Alert"
        assert inc["description"] == "Storage exceeded threshold causing upload failures"
        assert inc["resolution_notes"] == "Archive old media files and expand storage volume"

    def test_search_text_combines_title_and_description(self, tmp_path):
        path = _make_xlsx(tmp_path, [_valid_row()])
        incidents, _ = preprocess_xlsx(path)
        assert "Disk Space Alert" in incidents[0]["search_text"]
        assert "Storage exceeded" in incidents[0]["search_text"]

    def test_multiple_rows(self, tmp_path):
        rows = [
            _valid_row(**{"Incident ID": f"INC-{i}", "Ticket ID": f"TKT-{i}"})
            for i in range(10)
        ]
        path = _make_xlsx(tmp_path, rows)
        incidents, skipped = preprocess_xlsx(path)
        assert len(incidents) == 10
        assert len(skipped) == 0

    def test_skips_row_with_blank_description(self, tmp_path):
        rows = [
            _valid_row(),
            _valid_row(**{"Incident ID": "INC-9999", "Description": ""}),
        ]
        path = _make_xlsx(tmp_path, rows)
        incidents, skipped = preprocess_xlsx(path)
        assert len(incidents) == 1
        assert len(skipped) == 1

    def test_skips_row_with_blank_incident_id(self, tmp_path):
        rows = [
            _valid_row(),
            _valid_row(**{"Incident ID": ""}),
        ]
        path = _make_xlsx(tmp_path, rows)
        incidents, skipped = preprocess_xlsx(path)
        assert len(incidents) == 1
        assert len(skipped) == 1

    def test_skips_row_with_blank_solution(self, tmp_path):
        rows = [
            _valid_row(),
            _valid_row(**{"Incident ID": "INC-9999", "Solution": ""}),
        ]
        path = _make_xlsx(tmp_path, rows)
        incidents, skipped = preprocess_xlsx(path)
        assert len(incidents) == 1
        assert len(skipped) == 1

    def test_raises_for_csv_extension(self, tmp_path):
        with pytest.raises(InvalidFileFormatError):
            preprocess_xlsx(str(tmp_path / "data.csv"))

    def test_raises_for_empty_dataframe(self, tmp_path):
        """All rows skipped → EmptyDatasetError."""
        rows = [
            _valid_row(**{"Incident ID": "", "Description": ""}),
        ]
        path = _make_xlsx(tmp_path, rows)
        with pytest.raises(EmptyDatasetError):
            preprocess_xlsx(path)

    def test_raises_for_missing_required_column(self, tmp_path):
        """XLSX missing 'Description' column → InvalidFileFormatError."""
        df = pd.DataFrame([{"Incident ID": "INC-1", "Solution": "fix"}])
        path = str(tmp_path / "bad.xlsx")
        df.to_excel(path, index=False, engine="openpyxl")
        with pytest.raises(InvalidFileFormatError):
            preprocess_xlsx(path)

    def test_loads_real_dataset(self):
        """Smoke-test against the actual ITSM data file."""
        real_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "data", "incidents.xlsx",
        )
        real_path = os.path.normpath(real_path)
        if not os.path.exists(real_path):
            pytest.skip("Real XLSX not found at data/incidents.xlsx")
        incidents, skipped = preprocess_xlsx(real_path)
        assert len(incidents) > 0
        assert all("incident_id" in inc for inc in incidents)
        assert all("search_text" in inc for inc in incidents)
        assert all(len(inc["search_text"]) > 0 for inc in incidents)
