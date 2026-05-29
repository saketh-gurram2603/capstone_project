"""
Shared pytest fixtures and configuration.
"""

import os
import sys
import pytest

# Ensure backend/ is on the Python path so `from src.xxx` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    """
    Safety net: prevent any real network calls in unit tests.
    Integration tests override this fixture explicitly.
    """
    pass  # Will add blocking monkeypatches when integration tests are written


@pytest.fixture
def sample_incident_data():
    """Reusable valid incident payload for tests."""
    return {
        "number": "INC0010001",
        "state": "Open",
        "impact": "High",
        "urgency": "High",
        "priority": "P1",
        "category": "Database",
        "description": "Database connection pool exhausted under peak load causing 502 errors",
        "resolution_notes": "Increased connection pool size from 10 to 50 and added read replica",
        "assigned_to": "DBA Team",
    }


@pytest.fixture
def sample_search_payload():
    return {
        "query": "database connection timeout error",
        "filters": {"priority": "P1", "impact": "High"},
    }


@pytest.fixture
def sample_triage_payload():
    return {
        "description": "Service keeps throwing OutOfMemoryError every 30 minutes under load",
        "impact": "High",
        "urgency": "High",
    }
