"""
Unit tests for the custom exception hierarchy.
Verifies correct status codes, messages, and repr output.
"""

import pytest

from src.exceptions.custom_exceptions import (
    IncidentKBException,
    IngestionError,
    InvalidFileFormatError,
    EmptyDatasetError,
    RetrievalError,
    VectorDBUnavailableError,
    IndexNotFoundError,
    AgentError,
    EscalationError,
    LLMError,
    LLMTimeoutError,
    EmbeddingError,
    ConfigurationError,
    EvaluationError,
)


class TestBaseException:

    def test_default_status_code(self):
        exc = IncidentKBException("something broke")
        assert exc.status_code == 500
        assert exc.message == "something broke"
        assert exc.details == {}

    def test_custom_status_and_details(self):
        exc = IncidentKBException("not found", status_code=404, details={"id": "INC001"})
        assert exc.status_code == 404
        assert exc.details["id"] == "INC001"

    def test_is_exception(self):
        exc = IncidentKBException("test")
        assert isinstance(exc, Exception)

    def test_repr_contains_class_name(self):
        exc = IncidentKBException("test")
        assert "IncidentKBException" in repr(exc)


class TestIngestionExceptions:

    def test_ingestion_error_status_422(self):
        exc = IngestionError("bad csv file")
        assert exc.status_code == 422

    def test_invalid_file_format(self):
        exc = InvalidFileFormatError("report.xlsx")
        assert exc.status_code == 422
        assert "report.xlsx" in exc.message
        assert exc.details["filename"] == "report.xlsx"

    def test_empty_dataset_error(self):
        exc = EmptyDatasetError()
        assert exc.status_code == 422
        assert "empty" in exc.message.lower()

    def test_ingestion_inherits_from_base(self):
        exc = IngestionError("test")
        assert isinstance(exc, IncidentKBException)


class TestRetrievalExceptions:

    def test_retrieval_error_status_503(self):
        exc = RetrievalError("search failed")
        assert exc.status_code == 503

    def test_vector_db_unavailable(self):
        exc = VectorDBUnavailableError(reason="connection refused")
        assert exc.status_code == 503
        assert "vector database" in exc.message.lower()
        assert exc.details["reason"] == "connection refused"

    def test_index_not_found_status_409(self):
        exc = IndexNotFoundError()
        assert exc.status_code == 409
        assert "ingest" in exc.message.lower()


class TestAgentExceptions:

    def test_agent_error_status_500(self):
        exc = AgentError("graph failed")
        assert exc.status_code == 500

    def test_escalation_error(self):
        exc = EscalationError(reason="DB write failed")
        assert "escalation" in exc.message.lower()
        assert exc.details["reason"] == "DB write failed"


class TestLLMExceptions:

    def test_llm_error_status_503(self):
        exc = LLMError("both models failed")
        assert exc.status_code == 503

    def test_llm_timeout_with_model(self):
        exc = LLMTimeoutError(model="gpt-4o")
        assert "gpt-4o" in exc.message
        assert exc.details["model"] == "gpt-4o"

    def test_llm_timeout_without_model(self):
        exc = LLMTimeoutError()
        assert exc.status_code == 503

    def test_embedding_error(self):
        exc = EmbeddingError(reason="API rate limit")
        assert "embedding" in exc.message.lower()


class TestConfigurationException:

    def test_missing_key(self):
        exc = ConfigurationError("OPENAI_API_KEY")
        assert "OPENAI_API_KEY" in exc.message
        assert exc.status_code == 500
        assert exc.details["missing_key"] == "OPENAI_API_KEY"


class TestExceptionHierarchy:
    """Verify isinstance checks work correctly for except clause catching."""

    def test_ingestion_caught_as_base(self):
        with pytest.raises(IncidentKBException):
            raise IngestionError("test")

    def test_retrieval_caught_as_base(self):
        with pytest.raises(IncidentKBException):
            raise RetrievalError("test")

    def test_llm_error_caught_as_base(self):
        with pytest.raises(IncidentKBException):
            raise LLMTimeoutError()

    def test_invalid_file_caught_as_ingestion(self):
        with pytest.raises(IngestionError):
            raise InvalidFileFormatError("bad.xlsx")
