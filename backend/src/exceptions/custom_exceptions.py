"""
Custom exception hierarchy for the Incident Knowledge Base Assistant.
All exceptions carry an HTTP status code so the central handler can
convert them to consistent JSON error responses.
"""


class IncidentKBException(Exception):
    """Base exception for all application errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: dict | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"status_code={self.status_code}, "
            f"details={self.details})"
        )


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestionError(IncidentKBException):
    """Raised when CSV ingestion fails (parsing, embedding, or upsert)."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message, status_code=422, details=details)


class InvalidFileFormatError(IngestionError):
    """Raised when the uploaded file is not a supported format (CSV only)."""

    def __init__(self, filename: str) -> None:
        super().__init__(
            message=f"Unsupported file format: '{filename}'. Only CSV files are accepted.",
            details={"filename": filename},
        )


class EmptyDatasetError(IngestionError):
    """Raised when the CSV file contains no processable rows."""

    def __init__(self) -> None:
        super().__init__(message="The uploaded CSV file is empty or contains no valid rows.")


# ── Retrieval ────────────────────────────────────────────────────────────────

class RetrievalError(IncidentKBException):
    """Raised when hybrid search fails entirely (both BM25 and vector unavailable)."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message, status_code=503, details=details)


class VectorDBUnavailableError(RetrievalError):
    """Raised when Qdrant is unreachable. System falls back to BM25-only."""

    def __init__(self, reason: str = "") -> None:
        super().__init__(
            message="Vector database is currently unavailable. Falling back to keyword search.",
            details={"reason": reason},
        )


class IndexNotFoundError(RetrievalError):
    """Raised when BM25 index has not been built yet (ingest not run)."""

    def __init__(self) -> None:
        # Bypass RetrievalError.__init__ to set a custom 409 status code
        IncidentKBException.__init__(
            self,
            message="Search index not found. Please run /ingest before searching.",
            status_code=409,
        )


# ── Agent / Triage ────────────────────────────────────────────────────────────

class AgentError(IncidentKBException):
    """Raised when the triage agent graph fails to produce a result."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message, status_code=500, details=details)


class EscalationError(AgentError):
    """Raised when L3 escalation ticket creation fails."""

    def __init__(self, reason: str = "") -> None:
        super().__init__(
            message="Failed to create escalation ticket.",
            details={"reason": reason},
        )


# ── LLM / Embeddings ─────────────────────────────────────────────────────────

class LLMError(IncidentKBException):
    """Raised when both primary LLM and local fallback fail."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message, status_code=503, details=details)


class LLMTimeoutError(LLMError):
    """Raised when the LLM API call times out and fallback also fails."""

    def __init__(self, model: str = "") -> None:
        super().__init__(
            message=f"LLM request timed out{f' for model {model}' if model else ''}.",
            details={"model": model},
        )


class EmbeddingError(LLMError):
    """Raised when both ada-002 and local MiniLM embedding generation fail."""

    def __init__(self, reason: str = "") -> None:
        super().__init__(
            message="Embedding generation failed for both primary and fallback models.",
            details={"reason": reason},
        )


# ── Configuration ─────────────────────────────────────────────────────────────

class ConfigurationError(IncidentKBException):
    """Raised when required configuration keys are missing at startup."""

    def __init__(self, missing_key: str) -> None:
        super().__init__(
            message=f"Required configuration key '{missing_key}' is missing.",
            status_code=500,
            details={"missing_key": missing_key},
        )


# ── Evaluation ───────────────────────────────────────────────────────────────

class EvaluationError(IncidentKBException):
    """Raised when the evaluation pipeline fails."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message, status_code=500, details=details)
