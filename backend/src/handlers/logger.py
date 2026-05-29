"""
Logging configuration — mirrors Synapt-PersonalizedRAG-API pattern.
Two rotating file loggers:
  - app logger  : INFO + WARNING  → logs/app.log
  - error logger: ERROR + CRITICAL → logs/error.log
No print() statements anywhere in the codebase — use these loggers only.
"""

import logging
import os
import traceback
from logging.handlers import RotatingFileHandler

# ── Constants ────────────────────────────────────────────────────────────────
_APP_LOGGER_NAME = "IncidentKBApp"
_ERROR_LOGGER_NAME = "IncidentKBError"

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _build_rotating_handler(
    filepath: str,
    max_bytes: int,
    backup_count: int,
    level: int,
    log_filter: logging.Filter | None = None,
) -> RotatingFileHandler:
    """Create a rotating file handler with optional filter."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    handler = RotatingFileHandler(
        filepath,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    if log_filter:
        handler.addFilter(log_filter)
    return handler


def _build_console_handler(level: int) -> logging.StreamHandler:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    return handler


# ── Logger factory ───────────────────────────────────────────────────────────

def setup_loggers(
    log_dir: str = "logs",
    max_bytes: int = 5_242_880,  # 5 MB
    backup_count: int = 5,
    log_level: str = "INFO",
) -> tuple[logging.Logger, logging.Logger]:
    """
    Initialise and return (app_logger, error_logger).
    Call once at startup inside FastAPI lifespan.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # ── App logger (INFO + WARNING → app.log + console) ──────────────────────
    app_logger = logging.getLogger(_APP_LOGGER_NAME)
    app_logger.setLevel(numeric_level)
    if not app_logger.handlers:
        app_handler = _build_rotating_handler(
            filepath=os.path.join(log_dir, "app.log"),
            max_bytes=max_bytes,
            backup_count=backup_count,
            level=numeric_level,
        )
        app_logger.addHandler(app_handler)
        app_logger.addHandler(_build_console_handler(numeric_level))
    app_logger.propagate = False

    # ── Error logger (ERROR + CRITICAL → error.log) ──────────────────────────
    error_logger = logging.getLogger(_ERROR_LOGGER_NAME)
    error_logger.setLevel(logging.ERROR)
    if not error_logger.handlers:
        error_handler = _build_rotating_handler(
            filepath=os.path.join(log_dir, "error.log"),
            max_bytes=max_bytes,
            backup_count=backup_count,
            level=logging.ERROR,
        )
        error_logger.addHandler(error_handler)
        error_logger.addHandler(_build_console_handler(logging.ERROR))
    error_logger.propagate = False

    return app_logger, error_logger


# ── Module-level convenience accessors ───────────────────────────────────────
# These are populated by setup_loggers() at startup.

_app_logger: logging.Logger | None = None
_error_logger: logging.Logger | None = None


def init_loggers(log_dir: str, max_bytes: int, backup_count: int, log_level: str) -> None:
    """Called once from main.py lifespan."""
    global _app_logger, _error_logger
    _app_logger, _error_logger = setup_loggers(log_dir, max_bytes, backup_count, log_level)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the app logger (or the root app logger if no name given)."""
    base = _app_logger or logging.getLogger(_APP_LOGGER_NAME)
    return base.getChild(name) if name else base


def get_error_logger() -> logging.Logger:
    return _error_logger or logging.getLogger(_ERROR_LOGGER_NAME)


# ── Convenience wrappers (match Synapt style) ────────────────────────────────

def log_info(msg: str, *args, **kwargs) -> None:
    try:
        get_logger().info(msg, *args, **kwargs)
    except Exception:
        print("Logging error (info):", traceback.format_exc())


def log_warning(msg: str, *args, **kwargs) -> None:
    try:
        get_logger().warning(msg, *args, **kwargs)
    except Exception:
        print("Logging error (warning):", traceback.format_exc())


def log_error(msg: str, *args, **kwargs) -> None:
    try:
        get_error_logger().error(msg, *args, **kwargs)
    except Exception:
        print("Logging error (error):", traceback.format_exc())


def log_critical(msg: str, *args, **kwargs) -> None:
    try:
        get_error_logger().critical(msg, *args, **kwargs)
    except Exception:
        print("Logging error (critical):", traceback.format_exc())
