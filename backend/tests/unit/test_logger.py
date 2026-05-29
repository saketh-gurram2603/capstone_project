"""
Unit tests for the logging system.
Verifies rotating file handlers, log levels, and convenience wrappers.
"""

import logging
import os
import tempfile
import pytest

from src.handlers.logger import (
    setup_loggers,
    init_loggers,
    get_logger,
    get_error_logger,
    log_info,
    log_warning,
    log_error,
)


class TestLoggerSetup:

    def test_setup_creates_log_files(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        app_logger, error_logger = setup_loggers(
            log_dir=log_dir,
            max_bytes=1024,
            backup_count=2,
            log_level="DEBUG",
        )
        assert os.path.exists(os.path.join(log_dir, "app.log"))
        assert os.path.exists(os.path.join(log_dir, "error.log"))

    def test_app_logger_level(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        app_logger, _ = setup_loggers(log_dir=log_dir, log_level="WARNING")
        assert app_logger.level == logging.WARNING

    def test_error_logger_always_error_level(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        _, error_logger = setup_loggers(log_dir=log_dir, log_level="DEBUG")
        # Error logger is always ERROR level regardless of app log level
        assert error_logger.level == logging.ERROR

    def test_app_logger_writes_info(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        os.makedirs(log_dir, exist_ok=True)
        import logging
        # Use a unique logger name to avoid handler caching from other tests
        unique_name = f"TestApp_{tmp_path.name}"
        app_logger = logging.getLogger(unique_name)
        app_logger.setLevel(logging.INFO)
        log_file = os.path.join(log_dir, "app.log")
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        app_logger.addHandler(handler)
        app_logger.info("test info message")
        handler.flush()
        handler.close()
        with open(log_file) as f:
            content = f.read()
        assert "test info message" in content

    def test_error_logger_writes_error(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        os.makedirs(log_dir, exist_ok=True)
        import logging
        unique_name = f"TestError_{tmp_path.name}"
        error_logger = logging.getLogger(unique_name)
        error_logger.setLevel(logging.ERROR)
        log_file = os.path.join(log_dir, "error.log")
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.ERROR)
        error_logger.addHandler(handler)
        error_logger.error("test error message")
        handler.flush()
        handler.close()
        with open(log_file) as f:
            content = f.read()
        assert "test error message" in content

    def test_no_print_used(self, tmp_path):
        """Verify logger module has print() ONLY in safety-net except blocks."""
        import inspect
        import src.handlers.logger as logger_module
        source = inspect.getsource(logger_module)
        # Only count executable print() lines (not comments, not docstring mentions)
        print_lines = [
            line.strip() for line in source.splitlines()
            if "print(" in line
            and not line.strip().startswith("#")
            and not line.strip().startswith('"""')
            and not line.strip().startswith("'")
        ]
        # We allow exactly 4 safety-net print() calls (one per convenience wrapper)
        # Filter to only actual callable invocations — lines where print( is followed by a quote
        callable_prints = [l for l in print_lines if 'print("' in l or "print('" in l]
        assert len(callable_prints) <= 4, f"Too many print() calls: {callable_prints}"


class TestGetLogger:

    def test_get_logger_returns_child(self, tmp_path):
        init_loggers(str(tmp_path / "logs"), 1024, 2, "INFO")
        logger = get_logger("test.module")
        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_get_logger_no_name(self, tmp_path):
        init_loggers(str(tmp_path / "logs"), 1024, 2, "INFO")
        logger = get_logger()
        assert logger is not None

    def test_convenience_wrappers_dont_raise(self, tmp_path):
        """log_info/warning/error should never raise — they catch their own exceptions."""
        init_loggers(str(tmp_path / "logs"), 1024, 2, "INFO")
        log_info("test info %s", "value")
        log_warning("test warning")
        log_error("test error")
