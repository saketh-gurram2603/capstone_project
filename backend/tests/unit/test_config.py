"""
Unit tests for the configuration loader.
Verifies JSON loading, env fallback, require_env, and get_env.
"""

import json
import os
import pytest
from unittest.mock import patch

from src.core.config import load_app_config, load_env_config, require_env, get_env
from src.exceptions.custom_exceptions import ConfigurationError


class TestLoadAppConfig:

    def test_loads_successfully(self):
        cfg = load_app_config()
        assert "APP_NAME" in cfg
        assert "RETRIEVAL" in cfg
        assert "LLM" in cfg
        assert "QDRANT" in cfg

    def test_retrieval_keys_present(self):
        cfg = load_app_config()
        r = cfg["RETRIEVAL"]
        assert "K_MIN" in r
        assert "K_MAX" in r
        assert "RRF_K" in r
        assert "L1_CONFIDENCE_THRESHOLD" in r

    def test_k_min_less_than_k_max(self):
        cfg = load_app_config()
        assert cfg["RETRIEVAL"]["K_MIN"] < cfg["RETRIEVAL"]["K_MAX"]

    def test_l1_confidence_threshold_in_range(self):
        cfg = load_app_config()
        t = cfg["RETRIEVAL"]["L1_CONFIDENCE_THRESHOLD"]
        assert 0.0 < t < 1.0

    def test_llm_models_present(self):
        cfg = load_app_config()
        assert cfg["LLM"]["L1_MODEL"] == "gpt-4o-mini"
        assert cfg["LLM"]["L2_MODEL"] == "gpt-4o"
        assert cfg["LLM"]["EMBEDDING_MODEL"] == "text-embedding-ada-002"


class TestLoadEnvConfig:

    def test_loads_development(self):
        cfg = load_env_config("development")
        assert "qdrant_url" in cfg
        assert "redis_url" in cfg
        assert "postgres_host" in cfg

    def test_loads_docker(self):
        cfg = load_env_config("docker")
        assert cfg["qdrant_url"] == "http://qdrant:6333"
        assert cfg["redis_url"] == "redis://redis:6379"

    def test_unknown_env_falls_back_to_development(self):
        cfg = load_env_config("nonexistent_env")
        # Should not raise — falls back gracefully
        assert "qdrant_url" in cfg

    def test_development_debug_true(self):
        cfg = load_env_config("development")
        assert cfg["debug"] is True

    def test_production_debug_false(self):
        cfg = load_env_config("production")
        assert cfg["debug"] is False


class TestRequireEnv:

    def test_returns_value_when_set(self):
        with patch.dict(os.environ, {"TEST_KEY": "test_value"}):
            assert require_env("TEST_KEY") == "test_value"

    def test_raises_when_missing(self):
        key = "DEFINITELY_NOT_SET_XYZ_123"
        os.environ.pop(key, None)
        with pytest.raises(ConfigurationError) as exc_info:
            require_env(key)
        assert key in exc_info.value.message

    def test_raises_when_empty_string(self):
        with patch.dict(os.environ, {"EMPTY_KEY": ""}):
            with pytest.raises(ConfigurationError):
                require_env("EMPTY_KEY")


class TestGetEnv:

    def test_returns_value_when_set(self):
        with patch.dict(os.environ, {"SOME_KEY": "hello"}):
            assert get_env("SOME_KEY") == "hello"

    def test_returns_default_when_missing(self):
        key = "DEFINITELY_NOT_SET_ABC_456"
        os.environ.pop(key, None)
        assert get_env(key, "default_val") == "default_val"

    def test_returns_empty_string_as_default(self):
        key = "DEFINITELY_NOT_SET_ABC_789"
        os.environ.pop(key, None)
        assert get_env(key) == ""
