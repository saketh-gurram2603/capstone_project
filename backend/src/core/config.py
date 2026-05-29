"""
Configuration loader — mirrors Synapt-PersonalizedRAG-API pattern.
Loads:
  1. app_config.json  — static application constants
  2. config.json      — per-environment URLs / settings
  3. env/*.env        — secrets via python-dotenv (never hardcoded)
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.exceptions.custom_exceptions import ConfigurationError
from src.handlers.logger import log_info, log_warning

# ── Paths (relative to backend/) ─────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
_CONFIG_DIR = _BASE_DIR / "configuration"


def load_app_config() -> dict:
    """Load static app constants from configuration/app_config.json."""
    path = _CONFIG_DIR / "app_config.json"
    if not path.exists():
        raise ConfigurationError("configuration/app_config.json")
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    log_info("app_config loaded from %s", path)
    return config


def load_env_config(environment: str) -> dict:
    """
    Load environment-specific settings from configuration/config.json.
    Falls back to 'development' if the requested env is not found.
    """
    path = _CONFIG_DIR / "config.json"
    if not path.exists():
        raise ConfigurationError("configuration/config.json")
    with open(path, encoding="utf-8") as f:
        all_configs: dict = json.load(f)

    if environment not in all_configs:
        log_warning(
            "Environment '%s' not found in config.json — falling back to 'development'",
            environment,
        )
        environment = "development"

    env_cfg = all_configs[environment]
    log_info("env_config loaded for environment='%s'", environment)
    return env_cfg


def load_environment(environment: str, env_config: dict) -> None:
    """
    Load secrets from the .env file specified in env_config.
    Secrets are injected into os.environ — access via os.getenv().
    """
    env_file = env_config.get("env_file", f"./env/{environment}.env")
    env_path = _BASE_DIR / env_file

    if not env_path.exists():
        log_warning("Env file '%s' not found — secrets must be set in environment", env_path)
        return

    load_dotenv(dotenv_path=env_path, override=False)
    log_info("Secrets loaded from '%s'", env_path)


def require_env(key: str) -> str:
    """
    Get a required environment variable.
    Raises ConfigurationError if missing — fail fast at startup.
    """
    value = os.getenv(key)
    if not value:
        raise ConfigurationError(key)
    return value


def get_env(key: str, default: str = "") -> str:
    """Get an optional environment variable with a default."""
    return os.getenv(key, default)
