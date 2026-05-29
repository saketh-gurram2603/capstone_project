"""
LLM integration with:
  - Circuit breaker (pybreaker): opens after 5 failures in 60 s
  - Retry logic    (tenacity):   3 attempts, exponential back-off 2s → 4s → 8s
  - Local fallback (Flan-T5):   loaded at startup, used when circuit is open
"""

import asyncio
import os
from typing import Any, Optional

import pybreaker
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.handlers.logger import get_logger, log_error, log_info, log_warning

# transformers / torch imported lazily inside init_llm() and _flan_generate()
# so a torch/torchvision ABI mismatch never crashes the app at startup.

logger = get_logger("integrations.llm")

# ── Module-level state ────────────────────────────────────────────────────────
_openai_client: Optional[AsyncOpenAI] = None
_flan_tokenizer = None
_flan_model = None
_l1_model: str = "gpt-4o-mini"
_l2_model: str = "gpt-4o"
_request_timeout: float = 30.0
_max_retries: int = 3
_retry_base_delay: float = 2.0

# ── Circuit breaker ────────────────────────────────────────────────────────────
_breaker = pybreaker.CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    name="openai_llm",
)


def init_llm(
    openai_api_key: str,
    l1_model: str = "gpt-4o-mini",
    l2_model: str = "gpt-4o",
    fallback_model: str = "google/flan-t5-base",
    request_timeout: float = 30.0,
    max_retries: int = 3,
    retry_base_delay: float = 2.0,
) -> None:
    """Load Flan-T5 and configure OpenAI client. Called once at startup."""
    global _openai_client, _flan_tokenizer, _flan_model
    global _l1_model, _l2_model, _request_timeout, _max_retries, _retry_base_delay

    _l1_model = l1_model
    _l2_model = l2_model
    _request_timeout = request_timeout
    _max_retries = max_retries
    _retry_base_delay = retry_base_delay

    _openai_client = AsyncOpenAI(api_key=openai_api_key)
    log_info("OpenAI LLM client initialised | l1=%s l2=%s", l1_model, l2_model)

    log_info("Loading Flan-T5 fallback model '%s' ...", fallback_model)
    try:
        import torch  # lazy import
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # lazy import
        _flan_tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        _flan_model = AutoModelForSeq2SeqLM.from_pretrained(
            fallback_model,
            torch_dtype=torch.float32,
        )
        _flan_model.eval()
        log_info("Flan-T5 fallback model loaded successfully")
    except Exception as exc:
        log_warning(
            "Flan-T5 fallback model could not be loaded (torch/torchvision issue?) "
            "— OpenAI is the only LLM available | error=%s",
            exc,
        )
        _flan_tokenizer = None
        _flan_model = None


# ── Core LLM call with circuit breaker + retry ───────────────────────────────

async def chat_completion(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> tuple[str, bool]:
    """
    Send a chat completion request.
    Returns: (response_text, fallback_used)
    Falls back to Flan-T5 if OpenAI is unavailable.
    """
    target_model = model or _l1_model

    # ── Try OpenAI with circuit breaker + retry ───────────────────────────────
    try:
        response_text = await _openai_with_retry(messages, target_model, temperature, max_tokens)
        return response_text, False
    except pybreaker.CircuitBreakerError:
        log_warning("Circuit breaker OPEN — routing to Flan-T5 fallback")
    except Exception as exc:
        log_warning("OpenAI call failed after retries | error=%s — routing to Flan-T5", exc)

    # ── Flan-T5 fallback ──────────────────────────────────────────────────────
    try:
        prompt = _messages_to_prompt(messages)
        fallback_text = _flan_generate(prompt)
        log_info("Flan-T5 fallback response generated (len=%d)", len(fallback_text))
        return fallback_text, True
    except Exception as exc:
        log_error("Flan-T5 fallback also failed | error=%s", exc)
        raise RuntimeError("Both primary LLM and local fallback failed.") from exc


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type((asyncio.TimeoutError, Exception)),
    reraise=True,
)
async def _openai_with_retry(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """OpenAI call wrapped in tenacity retry + pybreaker circuit breaker."""
    @_breaker
    async def _call():
        return await asyncio.wait_for(
            _openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            timeout=_request_timeout,
        )

    response = await _call()
    return response.choices[0].message.content


def _flan_generate(prompt: str, max_new_tokens: int = 256) -> str:
    """Synchronous Flan-T5 generation (runs on CPU, ~20ms)."""
    if _flan_model is None or _flan_tokenizer is None:
        raise RuntimeError(
            "Flan-T5 fallback is unavailable (load failed at startup). "
            "Check torch/torchvision compatibility."
        )
    import torch  # lazy import — safe here since model already loaded
    inputs = _flan_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = _flan_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return _flan_tokenizer.decode(outputs[0], skip_special_tokens=True)


def _messages_to_prompt(messages: list[dict]) -> str:
    """Convert chat messages to a flat prompt string for Flan-T5."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role.capitalize()}: {content}")
    return "\n".join(parts) + "\nAssistant:"


async def llm_health_check() -> bool:
    """Return True if OpenAI API is reachable (does NOT count against circuit breaker)."""
    if _openai_client is None:
        return False
    try:
        await asyncio.wait_for(
            _openai_client.models.list(),
            timeout=5.0,
        )
        return True
    except Exception:
        return False
