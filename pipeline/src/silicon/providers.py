"""Provider clients + a single logged call interface.

Every API call is appended as one JSON line to runs/<run_id>.jsonl — the benchmark
requires raw output logs for Tier 1-2 deposits, so logging is not optional here.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import threading

import openai as _openai_mod
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_RETRYABLE = (
    _openai_mod.RateLimitError,
    _openai_mod.APIConnectionError,
    _openai_mod.APITimeoutError,
    _openai_mod.InternalServerError,
)

from .config import RUNS_DIR

load_dotenv()

_BASE_URLS = {
    "openai": None,  # SDK default
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
_KEY_VARS = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _openai_compatible(provider: str):
    from openai import OpenAI

    return OpenAI(api_key=os.environ[_KEY_VARS[provider]], base_url=_BASE_URLS[provider])


@dataclass
class CallLogger:
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def path(self) -> Path:
        return RUNS_DIR / f"{self.run_id}.jsonl"

    def log(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str) + "\n"
        with self._lock, self.path.open("a") as f:
            f.write(line)


_clients: dict[str, Any] = {}


def client_for(provider: str):
    if provider not in _clients:
        _clients[provider] = _openai_compatible(provider)
    return _clients[provider]


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, max=60),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
def call(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    logger: CallLogger,
    *,
    tag: str = "",
    max_tokens: int = 2048,
    logprobs: bool = False,
    top_logprobs: int = 20,
    require_logprobs_provider: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """One logged model call. Returns {"text", "logprobs", "raw_usage"}.

    Sampling is cold by default (no temperature passed -> provider default) per the
    Silicon Crowds recommendation: vary prompts, not draws; read out when possible.
    """
    t0 = time.time()
    extra: dict[str, Any] = {}
    if logprobs:
        extra["logprobs"] = True
        extra["top_logprobs"] = top_logprobs
    if provider == "openrouter" and require_logprobs_provider:
        extra["extra_body"] = {"provider": {"require_parameters": True}}
    # OpenAI's current models take max_completion_tokens; Groq/OpenRouter take max_tokens.
    tok_key = "max_completion_tokens" if provider == "openai" else "max_tokens"
    resp = client_for(provider).chat.completions.create(
        model=model, messages=messages, **{tok_key: max_tokens}, **extra, **kwargs
    )
    # Some OpenRouter hosts return HTTP 200 with an error body and no choices.
    if not getattr(resp, "choices", None):
        raise RuntimeError(f"{provider}/{model}: no choices in response (error={getattr(resp, 'error', None)})")
    choice = resp.choices[0]
    lp = None
    if logprobs and choice.logprobs is not None:
        lp = [
            {
                "token": t.token,
                "logprob": t.logprob,
                "top": [{"token": a.token, "logprob": a.logprob} for a in (t.top_logprobs or [])],
            }
            for t in (choice.logprobs.content or [])
        ]
    out = {
        "text": choice.message.content or "",
        "logprobs": lp,
        "raw_usage": resp.usage.model_dump() if resp.usage else None,
        "model_served": getattr(resp, "model", None),
        # OpenRouter reports which upstream host served the call — load-bearing for readout.
        "provider_served": getattr(resp, "provider", None),
    }

    logger.log(
        {
            "ts": t0,
            "elapsed": time.time() - t0,
            "tag": tag,
            "provider": provider,
            "model": model,
            "messages": messages,
            "kwargs": {k: v for k, v in kwargs.items() if k != "extra_body"},
            "response": out,
        }
    )
    return out
