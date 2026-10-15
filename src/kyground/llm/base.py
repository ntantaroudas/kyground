"""Provider-neutral model interface with disk caching, retries and usage
accounting.

Messages use the OpenAI chat format::

    {"role": "system" | "user" | "assistant" | "tool", "content": str,
     "tool_calls": [...], "tool_call_id": str}

Every call is cached under ``<cache_dir>/<model>/<k[:2]>/<k>.json`` where the
key hashes the model, the messages, the tool schema, the parameters and a
``cache_tag`` (which carries the repetition index). Re-running an experiment
therefore resumes where it stopped and reproduces earlier outputs exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw: str = ""


@dataclass
class LLMResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0
    finish_reason: str = ""
    from_cache: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LLMResult":
        d = dict(d)
        d["tool_calls"] = [ToolCall(**t) for t in d.get("tool_calls", [])]
        return cls(**d)


class LLMClient:
    """Base class. Subclasses implement :meth:`_call`."""

    provider = "abstract"

    def __init__(self, model: str, cache_dir: str | Path | None = None, max_retries: int = 6,
                 timeout_s: float = 180.0, **defaults: Any) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.defaults = defaults
        self._lock = threading.Lock()

    # ------------------------------------------------------------- caching
    def _key(self, payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        if not self.cache_dir:
            return None
        safe_model = self.model.replace("/", "_").replace(":", "_")
        return self.cache_dir / safe_model / key[:2] / f"{key}.json"

    def complete(self, messages: list[dict], *, tools: list[dict] | None = None, tool_choice: Any = None,
                 parallel_tool_calls: bool | None = None, reasoning_effort: str | None = None,
                 max_output_tokens: int | None = None, response_schema: dict | None = None,
                 seed: int | None = None, cache_tag: str = "", use_cache: bool = True) -> LLMResult:
        params = dict(self.defaults)
        params.update({k: v for k, v in dict(
            tool_choice=tool_choice, parallel_tool_calls=parallel_tool_calls, reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens, response_schema=response_schema, seed=seed).items()
            if v is not None})
        payload = {"provider": self.provider, "model": self.model, "messages": messages, "tools": tools,
                   "params": params, "tag": cache_tag}
        key = self._key(payload)
        cp = self._cache_path(key)
        if use_cache and cp is not None and cp.exists():
            try:
                res = LLMResult.from_dict(json.loads(cp.read_text(encoding="utf-8")))
                res.from_cache = True
                return res
            except Exception:
                pass
        if getattr(self, "deferred", False):      # session generation: queue the request (raises PendingRequest)
            self.defer(key, payload, cp)
        last_err = ""
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                res = self._call(messages, tools=tools, params=params)
                res.latency_s = round(time.time() - t0, 3)
                res.model = res.model or self.model
                res.provider = self.provider
                if cp is not None and not res.error:
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    tmp = cp.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
                    tmp.write_text(json.dumps(res.to_dict(), ensure_ascii=False), encoding="utf-8")
                    os.replace(tmp, cp)
                return res
            except Exception as exc:  # retry on transient errors
                last_err = f"{type(exc).__name__}: {exc}"
                if not self._retryable(exc) or attempt == self.max_retries - 1:
                    break
                time.sleep(min(60.0, (2 ** attempt) + random.random()))
        return LLMResult(text="", model=self.model, provider=self.provider, error=last_err)

    def _retryable(self, exc: Exception) -> bool:
        name = type(exc).__name__
        return any(s in name for s in ("RateLimit", "Timeout", "Connection", "InternalServer", "ServiceUnavailable",
                                       "Overloaded", "APIStatusError"))

    def _call(self, messages: list[dict], *, tools: list[dict] | None, params: dict) -> LLMResult:
        raise NotImplementedError


def make_client(spec: dict, cache_dir: str | Path | None = None) -> LLMClient:
    """Build a client from a config dict: ``{provider, model, base_url?, ...}``."""
    provider = spec.get("provider", "openai")
    model = spec["model"]
    extra = {k: v for k, v in spec.items() if k not in ("provider", "model", "base_url", "api_key_env", "pricing")}
    if provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(model, cache_dir=cache_dir, api_key_env=spec.get("api_key_env", "OPENAI_API_KEY"),
                            base_url=spec.get("base_url"), **extra)
    if provider in ("openai_compatible", "compat", "vllm", "ollama"):
        from .openai_client import OpenAIClient
        return OpenAIClient(model, cache_dir=cache_dir, api_key_env=spec.get("api_key_env", "COMPAT_API_KEY"),
                            base_url=spec.get("base_url"), compat=True, **extra)
    if provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(model, cache_dir=cache_dir, api_key_env=spec.get("api_key_env", "ANTHROPIC_API_KEY"),
                               **extra)
    if provider == "mock":
        from .mock import MockClient
        return MockClient(model, cache_dir=None, **extra)
    if provider == "session":
        from .session import SessionClient
        extra = {k: v for k, v in extra.items() if k not in ("queue", "role")}
        return SessionClient(model, cache_dir=cache_dir, queue=spec.get("queue"), role=spec.get("role", ""), **extra)
    raise ValueError(f"unknown provider {provider}")


def usage_add(acc: dict[str, dict[str, int]], res: LLMResult) -> None:
    """Accumulate token usage per model."""
    m = acc.setdefault(res.model or "?", {})
    for k, v in (res.usage or {}).items():
        m[k] = m.get(k, 0) + int(v or 0)
    m["calls"] = m.get("calls", 0) + 1
    if res.from_cache:
        m["cached_calls"] = m.get("cached_calls", 0) + 1
