"""OpenAI Chat Completions client (also used for OpenAI-compatible servers
such as vLLM, Ollama or LM Studio, e.g. to serve Llama-Krikri-8B)."""

from __future__ import annotations

import json
import os
from typing import Any

from .base import LLMClient, LLMResult, ToolCall

REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(self, model: str, cache_dir=None, api_key_env: str = "OPENAI_API_KEY",
                 base_url: str | None = None, compat: bool = False, **defaults: Any) -> None:
        super().__init__(model, cache_dir=cache_dir, **defaults)
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.compat = compat
        if compat:
            self.provider = "openai_compatible"
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            key = os.environ.get(self.api_key_env) or ("not-needed" if self.compat else None)
            if not key:
                raise RuntimeError(f"Set {self.api_key_env} in your environment (.env)")
            self._client = OpenAI(api_key=key, base_url=self.base_url, timeout=self.timeout_s, max_retries=0)
        return self._client

    @property
    def is_reasoning(self) -> bool:
        return (not self.compat) and self.model.startswith(REASONING_PREFIXES)

    def _call(self, messages: list[dict], *, tools: list[dict] | None, params: dict) -> LLMResult:
        kw: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = params.get("tool_choice", "auto")
            if params.get("parallel_tool_calls") is not None and kw["tool_choice"] != "none":
                kw["parallel_tool_calls"] = params["parallel_tool_calls"]
        if params.get("reasoning_effort") and self.is_reasoning:
            kw["reasoning_effort"] = params["reasoning_effort"]
        if params.get("verbosity") and self.is_reasoning:
            kw["verbosity"] = params["verbosity"]
        if params.get("max_output_tokens"):
            kw["max_completion_tokens" if not self.compat else "max_tokens"] = params["max_output_tokens"]
        if params.get("temperature") is not None and not self.is_reasoning:
            kw["temperature"] = params["temperature"]
        if params.get("seed") is not None:
            kw["seed"] = params["seed"]
        if params.get("response_schema"):
            kw["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": "result", "schema": params["response_schema"],
                                                     "strict": True}}
        resp = self.client.chat.completions.create(**kw)
        ch = resp.choices[0]
        msg = ch.message
        calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            raw = tc.function.arguments or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {"_unparsed": raw}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args, raw=raw))
        u = resp.usage
        usage: dict[str, int] = {}
        if u is not None:
            usage["input_tokens"] = int(getattr(u, "prompt_tokens", 0) or 0)
            usage["output_tokens"] = int(getattr(u, "completion_tokens", 0) or 0)
            ptd = getattr(u, "prompt_tokens_details", None)
            ctd = getattr(u, "completion_tokens_details", None)
            usage["cached_input_tokens"] = int(getattr(ptd, "cached_tokens", 0) or 0) if ptd else 0
            usage["reasoning_tokens"] = int(getattr(ctd, "reasoning_tokens", 0) or 0) if ctd else 0
        return LLMResult(text=msg.content or "", tool_calls=calls, model=getattr(resp, "model", self.model),
                         usage=usage, finish_reason=ch.finish_reason or "")
