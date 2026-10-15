"""Anthropic Messages client (used for the LLM judge by default).

OpenAI-style messages and tool schemas are converted to the Anthropic format.
Structured output uses ``output_config={"format": {"type": "json_schema", ...}}``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .base import LLMClient, LLMResult, ToolCall


def _to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            system_parts.append(m.get("content") or "")
        elif role == "user":
            out.append({"role": "user", "content": m.get("content") or ""})
        elif role == "assistant":
            blocks: list[dict] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                blocks.append({"type": "tool_use", "id": tc.get("id"), "name": fn.get("name"), "input": args})
            out.append({"role": "assistant", "content": blocks or ""})
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.get("tool_call_id"), "content": m.get("content") or ""}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return "\n\n".join(system_parts), out


def _tools_to_anthropic(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    res = []
    for t in tools:
        fn = t.get("function", t)
        res.append({"name": fn["name"], "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}})})
    return res


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str, cache_dir=None, api_key_env: str = "ANTHROPIC_API_KEY", **defaults: Any) -> None:
        super().__init__(model, cache_dir=cache_dir, **defaults)
        self.api_key_env = api_key_env
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"Set {self.api_key_env} in your environment (.env)")
            self._client = anthropic.Anthropic(api_key=key, timeout=self.timeout_s, max_retries=0)
        return self._client

    def _call(self, messages: list[dict], *, tools: list[dict] | None, params: dict) -> LLMResult:
        system, msgs = _to_anthropic(messages)
        kw: dict[str, Any] = {"model": self.model, "messages": msgs,
                              "max_tokens": int(params.get("max_output_tokens") or 2048)}
        if system:
            kw["system"] = system
        atools = _tools_to_anthropic(tools)
        if atools and params.get("tool_choice") != "none":
            kw["tools"] = atools
            kw["tool_choice"] = {"type": "auto"}
        out_cfg: dict[str, Any] = {}
        if params.get("response_schema"):
            out_cfg["format"] = {"type": "json_schema", "schema": params["response_schema"]}
        if params.get("effort"):
            out_cfg["effort"] = params["effort"]
        if out_cfg:
            kw["output_config"] = out_cfg
        resp = self.client.messages.create(**kw)
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for b in resp.content:
            if getattr(b, "type", "") == "text":
                text_parts.append(b.text)
            elif getattr(b, "type", "") == "tool_use":
                calls.append(ToolCall(id=b.id, name=b.name, arguments=dict(b.input or {}),
                                      raw=json.dumps(b.input, ensure_ascii=False)))
        u = resp.usage
        usage = {
            "input_tokens": int(getattr(u, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(u, "output_tokens", 0) or 0),
            "cached_input_tokens": int(getattr(u, "cache_read_input_tokens", 0) or 0),
        }
        return LLMResult(text="".join(text_parts), tool_calls=calls, model=getattr(resp, "model", self.model),
                         usage=usage, finish_reason=getattr(resp, "stop_reason", "") or "")
