"""Shared machinery for the compared systems."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..llm.base import LLMClient, LLMResult, usage_add
from ..llm.session import PendingRequest

ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_prompts(path: str | Path | None = None) -> dict:
    return load_yaml(path or ROOT / "configs" / "prompts.yaml")


def _as_function(t: dict) -> dict:
    props = {}
    for name, p in (t.get("parameters") or {}).items():
        p = dict(p)
        p.pop("default", None)
        props[name] = p
    return {"type": "function", "function": {
        "name": t["name"], "description": " ".join(t["description"].split()),
        "parameters": {"type": "object", "properties": props, "required": list(t.get("required") or [])},
    }}


def load_tools(path: str | Path | None = None) -> list[dict]:
    """kythera tool definitions in OpenAI function-calling format."""
    spec = load_yaml(path or ROOT / "configs" / "tools.yaml")
    return [_as_function(t) for t in spec["tools"]]


def load_rag_tool(path: str | Path | None = None) -> dict | None:
    """The single search tool through which the router formulates C1q's vector-search query."""
    spec = load_yaml(path or ROOT / "configs" / "tools.yaml")
    return _as_function(spec["rag_tool"]) if spec.get("rag_tool") else None


@dataclass
class Thread:
    """A conversation. Only user and final assistant texts are kept in the
    history (the same policy for every condition)."""

    lang: str
    tid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    history: list[dict] = field(default_factory=list)
    max_history_turns: int = 6

    def window(self) -> list[dict]:
        return self.history[-2 * self.max_history_turns:]

    def add(self, user: str, answer: str) -> None:
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": answer})


@dataclass
class TurnResult:
    answer: str
    condition: str
    lang: str
    tool_calls: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    context_rids: list[str] = field(default_factory=list)
    rewritten_query: str = ""
    usage: dict[str, dict[str, int]] = field(default_factory=dict)
    latency_s: float = 0.0
    stage_latency: dict[str, float] = field(default_factory=dict)
    error: str = ""
    flags: list[str] = field(default_factory=list)   # e.g. router_no_tool, rewrite_no_query
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class System:
    """Base class: subclasses implement :meth:`_answer`."""

    condition = "?"

    def __init__(self, answer_llm: LLMClient, prompts: dict, answer_params: dict | None = None) -> None:
        self.answer_llm = answer_llm
        self.prompts = prompts
        self.answer_params = dict(answer_params or {})

    # prompt helpers --------------------------------------------------------
    def system_prompt(self, lang: str, grounding: str, extra: str = "") -> str:
        base = self.prompts["base"][lang].strip()
        g = self.prompts["grounding"][grounding][lang].strip()
        return f"{base}\n{g}" + (f"\n\n{extra}" if extra else "")

    def wrap_context(self, lang: str, context: str, question: str) -> str:
        return self.prompts["context_wrapper"][lang].format(context=context, question=question)

    # public API ------------------------------------------------------------
    def new_thread(self, lang: str) -> Thread:
        return Thread(lang=lang)

    def ask(self, thread: Thread, user_text: str, cache_tag: str = "") -> TurnResult:
        t0 = time.time()
        try:
            res = self._answer(thread, user_text, cache_tag)
        except PendingRequest:    # session generation: not answered yet; the runner retries in the next round
            raise
        except Exception as exc:  # never crash a whole run on one item
            res = TurnResult(answer="", condition=self.condition, lang=thread.lang,
                             error=f"{type(exc).__name__}: {exc}")
        res.latency_s = round(time.time() - t0, 3)
        thread.add(user_text, res.answer)
        return res

    def _answer(self, thread: Thread, user_text: str, cache_tag: str) -> TurnResult:  # pragma: no cover
        raise NotImplementedError

    # utilities ---------------------------------------------------------------
    def _final(self, messages: list[dict], cache_tag: str, tools: list[dict] | None = None,
               tool_choice: str | None = None) -> LLMResult:
        p = self.answer_params
        return self.answer_llm.complete(
            messages, tools=tools, tool_choice=tool_choice,
            reasoning_effort=p.get("reasoning_effort"), max_output_tokens=p.get("max_output_tokens", 4000),
            cache_tag=cache_tag)

    @staticmethod
    def _account(res: TurnResult, llm_res: LLMResult, stage: str) -> None:
        usage_add(res.usage, llm_res)
        res.stage_latency[stage] = round(res.stage_latency.get(stage, 0.0) + llm_res.latency_s, 3)
        res.from_cache = res.from_cache or llm_res.from_cache
        if llm_res.error:
            res.error = (res.error + " | " if res.error else "") + f"{stage}: {llm_res.error}"
