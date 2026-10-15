"""Implementations of conditions C0, C1, C1q, C2, C3 and C4."""

from __future__ import annotations

import json
import time
from typing import Any

from ..kb.replica_api import ToolBackend, result_ids
from ..kb.serialize import kb_text
from ..kb.store import KBStore
from ..llm.base import LLMClient
from ..retrieval.index import VectorIndex
from .base import System, Thread, TurnResult


# ---------------------------------------------------------------------------
# C0 — closed book
# ---------------------------------------------------------------------------

class ClosedBook(System):
    condition = "C0"

    def _answer(self, thread: Thread, user_text: str, cache_tag: str) -> TurnResult:
        res = TurnResult(answer="", condition=self.condition, lang=thread.lang)
        msgs = [{"role": "system", "content": self.system_prompt(thread.lang, "none")}, *thread.window(),
                {"role": "user", "content": user_text}]
        out = self._final(msgs, cache_tag)
        self._account(res, out, "answer")
        res.answer = out.text
        return res


# ---------------------------------------------------------------------------
# C1 / C1q — vector RAG (optionally with LLM query rewriting)
# ---------------------------------------------------------------------------

def format_chunks(hits) -> str:
    return "\n\n".join(f"[{i + 1}] {c.text}" for i, (c, _s) in enumerate(hits))


class VectorRAG(System):
    condition = "C1"

    def __init__(self, answer_llm: LLMClient, prompts: dict, index: VectorIndex, k: int = 5,
                 rewrite_llm: LLMClient | None = None, rewrite_params: dict | None = None,
                 answer_params: dict | None = None, rag_tool: dict | None = None) -> None:
        super().__init__(answer_llm, prompts, answer_params)
        self.index = index
        self.k = k
        self.rewrite_llm = rewrite_llm
        self.rewrite_params = dict(rewrite_params or {})
        self.rag_tool = rag_tool
        if rewrite_llm is not None:
            self.condition = "C1q"
            if rag_tool is None:
                raise ValueError("C1q needs the rag_tool definition (configs/tools.yaml)")

    def rewrite(self, res: TurnResult, thread: Thread, user_text: str, cache_tag: str) -> str:
        """C1q: the router model formulates the search query by calling one search tool, with the same base
        prompt, history and function-calling mechanism as the C2 router; the call's ``query`` is embedded."""
        msgs = [{"role": "system", "content": self.system_prompt(thread.lang, "rewrite")}, *thread.window(),
                {"role": "user", "content": user_text}]
        out = self.rewrite_llm.complete(
            msgs, tools=[self.rag_tool], tool_choice="required", parallel_tool_calls=False,
            reasoning_effort=self.rewrite_params.get("reasoning_effort"),
            max_output_tokens=self.rewrite_params.get("max_output_tokens", 2000), cache_tag=f"{cache_tag}|rewrite")
        self._account(res, out, "rewrite")
        q = ""
        for c in out.tool_calls:
            q = str((c.arguments or {}).get("query") or "").strip()
            if q:
                break
        if not q:
            res.flags.append("rewrite_no_query")
        return q or user_text

    def retrieve(self, res: TurnResult, thread: Thread, user_text: str, cache_tag: str) -> str:
        query = user_text
        if self.rewrite_llm is not None:
            query = self.rewrite(res, thread, user_text, cache_tag)
            res.rewritten_query = query
        t0 = time.time()
        hits = self.index.search(query, k=self.k)
        res.stage_latency["retrieve"] = round(time.time() - t0, 3)
        res.retrieved = [{"rid": c.rid, "cid": c.cid, "score": round(s, 4), "version": c.version} for c, s in hits]
        res.context_rids = list(dict.fromkeys(c.rid for c, _ in hits))
        return format_chunks(hits)

    def _answer(self, thread: Thread, user_text: str, cache_tag: str) -> TurnResult:
        res = TurnResult(answer="", condition=self.condition, lang=thread.lang)
        ctx = self.retrieve(res, thread, user_text, cache_tag)
        msgs = [{"role": "system", "content": self.system_prompt(thread.lang, "rag")}, *thread.window(),
                {"role": "user", "content": self.wrap_context(thread.lang, ctx, user_text)}]
        out = self._final(msgs, cache_tag)
        self._account(res, out, "answer")
        res.answer = out.text
        return res


# ---------------------------------------------------------------------------
# C2 — tool retrieval (two-model agent: router + answerer)
# ---------------------------------------------------------------------------

def shared_router_tag(cache_tag: str) -> str:
    """Cache tag of the router requests in the accuracy runs, without the condition and the label. The router sees
    identical messages in C2 and C3 and under every search normalisation (E4), so one router decision per query and
    step is generated and shared by them, as a deterministic router would give; a step whose input differs (other
    tool results) still gets its own request, because the messages are part of the cache key."""
    p = cache_tag.split("|")
    if len(p) >= 5 and p[0] == "acc":
        return "|".join(["acc", "router", "", *p[3:]])
    return cache_tag


class ToolAgent(System):
    """Mirrors the deployed pipeline: a router model (gpt-5-mini, minimal
    reasoning) selects up to ``max_tool_calls`` kythera tools and fills their
    parameters; the backend executes them against the data API (live or
    replica); the answer model (gpt-5-nano) composes the reply from the tool
    results."""

    condition = "C2"

    def __init__(self, answer_llm: LLMClient, prompts: dict, router_llm: LLMClient, tools: list[dict],
                 backend: ToolBackend, router_params: dict | None = None, answer_params: dict | None = None,
                 max_tool_calls: int = 2, max_router_steps: int = 2, max_tool_chars: int = 12000,
                 grounding: str = "tools") -> None:
        super().__init__(answer_llm, prompts, answer_params)
        self.router_llm = router_llm
        self.tools = tools
        self.backend = backend
        self.router_params = dict(router_params or {})
        self.max_tool_calls = max_tool_calls
        self.max_router_steps = max_router_steps
        self.max_tool_chars = max_tool_chars
        self.grounding = grounding

    def _route_and_call(self, res: TurnResult, msgs: list[dict], lang: str, cache_tag: str) -> list[dict]:
        tool_msgs: list[dict] = []
        done = 0
        for step in range(self.max_router_steps):
            out = self.router_llm.complete(
                msgs + tool_msgs, tools=self.tools, tool_choice="auto", parallel_tool_calls=True,
                reasoning_effort=self.router_params.get("reasoning_effort"),
                max_output_tokens=self.router_params.get("max_output_tokens", 2000),
                cache_tag=f"{shared_router_tag(cache_tag)}|router{step}")
            self._account(res, out, "router")
            calls = out.tool_calls[: max(0, self.max_tool_calls - done)]
            if not calls:
                if step == 0:
                    res.flags.append("router_error" if out.error else "router_no_tool")
                break
            tool_msgs.append({"role": "assistant", "content": out.text or None, "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                for c in calls]})
            for c in calls:
                t0 = time.time()
                try:
                    result = self.backend.call(c.name, c.arguments, lang)
                except Exception as exc:  # backend error is returned to the model, as an API would
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                dt = round(time.time() - t0, 3)
                ids = result_ids(result)
                res.tool_calls.append({"name": c.name, "args": c.arguments, "n_results": result.get("count", 0),
                                       "ids": ids, "latency_s": dt, "step": step,
                                       "error": result.get("error", "")})
                res.stage_latency["tools"] = round(res.stage_latency.get("tools", 0.0) + dt, 3)
                content = json.dumps(result, ensure_ascii=False)
                if len(content) > self.max_tool_chars:
                    content = content[: self.max_tool_chars] + " …[truncated]"
                tool_msgs.append({"role": "tool", "tool_call_id": c.id, "content": content})
                for i in ids:
                    if i not in res.context_rids:
                        res.context_rids.append(i)
            done += len(calls)
            if done >= self.max_tool_calls:
                break
        return tool_msgs

    def _router_messages(self, thread: Thread, user_text: str) -> list[dict]:
        """What the router sees: identical in C2 and C3."""
        return [{"role": "system", "content": self.system_prompt(thread.lang, "tools")}, *thread.window(),
                {"role": "user", "content": user_text}]

    def _messages(self, thread: Thread, user_text: str, res: TurnResult, cache_tag: str) -> list[dict]:
        """What the answer model sees before the tool results."""
        return [{"role": "system", "content": self.system_prompt(thread.lang, self.grounding)}, *thread.window(),
                {"role": "user", "content": user_text}]

    def _answer(self, thread: Thread, user_text: str, cache_tag: str) -> TurnResult:
        res = TurnResult(answer="", condition=self.condition, lang=thread.lang)
        tool_msgs = self._route_and_call(res, self._router_messages(thread, user_text), thread.lang, cache_tag)
        msgs = self._messages(thread, user_text, res, cache_tag)
        if tool_msgs:
            out = self._final(msgs + tool_msgs, cache_tag, tools=self.tools, tool_choice="none")
        else:
            out = self._final(msgs, cache_tag)
        self._account(res, out, "answer")
        res.answer = out.text
        return res


# ---------------------------------------------------------------------------
# C3 — hybrid: tools + vector context
# ---------------------------------------------------------------------------

class Hybrid(ToolAgent):
    condition = "C3"

    def __init__(self, *args: Any, index: VectorIndex, k: int = 5, **kw: Any) -> None:
        kw.setdefault("grounding", "hybrid")
        super().__init__(*args, **kw)
        self.index = index
        self.k = k

    def _messages(self, thread: Thread, user_text: str, res: TurnResult, cache_tag: str) -> list[dict]:
        t0 = time.time()
        hits = self.index.search(user_text, k=self.k)
        res.stage_latency["retrieve"] = round(time.time() - t0, 3)
        res.retrieved = [{"rid": c.rid, "cid": c.cid, "score": round(s, 4), "version": c.version} for c, s in hits]
        for c, _ in hits:
            if c.rid not in res.context_rids:
                res.context_rids.append(c.rid)
        ctx = format_chunks(hits)
        return [{"role": "system", "content": self.system_prompt(thread.lang, self.grounding)}, *thread.window(),
                {"role": "user", "content": self.wrap_context(thread.lang, ctx, user_text)}]


# ---------------------------------------------------------------------------
# C4 — full context (reference condition)
# ---------------------------------------------------------------------------

class FullContext(System):
    """The entire published KB in the system prompt, rebuilt from the store at
    every turn (``live=True``) so that it is as fresh as the tool path, or
    frozen at construction (``live=False``)."""

    condition = "C4"

    def __init__(self, answer_llm: LLMClient, prompts: dict, store: KBStore, live: bool = True,
                 answer_params: dict | None = None) -> None:
        super().__init__(answer_llm, prompts, answer_params)
        self.store = store
        self.live = live
        self._frozen = kb_text(store.public())

    def _answer(self, thread: Thread, user_text: str, cache_tag: str) -> TurnResult:
        res = TurnResult(answer="", condition=self.condition, lang=thread.lang)
        kb = kb_text(self.store.public()) if self.live else self._frozen
        extra = self.prompts["kb_wrapper"].format(kb=kb)
        msgs = [{"role": "system", "content": self.system_prompt(thread.lang, "full", extra)}, *thread.window(),
                {"role": "user", "content": user_text}]
        out = self._final(msgs, cache_tag)
        self._account(res, out, "answer")
        res.answer = out.text
        res.context_rids = [r.rid for r in self.store.public()]
        return res
