"""Deterministic mock model for tests and dry runs (no network, no cost).

It is *not* a baseline: its only purpose is to exercise every code path
(tool routing, retrieval, multi-turn threads, scoring, statistics, figures)
end to end. Outputs are labelled ``[MOCK]`` in the run metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .. import textnorm as tn
from .base import LLMClient, LLMResult, ToolCall

_ROUTES = [
    ("get_seasonal_calendar", ["ημερολογιο", "calendar", "μηνα ", "month", "εργασιεσ", "tasks"]),
    ("lookup_glossary", ["σημαινει", "ορο", "οροσ", "term", "mean", "λεξη", "word"]),
    ("get_crop_info", ["καλλιεργεια", "crop", "φυτευ", "planting", "επιστημονικο", "scientific", "συγκομιδησ"]),
    ("search_experiences", ["εμπειρι", "experience", "ξεναγ", "tour", "διαδρομη", "οικοδεσποτ"]),
    ("list_training_modules", ["προγραμμα", "μαθημα", "module", "course", "lesson", "εκπαιδευ", "training"]),
    ("search_local_products", ["προιον", "product", "παραγει", "produces", "κοστιζει", "cost", "τιμη", "price",
                               "μελι", "honey", "λαδι", "oil", "αλατι", "salt", "κρασι", "wine", "πουλαει", "sell"]),
]


def _flatten(obj: Any, prefix: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_flatten(v, f"{k}: "))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_flatten(v, prefix))
    elif obj not in (None, ""):
        out.append(f"{prefix}{obj}")
    return out


class MockClient(LLMClient):
    provider = "mock"

    def _call(self, messages: list[dict], *, tools: list[dict] | None, params: dict) -> LLMResult:
        system = " ".join(m.get("content") or "" for m in messages if m["role"] == "system")
        lang = "en" if "Interface language: English" in system else "el"
        last_user_idx = max(i for i, m in enumerate(messages) if m["role"] == "user")
        user = messages[last_user_idx].get("content") or ""
        question = re.split(r"(?:Ερώτηση|Question)\s*:\s*", user)[-1]
        after = messages[last_user_idx + 1:]
        has_tool_results = any(m["role"] == "tool" for m in after)

        # judge / structured output
        if params.get("response_schema"):
            schema = params["response_schema"]
            out = {}
            for k, spec in schema.get("properties", {}).items():
                if spec.get("enum"):
                    out[k] = spec["enum"][0]
                elif spec.get("type") == "boolean":
                    out[k] = False
                elif spec.get("type") in ("number", "integer"):
                    out[k] = 0
                else:
                    out[k] = "mock"
            return self._res(json.dumps(out), messages)

        # router
        if tools and params.get("tool_choice") != "none" and not has_tool_results:
            nq = tn.norm(question) + " "
            available = [t["function"]["name"] for t in tools]
            name = "search_kythera_knowledge"
            for tool, keys in _ROUTES:
                if any(k in nq for k in keys):
                    name = tool
                    break
            if name not in available:
                name = available[0]
            qt = re.findall(r"«([^»]+)»", question)
            toks = sorted(tn.content_tokens(question), key=len, reverse=True)
            query = qt[0] if qt else (toks[0] if toks else "")
            args: dict[str, Any] = {"lang": lang}
            if name == "get_seasonal_calendar":
                ms = tn.months_in(question)
                if ms:
                    args["month"] = ms[0]
            elif name == "get_crop_info":
                args["crop_type"] = query
            elif name == "lookup_glossary":
                args["term"] = query
            elif name == "list_training_modules":
                pass
            else:
                args["query"] = query
            return LLMResult(text="", tool_calls=[ToolCall(id="call_" + hashlib.md5(question.encode()).hexdigest()[:8], name=name,
                                                           arguments=args, raw=json.dumps(args, ensure_ascii=False))],
                             usage=self._usage(messages, ""))

        # answerer: extract the most relevant lines from the context
        ctx_lines: list[str] = []
        for m in after:
            if m["role"] == "tool":
                try:
                    ctx_lines.extend(_flatten(json.loads(m.get("content") or "{}")))
                except json.JSONDecodeError:
                    ctx_lines.extend((m.get("content") or "").split("\n"))
        if "<CONTEXT>" in user or "Αποσπάσματα" in user or "Excerpts" in user:
            ctx_lines.extend(user.split("\n"))
        if "<KNOWLEDGE_BASE>" in system:
            kb = system.split("<KNOWLEDGE_BASE>", 1)[1]
            ctx_lines.extend(kb.split("\n"))
        qst = set(tn.stems(question))
        scored = []
        for ln in ctx_lines:
            st = set(tn.stems(ln))
            ov = len(qst & st)
            if ov:
                scored.append((ov, len(ln), ln))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if not scored:
            text = "Δεν βρήκα σχετικές πληροφορίες στα τοπικά δεδομένα." if lang == "el" else \
                "I couldn't find relevant information in the local data."
        else:
            text = " ".join(dict.fromkeys(s[2] for s in scored[:3]))
        return self._res(text, messages)

    def _usage(self, messages: list[dict], text: str) -> dict:
        n_in = sum(len(m.get("content") or "") for m in messages) // 4
        return {"input_tokens": n_in, "output_tokens": len(text) // 4, "cached_input_tokens": 0}

    def _res(self, text: str, messages: list[dict]) -> LLMResult:
        return LLMResult(text=text, usage=self._usage(messages, text), finish_reason="stop")
