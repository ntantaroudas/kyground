"""Builds stores, backends, indexes and systems from ``configs/study.yaml``."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..kb.replica_api import KYTHERA_WKT, ReplicaAPI, ReplicaBackend, ToolBackend
from ..kb.store import KBStore
from ..llm.base import LLMClient, make_client
from ..retrieval.embedder import Embedder, make_embedder
from ..retrieval.index import VectorIndex
from ..systems.base import load_prompts, load_rag_tool, load_tools
from ..systems.conditions import ClosedBook, FullContext, Hybrid, ToolAgent, VectorRAG

ROOT = Path(__file__).resolve().parents[3]


def load_env(path: str | Path | None = None) -> None:
    """Minimal .env loader (KEY=VALUE lines); existing variables win."""
    p = Path(path) if path else ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    p = Path(path) if path else ROOT / "configs" / "study.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


def _deep_update(d: dict, u: dict) -> None:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def rpath(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


@dataclass
class Env:
    """Everything an experiment needs, built once per configuration."""

    cfg: dict
    mock: bool = False
    prompts: dict = field(default_factory=dict)
    tools: list[dict] = field(default_factory=list)
    rag_tool: dict | None = None
    clients: dict[str, LLMClient] = field(default_factory=dict)
    embedder: Embedder | None = None
    quiet: bool = False

    @classmethod
    def build(cls, cfg: dict, mock: bool = False) -> "Env":
        load_env()
        env = cls(cfg=cfg, mock=mock)
        env.prompts = load_prompts(rpath(cfg.get("prompts", "configs/prompts.yaml")))
        env.tools = load_tools(rpath(cfg.get("tools", "configs/tools.yaml")))
        env.rag_tool = load_rag_tool(rpath(cfg.get("tools", "configs/tools.yaml")))
        cache = rpath(cfg["paths"]["cache_dir"])
        for role, spec in cfg["models"].items():
            spec = dict(spec)
            if mock:
                spec = {"provider": "mock", "model": f"mock-{role}"}
            spec = {k: v for k, v in spec.items() if k not in ("reasoning_effort", "max_output_tokens", "verbosity")}
            if spec.get("enabled", True) is False:
                continue
            spec.pop("enabled", None)
            if spec.get("provider") == "session":      # Claude sub-agents through the request cache
                spec["queue"] = str(rpath(cfg.get("session", {}).get("queue", "runs/session/pending.jsonl")))
                spec["role"] = role
            env.clients[role] = make_client(spec, cache_dir=None if mock else cache / "llm")
        emb_spec = {"provider": "hash"} if mock else cfg["embedder"]
        env.embedder = make_embedder(emb_spec, cache_dir=None if mock else cache)
        return env

    # ------------------------------------------------------------------ parts
    def params(self, role: str) -> dict:
        spec = self.cfg["models"].get(role, {})
        return {k: spec[k] for k in ("reasoning_effort", "max_output_tokens", "verbosity") if k in spec}

    def store(self, snapshot: str | Path | None = None) -> KBStore:
        return KBStore.from_jsonl(rpath(snapshot or self.cfg["paths"]["snapshot"]))

    def replica_backend(self, store: KBStore, search_mode: str | None = None) -> ReplicaBackend:
        rc = self.cfg.get("replica", {})
        api = ReplicaAPI(store, search_mode=search_mode or rc.get("search_mode", "phrase_casefold"),
                         spatial_missing_coords=rc.get("spatial_missing_coords", "include"),
                         default_limit=int(rc.get("default_limit", 5)),
                         filter_mode=rc.get("filter_mode", "unaccent"), ranking=rc.get("ranking", "relevance"))
        return ReplicaBackend(api, wkt=rc.get("wkt", KYTHERA_WKT))

    def live_backend(self) -> ToolBackend:
        from ..sources.api_client import LiveBackend, LiveAPI
        lc = self.cfg.get("live_api", {})
        return LiveBackend(LiveAPI.from_config(lc), wkt=self.cfg.get("replica", {}).get("wkt", KYTHERA_WKT))

    def index(self, store: KBStore, policy: str = "never", source: str = "public") -> VectorIndex:
        rc = self.cfg.get("retrieval", {})
        idx = VectorIndex(self.embedder, policy=policy, source=source, max_chars=int(rc.get("max_chars", 1500)))
        idx.build(store)
        if policy in ("immediate_upsert", "immediate_append"):
            idx.attach(store)
        return idx

    def system(self, condition: str, store: KBStore, *, backend: ToolBackend | None = None,
               index: VectorIndex | None = None, answer_role: str = "answer") -> Any:
        """Instantiate one condition. ``answer_role`` selects the answer model
        (``answer`` = production gpt-5-nano; ``answer_open`` = Krikri)."""
        a_llm = self.clients[answer_role]
        a_par = self.params(answer_role)
        k = int(self.cfg.get("retrieval", {}).get("k", 5))
        ag = self.cfg.get("agent", {})
        if condition == "C0":
            return ClosedBook(a_llm, self.prompts, answer_params=a_par)
        if condition in ("C1", "C1q"):
            idx = index or self.index(store)
            rw = self.clients["router"] if condition == "C1q" else None
            return VectorRAG(a_llm, self.prompts, idx, k=k, rewrite_llm=rw, rewrite_params=self.params("router"),
                             answer_params=a_par, rag_tool=self.rag_tool)
        if condition in ("C2", "C3"):
            be = backend or self.replica_backend(store)
            common = dict(router_llm=self.clients["router"], tools=self.tools, backend=be,
                          router_params=self.params("router"), answer_params=a_par,
                          max_tool_calls=int(ag.get("max_tool_calls", 2)),
                          max_router_steps=int(ag.get("max_router_steps", 2)),
                          max_tool_chars=int(ag.get("max_tool_chars", 12000)))
            if condition == "C2":
                return ToolAgent(a_llm, self.prompts, **common)
            idx = index or self.index(store)
            return Hybrid(a_llm, self.prompts, index=idx, k=k, **common)
        if condition == "C4":
            return FullContext(a_llm, self.prompts, store, live=True, answer_params=a_par)
        raise ValueError(f"unknown condition {condition}")
