"""Exact-search vector index over the knowledge store, with refresh policies.

Refresh policies (how edits reach the index):

- ``never``            the index reflects the store as it was at build time
- ``manual``           the experiment calls :meth:`VectorIndex.refresh` at
                       scheduled points (e.g. a nightly batch job)
- ``immediate_upsert`` on every edit the record's old chunks are deleted and
                       new chunks embedded (best case for vector RAG)
- ``immediate_append`` on every edit new chunks are added but old ones are
                       kept (append-only stores; old and new versions coexist)

``source="public"`` indexes the gated public content (the fair comparison);
``source="raw"`` indexes every record including drafts and hidden prices (the
misconfigured-indexer ablation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..kb.serialize import chunk_record, ungated_view
from ..kb.store import KBStore, Mutation
from ..schema import Record
from .embedder import Embedder

POLICIES = ("never", "manual", "immediate_upsert", "immediate_append")


@dataclass
class Chunk:
    cid: str
    rid: str
    rtype: str
    text: str
    version: int


@dataclass
class VectorIndex:
    embedder: Embedder
    policy: str = "never"
    source: str = "public"            # public | raw
    max_chars: int = 1500
    chunks: list[Chunk] = field(default_factory=list)
    _mat: np.ndarray | None = None
    _store: KBStore | None = None
    pending: set[str] = field(default_factory=set)
    built_version: int = 0
    stats: dict = field(default_factory=lambda: {"builds": 0, "upserts": 0, "appends": 0, "embedded_chunks": 0})

    # ------------------------------------------------------------- building
    def _chunks_for(self, r: Record, version: int) -> list[Chunk]:
        if self.source == "raw":  # misconfigured indexer: same chunking as C1, gating flags ignored
            if r.exclude_reason:      # records excluded from the study (e.g. test content) stay excluded
                return []
            texts = chunk_record(ungated_view(r), max_chars=self.max_chars)
        else:
            if not r.is_public:
                return []
            texts = chunk_record(r, max_chars=self.max_chars)
        return [Chunk(cid=f"{r.rid}#v{version}#{i}", rid=r.rid, rtype=r.rtype, text=t, version=version)
                for i, t in enumerate(texts)]

    def build(self, store: KBStore) -> None:
        self._store = store
        recs = store.all(include_gated=(self.source == "raw"))
        self.chunks = [c for r in recs for c in self._chunks_for(r, store.version)]
        self._reembed()
        self.built_version = store.version
        self.pending.clear()
        self.stats["builds"] += 1

    def attach(self, store: KBStore) -> None:
        self._store = store
        store.subscribe(self._on_mutation)

    def _reembed(self) -> None:
        if not self.chunks:
            self._mat = np.zeros((0, 1), dtype=np.float32)
            return
        self._mat = self.embedder.embed([c.text for c in self.chunks], kind="passage")
        self.stats["embedded_chunks"] += len(self.chunks)

    # ------------------------------------------------------------ refreshes
    def _on_mutation(self, m: Mutation, before: Record | None, after: Record | None, version: int) -> None:
        self.pending.add(m.rid)
        if self.policy == "immediate_upsert":
            self._upsert(m.rid, after, version)
        elif self.policy == "immediate_append":
            if after is not None and (self.source == "raw" or after.is_public):
                self.chunks.extend(self._chunks_for(after, version))
                self._reembed()
                self.stats["appends"] += 1
            self.pending.discard(m.rid)

    def _upsert(self, rid: str, after: Record | None, version: int) -> None:
        self.chunks = [c for c in self.chunks if c.rid != rid]
        if after is not None:
            self.chunks.extend(self._chunks_for(after, version))
        self._reembed()
        self.pending.discard(rid)
        self.stats["upserts"] += 1

    def refresh(self) -> None:
        """Batch re-index (e.g. nightly job): rebuild from the current store."""
        if self._store is not None:
            self.build(self._store)

    # --------------------------------------------------------------- search
    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if self._mat is None or not self.chunks:
            return []
        q = self.embedder.embed([query], kind="query")[0]
        sims = self._mat @ q
        order = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in order]
