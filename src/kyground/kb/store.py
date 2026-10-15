"""Versioned in-memory knowledge store with server-side editorial gating.

The store is the single source of truth inside an experiment. Every condition
reads from the same store:

- the tool-calling agent through :class:`~kyground.kb.replica_api.ReplicaAPI`,
  which applies gating on every call (as the deployed API does);
- the vector index, which is built from ``store.public()`` and is notified of
  mutations so that its refresh policy can decide when to re-embed;
- the full-context condition, which serialises ``store.public()``.

Freshness and governance experiments mutate the store through
:meth:`KBStore.apply`, which is logged and versioned. Mutations never touch the
production platform.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from ..schema import Record, load_records

EPOCH = datetime(2026, 10, 1, tzinfo=timezone.utc)


@dataclass
class Mutation:
    """A single editorial action.

    ops:
      - ``set_text``: ``key`` in {title, body, summary}, ``lang``, ``value``
      - ``set_attr``: ``key`` attribute; ``lang`` for bilingual attrs; ``value``
      - ``set_status``: ``value`` in {published, draft}
      - ``set_active``: ``value`` bool
      - ``set_price``: ``value`` float or None
      - ``set_price_visible``: ``value`` bool
      - ``add``: ``record`` (dict)
      - ``remove``: hard delete
    """

    op: str
    rid: str
    key: str = ""
    lang: str | None = None
    value: Any = None
    record: dict | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


Listener = Callable[[Mutation, "Record | None", "Record | None", int], None]


@dataclass
class KBStore:
    records: dict[str, Record] = field(default_factory=dict)
    version: int = 0
    log: list[dict] = field(default_factory=list)
    _listeners: list[Listener] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_records(cls, records: list[Record]) -> "KBStore":
        return cls(records={r.rid: copy.deepcopy(r) for r in records})

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "KBStore":
        return cls.from_records(load_records(path))

    def clone(self) -> "KBStore":
        return KBStore(records=copy.deepcopy(self.records), version=self.version, log=list(self.log))

    # -------------------------------------------------------------- queries
    def all(self, include_gated: bool = False) -> list[Record]:
        return [r for r in self.records.values() if include_gated or r.is_public]

    def public(self) -> list[Record]:
        return self.all(include_gated=False)

    def get(self, rid: str, include_gated: bool = False) -> Record | None:
        r = self.records.get(rid)
        if r is None:
            return None
        if not include_gated and not r.is_public:
            return None
        return r

    def by_type(self, rtype: str, include_gated: bool = False) -> list[Record]:
        return [r for r in self.all(include_gated) if r.rtype == rtype]

    def fingerprint(self) -> str:
        """Hash of the public content (detects drift between runs)."""
        h = hashlib.sha256()
        for rid in sorted(r.rid for r in self.public()):
            h.update(self.records[rid].content_hash().encode())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------ mutation
    def subscribe(self, fn: Listener) -> None:
        self._listeners.append(fn)

    def logical_time(self) -> str:
        return (EPOCH + timedelta(minutes=self.version)).isoformat(timespec="seconds")

    def apply(self, m: Mutation) -> Record | None:
        before = copy.deepcopy(self.records.get(m.rid))
        if m.op == "add":
            assert m.record is not None
            rec = Record.from_dict(m.record)
            self.records[rec.rid] = rec
        elif m.op == "remove":
            self.records.pop(m.rid, None)
        else:
            rec = self.records[m.rid]
            if m.op == "set_text":
                getattr(rec, m.key)[m.lang or "el"] = m.value
            elif m.op == "set_attr":
                if m.lang:
                    cur = rec.attrs.get(m.key)
                    if not isinstance(cur, dict):
                        cur = {}
                    cur[m.lang] = m.value
                    rec.attrs[m.key] = cur
                else:
                    rec.attrs[m.key] = m.value
            elif m.op == "set_status":
                rec.status = m.value
            elif m.op == "set_active":
                rec.is_active = bool(m.value)
            elif m.op == "set_price":
                rec.price = m.value
            elif m.op == "set_price_visible":
                rec.price_visible = bool(m.value)
            else:
                raise ValueError(f"unknown mutation op {m.op}")
        self.version += 1
        after = self.records.get(m.rid if m.op != "add" else (m.record or {}).get("rid", m.rid))
        if after is not None:
            after.updated_at = self.logical_time()
        self.log.append({"version": self.version, **m.to_dict()})
        for fn in self._listeners:
            fn(m, before, copy.deepcopy(after) if after else None, self.version)
        return after

    def apply_all(self, ms: list[Mutation]) -> None:
        for m in ms:
            self.apply(m)

    def dump_log(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.log, ensure_ascii=False, indent=1), encoding="utf-8")
