"""What changed on the platform since the frozen snapshot.

Every experiment runs on the frozen snapshot, so later edits on the platform cannot affect it. The live checks
(C2 against the deployed data interface, and the replay of tool calls) do see the platform as it is on the day they
run. Re-scrape the public pages just before them (``kyground scrape-public``) and compare the two snapshots:

    kyground snapshot-diff --new data/snapshots/public_YYYYMMDD/records.jsonl --out runs/<id>/snapshot_diff.json

``kyground report`` then leaves out of the live comparison every item whose gold records were changed or removed,
and reports how many. Readers without the frozen snapshot can compare a new snapshot with the released content
hashes of its records (``kyground snapshot-hashes``; ``--old snapshot_record_hashes.json``).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..schema import Record

CONTENT_FIELDS = ("title", "body", "summary", "attrs", "price", "price_visible", "status", "is_active", "lat", "lon",
                  "parent", "exclude_reason")


def content_hash(r: Record) -> str:
    """Hash of a record's content fields only (not its URL, source or retrieval time)."""
    payload = json.dumps({f: getattr(r, f) for f in CONTENT_FIELDS}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def record_hashes(records: list[Record]) -> dict[str, str]:
    """Identifier and content hash of every published record: released in place of the snapshot itself, so that
    anyone can tell which records of a new snapshot are unchanged without receiving the content."""
    return {r.rid: content_hash(r) for r in sorted(records, key=lambda x: x.rid) if r.is_public}


def diff_snapshots(old: list[Record] | dict[str, str], new: list[Record]) -> dict[str, Any]:
    """Published records added, removed or changed between two snapshots. ``old`` may be a list of records or the
    released hashes (``record_hashes``); with hashes, changed fields cannot be named."""
    n = {r.rid: r for r in new if r.is_public}
    changed = []
    if isinstance(old, dict):
        o_ids = set(old)
        for rid in sorted(o_ids & set(n)):
            if old[rid] != content_hash(n[rid]):
                changed.append({"rid": rid, "fields": ["content"]})
    else:
        o = {r.rid: r for r in old if r.is_public}
        o_ids = set(o)
        for rid in sorted(o_ids & set(n)):
            fields = [f for f in CONTENT_FIELDS if getattr(o[rid], f) != getattr(n[rid], f)]
            if fields:
                changed.append({"rid": rid, "fields": fields})
    return {"n_old": len(o_ids), "n_new": len(n), "added": sorted(set(n) - o_ids), "removed": sorted(o_ids - set(n)),
            "changed": changed}


def affected_items(diff: dict, items: list) -> set[str]:
    """Items with a gold record that was changed or removed."""
    bad = set(diff.get("removed", [])) | {c["rid"] for c in diff.get("changed", [])}
    return {it.iid for it in items if set(it.gold_rids) & bad}
