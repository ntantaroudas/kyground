"""Session generation: requests queued by the runners, exported in batches, answered (here by the mock model, in
the sub-agents' output format), imported into the cache, and the experiments completed over several rounds."""

import json
from pathlib import Path

import pandas as pd

from kyground.bench.build import build
from kyground.experiments.accuracy import run_accuracy, score_accuracy
from kyground.experiments.factory import Env, load_config
from kyground.experiments.freshness import run_freshness
from kyground.experiments.governance import run_governance
from kyground.llm.mock import MockClient
from kyground.llm.session import export_queue, import_outputs, open_batches, request_kind, subagent_prompt
from kyground.schema import load_items, read_jsonl, write_jsonl
from test_kb import _kb


def _fake_subagents(bdir: Path) -> int:
    """Answer every open batch the way a sub-agent would: one JSON line per request, in the batch's order."""
    payloads = {}
    for q in bdir.parent.glob("*.exported*.jsonl"):
        for line in q.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            payloads[e["key"]] = e["payload"]
    mock = MockClient("fake-subagent")
    n = 0
    manifest = [json.loads(l) for l in (bdir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    for m in manifest:
        out = bdir / f"{m['batch']}.out.jsonl"
        if out.exists():
            continue
        lines = []
        for rid, key in m["ids"].items():
            p = payloads[key]
            res = mock._call(p["messages"], tools=p.get("tools"), params=p.get("params") or {})
            calls = [{"name": c.name, "arguments": c.arguments} for c in res.tool_calls]
            if m["kind"] == "rewrite":
                calls = calls[:1] or [{"name": p["tools"][0]["function"]["name"], "arguments": {"query": "x"}}]
            lines.append(json.dumps({"id": rid, "text": "" if calls else res.text, "tool_calls": calls}
                                    if m["kind"] != "answer" else {"id": rid, "text": res.text}, ensure_ascii=False))
        out.write_text("\n".join([json.dumps({"checks": m["checks"]})] + lines) + "\n", encoding="utf-8")
        n += 1
    return n


def test_rounds_complete_the_experiments(tmp_path: Path):
    snap = tmp_path / "records.jsonl"
    write_jsonl(snap, _kb().all(include_gated=True))
    bench = tmp_path / "bench"
    build(list(_kb().all(include_gated=True)), None, bench, dev_frac=0.0)
    queue, bdir = tmp_path / "session" / "pending.jsonl", tmp_path / "session" / "batches"
    cfg = load_config(overrides={
        "paths": {"snapshot": str(snap), "benchmark_dir": str(bench), "runs_dir": str(tmp_path / "runs"),
                  "cache_dir": str(tmp_path / "cache")},
        "models": {"answer": {"provider": "session", "model": "sess-model", "max_output_tokens": 4000},
                   "router": {"provider": "session", "model": "sess-model", "max_output_tokens": 2000},
                   "judge": {"provider": "mock", "model": "mock-judge"}},
        "session": {"queue": str(queue), "batches": str(bdir)},
        "embedder": {"provider": "hash"}})
    items = load_items(bench / "items.jsonl")
    clusters = {i.iid: i.cluster for i in items}
    run = tmp_path / "runs" / "r"
    conds = ["C0", "C1", "C1q", "C2", "C3", "C4"]
    rounds = 0
    for rounds in range(1, 25):   # a C2 conversation turn takes up to three rounds (two router steps, answer)
        env = Env.build(cfg)
        env.quiet = True
        run_accuracy(env, run, conds, split="test", forms=["el", "en"], workers=2)
        run_accuracy(env, run, ["C2"], split="test", forms=["el"], workers=2, search_mode="phrase_unaccent",
                     label="search=phrase_unaccent")
        run_freshness(env, run, ["C2", "C1-batch"], n_events=2, workers=1)
        run_governance(env, run, ["C2", "C1-batch"], n_per_kind=1, workers=1,
                       phases={"C2": ["G1", "G23"], "C1-batch": ["G1"]})
        s = export_queue(queue, bdir, clusters, max_chars=20000)
        if not s["requests"]:
            break
        assert _fake_subagents(bdir) == s["batches"]
        r = import_outputs(bdir, "sess-label")
        assert r["stored"] == s["requests"] and not r["batches_to_redo"]
    assert 3 <= rounds < 24 and not open_batches(bdir)
    # every query of every condition answered once; labelled rows kept apart
    df = pd.DataFrame(list(read_jsonl(score_accuracy(run, bench / "items.jsonl"))))
    n_q = len({q for q in df[df["label"] == ""]["qid"]})
    assert df[df["label"] == ""].groupby("condition").size().eq(n_q).all() and set(conds) <= set(df["condition"])
    assert (df["error"] == "").all() and not df.duplicated(["condition", "qid", "rep"]).any()
    fr = list(read_jsonl(run / "freshness.jsonl"))
    assert len(fr) == 2 * 2 * 5                      # variants x events x phases
    gv = pd.DataFrame(list(read_jsonl(run / "governance.jsonl")))
    assert set(gv[gv["variant"] == "C1-batch"]["phase"]) == {"G1_static"}
    assert {"G0_visible", "G3_in_thread", "G2_revoked_after_refresh"} <= set(gv[gv["variant"] == "C2"]["phase"])
    # batches: one kind, one system prompt, no two requests about the same cluster, edit or canary
    manifest = [json.loads(l) for l in (bdir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    payloads = {}
    for q in queue.parent.glob("*.exported*.jsonl"):
        for line in q.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            payloads[e["key"]] = e
    from kyground.llm.session import _topic
    for m in manifest:
        es = [payloads[k] for k in m["ids"].values()]
        assert {request_kind(e["payload"]) for e in es} == {m["kind"]}
        assert len({e["payload"]["messages"][0]["content"] for e in es}) == 1
        topics = [_topic(e["payload"]["tag"], clusters) for e in es]
        assert len(topics) == len(set(topics))
        text = (bdir / f"{m['batch']}.md").read_text(encoding="utf-8")
        assert max(len(l) for l in text.splitlines()) <= 1200 and "## END OF REQUESTS" in text
    # C2 and C3 share the router's decisions: one first-step router request per accuracy query
    routers = [e for e in payloads.values() if e["payload"]["tag"].startswith("acc|") and
               e["payload"]["tag"].endswith("|router0")]
    n_c2 = len(df[df["condition"] == "C2"])
    assert len(routers) == n_c2
    # the ablation reuses the main run's requests whenever its tool results are the same
    abl = [e for e in payloads.values() if e["payload"]["tag"].startswith("acc|C2||") and "|router" not in
           e["payload"]["tag"]]
    assert len(abl) < n_c2 + len(df[df["condition"] == "C2@search=phrase_unaccent"])
    # a further round replays everything from the cache and queues nothing
    env = Env.build(cfg)
    env.quiet = True
    run_accuracy(env, run, conds, split="test", forms=["el", "en"], workers=2)
    assert not queue.exists()
    assert "Read tool" in subagent_prompt(bdir, manifest[0]["batch"])
    m0 = manifest[0]
    assert m0["checks"] and all(f": {c}\u27e7" in (bdir / f"{m0['batch']}.md").read_text(encoding="utf-8")
                                for c in m0["checks"])


def test_wrong_check_codes_reject_the_batch(tmp_path: Path):
    from kyground.llm.session import SessionClient
    queue, bdir = tmp_path / "q.jsonl", tmp_path / "b"
    cl = SessionClient("m", cache_dir=tmp_path / "cache", queue=queue)
    import pytest
    from kyground.llm.session import PendingRequest
    for i in range(3):
        with pytest.raises(PendingRequest):
            cl.complete([{"role": "system", "content": "S"}, {"role": "user", "content": f"q{i}"}],
                        cache_tag=f"acc|C0||it{i}#el|rep0")
    s = export_queue(queue, bdir, {})
    m = json.loads((bdir / "manifest.jsonl").read_text().splitlines()[0])
    lines = [json.dumps({"id": rid, "text": "x"}) for rid in m["ids"]]
    (bdir / f"{m['batch']}.out.jsonl").write_text("\n".join([json.dumps({"checks": ["ZZZZ"]})] + lines))
    r = import_outputs(bdir, "lbl")
    assert r["stored"] == 0 and m["batch"] in r["batches_to_redo"]
    (bdir / f"{m['batch']}.out.jsonl").write_text("\n".join([json.dumps({"checks": m["checks"]})] + lines))
    assert import_outputs(bdir, "lbl")["stored"] == s["requests"] == 3
