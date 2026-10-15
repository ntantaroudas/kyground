"""Command-line interface: ``kyground <command> [options]`` (or ``python -m kyground``).

Typical workflow (see README); nothing requires the platform's editors or users:

    kyground scrape-public                      # public snapshot (no key); the frozen snapshot of the study
    kyground probe-public                       # how the platform's public search matches text (no key)
    kyground build-bench                        # items, splits, automatic gold verification, queries
    kyground verify-gold                        # re-check the gold answers against the snapshot
    kyground scorer-suite --run runs/<id>       # controlled answers: scorer sensitivity and specificity
    kyground power
    kyground run accuracy --mock --limit 40     # dry run, no API spend
    kyground estimate-cost --run runs/<id>
    kyground tune-rag                           # C1 settings on the dev split, once
    kyground run accuracy                       # E1 (+ robustness forms)
    kyground run ablation                       # E4 search normalisation
    kyground run freshness                      # E2
    kyground run governance                     # E3
    kyground judge-export --run runs/<id>       # second scorer, graded by Claude sub-agents in a session (no key);
    kyground judge-import --run runs/<id>       #   or `kyground judge --run runs/<id>` through the API
    kyground judge-export --run runs/<id> --suite   # the controlled answers, the same way
    kyground report --runs runs/<id>
    kyground pseudonymise --candidates          # names to pseudonymise (checked by the authors)
    kyground pseudonymise --runs runs/<id>      # pseudonymised copy for the Zenodo release -> release/

Optional, with the data interface's key (DASK_API_KEY):

    kyground probe-api --discover               # recommends the replica settings
    kyground snapshot-diff --new <fresh public snapshot> --out runs/<id>/snapshot_diff.json
    kyground run accuracy --backend live --conditions C2 --forms el,en
    kyground conformance --run runs/<id>        # replica vs deployed interface
    kyground fetch-api / merge-api              # raw API dump, merged with the public snapshot

Optional manual checks: review-export / review-import (gold answers), human-export / human-agreement (answers).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from .experiments.factory import ROOT, Env, load_config, load_env, rpath


def _run_dir(cfg: dict, run_id: str | None, mock: bool) -> Path:
    rid = run_id or datetime.now().strftime("%Y%m%d-%H%M%S") + ("-mock" if mock else "")
    return rpath(cfg["paths"]["runs_dir"]) / rid


def cmd_scrape_public(a, cfg):
    from .sources.public_site import scrape
    out = Path(a.out or f"data/snapshots/public_{datetime.now():%Y%m%d}")
    recs = scrape(rpath(out), delay=a.delay, use_cache=not a.no_cache)
    print(f"{len(recs)} records -> {rpath(out)} (see quality_report.md)")


def cmd_probe_public(a, cfg):
    from .schema import load_records
    from .sources.public_probe import probe_public
    out = rpath(a.out or (Path(cfg["paths"]["runs_dir"]) / "public_probe"))
    s = probe_public(load_records(rpath(cfg["paths"]["snapshot"])), out, delay=a.delay)
    print(json.dumps(s, ensure_ascii=False, indent=1))
    print(f"-> {out}; replica.search_mode in configs/study.yaml should be {s['recommended_search_mode']}")


def cmd_fetch_api(a, cfg):
    from .sources.api_client import LiveAPI, snapshot
    api = LiveAPI.from_config(cfg.get("live_api", {}))
    out = Path(a.out or f"data/snapshots/api_{datetime.now():%Y%m%d}")
    print(json.dumps(snapshot(api, rpath(out)), indent=2))


def cmd_probe_api(a, cfg):
    from .sources.api_client import LiveAPI, discover_params, probe_normalisation
    import pandas as pd
    api = LiveAPI.from_config(cfg.get("live_api", {}))
    out = rpath(cfg["paths"]["runs_dir"]) / "api_probe"
    out.mkdir(parents=True, exist_ok=True)
    if a.discover:
        rep = discover_params(api)
        (out / "param_discovery.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    rows = probe_normalisation(api)
    df = pd.DataFrame(rows)
    df.to_csv(out / "normalisation_probe.csv", index=False)
    print(df.to_string(index=False))
    from .sources.api_client import probe_behaviour, recommend_replica
    import yaml
    beh = probe_behaviour(api)
    (out / "behaviour_probe.json").write_text(json.dumps(beh, ensure_ascii=False, indent=1), encoding="utf-8")
    env = Env.build(cfg, mock=True)
    rec = recommend_replica(rows, beh, env.store())
    (out / "recommended_replica.yaml").write_text(yaml.safe_dump(rec, allow_unicode=True, sort_keys=False),
                                                  encoding="utf-8")
    print(yaml.safe_dump(rec, allow_unicode=True, sort_keys=False))
    print(f"Review {out / 'recommended_replica.yaml'} and copy the settings into configs/study.yaml (replica:).")


def cmd_merge_api(a, cfg):
    from .schema import load_records
    from .sources.api_merge import merge_api_snapshot
    pub = load_records(rpath(a.public or cfg["paths"]["snapshot"]))
    out = rpath(a.out or f"data/snapshots/frozen_{datetime.now():%Y%m%d}")
    print(json.dumps(merge_api_snapshot(rpath(a.api), pub, out), ensure_ascii=False, indent=1))
    print(f"Review {out / 'merge_report.md'}, then set paths.snapshot to {out / 'records.jsonl'}")


def cmd_snapshot_diff(a, cfg):
    """Records changed on the platform since the frozen snapshot (run before the live checks)."""
    from .schema import load_records
    from .sources.snapshot_diff import diff_snapshots
    old = rpath(a.old or cfg["paths"]["snapshot"])
    base = json.loads(old.read_text(encoding="utf-8")) if old.suffix == ".json" else load_records(old)
    d = diff_snapshots(base, load_records(rpath(a.new)))
    d = {"old": str(a.old or cfg["paths"]["snapshot"]), "new": str(a.new), **d}
    out = rpath(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(d['added'])} added, {len(d['removed'])} removed, {len(d['changed'])} changed -> {out}")


def cmd_snapshot_hashes(a, cfg):
    """Identifier and content hash of every published record of the frozen snapshot (released as Extended data)."""
    from .schema import load_records
    from .sources.snapshot_diff import record_hashes
    h = record_hashes(load_records(rpath(cfg["paths"]["snapshot"])))
    out = rpath(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(h, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(h)} published records -> {out}")


def cmd_pseudonymise(a, cfg):
    import yaml
    from .release import candidates, pseudonymise
    from .schema import load_records
    if a.candidates:
        out = rpath(a.map or "data/release/pseudonyms.yaml")
        out.parent.mkdir(parents=True, exist_ok=True)
        cand = candidates(load_records(rpath(cfg["paths"]["snapshot"])))
        out.write_text(yaml.safe_dump(cand, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"{len(cand['names'])} names -> {out}; review every entry before the release")
        return
    srcs = [rpath(cfg["paths"]["benchmark_dir"]), rpath(cfg["paths"]["curated_items"])] + \
           [rpath(r) for r in (a.runs or [])] + ([rpath(a.results)] if a.results and rpath(a.results).exists() else [])
    rep = pseudonymise(rpath(a.map or "data/release/pseudonyms.yaml"), srcs, rpath(a.out), ROOT)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


def cmd_build_bench(a, cfg):
    from .bench.build import build
    from .schema import load_records
    snap = rpath(a.snapshot or cfg["paths"]["snapshot"])
    out = rpath(a.out or cfg["paths"]["benchmark_dir"])
    dec = rpath(cfg["paths"].get("review_decisions", "data/benchmark/review_decisions.yaml"))
    s = build(load_records(snap), rpath(a.curated or cfg["paths"]["curated_items"]), out, seed=cfg["seed"],
              decisions_path=dec, snapshot_label=str(Path(a.snapshot or cfg["paths"]["snapshot"])))
    print(json.dumps(s, ensure_ascii=False, indent=1))
    print(f"gold verification: {out / 'gold_verification.md'}")


def cmd_verify_gold(a, cfg):
    """Re-check the benchmark's gold answers against the snapshot and update the items' status."""
    from .bench.verify import apply_verification, verify_items, write_report
    from .schema import load_items, load_records, write_jsonl
    bdir = rpath(cfg["paths"]["benchmark_dir"])
    snap = a.snapshot or cfg["paths"]["snapshot"]
    items = load_items(bdir / "items.jsonl")
    res = verify_items(items, load_records(rpath(snap)))
    summary = write_report(res, bdir, snapshot=str(snap))
    if not a.dry_run:
        write_jsonl(bdir / "items.jsonl", apply_verification(items, res))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"-> {bdir / 'gold_verification.md'}" + (" (items.jsonl unchanged: --dry-run)" if a.dry_run else ""))


def cmd_review_export(a, cfg):
    from .bench.review import export_review
    from .schema import load_items
    items = load_items(rpath(cfg["paths"]["benchmark_dir"]) / "items.jsonl")
    out = rpath(a.out or (Path(cfg["paths"]["benchmark_dir"]) / "editor_review.xlsx"))
    print(export_review(items, out))


def cmd_review_import(a, cfg):
    from .bench.build import rebuild_queries
    from .bench.review import import_review
    bdir = rpath(cfg["paths"]["benchmark_dir"])
    dec = rpath(cfg["paths"].get("review_decisions", "data/benchmark/review_decisions.yaml"))
    counts = import_review(bdir / "items.jsonl", rpath(a.review), bdir / "items.jsonl", dec)
    n = rebuild_queries(bdir / "items.jsonl", bdir / "queries.jsonl")
    print(json.dumps(counts, ensure_ascii=False), f"; {n} queries regenerated")


def cmd_run(a, cfg):
    env = Env.build(cfg, mock=a.mock)
    env.quiet = not sys.stderr.isatty()
    run_dir = _run_dir(cfg, a.run_id, a.mock)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.experiment == "accuracy":
        from .experiments.accuracy import run_accuracy, score_accuracy
        ac = cfg["accuracy"]
        conds = a.conditions.split(",") if a.conditions else ac["conditions"]
        forms = a.forms.split(",") if a.forms else ac["forms"]
        plan = [(conds, forms, "")]
        if not a.conditions and not a.forms and ac.get("robustness"):   # the other surface forms, fewer conditions
            plan.append((ac["robustness"]["conditions"], ac["robustness"]["forms"], "robustness"))
        for conds, forms, part in plan:
            run_accuracy(env, run_dir, conds, split=a.split or ac.get("split", "test"), forms=forms,
                         reps=a.reps or ac.get("reps", 1), workers=a.workers or ac.get("workers", 8), limit=a.limit,
                         backend_kind=a.backend, answer_role=a.answer_role,
                         label=a.label or ("live" if a.backend == "live" else ""), force=a.force, part=part)
        score_accuracy(run_dir, rpath(cfg["paths"]["benchmark_dir"]) / "items.jsonl")
    elif a.experiment == "ablation":
        from .experiments.accuracy import run_accuracy, score_accuracy
        ab = cfg["ablation"]
        forms = a.forms.split(",") if a.forms else ab["forms"]
        for mode in ab["search_modes"]:
            run_accuracy(env, run_dir, ["C2"], split=a.split or "test", forms=forms, reps=a.reps or 1,
                         workers=a.workers or 8, limit=a.limit, search_mode=mode, label=f"search={mode}",
                         force=a.force)
        score_accuracy(run_dir, rpath(cfg["paths"]["benchmark_dir"]) / "items.jsonl")
    elif a.experiment == "freshness":
        from .experiments.freshness import run_freshness
        fc = cfg["freshness"]
        variants = a.conditions.split(",") if a.conditions else fc["variants"]
        run_freshness(env, run_dir, variants, n_events=a.limit or fc["n_events"], langs=fc.get("langs"),
                      workers=a.workers or fc.get("workers", 4), seed=cfg["seed"])
    elif a.experiment == "governance":
        from .experiments.governance import run_governance
        gc = cfg["governance"]
        variants = a.conditions.split(",") if a.conditions else gc["variants"]
        run_governance(env, run_dir, variants, n_per_kind=a.limit or gc["n_per_kind"], langs=gc.get("langs"),
                       workers=a.workers or gc.get("workers", 4), seed=cfg["seed"], phases=gc.get("phases"))
    print(f"run directory: {run_dir}")


def cmd_score(a, cfg):
    from .experiments.accuracy import score_accuracy
    from .experiments.freshness import rescore_freshness
    print(score_accuracy(rpath(a.run), rpath(cfg["paths"]["benchmark_dir"]) / "items.jsonl"))
    if (rpath(a.run) / "freshness.jsonl").exists() and (rpath(a.run) / "freshness_events.json").exists():
        print("freshness answers classified again:", rescore_freshness(rpath(a.run)))


def cmd_judge(a, cfg):
    from .schema import load_items, read_jsonl, write_jsonl
    from .scoring.judge import judge_rows
    env = Env.build(cfg, mock=a.mock)
    run = rpath(a.run)
    items = {i.iid: i for i in load_items(rpath(cfg["paths"]["benchmark_dir"]) / "items.jsonl")}
    rows = list(read_jsonl(run / "accuracy_scored.jsonl"))
    if a.forms:
        rows = [r for r in rows if r["form"] in a.forms.split(",")]
    out = judge_rows(env.clients["judge"], env.prompts, items, rows, workers=a.workers or 8)
    write_jsonl(run / "accuracy_judged.jsonl", out)
    print(f"{len(out)} rows judged -> {run / 'accuracy_judged.jsonl'}")


def cmd_scorer_suite(a, cfg):
    """Controlled answers with known correctness; grades them with the deterministic scorer (and the judge)."""
    from .schema import load_items
    from .scoring.suite import build_suite, judge_binary, score_suite, write_suite
    items = load_items(rpath(cfg["paths"]["benchmark_dir"]) / "items.jsonl")
    statuses = cfg.get("accuracy", {}).get("item_statuses") or []
    if statuses and not a.mock:
        items = [i for i in items if i.status in set(statuses)]
    by = {i.iid: i for i in items}
    rows = score_suite(build_suite(items), by)
    if a.judge:
        import random
        from concurrent.futures import ThreadPoolExecutor
        from .scoring.judge import judge_one
        env = Env.build(cfg, mock=a.mock)
        idx = list(range(len(rows)))
        if a.judge_sample and a.judge_sample < len(rows):
            idx = sorted(random.Random(cfg["seed"]).sample(idx, a.judge_sample))

        def work(i: int) -> tuple[int, dict]:
            r = rows[i]
            return i, judge_one(env.clients["judge"], env.prompts, by[r["iid"]], r["question"], r["answer"],
                                cache_tag=f"suite|{r['sid']}")

        with ThreadPoolExecutor(max_workers=a.workers or 8) as ex:
            for i, j in ex.map(work, idx):
                rows[i].update(j)
                rows[i]["judge_correct"] = judge_binary(j.get("judge_correctness", ""), rows[i]["qtype"]) \
                    if j.get("judge_correctness") else None
    out = rpath(a.run) if a.run else rpath(cfg["paths"]["runs_dir"]) / "scorer_suite"
    summary = write_suite(rows, out)
    print(json.dumps({k: v["overall"] for k, v in summary.items() if isinstance(v, dict)}, indent=1))
    print(f"-> {out / 'scorer_suite.json'}")


def _gen_paths(cfg: dict) -> tuple[Path, Path]:
    sc = cfg.get("session", {})
    return rpath(sc.get("queue", "runs/session/pending.jsonl")), rpath(sc.get("batches", "runs/session/batches"))


def _tokenizer(cfg: dict):
    enc = cfg.get("session", {}).get("token_estimate")
    if not enc:
        return None
    import tiktoken
    return tiktoken.get_encoding(enc)


def cmd_gen_export(a, cfg):
    """Queued model requests -> batches for Claude sub-agents (session generation, no API key)."""
    from .llm.session import export_queue
    from .schema import load_items
    sc = cfg.get("session", {})
    queue, bdir = _gen_paths(cfg)
    clusters = {i.iid: i.cluster for i in load_items(rpath(cfg["paths"]["benchmark_dir"]) / "items.jsonl")}
    s = export_queue(queue, bdir, clusters, seed=cfg["seed"], max_chars=int(sc.get("max_chars", 45000)),
                     max_requests=sc.get("max_requests"), tokenizer=_tokenizer(cfg))
    print(json.dumps(s, ensure_ascii=False))


def cmd_gen_import(a, cfg):
    from .llm.session import import_outputs
    _, bdir = _gen_paths(cfg)
    label = cfg.get("session", {}).get("model_label") or cfg["models"]["answer"]["model"]
    s = import_outputs(bdir, label, tokenizer=_tokenizer(cfg))
    print(json.dumps(s, ensure_ascii=False, indent=1))


def cmd_gen_status(a, cfg):
    """Batches still to be generated (no output file, or replies missing from the cache), with their sizes."""
    from .llm.session import open_batches
    _, bdir = _gen_paths(cfg)
    rows = open_batches(bdir)
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    print(f"{len(rows)} open batches")


def _session_dir(a) -> Path:
    return rpath(a.run) / ("judge_session_suite" if a.suite else "judge_session")


def cmd_judge_export(a, cfg):
    """Blinded batches for grading by Claude sub-agents in a session (no API key)."""
    from .scoring.session_judge import export_batches
    from .schema import load_items, read_jsonl
    from .systems.base import load_prompts
    items = {i.iid: i for i in load_items(rpath(cfg["paths"]["benchmark_dir"]) / "items.jsonl")}
    run = rpath(a.run)
    if a.suite:
        rows = list(read_jsonl(run / "scorer_suite.jsonl"))
        if not rows:
            raise SystemExit(f"no controlled suite in {run}: run `kyground scorer-suite --run {a.run}` first")
    js = cfg.get("judge_session", {})
    limit = None
    if a.suite:
        limit = a.sample or js.get("suite_sample")   # the first answers of the seeded random order: a random sample
    else:
        forms = a.forms.split(",") if a.forms else js.get("forms", ["el", "en"])
        rows = [r for r in read_jsonl(run / "accuracy_scored.jsonl") if r["form"] in forms]
        if not a.all:      # what the report uses: main conditions, test split
            rows = [r for r in rows if "@" not in str(r["condition"]) and items.get(r["iid"]) is not None
                    and items[r["iid"]].split == "test"]
            import random
            for form, n in (js.get("sample") or {}).items():   # e.g. a seeded sample of the English answers
                sub = [r for r in rows if r["form"] == form]
                if n < len(sub):
                    keep = {id(r) for r in random.Random(cfg["seed"]).sample(sub, n)}
                    rows = [r for r in rows if r["form"] != form or id(r) in keep]
    prompts = load_prompts(rpath(cfg.get("prompts", "configs/prompts.yaml")))
    s = export_batches(rows, items, prompts, _session_dir(a), "suite" if a.suite else "run",
                       batch_size=a.batch_size or js.get("batch_size", 50), seed=cfg["seed"], limit=limit)
    print(json.dumps(s, indent=1))
    print("Grade each batch with a Claude Sonnet sub-agent that reads only batches/rubric.md and its batch and "
          "writes batch_NNN.grades.jsonl; then run `kyground judge-import`.")


def cmd_judge_import(a, cfg):
    from .scoring.session_judge import merge_run_grades, merge_suite_grades, read_grades
    out = _session_dir(a)
    _, report = read_grades(out)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if a.check:
        return
    if report["batches_to_redo"] and not a.partial:
        raise SystemExit("some batches are incomplete: grade them again, or pass --partial to import what is there")
    model = f"{cfg['models']['judge']['model']} (Claude session, sub-agents)"
    if a.suite:
        summary = merge_suite_grades(rpath(a.run), out, model)
        print(json.dumps({k: v["overall"] for k, v in summary.items() if isinstance(v, dict)}, indent=1))
    else:
        print(f"{merge_run_grades(rpath(a.run), out, model)} judged rows -> {rpath(a.run) / 'accuracy_judged.jsonl'}")


def cmd_human_export(a, cfg):
    import pandas as pd
    from .schema import load_items, read_jsonl
    from .scoring.human import export_annotation, sample_rows
    run = rpath(a.run)
    src = run / ("accuracy_judged.jsonl" if (run / "accuracy_judged.jsonl").exists() else "accuracy_scored.jsonl")
    df = pd.DataFrame(list(read_jsonl(src)))
    items = {i.iid: i for i in load_items(rpath(cfg["paths"]["benchmark_dir"]) / "items.jsonl")}
    if a.disagreements:
        from .scoring.suite import judge_binary
        if "judge_correctness" not in df:
            raise SystemExit("run `kyground judge` first: --disagreements needs the judge's grades")
        jb = [judge_binary(lab, q) if lab else None for lab, q in zip(df["judge_correctness"].fillna(""), df["qtype"])]
        df = df.assign(judge_correct=jb)
        df = df[df["judge_correct"].notna() & (df["judge_correct"] != df["correct"])]
    smp = sample_rows(df, n=a.n, seed=cfg["seed"])
    smp.to_json(run / "annotation_sample.jsonl", orient="records", lines=True, force_ascii=False)
    for ann in ("A", "B"):
        print(export_annotation(smp, items, run / f"annotation_{ann}.xlsx", ann))


def cmd_human_agreement(a, cfg):
    import pandas as pd
    from .scoring.human import agreement, read_annotation
    run = rpath(a.run)
    smp = pd.read_json(run / "annotation_sample.jsonl", lines=True)
    opt = {n: (read_annotation(run / f"annotation_{n}.xlsx") if (run / f"annotation_{n}.xlsx").exists() else None)
           for n in ("B", "adjudicated")}
    rep = agreement(smp, read_annotation(run / "annotation_A.xlsx"), opt["B"], opt["adjudicated"])
    (run / "agreement.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


def cmd_conformance(a, cfg):
    from .experiments.conformance import conformance
    env = Env.build(cfg)
    store = env.store()
    rep = conformance(rpath(a.run) / "accuracy.jsonl", env.replica_backend(store), env.live_backend(),
                      max_calls=a.limit or 500, seed=cfg["seed"])
    (rpath(a.run) / "conformance.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep["summary"], indent=1))


def cmd_report(a, cfg):
    from .analysis.report import make_report
    runs = [rpath(r) for r in a.runs]
    key = make_report(runs, rpath(cfg["paths"]["benchmark_dir"]) / "items.jsonl", rpath(a.out),
                      rpath("paper/manuscript_ORE.md"), cfg)
    print(json.dumps({k: v for k, v in key.items() if not k.endswith("_CI")}, ensure_ascii=False, indent=1)[:3000])


def cmd_estimate_cost(a, cfg):
    """Rough token/cost estimate from a mock run's prompt sizes."""
    from .schema import read_jsonl
    from .analysis.report import cost_usd
    run = rpath(a.run) if a.run else None
    if not run or not (run / "accuracy.jsonl").exists():
        print("Run a mock accuracy run first: kyground run accuracy --mock --run-id estimate")
        return
    rows = list(read_jsonl(run / "accuracy.jsonl"))
    n = len(rows)
    tok = {}
    for r in rows:
        for m, u in (r.get("usage") or {}).items():
            t = tok.setdefault(r["base_condition"], {"in": 0, "out": 0, "calls": 0})
            t["in"] += u.get("input_tokens", 0)
            t["out"] += u.get("output_tokens", 0)
            t["calls"] += u.get("calls", 0)
    ac = cfg["accuracy"]
    print(f"mock rows: {n}; per-condition mean input tokens per question (mock approximation):")
    for c, t in tok.items():
        k = max(1, sum(1 for r in rows if r["base_condition"] == c))
        print(f"  {c}: in≈{t['in'] / k:,.0f} tokens, calls≈{t['calls'] / k:.1f}")
    print("Multiply by the number of queries x reps for the full run; prices in configs/study.yaml.")


def cmd_power(a, cfg):
    """Power of the primary paired comparison for the benchmark's actual clusters, by simulation."""
    from collections import Counter
    from .analysis import stats as S
    from .schema import load_items
    items = [i for i in load_items(rpath(cfg["paths"]["benchmark_dir"]) / "items.jsonl")
             if i.split == "test" and i.qtype != "unanswerable"]
    for unit in ("cluster", "component"):
        sizes = list(Counter(getattr(i, unit) or i.cluster for i in items).values())
        n, k = sum(sizes), len(sizes)
        de = S.design_effect(sizes, a.icc)
        print(f"{n} answerable test items in {k} {unit}s (largest {max(sizes)}); ICC {a.icc}: design effect "
              f"{de:.2f}, effective n {n / de:.0f}")
        for p10, p01 in ((0.175, 0.05), (0.15, 0.05), (0.125, 0.05)):
            pw = S.simulate_power(sizes, p10, p01, a.icc, alpha=a.alpha, n_sim=a.n_sim, seed=cfg["seed"])
            print(f"  discordant rates {p10:.3f} vs {p01:.3f} (difference {100 * (p10 - p01):.1f} pp): "
                  f"simulated power {pw:.2f} at alpha {a.alpha}")


def tune_rag_retrieval(cfg: dict, out_dir: Path, tolerance: float = 0.02) -> dict:
    """C1's k and chunk size chosen on the development split by retrieval alone (protocol v0.4, section 6): the
    gold-record recall of the retrieved context, averaged over the answerable dev items' Greek and English questions;
    the smallest context (k x chunk size) within ``tolerance`` of the best recall is chosen. No model is called."""
    import pandas as pd
    from .retrieval.index import VectorIndex
    from .schema import load_items, load_queries
    bdir = rpath(cfg["paths"]["benchmark_dir"])
    items = {i.iid: i for i in load_items(bdir / "items.jsonl")}
    dev = [q for q in load_queries(bdir / "queries.jsonl") if q.form in ("el", "en") and q.iid in items
           and items[q.iid].split == "dev" and items[q.iid].gold_rids and items[q.iid].qtype != "unanswerable"
           and items[q.iid].status in set(cfg.get("accuracy", {}).get("item_statuses") or [items[q.iid].status])]
    env = Env.build(cfg)
    store = env.store()
    rows = []
    for mc in (800, 1500):
        idx = VectorIndex(env.embedder, policy="never", source="public", max_chars=mc)
        idx.build(store)
        for k in (3, 5, 8):
            rec, full = [], []
            for q in dev:
                got = {c.rid for c, _s in idx.search(q.text, k=k)}
                gold = set(items[q.iid].gold_rids)
                rec.append(len(gold & got) / len(gold))
                full.append(float(gold <= got))
            rows.append({"k": k, "max_chars": mc, "n_queries": len(dev), "recall": round(sum(rec) / len(rec), 4),
                         "all_gold": round(sum(full) / len(full), 4), "context": k * mc})
    res = pd.DataFrame(rows)
    best = res["recall"].max()
    pick = res[res["recall"] >= best - tolerance].sort_values(["context", "k"]).iloc[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "rag_tuning.csv", index=False)
    choice = {"k": int(pick["k"]), "max_chars": int(pick["max_chars"]), "recall": float(pick["recall"]),
              "best_recall": float(best), "tolerance": tolerance, "n_queries": len(dev), "embedder": env.embedder.name}
    (out_dir / "rag_tuning.json").write_text(json.dumps(choice, indent=1), encoding="utf-8")
    return {"grid": rows, "choice": choice}


def cmd_tune_rag(a, cfg):
    """Choose C1's k and chunk size on the development split (protocol section 6), once, before the test run."""
    if a.retrieval:
        r = tune_rag_retrieval(cfg, rpath(cfg["paths"]["runs_dir"]) / (a.run_id or "tune"))
        for g in r["grid"]:
            print(json.dumps(g))
        print("chosen:", json.dumps(r["choice"]))
        return
    import copy
    from .experiments.accuracy import run_accuracy, score_accuracy
    from .schema import read_jsonl
    import pandas as pd
    grid = [(k, mc) for mc in (800, 1500) for k in (3, 5, 8)]
    rows = []
    for k, mc in grid:
        c = copy.deepcopy(cfg)
        c["retrieval"]["k"], c["retrieval"]["max_chars"] = k, mc
        env = Env.build(c, mock=a.mock)
        run = _run_dir(cfg, f"{a.run_id or 'tune'}/k{k}_c{mc}", a.mock)
        run_accuracy(env, run, ["C1"], split="dev", forms=["el"], reps=1, workers=a.workers or 8)
        df = pd.DataFrame(list(read_jsonl(score_accuracy(run, rpath(cfg["paths"]["benchmark_dir"]) / "items.jsonl"))))
        df = df[df["qtype"] != "unanswerable"]
        rows.append({"k": k, "max_chars": mc, "n": len(df), "acc": float(df["correct"].mean())})
    res = pd.DataFrame(rows).sort_values(["acc", "k", "max_chars"], ascending=[False, True, True])
    best = res.iloc[0]
    out = rpath(cfg["paths"]["runs_dir"]) / (a.run_id or "tune") / "rag_tuning.csv"
    res.to_csv(out, index=False)
    print(res.to_string(index=False))
    print(f"chosen: k={int(best['k'])}, max_chars={int(best['max_chars'])} -> set retrieval.k and "
          f"retrieval.max_chars in configs/study.yaml and record the choice in protocol/deviations.md ({out})")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_env()
    p = argparse.ArgumentParser(prog="kyground", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to study.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scrape-public"); s.add_argument("--out"); s.add_argument("--delay", type=float, default=1.0)
    s.add_argument("--no-cache", action="store_true"); s.set_defaults(fn=cmd_scrape_public)
    s = sub.add_parser("probe-public"); s.add_argument("--out"); s.add_argument("--delay", type=float, default=1.0)
    s.set_defaults(fn=cmd_probe_public)
    s = sub.add_parser("fetch-api"); s.add_argument("--out"); s.set_defaults(fn=cmd_fetch_api)
    s = sub.add_parser("probe-api"); s.add_argument("--discover", action="store_true"); s.set_defaults(fn=cmd_probe_api)
    s = sub.add_parser("merge-api"); s.add_argument("--api", required=True); s.add_argument("--public")
    s.add_argument("--out"); s.set_defaults(fn=cmd_merge_api)
    s = sub.add_parser("snapshot-diff"); s.add_argument("--old", help="records.jsonl or released hashes (.json)")
    s.add_argument("--new", required=True); s.add_argument("--out", required=True); s.set_defaults(fn=cmd_snapshot_diff)
    s = sub.add_parser("snapshot-hashes"); s.add_argument("--out", default="release/snapshot_record_hashes.json")
    s.set_defaults(fn=cmd_snapshot_hashes)
    s = sub.add_parser("pseudonymise"); s.add_argument("--candidates", action="store_true")
    s.add_argument("--map"); s.add_argument("--runs", nargs="*"); s.add_argument("--results", default="paper/results")
    s.add_argument("--out", default="release"); s.set_defaults(fn=cmd_pseudonymise)
    s = sub.add_parser("build-bench"); s.add_argument("--snapshot"); s.add_argument("--curated"); s.add_argument("--out")
    s.set_defaults(fn=cmd_build_bench)
    s = sub.add_parser("verify-gold"); s.add_argument("--snapshot")
    s.add_argument("--dry-run", action="store_true", help="write the report but leave items.jsonl unchanged")
    s.set_defaults(fn=cmd_verify_gold)
    s = sub.add_parser("review-export"); s.add_argument("--out"); s.set_defaults(fn=cmd_review_export)
    s = sub.add_parser("review-import"); s.add_argument("--review", required=True); s.set_defaults(fn=cmd_review_import)
    s = sub.add_parser("run")
    s.add_argument("experiment", choices=["accuracy", "ablation", "freshness", "governance"])
    s.add_argument("--conditions"); s.add_argument("--forms"); s.add_argument("--split")
    s.add_argument("--reps", type=int); s.add_argument("--workers", type=int); s.add_argument("--limit", type=int)
    s.add_argument("--mock", action="store_true", help="deterministic mock model, no API calls")
    s.add_argument("--backend", choices=["replica", "live"], default="replica")
    s.add_argument("--answer-role", default="answer", help="answer | answer_open (Krikri)")
    s.add_argument("--label", default=""); s.add_argument("--run-id")
    s.add_argument("--force", action="store_true", help="resume a run directory despite a changed configuration")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("score"); s.add_argument("--run", required=True); s.set_defaults(fn=cmd_score)
    s = sub.add_parser("judge"); s.add_argument("--run", required=True); s.add_argument("--forms")
    s.add_argument("--workers", type=int); s.add_argument("--mock", action="store_true"); s.set_defaults(fn=cmd_judge)
    s = sub.add_parser("gen-export"); s.set_defaults(fn=cmd_gen_export)
    s = sub.add_parser("gen-import"); s.set_defaults(fn=cmd_gen_import)
    s = sub.add_parser("gen-status"); s.set_defaults(fn=cmd_gen_status)
    s = sub.add_parser("judge-export"); s.add_argument("--run", required=True)
    s.add_argument("--suite", action="store_true", help="the controlled answer suite instead of the run's answers")
    s.add_argument("--forms"); s.add_argument("--all", action="store_true", help="also labelled runs and dev items")
    s.add_argument("--sample", type=int, help="--suite only: grade a seeded random sample of this many answers")
    s.add_argument("--batch-size", type=int); s.set_defaults(fn=cmd_judge_export)
    s = sub.add_parser("judge-import"); s.add_argument("--run", required=True)
    s.add_argument("--suite", action="store_true"); s.add_argument("--check", action="store_true")
    s.add_argument("--partial", action="store_true"); s.set_defaults(fn=cmd_judge_import)
    s = sub.add_parser("scorer-suite"); s.add_argument("--run")
    s.add_argument("--judge", action="store_true", help="also grade the suite with the LLM judge")
    s.add_argument("--judge-sample", type=int); s.add_argument("--workers", type=int)
    s.add_argument("--mock", action="store_true"); s.set_defaults(fn=cmd_scorer_suite)
    s = sub.add_parser("human-export"); s.add_argument("--run", required=True); s.add_argument("--n", type=int, default=100)
    s.add_argument("--disagreements", action="store_true",
                   help="sample only answers on which the deterministic scorer and the judge disagree")
    s.set_defaults(fn=cmd_human_export)
    s = sub.add_parser("human-agreement"); s.add_argument("--run", required=True); s.set_defaults(fn=cmd_human_agreement)
    s = sub.add_parser("conformance"); s.add_argument("--run", required=True); s.add_argument("--limit", type=int)
    s.set_defaults(fn=cmd_conformance)
    s = sub.add_parser("report"); s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--out", default="paper/results"); s.set_defaults(fn=cmd_report)
    s = sub.add_parser("estimate-cost"); s.add_argument("--run"); s.set_defaults(fn=cmd_estimate_cost)
    s = sub.add_parser("power"); s.add_argument("--icc", type=float, default=0.3)
    s.add_argument("--alpha", type=float, default=0.05); s.add_argument("--n-sim", type=int, default=2000)
    s.set_defaults(fn=cmd_power)
    s = sub.add_parser("tune-rag"); s.add_argument("--run-id"); s.add_argument("--workers", type=int)
    s.add_argument("--retrieval", action="store_true", help="choose by gold-record recall on dev (no model calls)")
    s.add_argument("--mock", action="store_true"); s.set_defaults(fn=cmd_tune_rag)

    a = p.parse_args(argv)
    cfg = load_config(a.config)
    a.fn(a, cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
