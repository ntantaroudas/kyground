"""LLM-as-judge (second scorer).

A model from a different family than the answer models (Anthropic Claude by
default; answers come from OpenAI GPT-5-family and, optionally, Llama-Krikri)
grades each answer against the reference facts with a fixed rubric
(``configs/prompts.yaml: judge``) and structured JSON output. Its agreement with
the deterministic scorer on the experiments' answers, its accuracy on the
controlled answer suite (``scoring/suite.py``) and the primary contrasts graded
by it are reported as a sensitivity analysis of the primary outcome.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from tqdm import tqdm

from ..bench.nuggets import describe
from ..llm.base import LLMClient
from ..schema import Item


def reference_text(item: Item) -> str:
    facts = "; ".join(describe(n, "el") for n in item.nuggets)
    ref = item.answer_ref.get("el") or ""
    ref_en = item.answer_ref.get("en") or ""
    out = f"{ref} [EN: {ref_en}] — facts: {facts}"
    return out.strip()


def judge_one(client: LLMClient, prompts: dict, item: Item, question: str, answer: str,
              cache_tag: str = "") -> dict:
    j = prompts["judge"]
    msgs = [{"role": "system", "content": j["system"]},
            {"role": "user", "content": j["user"].format(question=question, qtype=item.qtype,
                                                         reference=reference_text(item), answer=answer or "(empty)")}]
    res = client.complete(msgs, response_schema=j["schema"], max_output_tokens=800, cache_tag=cache_tag)
    out = {"judge_model": res.model, "judge_error": res.error}
    try:
        data = json.loads(res.text)
        out.update({"judge_correctness": data.get("correctness", ""),
                    "judge_unsupported": int(bool(data.get("unsupported_specifics"))),
                    "judge_rationale": data.get("rationale", "")})
    except (json.JSONDecodeError, TypeError):
        out.update({"judge_correctness": "", "judge_unsupported": None, "judge_rationale": res.text[:300]})
    return out


def judge_rows(client: LLMClient, prompts: dict, items: dict[str, Item], rows: Iterable[dict],
               workers: int = 8) -> list[dict]:
    rows = list(rows)

    def work(r: dict) -> dict:
        it = items[r["iid"]]
        return {**r, **judge_one(client, prompts, it, r["question"], r.get("answer", ""),
                                 cache_tag=f"judge|{r['condition']}|{r['qid']}|{r.get('rep', 0)}")}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(tqdm(ex.map(work, rows), total=len(rows), desc="judge"))
