# KyGround evaluation harness

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23388561.svg)](https://doi.org/10.5281/zenodo.23388561)

Software of the KyGround study, which compares tool-calling retrieval with vector retrieval-augmented generation
(RAG) for a small, community-edited Greek–English agricultural knowledge base, the DASK platform of the Kythera pilot
of the PoliRURAL Plus project.

The harness downloads a snapshot of the platform's public records, builds the bilingual benchmark and verifies every
reference answer against the snapshot, runs six retrieval conditions in four experiments, scores the answers with a
Greek-aware rule-based scorer and an LLM judge, audits batched generation for cross-request contamination and writes
the result tables and figures.

| Code | Condition | What the answer model sees |
|---|---|---|
| C0 | closed book | nothing local |
| C1 | vector RAG | the 8 chunks (up to 800 characters) nearest to the latest user message (BGE-M3, exact cosine search) |
| C1q | vector RAG, router-written query | as C1, for the query the router writes when calling one search tool |
| C2 | tool retrieval (the deployed design, reconstructed) | results of up to two `kythera` tools chosen and filled by the router, run by a replica of the data interface |
| C3 | hybrid | the C2 tool results and the C1 chunks |
| C4 | full context | the whole published knowledge base in the prompt |

The experiments are E1, accuracy on 198 bilingual questions in up to nine forms (standard Greek, English, Greek
without accents, Greek in capitals, three Greeklish schemes, another grammatical form and colloquial wording), E2,
freshness after edits, E3, disclosure of hidden content, and E4, the search settings of the data interface.

## Related outputs

| Output | Link |
|---|---|
| Data deposit (benchmark, run outputs, extended data, CC BY 4.0) | https://doi.org/10.5281/zenodo.23339394 |
| Preprint, accuracy and robustness to how users type Greek | https://doi.org/10.48550/arXiv.2610.08205 |
| Preprint, freshness and access control | https://doi.org/10.20944/preprints202610.0645.v1 |
| Archived release of this software, version 1.0.0 | https://doi.org/10.5281/zenodo.23388561 |
| All versions of this software | https://doi.org/10.5281/zenodo.23388560 |

## Installation

Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                              # offline, with a stand-in model
```

Tests that need the frozen snapshot or the benchmark are skipped when `data/` is absent. The optional extras
`local-embeddings` (sentence-transformers, for BGE-M3) and `session` (tiktoken, for token counts) are installed with
`pip install -e ".[local-embeddings,session]"`.

## Reproducing the result tables from the data deposit

Download and unpack `KyGround_data_v1.0.zip` from the data deposit, then, in the folder of this repository
(Linux, macOS or WSL):

```bash
D=path/to/KyGround_data_v1.0/underlying_data
mkdir -p data/benchmark/v1.0 runs/suite
cp "$D"/benchmark/items.jsonl "$D"/benchmark/queries.jsonl "$D"/benchmark/summary.json data/benchmark/v1.0/
cp -r "$D"/runs/main runs/main
cp "$D"/scorer_suite/scorer_suite.json "$D"/scorer_suite/scorer_suite.jsonl runs/suite/
cp -r "$D"/scorer_suite/judge_session runs/suite/judge_session_suite
kyground report --runs runs/main runs/suite      # -> paper/results/tables and paper/figures
```

Every point estimate equals the published one. The bootstrap confidence intervals can differ in the last digit,
because the resampling is drawn over question clusters in sorted order and pseudonymisation renamed a few clusters.
Run on the original, unpseudonymised data, this code reproduces every table of `underlying_data/results` exactly.

## Running the experiments again

`kyground --help` and the docstring of `src/kyground/cli.py` list the commands, from `kyground scrape-public`
(a snapshot of the platform's current public pages) and `kyground build-bench` to `kyground run`, `kyground score`,
`kyground judge-export`/`judge-import` and `kyground report`. Models are reached through sub-agents of a Claude
session, as in the study, which needs no API key, or through the Anthropic API, the OpenAI API or any
OpenAI-compatible server. Copy `.env.example` to `.env` for the keys of these back-ends, and never commit `.env`.
The platform is updated continuously, so a new snapshot may differ from the frozen one, and generating the replies
again does not reproduce them word for word. The protocol, its deviations and the record hashes of the frozen
snapshot are in the data deposit, in `extended_data`.

## Data protection

The snapshot, the benchmark before pseudonymisation and the run outputs name producers, contact persons and hosts,
some of whom are natural persons. Keep `data/`, `runs/` and `release/` out of any public repository, which
`.gitignore` does. `kyground pseudonymise` and `tools/make_zenodo_release.py` made the public data deposit.

In this public release, the names of real persons in test fixtures and code comments and an invented telephone
number of the island in the controlled scorer suite were replaced by fictitious ones. This changes no result.

## Licence and citation

The code is released under the MIT License (`LICENSE`). Please cite the archived release and the article,
as given in `CITATION.cff`.

The work was supported by the European Union's Horizon Europe research and innovation programme through the
PoliRURAL Plus project (Grant Agreement No. 101136910), via its DEVELOP cascade-funding open call.
