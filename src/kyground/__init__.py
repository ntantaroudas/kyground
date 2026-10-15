"""KyGround: tool-calling retrieval versus vector RAG on a community-edited,
bilingual (Greek/English) agricultural knowledge base.

The package is organised as follows:

- ``schema``       canonical records, benchmark items, nuggets and queries
- ``textnorm``     Greek text normalisation, Greeklish schemes, months, numbers
- ``sources``      public-site scraper, public-search probe, snapshot comparison, live data-API adapter
- ``kb``           knowledge store with editorial gating, replica of the data API,
                   serialisation, governance canaries and edit events
- ``bench``        benchmark construction (templates, surface forms) and automatic gold verification
- ``llm``          model clients (Claude-session sub-agents, OpenAI, Anthropic, OpenAI-compatible, mock)
- ``retrieval``    embedders and a vector index with refresh policies
- ``systems``      the compared conditions C0-C4
- ``experiments``  accuracy/robustness, freshness, governance, API probes
- ``scoring``      deterministic nugget scoring, leakage detection, LLM judge, controlled answer suite
- ``analysis``     statistics, tables and figures
"""

__version__ = "1.0.0"
