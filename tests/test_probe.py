"""The probe battery must recover the deployed interface's behaviour; here a replica with known settings plays
the deployed interface."""

from pathlib import Path

import pytest

from kyground.kb.replica_api import ReplicaAPI
from kyground.kb.store import KBStore
from kyground.schema import load_records
from kyground.sources.api_client import probe_behaviour, probe_normalisation, recommend_replica

SNAP = Path(__file__).resolve().parents[1] / "data/snapshots/public_20261005/records.jsonl"


class FakeLive:
    """Answers like LiveAPI.get, from a replica with fixed settings."""

    def __init__(self, api: ReplicaAPI) -> None:
        self.api = api

    def get(self, endpoint, path_args=None, **kw):
        a, lang, lim = self.api, kw.get("lang", "el"), kw.get("limit")
        if endpoint == "knowledge/search":
            r = a.knowledge_search(kw.get("query", ""), lang, lim)
        elif endpoint == "products/search":
            r = a.product_search(kw.get("query", ""), lang, kw.get("category"), lim)
        elif endpoint == "experiences/search":
            r = a.experience_search(kw.get("query", ""), lang, kw.get("wkt"), lim)
        elif endpoint == "glossary/lookup":
            r = a.glossary_lookup(kw.get("term", ""), lang, lim)
        elif endpoint == "crops/{type}":
            r = a.crop_info(path_args["type"], lang)
        elif endpoint == "knowledge/seasonal-calendar":
            r = a.seasonal_calendar(kw.get("month"), kw.get("crop_type"), lang)
        else:
            r = {"results": []}
        return {**r, "_status": 200}


@pytest.mark.skipif(not SNAP.exists(), reason="public snapshot not available")
@pytest.mark.parametrize("truth", [
    {"search_mode": "phrase_unaccent", "filter_mode": "unaccent", "spatial_missing_coords": "exclude"},
    {"search_mode": "phrase_casefold", "filter_mode": "casefold", "spatial_missing_coords": "include"},
])
def test_probe_recovers_replica_settings(truth):
    store = KBStore.from_records(load_records(SNAP))
    live = FakeLive(ReplicaAPI(store, **truth))
    rec = recommend_replica(probe_normalisation(live), probe_behaviour(live), store)["replica"]
    for k, v in truth.items():
        assert rec[k] == v, (k, rec)
