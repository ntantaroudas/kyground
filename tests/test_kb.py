"""Store, gating, replica API semantics, index refresh policies, canaries."""

from kyground.kb.canaries import insert_canaries, make_canaries, make_edit_events, old_new_state
from kyground.kb.replica_api import ReplicaAPI, ReplicaBackend, parse_wkt_bbox
from kyground.kb.serialize import kb_text, public_view, raw_text
from kyground.kb.store import KBStore, Mutation
from kyground.retrieval.embedder import HashEmbedder
from kyground.retrieval.index import VectorIndex
from kyground.schema import Record, bil


def _kb() -> KBStore:
    recs = [
        Record(rid="product:honey", rtype="product", slug="honey", title=bil("Θυμαρίσιο Μέλι Κυθήρων 450gr"),
               body=bil("Μέλι από άγριο θυμάρι."), attrs={"producer": bil("ΠΛΑΣΜΑΤΙΚΟΣ"), "category": bil("ΜΕΛΙ")},
               price=12.5, price_visible=False),
        Record(rid="product:oil", rtype="product", slug="oil", title=bil("ΜΟΥΖΟΥΡΙ 750ml", "MOUZOURI 750ml"),
               body=bil("Ελαιόλαδο από ελιές Κορωνέικης ποικιλίας."), attrs={"producer": bil("Συνεταιρισμός Λιβαδίου")},
               price=9.9, price_visible=True),
        Record(rid="product:draft", rtype="product", slug="draft", title=bil("Μέλι Σπάρτου"), status="draft",
               attrs={"producer": bil("Κρυφός Παραγωγός")}),
        Record(rid="experience:bees", rtype="experience", slug="bees", title=bil("Γνωρίστε τις μέλισσες"),
               attrs={"location_name": bil("Μητάτα"), "duration_hours": 2.5, "max_people": 12},
               lat=36.246, lon=22.993),
        Record(rid="experience:far", rtype="experience", slug="far", title=bil("Εμπειρία στην Αθήνα με μέλισσες"),
               attrs={"duration_hours": 1.0}, lat=37.98, lon=23.72),
        Record(rid="crop:olive", rtype="crop", slug="olive", title=bil("Ελιά", "Olive"),
               attrs={"harvest_period": bil("Νοέμβριος - Ιανουάριος", "November - January"),
                      "crop_type": bil("Ελιά", "Olive")}),
        Record(rid="calendar:m11", rtype="calendar", slug="m11", title=bil("Έναρξη συγκομιδής — Ελιά"),
               attrs={"month": 11, "activity": bil("Έναρξη συγκομιδής"), "crop_name": bil("Ελιά"), "crop_slug": "olive"}),
        Record(rid="faq:1", rtype="faq", slug="1", title=bil("Πότε μαζεύουμε τις ελιές;"),
               body=bil("Από τα μέσα Νοεμβρίου έως τον Ιανουάριο.")),
    ]
    return KBStore.from_records(recs)


def test_gating_and_views():
    s = _kb()
    assert "product:draft" not in {r.rid for r in s.public()}
    v = public_view(s.get("product:honey"), "el")
    assert v["price"] == "Τιμή κατόπιν επικοινωνίας" and "12" not in str(v)
    assert public_view(s.get("product:oil"), "en")["price"] == "€9.90"
    assert "Κρυφός" not in kb_text(s.public())
    assert "internal price: 12.50" in raw_text(s.records["product:honey"])


def test_replica_search_modes():
    s = _kb()
    api = ReplicaAPI(s, search_mode="phrase_casefold")
    assert api.product_search("μέλι")["count"] == 1
    assert api.product_search("θυμαρισιο")["count"] == 0     # accent-sensitive like SQL ILIKE
    assert api.product_search("μελι")["count"] == 1          # ...but the category field is "ΜΕΛΙ" (no accent)
    assert api.product_search("Σπάρτου")["count"] == 0       # draft never returned
    api.search_mode = "phrase_unaccent"
    assert api.product_search("θυμαρισιο")["count"] == 1
    api.search_mode = "tokens_all_stem"
    assert api.product_search("θυμαρίσιου μελιού")["count"] == 1
    assert api.knowledge_search("ελιές")["count"] == 1


def test_replica_structured_and_spatial():
    s = _kb()
    api = ReplicaAPI(s)
    assert api.seasonal_calendar(month=11)["count"] == 1
    assert api.seasonal_calendar(month=5)["count"] == 0
    ci = api.crop_info("olive")
    assert ci["count"] == 1 and any(x["id"] == "calendar:m11" for x in ci["related"])
    assert parse_wkt_bbox("POLYGON((22.85 35.80, 23.35 35.80, 23.35 36.40, 22.85 36.40, 22.85 35.80))") == \
        (22.85, 35.8, 23.35, 36.4)
    res = api.experience_search("μέλισσ")
    ids = [x["id"] for x in res["results"]]
    assert ids == ["experience:bees"]                         # the Athens experience is outside the envelope
    be = ReplicaBackend(api)
    out = be.call("search_local_products", {"query": "ΜΟΥΖΟΥΡΙ", "lang": "en"}, "el")
    assert out["results"][0]["title"] == "MOUZOURI 750ml"


def test_index_refresh_policies():
    for policy, expect_new in (("never", False), ("manual", False), ("immediate_upsert", True)):
        s = _kb()
        idx = VectorIndex(HashEmbedder(dim=256), policy=policy)
        idx.build(s)
        if policy == "immediate_upsert":
            idx.attach(s)
        s.apply(Mutation("set_text", "faq:1", key="body", lang="el", value="Από τον Οκτώβριο έως τον Δεκέμβριο."))
        texts = " ".join(c.text for c in idx.chunks)
        assert ("Οκτώβριο" in texts) == expect_new
        if policy == "manual":
            idx.refresh()
            assert "Οκτώβριο" in " ".join(c.text for c in idx.chunks)
    s = _kb()
    idx = VectorIndex(HashEmbedder(dim=256), policy="immediate_append")
    idx.build(s)
    idx.attach(s)
    s.apply(Mutation("set_text", "faq:1", key="body", lang="el", value="Από τον Οκτώβριο έως τον Δεκέμβριο."))
    texts = " ".join(c.text for c in idx.chunks)
    assert "Οκτώβριο" in texts and "Νοεμβρίου" in texts      # both versions coexist


def test_revocation_removes_from_upsert_index():
    s = _kb()
    idx = VectorIndex(HashEmbedder(dim=256), policy="immediate_upsert")
    idx.build(s)
    idx.attach(s)
    assert any(c.rid == "product:oil" for c in idx.chunks)
    s.apply(Mutation("set_status", "product:oil", value="draft"))
    assert not any(c.rid == "product:oil" for c in idx.chunks)


def test_canaries_and_edits():
    cs = make_canaries(n_per_kind=2, seed=1)
    assert len(cs) == 8 and len({c.cid for c in cs}) == 8
    for c in cs:
        for p in c.probes:          # no marker may occur in the probe text
            for m in c.markers:
                assert m not in p["text"], (c.cid, m)
    s = _kb()
    insert_canaries(s, cs, gated=True)
    pub = {r.rid for r in s.public()}
    assert all(c.record.rid not in pub for c in cs if c.kind != "hidden_price")
    hp = [c for c in cs if c.kind == "hidden_price"][0]
    assert hp.record.rid in pub and not s.records[hp.record.rid].price_visible
    evs = make_edit_events(_kb(), n=5, seed=3)
    assert evs, "no edit events generated"
    for ev in evs:
        assert ev.item_old.question == ev.item_new.question
    ev = [e for e in evs if e.field == "harvest_period"] or evs
    e = ev[0]
    st = old_new_state("Η συγκομιδή γίνεται από τον Ιανουάριο έως τον Μάρτιο.", e)
    assert st["state"] in ("updated", "stale", "mixed", "neither")


def test_canary_markers_unique_and_absent_elsewhere():
    """No canary marker may occur in another canary, in the real knowledge base, or in the public content
    available while that canary is gated (otherwise disclosures would be counted by construction)."""
    from pathlib import Path
    from kyground.scoring.leakage import leaks
    from kyground.schema import load_records
    snap = Path(__file__).resolve().parents[1] / "data/snapshots/public_20261005/records.jsonl"
    base = KBStore.from_records(load_records(snap)) if snap.exists() else _kb()
    cs = make_canaries(n_per_kind=12, seed=20261005)
    markers = [m for c in cs for m in c.markers]
    assert len(markers) == len(set(markers))
    assert len({c.hidden_price for c in cs if c.hidden_price is not None}) == 24
    assert len({c.hidden_hours for c in cs if c.hidden_hours is not None}) == 12
    real = kb_text(base.public())
    for c in cs:  # absent from the real knowledge base
        assert not leaks(real, c.markers, c.hidden_price, hidden_hours=c.hidden_hours)["leak"], c.cid
    # G1: every canary gated
    s = base.clone()
    insert_canaries(s, cs, gated=True)
    txt = kb_text(s.public())
    for c in cs:
        assert not leaks(txt, c.markers, c.hidden_price, hidden_hours=c.hidden_hours)["leak"], c.cid
    # G2/G3: all canaries public except the one revoked
    for c in cs:
        s = base.clone()
        insert_canaries(s, cs, gated=False)
        s.apply_all(c.gate)
        txt = kb_text(s.public())
        assert not leaks(txt, c.markers, c.hidden_price, hidden_hours=c.hidden_hours)["leak"], c.cid
