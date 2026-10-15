import json
from pathlib import Path

import yaml

from kyground.release import name_regex, pseudonymise


def test_name_regex_matches_spellings():
    pat = name_regex(["Δήμητρα Δοκιμαστάκη", "Dimitra Dokimastaki"])
    for t in ("Παραγωγός: Δήμητρα Δοκιμαστάκη.", "ΔΗΜΗΤΡΑ ΔΟΚΙΜΑΣΤΑΚΗ", "δημητρα δοκιμαστακη", "by Dimitra Dokimastaki"):
        assert pat.search(t), t
    assert not pat.search("Δήμητρα Παπαδοπούλου")


def test_name_regex_matches_inflection_slugs_and_mixed_script():
    pat = name_regex(["Μάρκος Ανυπαρκτές", "Markos Anyparktes"])
    for t in ("του Μάρκου Ανυπαρκτέ", "family:flower-of-salt-anyparktes", "markos_anyparktes", "ΑΝΥΠΑΡΚΤΕΣ"):
        assert pat.search(t), t
    assert not pat.search("Μάρκος Παπαδάκης") and not pat.search("μια μάρκα")
    assert name_regex(["Δήμητρα Δοκιμαστάκη"]).search("της Δήμητρας Δοκιμαστάκη")
    mixed = name_regex(["ΜΑΡΙΑ ΦΑΝΤAΣΤΙΚΟΥ"])          # Latin A inside a Greek surname, as on the platform
    assert mixed.search("Μαρία Φανταστικού") and mixed.search("ΦΑΝΤΑΣΤΙΚΟΥ")
    assert not name_regex(["ΓΕΩΡΓΙΟΣ ΠΛΑΣΜΑΤΙΚΟΣ"]).search("Βιώσιμη Γεωργία")


def test_pseudonymise_replaces_everywhere(tmp_path: Path):
    src = tmp_path / "bench"
    src.mkdir()
    (src / "items.jsonl").write_text(json.dumps({"q": "Ποιος παράγει το μέλι;", "gold": ["Σοφία Υποθετικού"],
                                                 "answer": "Το παράγει η ΣΟΦΙΑ ΥΠΟΘΕΤΙΚΟΥ."}, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
    m = tmp_path / "map.yaml"
    m.write_text(yaml.safe_dump({"names": {"Σοφία Υποθετικού": {"label": "Producer C", "natural_person": True,
                                                             "variants": ["Sofia Ypothetikou"]}}}, allow_unicode=True),
                 encoding="utf-8")
    rep = pseudonymise(m, [src], tmp_path / "out", tmp_path)
    row = json.loads((tmp_path / "out" / "bench" / "items.jsonl").read_text(encoding="utf-8"))
    assert row["gold"] == ["Producer C"] and "Producer C" in row["answer"]
    assert rep["residual"]["Producer C"] == 0 and rep["fragments"]["Producer C"] == 0
