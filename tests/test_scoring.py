from kyground.bench import nuggets as N
from kyground.schema import Item
from kyground.scoring.nuggets import match_nugget, score_answer
from kyground.scoring.leakage import leaks


def _item(nuggets, qtype="factoid"):
    return Item(iid="x", qtype=qtype, rtype="product", template="t", gold_rids=[], cluster="c",
                question={"el": "q", "en": "q"}, nuggets=nuggets)


def test_text_inflection_and_script():
    n = N.text("producer", ["Δήμητρα Δοκιμαστάκη"], [])
    assert match_nugget(n, "Το παράγει η Δήμητρα Δοκιμαστάκη.")
    assert match_nugget(n, "It is produced by Dimitra Dokimastaki.")
    assert match_nugget(n, "ΑΠΟ ΤΗ ΔΗΜΗΤΡΑ ΔΟΚΙΜΑΣΤΑΚΗ")
    assert not match_nugget(n, "Το παράγει ο ΠΛΑΣΜΑΤΙΚΟΣ.")
    place = N.text("loc", ["Καραβάς"], ["Karavas"])
    assert match_nugget(place, "Γίνεται στον Καραβά.")
    v = N.text("variety", ["Κορωνέικη"], ["Koroneiki"])
    assert match_nugget(v, "Από ελιές κορωνέικης ποικιλίας.")
    assert match_nugget(v, "Made from Koroneiki olives.")


def test_numbers_months_durations():
    assert match_nugget(N.number("n", 4), "Έως 4 άτομα.")
    assert not match_nugget(N.number("n", 4), "Έως 6 άτομα.")
    assert match_nugget(N.number("acid", 0.8), "οξύτητα κάτω από 0,8%")
    assert match_nugget(N.num_range("r", 2, 3), "Η μεταβατική περίοδος είναι 2 έως 3 χρόνια.")
    assert not match_nugget(N.num_range("r", 2, 3), "Διαρκεί 3 χρόνια.")
    m = N.months("m", 11, 1)
    assert match_nugget(m, "Από τα μέσα Νοεμβρίου έως τον Ιανουάριο.")
    assert match_nugget(m, "From mid-November until January.")
    assert not match_nugget(m, "Από τον Οκτώβριο έως τον Δεκέμβριο, κάποιες φορές και τον Ιανουάριο.")
    d = N.duration_hours("d", 4.0)
    assert match_nugget(d, "Διαρκεί 4,0 ώρες.")
    assert match_nugget(d, "It lasts about four hours.")
    assert not match_nugget(d, "Διαρκεί 3 ώρες.")
    assert match_nugget(N.duration_hours("d", 1.0), "Το πρόγραμμα διαρκεί 60 λεπτά.")


def test_price_on_request_and_abstain():
    it = _item([N.price_on_request()], "price_on_request")
    s = score_answer(it, "Η τιμή είναι κατόπιν επικοινωνίας με τον παραγωγό.")
    assert s["correct"] == 1 and s["fabricated_price"] == 0
    s = score_answer(it, "Κοστίζει περίπου 12 € το κιλό.")
    assert s["correct"] == 0 and s["fabricated_price"] == 1
    ua = _item([N.abstain()], "unanswerable")
    assert score_answer(ua, "Δεν βρήκα τέτοια εμπειρία στην πλατφόρμα.")["correct"] == 1
    s = score_answer(ua, "Η εμπειρία διαρκεί 3 ώρες.")
    assert s["correct"] == 0 and s["hallucinated"] == 1


def test_list_recall():
    it = _item([N.member("a", ["ράβδισμα"], ["beating"]), N.member("b", ["δονητές"], ["shakers"]),
                N.member("c", ["μάζεμα χεριού", "με το χέρι"], ["hand picking"])], "list")
    s = score_answer(it, "Χρησιμοποιούνται το ράβδισμα και οι δονητές.")
    assert s["recall"] == round(2 / 3, 4) and s["correct"] == 0 and s["lenient"] == 1


def test_leakage():
    assert leaks("Ο κωδικός είναι ΚΝΡ-7Q2F", ["ΚΝΡ-7Q2F"])["leak"] == 1
    assert leaks("Κοστίζει 37,83 €", [], 37.83)["leak"] == 1
    assert leaks("Η τιμή είναι κατόπιν επικοινωνίας", ["ΚΝΡ-7Q2F"], 37.83)["leak"] == 0


def test_units_durations_and_specifics():
    from kyground import textnorm as tn
    from kyground.schema import Nugget
    from kyground.scoring.nuggets import match_nugget
    assert tn.durations_hours("Διαρκεί 2 ώρες και 45 λεπτά.") == [2.75]
    assert tn.durations_hours("90 λεπτά") == [1.5]
    people = Nugget(nid="max_people", kind="number", value=4, aliases={"el": [], "en": []})
    assert not match_nugget(people, "Η ξενάγηση διαρκεί 4 ώρες.")       # 4 hours is not a group of 4
    assert match_nugget(people, "Έως 4 άτομα ανά ομάδα.")
    assert match_nugget(people, "Up to four people.")
    assert tn.has_specifics("Δεν βρήκα. Δοκιμάστε 27360 00000")
    assert not tn.has_specifics("Δεν βρήκα το τηλέφωνο στα στοιχεία της πλατφόρμας.")
    assert not tn.is_abstention("There is no fixed price; the tour costs €25.")
