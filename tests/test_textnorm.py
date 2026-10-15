from kyground import textnorm as tn


def test_norm_accents_sigma_case():
    assert tn.norm("Ελιές, ΚΥΘΉΡΩΝ!") == "ελιεσ κυθηρων"
    assert tn.norm("ελιες") == tn.norm("ελιές") == tn.norm("ΕΛΙΕΣ")
    assert tn.norm("Μαΐου") == "μαιου"


def test_casefold_only_keeps_accents():
    # casefold also folds final sigma (more lenient than SQL ILIKE); accents are kept
    assert tn.casefold_only("ΕΛΙΈΣ") == tn.casefold_only("ελιές") == "ελιέσ"
    assert "ελιεσ" not in tn.casefold_only("Ελιές")


def test_stem_inflection():
    assert tn.stem("ελιεσ") == tn.stem("ελια") == tn.stem("ελιων") == "ελι"
    assert tn.stem("κυθηρων") == tn.stem("κυθηρα")
    assert tn.stem("καραβασ") == tn.stem("καραβα")


def test_stem_match():
    assert tn.stem_match("Καραβάς", "Η εμπειρία γίνεται στον Καραβά.")
    assert tn.stem_match("θυμαρίσιο μέλι", "του θυμαρίσιου μελιού")
    assert not tn.stem_match("μέλι", "Γνωρίστε τις μέλισσες")
    assert not tn.stem_match("Καραβάς", "Αγία Πελαγία")


def test_greeklish_schemes():
    q = "Πότε μαζεύουμε τις ελιές;"
    assert tn.to_greeklish_phon(q) == "pote mazevoume tis elies;"
    assert tn.to_greeklish_visual(q) == "pote mazeuoume tis elies;"
    assert tn.to_elot(q) == "Pote mazevoume tis elies;"
    assert tn.to_elot("ΜΟΥΖΟΥΡΙ") == "MOUZOURI"
    assert tn.to_greeklish_visual("θυμάρι") == "8umari"
    assert tn.to_greeklish_phon("Χωριό") == "xorio"
    assert tn.latin_key("ΜΟΥΖΟΥΡΙ") == tn.norm("Mouzouri")


def test_caps_and_sigma():
    assert tn.caps_no_accents("Πότε μαζεύουμε τις ελιές;") == "ΠΟΤΕ ΜΑΖΕΥΟΥΜΕ ΤΙΣ ΕΛΙΕΣ;"
    assert tn.sigma_error("τις ελιές") == "τισ ελιέσ"


def test_months():
    assert tn.months_in("Χειμώνας (Νοέμβριος-Ιανουάριος)") == [11, 1]
    assert tn.parse_month_range("Φθινόπωρο (Οκτώβριος-Νοέμβριος)") == (10, 11)
    assert tn.month_set("from mid-November to January") == {11, 1}
    assert tn.month_set("τον Γενάρη και τον Μάη") == {1, 5}
    assert tn.months_between(11, 1) == [11, 12, 1]
    # "may" as modal verb is accepted as May (documented limitation), "mar" is not
    assert 3 not in tn.month_set("mar the surface")


def test_numbers_and_durations():
    assert 4.0 in tn.numbers_in("4,0 ώρες")
    assert 0.8 in tn.numbers_in("κάτω από 0.8%")
    assert 950.0 in tn.numbers_in("ΜΕΛΙ 950gr")
    assert set(tn.numbers_in("50-100 λίτρα")) >= {50.0, 100.0}
    assert tn.durations_hours("Διαρκεί 4,0 ώρες") == [4.0]
    assert tn.durations_hours("It lasts four hours") == [4.0]
    assert tn.durations_hours("περίπου 90 λεπτά") == [1.5]
    assert tn.currency_amounts("κοστίζει 12,50 €") == [12.5]
    assert tn.currency_amounts("price is €8") == [8.0]
    assert tn.currency_amounts("4,0 ώρες") == []


def test_cues():
    assert tn.is_abstention("Δεν βρήκα πληροφορίες για αυτό τον παραγωγό.")
    assert tn.is_abstention("I couldn't find that producer in the platform.")
    assert tn.mentions_on_request("Η τιμή είναι κατόπιν επικοινωνίας με τον παραγωγό.")
    assert tn.mentions_on_request("The price is available on request.")
