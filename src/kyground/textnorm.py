"""Greek-aware text normalisation, Greeklish schemes, month and number parsing.

All matching in the harness (nugget scoring, replica search modes, leakage
detection) goes through :func:`norm`, which applies, in order: Unicode NFKC,
case folding, removal of combining marks (tonos, dialytika), folding of final
sigma to medial sigma, replacement of punctuation with spaces and whitespace
collapsing. ``norm("Ελιές, ΚΥΘΉΡΩΝ!") == "ελιεσ κυθηρων"``.

Three Latin-script ("Greeklish") schemes are provided for the robustness
experiment:

- :func:`to_elot`            ELOT 743 / ISO 843-style transcription
                             (``θ→th, χ→ch, ξ→x, η→i, υ→y, ου→ou``)
- :func:`to_greeklish_phon`  informal phonetic Greeklish
                             (``χ→x, ξ→ks, η/ι/υ/ει/οι→i, αι→e``)
- :func:`to_greeklish_visual` informal visual Greeklish
                             (``θ→8, ω→w, η→h, ξ→3, ψ→4, υ→u``)
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# ---------------------------------------------------------------------------
# Core normalisation
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w%€]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def strip_accents(s: str) -> str:
    """Remove combining marks (tonos, dialytika, ...) but keep case."""
    if not s:
        return ""
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


@lru_cache(maxsize=200_000)
def norm(s: str) -> str:
    """Canonical matching form: casefolded, accentless, final sigma folded."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()                      # casefold maps ς -> σ
    s = strip_accents(s)
    s = s.replace("ς", "σ")
    s = s.replace("_", " ")
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def casefold_only(s: str) -> str:
    """Case-insensitive but accent-sensitive form (emulates SQL ILIKE)."""
    if not s:
        return ""
    return unicodedata.normalize("NFC", s).casefold()


def tokens(s: str) -> list[str]:
    return norm(s).split()


STOP_EL = set(norm(w) for w in """
ο η το οι τα του της των τον την τη τους τις στο στη στην στον στα στους στις σε με για και να
που πού πως πώς ποιο ποια ποιος ποιες ποιοι ποιον ποιων ποιου ποιας πότε ποτέ ποσο πόσο πόσα πόσες
πόσοι πόση είναι ειναι έχει εχει από απο ένα ενα μία μια ένας ενας μου σου μας σας θα δεν μη μην αν ή
κατά κατα ως ως έως εως μέχρι μεχρι υπάρχει υπαρχει υπάρχουν υπαρχουν κάποιο κάποια κάποιος αυτό αυτή
αυτός αυτά αυτές αυτοί εκεί εδώ πολύ λίγο κάθε ήταν θέλω θελω μπορώ μπορω μπορείτε μπορεις μπορείς
ξέρετε ξερεις ξέρεις πείτε πες πείτε μου λίγο σχετικά σχετικα τι τί
""".split())

STOP_EN = set("""
a an the of to in on at for and or is are was were be been with by from as that this these those it its
what which who whom when where how much many does do did can could would should will i you we they he
she me my our your their there here about tell please any some
""".split())

STOPWORDS = STOP_EL | STOP_EN


def content_tokens(s: str, min_len: int = 3) -> list[str]:
    return [t for t in tokens(s) if t not in STOPWORDS and len(t) >= min_len]


# ---------------------------------------------------------------------------
# Light stemming (normalised tokens)
# ---------------------------------------------------------------------------

_EL_SUFFIXES = sorted(
    """ιουσ ιεσ ιων ιου ιοι ουσ ουν εων εωσ εισ ατοσ ματα ματοσ ματων εσ ων ου ασ ησ οσ οι αι ιο ια
    α ε η ι ο υ ω σ""".split(),
    key=len,
    reverse=True,
)
_EN_SUFFIXES = ["ies", "ing", "ed", "es", "s"]
_GREEK_RE = re.compile(r"[Ͱ-Ͽἀ-῿]")


def is_greek(s: str) -> bool:
    return bool(_GREEK_RE.search(s or ""))


@lru_cache(maxsize=200_000)
def stem(tok: str) -> str:
    """Very light suffix stripping on an already-normalised token.

    It is intentionally conservative (stems keep >= 3 characters); it is used
    only for prefix-containment matching, never for display.
    """
    if len(tok) <= 3:
        return tok
    if is_greek(tok):
        for suf in _EL_SUFFIXES:
            if tok.endswith(suf) and len(tok) - len(suf) >= 3:
                return tok[: -len(suf)]
        return tok
    for suf in _EN_SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def stems(s: str, min_len: int = 3) -> list[str]:
    return [stem(t) for t in content_tokens(s, min_len=min_len)]


def stem_match(needle: str, haystack: str, min_stem: int = 3) -> bool:
    """True if every content token of ``needle`` matches some token of
    ``haystack`` by stem-prefix containment (either direction)."""
    n_st = [stem(t) for t in content_tokens(needle, min_len=2)]
    if not n_st:
        return False
    h_tok = tokens(haystack)
    h_st = {stem(t) for t in h_tok}
    for ns in n_st:
        if len(ns) < min_stem:
            # very short tokens must match exactly as tokens
            if ns not in h_st and ns not in h_tok:
                return False
            continue
        if ns in h_st:
            continue
        ok = False
        for ht in h_tok:
            hs = stem(ht)
            # forward: answer token is an inflected form of the needle stem
            if ht.startswith(ns) and len(ht) <= len(ns) + 4:
                ok = True
                break
            # backward: needle stem carries an extra short ending
            if len(hs) >= max(min_stem, len(ns) - 2) and ns.startswith(hs):
                ok = True
                break
        if not ok:
            return False
    return True


# ---------------------------------------------------------------------------
# Surface-form perturbations
# ---------------------------------------------------------------------------

def caps_no_accents(s: str) -> str:
    """Greek all-caps typing convention (accents dropped)."""
    return strip_accents(s).upper()


_FINAL_SIGMA_RE = re.compile(r"ς")


def sigma_error(s: str) -> str:
    """Replace final sigma with medial sigma (a common typing/OCR error)."""
    return _FINAL_SIGMA_RE.sub("σ", s)


_VOICELESS = set("θκξπστφχψ")
_VOWELS = set("αεηιουω")


def _translit(s: str, scheme: str) -> str:
    """Shared engine for Greek -> Latin schemes. Works on accentless text."""
    src = strip_accents(s)
    out: list[str] = []
    i = 0
    n = len(src)

    def low(ch: str) -> str:
        return ch.lower()

    def nxt(k: int) -> str:
        return low(src[k]) if k < n else ""

    def at_word_start(k: int) -> bool:
        return k == 0 or not src[k - 1].isalpha()

    single = {
        "elot": dict(α="a", β="v", γ="g", δ="d", ε="e", ζ="z", η="i", θ="th", ι="i", κ="k", λ="l", μ="m",
                     ν="n", ξ="x", ο="o", π="p", ρ="r", σ="s", ς="s", τ="t", υ="y", φ="f", χ="ch", ψ="ps", ω="o"),
        "phon": dict(α="a", β="v", γ="g", δ="d", ε="e", ζ="z", η="i", θ="th", ι="i", κ="k", λ="l", μ="m",
                     ν="n", ξ="ks", ο="o", π="p", ρ="r", σ="s", ς="s", τ="t", υ="i", φ="f", χ="x", ψ="ps", ω="o"),
        "visual": dict(α="a", β="b", γ="g", δ="d", ε="e", ζ="z", η="h", θ="8", ι="i", κ="k", λ="l", μ="m",
                       ν="n", ξ="3", ο="o", π="p", ρ="r", σ="s", ς="s", τ="t", υ="u", φ="f", χ="x", ψ="4", ω="w"),
    }[scheme]

    while i < n:
        ch = src[i]
        c = low(ch)
        c2 = nxt(i + 1)
        if c not in single and c not in ("ϊ", "ϋ"):
            out.append(ch if scheme == "elot" else ch.lower())
            i += 1
            continue
        pair = c + c2
        rep = None
        consumed = 2
        if scheme in ("elot", "phon", "visual") and pair == "ου":
            rep = "ou" if scheme != "visual" else "ou"
        elif pair in ("αυ", "ευ", "ηυ") and scheme in ("elot", "phon"):
            base = {"αυ": "a", "ευ": "e", "ηυ": "i"}[pair]
            after = nxt(i + 2)
            voiceless = (after == "" or after in _VOICELESS or not after.isalpha())
            rep = base + ("f" if voiceless else "v")
        elif pair == "γγ" and scheme != "visual":
            rep = "ng"
        elif pair == "γκ" and scheme == "elot":
            rep = "gk"
        elif pair == "γκ" and scheme == "phon":
            rep = "g" if at_word_start(i) else "gk"
        elif pair == "γξ" and scheme == "elot":
            rep = "nx"
        elif pair == "γχ" and scheme == "elot":
            rep = "nch"
        elif pair == "μπ" and scheme in ("elot", "phon"):
            rep = "b" if at_word_start(i) else "mp"
        elif pair == "ντ" and scheme == "phon":
            rep = "d" if at_word_start(i) else "nt"
        elif scheme == "phon" and pair in ("αι",):
            rep = "e"
        elif scheme == "phon" and pair in ("ει", "οι", "υι"):
            rep = "i"
        if rep is None:
            consumed = 1
            if c in ("ϊ",):
                rep = "i"
            elif c in ("ϋ",):
                rep = "y" if scheme == "elot" else ("i" if scheme == "phon" else "u")
            else:
                rep = single[c]
        # case handling: ELOT keeps capitalisation, informal schemes are lower-case
        if scheme == "elot" and ch.isupper():
            nxt_is_upper = (i + 1 < n and src[i + 1].isupper())
            rep = rep.upper() if nxt_is_upper or len(rep) == 1 else rep.capitalize()
        out.append(rep)
        i += consumed
    return "".join(out)


def to_elot(s: str) -> str:
    return _translit(s, "elot")


def to_greeklish_phon(s: str) -> str:
    return _translit(s, "phon").lower()


def to_greeklish_visual(s: str) -> str:
    return _translit(s, "visual").lower()


def latin_key(s: str) -> str:
    """Normalised Latin rendering used to match Greek aliases against
    Latin-script answers (e.g. ``ΜΟΥΖΟΥΡΙ`` vs ``Mouzouri``)."""
    return norm(to_elot(s))


# ---------------------------------------------------------------------------
# Months
# ---------------------------------------------------------------------------

_MONTH_FORMS: dict[int, list[str]] = {
    1: ["ιανουαριοσ", "ιανουαριου", "ιανουαριο", "γεναρησ", "γεναρη", "ιαν", "january", "jan"],
    2: ["φεβρουαριοσ", "φεβρουαριου", "φεβρουαριο", "φλεβαρησ", "φλεβαρη", "φεβ", "february", "feb"],
    3: ["μαρτιοσ", "μαρτιου", "μαρτιο", "μαρτησ", "μαρτη", "μαρ", "march", "mar"],
    4: ["απριλιοσ", "απριλιου", "απριλιο", "απριλησ", "απριλη", "απρ", "april", "apr"],
    5: ["μαιοσ", "μαιου", "μαιο", "μαησ", "μαη", "may"],
    6: ["ιουνιοσ", "ιουνιου", "ιουνιο", "ιουνησ", "ιουνη", "ιουν", "june", "jun"],
    7: ["ιουλιοσ", "ιουλιου", "ιουλιο", "ιουλησ", "ιουλη", "ιουλ", "july", "jul"],
    8: ["αυγουστοσ", "αυγουστου", "αυγουστο", "αυγ", "august", "aug"],
    9: ["σεπτεμβριοσ", "σεπτεμβριου", "σεπτεμβριο", "σεπτεμβρησ", "σεπτεμβρη", "σεπ", "september", "sep", "sept"],
    10: ["οκτωβριοσ", "οκτωβριου", "οκτωβριο", "οκτωβρησ", "οκτωβρη", "οκτ", "october", "oct"],
    11: ["νοεμβριοσ", "νοεμβριου", "νοεμβριο", "νοεμβρησ", "νοεμβρη", "νοε", "november", "nov"],
    12: ["δεκεμβριοσ", "δεκεμβριου", "δεκεμβριο", "δεκεμβρησ", "δεκεμβρη", "δεκ", "december", "dec"],
}
# Short English forms that are also common words are only accepted in a
# month context; we drop the ambiguous ones from free-text matching.
_AMBIGUOUS_SHORT = {"mar", "may", "jun", "dec", "sep", "oct", "nov", "jan", "feb", "apr", "aug", "jul"}
_FORM_TO_MONTH: dict[str, int] = {}
for _m, _forms in _MONTH_FORMS.items():
    for _f in _forms:
        _FORM_TO_MONTH[_f] = _m

MONTH_NAME = {
    "el": {1: "Ιανουάριος", 2: "Φεβρουάριος", 3: "Μάρτιος", 4: "Απρίλιος", 5: "Μάιος", 6: "Ιούνιος",
           7: "Ιούλιος", 8: "Αύγουστος", 9: "Σεπτέμβριος", 10: "Οκτώβριος", 11: "Νοέμβριος", 12: "Δεκέμβριος"},
    "el_acc": {1: "Ιανουάριο", 2: "Φεβρουάριο", 3: "Μάρτιο", 4: "Απρίλιο", 5: "Μάιο", 6: "Ιούνιο",
               7: "Ιούλιο", 8: "Αύγουστο", 9: "Σεπτέμβριο", 10: "Οκτώβριο", 11: "Νοέμβριο", 12: "Δεκέμβριο"},
    "en": {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July",
           8: "August", 9: "September", 10: "October", 11: "November", 12: "December"},
}


def months_in(text: str, allow_short: bool = False) -> list[int]:
    """Months mentioned in ``text`` in order of appearance (duplicates kept)."""
    out: list[int] = []
    for tok in tokens(text):
        m = _FORM_TO_MONTH.get(tok)
        if m is None:
            continue
        if tok in _AMBIGUOUS_SHORT and not allow_short:
            # accept "May" (full word) but not "mar"/"dec" fragments
            if tok != "may":
                continue
        out.append(m)
    return out


def month_set(text: str) -> set[int]:
    return set(months_in(text))


def months_between(start: int, end: int) -> list[int]:
    """Inclusive month range, wrapping over the new year (11 -> 1)."""
    out = [start]
    m = start
    while m != end:
        m = m % 12 + 1
        out.append(m)
        if len(out) > 12:
            break
    return out


def parse_month_range(text: str) -> tuple[int, int] | None:
    """Parse strings such as ``Χειμώνας (Νοέμβριος-Ιανουάριος)``."""
    ms = months_in(text, allow_short=True)
    if not ms:
        return None
    return (ms[0], ms[-1])


# ---------------------------------------------------------------------------
# Numbers, durations, currency
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"(?<![\d.,])(\d{1,3}(?:\.\d{3})+|\d+)(?:([.,])(\d+))?(?!\d)")

_WORD_NUM = {
    # Greek (normalised)
    "ενα": 1, "μια": 1, "μια": 1, "ενασ": 1, "δυο": 2, "δυο": 2, "τρια": 3, "τρεισ": 3, "τεσσερα": 4,
    "τεσσερισ": 4, "πεντε": 5, "εξι": 6, "επτα": 7, "εφτα": 7, "οκτω": 8, "οχτω": 8, "εννεα": 9,
    "εννια": 9, "δεκα": 10, "εντεκα": 11, "δωδεκα": 12, "μιση": 0.5, "μισο": 0.5, "μιαμιση": 1.5,
    "ενα και μιση": 1.5, "δυομισι": 2.5, "τρεισημισι": 3.5, "τεσσερισημισι": 4.5, "πεντεμισι": 5.5,
    # English
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "half": 0.5,
}


def numbers_in(text: str) -> list[float]:
    """Numbers in ``text``; decimal comma and point both accepted.

    Ambiguous ``1.200`` yields both 1200 and 1.2.
    """
    out: list[float] = []
    if not text:
        return out
    for m in _NUM_RE.finditer(text):
        whole, sep, frac = m.group(1), m.group(2), m.group(3)
        whole_clean = whole.replace(".", "")
        try:
            if sep and frac is not None:
                out.append(float(f"{whole_clean}.{frac}"))
            else:
                out.append(float(whole_clean))
                if re.fullmatch(r"\d{1,3}\.\d{3}", whole):
                    out.append(float(whole))
        except ValueError:
            continue
    for tok in tokens(text):
        if tok in _WORD_NUM:
            out.append(float(_WORD_NUM[tok]))
    return out


OTHER_UNIT_RE = re.compile(r"\s*(?:ωρ|hour|hrs?\b|h\b|λεπτ|minute|mins?\b|€|ευρω|euro|eur\b|gr\b|g\b|kg|γρ|κιλ|km|"
                           r"χλμ|%|μετρ|στρεμ|ετ(?:ων|η|οσ)|year|ml\b|lt\b|λιτρ|θερμιδ|kcal|cal\b)")


def numbers_with_spans(text: str) -> list[tuple[float, int, int, str]]:
    """(value, start, end, normalised text) for the digit and word numbers of ``text``."""
    t = norm_keep_punct(text)
    out: list[tuple[float, int, int, str]] = []
    for m in _NUM_RE.finditer(t):
        whole, sep, frac = m.group(1), m.group(2), m.group(3)
        try:
            v = float(f"{whole.replace('.', '')}.{frac}") if sep and frac is not None else float(whole.replace(".", ""))
        except ValueError:
            continue
        out.append((v, m.start(), m.end(), t))
    for m in re.finditer(r"[a-zα-ω]+", t):
        if m.group(0) in _WORD_NUM:
            out.append((float(_WORD_NUM[m.group(0)]), m.start(), m.end(), t))
    return out


def number_with_unit(value: float, text: str, cues: list[str], window: int = 30) -> bool:
    """``value`` occurs next to one of the unit ``cues`` (e.g. 'ατομ', 'people'), or bare - not attached to a
    different unit such as hours, euros or grams."""
    for v, start, end, t in numbers_with_spans(text):
        if abs(v - value) > max(1e-6, 0.005 * abs(value)):
            continue
        ctx = t[max(0, start - window):end + window]
        if any(c in ctx for c in cues):
            return True
        if not OTHER_UNIT_RE.match(t[end:end + 12]):
            return True
    return False


# not part of a longer number ("1234.5"), but sentence punctuation may follow ("... 27360 00000.")
_PHONE_RE = re.compile(r"(?<!\d)(?<!\d[.,])(?:\+?30[\s-]?)?\d(?:[\s-]?\d){9}(?!\d)(?![.,]\d)")   # Greek numbers
_TIME_RE = re.compile(r"(?<!\d)(?<!\d[.,])(?:[01]?\d|2[0-3])[:.][0-5]\d(?!\d)(?![.,]\d)")
_KCAL_RE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:θερμιδ|kcal|cal\b|calories)")


def has_phone(text: str) -> bool:
    return bool(_PHONE_RE.search(norm_keep_punct(text or "")))


def has_time(text: str) -> bool:
    return bool(_TIME_RE.search(norm_keep_punct(text or "")))


def has_kcal(text: str) -> bool:
    return bool(_KCAL_RE.search(norm_keep_punct(text or "")))


def has_date(text: str) -> bool:
    """A day followed by a month name ("12 Μαΐου", "3 March")."""
    names = "|".join(sorted({norm(x) for d in ("el", "el_acc", "en") for x in MONTH_NAME[d].values() if x},
                            key=len, reverse=True))
    return bool(re.search(rf"(?<!\d)\d{{1,2}}\s+(?:{names})", norm_keep_punct(text or "")))


def has_specifics(text: str) -> bool:
    """Concrete specifics that an unanswerable question invites a model to invent: a phone number, a clock
    time, a date (day and month), a price, a duration or a calorie value."""
    if has_phone(text) or has_time(text) or has_kcal(text):
        return True
    if currency_amounts(text or "") or durations_hours(text or ""):
        return True
    return has_date(text)


def norm_keep_punct(s: str) -> str:
    """Casefold + accent strip + sigma fold, but keep punctuation (numbers)."""
    if not s:
        return ""
    s = strip_accents(unicodedata.normalize("NFKC", s).casefold()).replace("ς", "σ")
    return _WS_RE.sub(" ", s).strip()



_DUR_RE = re.compile(r"(\d+(?:[.,]\d+)?|[a-zα-ω]+)\s*(ωρ\w*|hours?|hrs?|h\b|λεπτ\w*|minutes?|mins?)\b")


def durations_hours(text: str) -> list[float]:
    """Durations expressed in hours (minutes converted), unit-anchored; an hours value directly followed by a
    minutes value ("2 ώρες και 45 λεπτά", "2 hours 45 minutes") is read as one duration (2.75)."""
    t = norm_keep_punct(text) + " "
    parts: list[tuple[float, bool, int, int]] = []   # (value, is_minutes, start, end)
    for m in _DUR_RE.finditer(t):
        num_s, unit = m.group(1), m.group(2)
        if re.fullmatch(r"\d+(?:[.,]\d+)?", num_s):
            val = float(num_s.replace(",", "."))
        else:
            val = _WORD_NUM.get(num_s)
            if val is None:
                continue
        parts.append((float(val), unit.startswith(("λεπτ", "min")), m.start(), m.end()))
    out: list[float] = []
    i = 0
    while i < len(parts):
        val, is_min, _s, end = parts[i]
        if not is_min and i + 1 < len(parts) and parts[i + 1][1]:
            gap = t[end:parts[i + 1][2]].strip(" ,")
            if gap in ("", "και", "κ.", "and", "&"):
                out.append(val + parts[i + 1][0] / 60.0)
                i += 2
                continue
        out.append(val / 60.0 if is_min else val)
        i += 1
    return out


_CURRENCY_RE = re.compile(
    r"(€\s?\d+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?\s?(?:€|ευρω|ευρώ|eur\b|euro|euros))",
    re.IGNORECASE,
)


def currency_amounts(text: str) -> list[float]:
    out: list[float] = []
    for m in _CURRENCY_RE.finditer(text or ""):
        nums = re.findall(r"\d+(?:[.,]\d{1,2})?", m.group(0))
        for x in nums:
            try:
                out.append(float(x.replace(",", ".")))
            except ValueError:
                pass
    return out


# ---------------------------------------------------------------------------
# Cue lexicons (normalised substrings)
# ---------------------------------------------------------------------------

ABSTAIN_CUES = [norm(x) for x in [
    "δεν βρήκα", "δεν μπόρεσα να βρω", "δεν μπορώ να βρω", "δεν υπάρχουν πληροφορίες", "δεν υπάρχει πληροφορία",
    "δεν υπάρχουν στοιχεία", "δεν έχω πληροφορίες", "δεν έχω στοιχεία", "δεν διαθέτω", "δεν γνωρίζω",
    "δεν είμαι σε θέση", "δεν αναφέρεται", "δεν εντόπισα", "δεν εντοπίστηκ", "δεν υπάρχει καταχώρηση",
    "δεν υπάρχει καταχωρημέν", "δεν είναι καταχωρημέν", "δεν υπάρχει τέτοι", "δεν φαίνεται να υπάρχει",
    "δεν περιλαμβάνεται", "δεν βρέθηκ", "δεν έχω πρόσβαση", "δεν είναι διαθέσιμ", "δεν διαθέτουμε",
    "no information", "couldn't find", "could not find", "can't find", "cannot find",
    "don't have information", "don't have any information", "do not have information", "don't have data",
    "do not have data", "no record", "isn't listed", "is not listed", "not listed", "unable to find",
    "i don't know", "i do not know", "no matching", "there is no information", "there is no record",
    "there are no records", "there are no results", "doesn't appear", "does not appear", "no data available",
    "not in the knowledge base", "not available in the", "is not available on the platform",
]]

ON_REQUEST_CUES = [norm(x) for x in [
    "κατόπιν επικοινωνίας", "επικοινωνήστε", "επικοινωνήσετε", "επικοινωνία με τον παραγωγό",
    "επικοινωνία με τον οικοδεσπότη", "δεν αναγράφεται", "δεν εμφανίζεται", "δεν δημοσιεύεται",
    "δεν είναι δημοσιευμένη", "δεν είναι διαθέσιμη", "δεν αναφέρεται τιμή", "δεν υπάρχει τιμή",
    "on request", "upon request", "contact the producer", "contact the host", "contact the seller",
    "not listed", "not published", "not shown", "not displayed", "not available", "no price",
]]


def has_cue(text: str, cues: list[str]) -> bool:
    t = norm(text)
    return any(c and c in t for c in cues)


# Negated availability, existence or knowledge, in Greek and English, matched on lower-cased, unaccented text
# with final sigma folded (``norm_keep_punct``). "There is no fixed price" is not an abstention; "there is no such
# producer", "the calendar does not include pistachios" and "Η πληροφορία δεν υπάρχει στην πλατφόρμα" are.
_ABSTAIN_RES = [re.compile(p) for p in (
    r"\bδεν\s+(?:(?:το|τα|την|τη|τον|μου|μασ)\s+)?(?:υπαρχ|αναφερ|αναγραφ|δημοσιευ|εμφανιζ|καταγραφ|καταχωρ|"
    r"δινετ|δινοντ|παρεχετ|παρεχοντ|προκυπτ|προσδιοριζ|διευκρινιζ|περιλαμβαν|βρεθηκ|εντοπ|βρηκα|βρισκω|γνωριζ|"
    r"ξερω)",
    r"\bδεν\s+(?:εχω|εχουμε|εχει|εχουν|διαθετω|διαθετουμε|διαθετει|διαθετουν)\s+(?:\w+\s+){0,3}?"
    r"(?:πληροφορ|στοιχει|δεδομεν|προσβαση|ενδειξ|καταχωρ)",
    r"\bδεν\s+ειναι\s+(?:\w+\s+)?(?:διαθεσιμ|καταχωρημεν|δημοσιευμεν|γνωστ|καταγεγραμμεν)",
    r"\bδεν\s+(?:μπορω|μπορεσα|ειμαι\s+σε\s+θεση)\s+να\s+(?:σασ\s+)?(?:βρω|εντοπισω|επιβεβαιωσω|απαντησω|δωσω|"
    r"προσδιορισω)",
    r"\b(?:καμια|καμμια)\s+(?:πληροφορια|αναφορα|καταχωρηση|ενδειξη)",
    r"\b(?:is|are|was|were|has|have)(?:\s+not|n't)(?:\s+been)?\s+(?:\w+\s+)?(?:mentioned|listed|specified|provided|"
    r"given|stated|included|published|available|shown|displayed|recorded|documented|found|indicated|disclosed)\b",
    r"\b(?:does|do|did)(?:\s+not|n't)\s+(?:\w+\s+)?(?:mention|list|specify|provide|give|state|include|contain|show|"
    r"have|exist|appear|indicate|offer|record|cover)\b",
    r"\bno\s+(?:\w+\s+)?(?:information|data|details|record|records|mention|listing|listings|entry|entries|results?|"
    r"history|indication|such)\b",
    r"\b(?:couldn't|could not|can't|cannot|unable to|wasn't able to|was not able to|am not able to)\s+(?:\w+\s+)?"
    r"(?:find|locate|confirm|verify|determine|identify|provide)\b",
    r"\b(?:i|we)\s+(?:don't|do not)\s+(?:know|have)\b",
    r"\bnot\s+(?:on|in|part of)\s+(?:the\s+)?(?:platform|glossary|calendar|knowledge base|catalogue|catalog|database|"
    r"data|records?)\b",
    r"(?:^|[.;:!?(]\s*)not\s+(?:\w+\s+)?(?:available|published|listed|mentioned|specified|given|provided|found|known|"
    r"included|shown|recorded)\b",
    r"\bnot\s+(?:available|published|listed|recorded)\s+(?:on|in)\b",
)]


def is_abstention(text: str) -> bool:
    if has_cue(text, ABSTAIN_CUES):
        return True
    t = norm_keep_punct(text or "").replace("\u2019", "'")
    return any(p.search(t) for p in _ABSTAIN_RES)


def mentions_on_request(text: str) -> bool:
    return has_cue(text, ON_REQUEST_CUES)
