"""
fertig.bindings — der Bindungs-Parser (GSM8K-Projekt, Kernstück).

Diagnose (Falsifikations-Ledger): Muster-Templates erreichen ~4% Präzision,
weil sie Zahlen ohne Bindung verarbeiten. Der Bindungs-Parser macht aus
einer Textaufgabe eine BINDUNGS-STRUKTUR:

    Zahl --(zählt)--> Objekt (clips)
    Zahl --(Einheit)--> clips/friends/dollars
    Zahl --(Rolle)--> qty | partitive | ratio | price | duration
    Frage --(Ziel)--> Objekt + Operation (sum/left/diff/...)

Prinzipien (Hausregeln):
  * Determinismus: gleiche Frage -> gleiche Bindung -> gleiche Antwort.
  * Abstinenz: unvollständige Bindung (Objekt oder Einheit fehlt) -> None.
    Kein Raten auf Zahlen allein.
  * Objekt-Konsistenz: eine Operation bindet nur Mengen DESSELBEN Objekts
    (oder explizit konvertierbarer Einheiten).
  * Verifikation: jede Zwischengröße trägt Objekt+Einheit; die Antwort
    wird nur abgegeben, wenn sie das Frageziel trifft.

Die Rolle 'ratio' referenziert ein VORHER gebundenes Objekt ("half as
many clips") — das ist die Kette, die Muster-Templates nicht können.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Datenmodell
# ---------------------------------------------------------------------------


@dataclass
class Quantity:
    """Eine gebundene Menge im Text."""

    value: Fraction            # numerischer Wert (exakt)
    text: str                  # Original-Text der Menge ("48", "half as many")
    unit: Optional[str]        # Einheit ("clips", "friends", "$", "days")
    obj: Optional[str]         # gebundenes Objekt (NP-Kopf, Singular)
    role: str                  # qty|partitive|ratio|price|duration|rate|each
    ref_obj: Optional[str] = None   # bei ratio: das referenzierte Objekt
    span: Tuple[int, int] = (0, 0)
    sentence: int = 0

    def __repr__(self) -> str:  # pragma: no cover - Debug
        return (f"Q({self.text}={self.value} {self.unit or ''} "
                f"obj={self.obj} role={self.role}"
                f"{' <- ' + self.ref_obj if self.ref_obj else ''})")


@dataclass
class QuestionTarget:
    """Was die Frage sucht: Objekt + Operation + Bezugszeitraum."""

    obj: Optional[str] = None          # "clips"
    op: str = "sum"                   # sum | left | diff | product | total
    period: Optional[str] = None       # "april and may"
    ok: bool = False

    def __repr__(self) -> str:  # pragma: no cover - Debug
        return f"Target(obj={self.obj}, op={self.op})"


@dataclass
class BindingResult:
    """Vollständiges Bindungsergebnis einer Aufgabe."""

    quantities: List[Quantity] = field(default_factory=list)
    target: QuestionTarget = field(default_factory=QuestionTarget)
    answer: Optional[str] = None
    ok: bool = False
    reason: str = ""

    def __repr__(self) -> str:  # pragma: no cover - Debug
        return f"BindingResult(ok={self.ok}, answer={self.answer}, {self.reason})"


# ---------------------------------------------------------------------------
# Lexikon (geschlossene Klassen — keine Hardcoding auf Benchmarks)
# ---------------------------------------------------------------------------

# Objekt-/Einheits-Köpfe: was Mengen tragen kann (allgemeine NP-Köpfe)
_SINGULAR = {
    "clips": "clip", "friends": "friend", "students": "student",
    "children": "child", "apples": "apple", "oranges": "orange",
    "books": "book", "pages": "page", "miles": "mile", "hours": "hour",
    "days": "day", "weeks": "week", "months": "month", "years": "year",
    "dollars": "dollar", "cents": "cent", "minutes": "minute",
    "seconds": "second", "bottles": "bottle", "cookies": "cookie",
    "candies": "candy", "tickets": "ticket", "cards": "card",
    "pencils": "pencil", "crayons": "crayon", "marbles": "marble",
    "stamps": "stamp", "toys": "toy", "games": "game", "points": "point",
    "miles per hour": "mph", "kilograms": "kilogram", "grams": "gram",
    "meters": "meter", "feet": "foot", "inches": "inch", "laps": "lap",
    "rows": "row", "chairs": "chair", "tables": "table", "rooms": "room",
    "floors": "floor", "flights": "flight", "steps": "step",
    "questions": "question", "problems": "problem", "answers": "answer",
    "tests": "test", "scores": "score", "runs": "run", "walks": "walk",
    "songs": "song", "pictures": "picture", "photos": "photo",
    "slices": "slice", "pieces": "piece", "pizzas": "pizza",
    "cakes": "cake", "pies": "pie", "loaves": "loaf", "eggs": "egg",
    "sandwiches": "sandwich", "burgers": "burger", "hot dogs": "hot dog",
    "coins": "coin", "quarters": "quarter", "dimes": "dime",
    "nickels": "nickel", "pennies": "penny", "bills": "bill",
    "bags": "bag", "boxes": "box", "baskets": "basket", "jars": "jar",
    "cans": "can", "packs": "pack", "cartons": "carton", "dozens": "dozen",
    "hours per day": "hour", "times": "time", "miles per gallon": "mpg",
    "beetles": "beetle", "birds": "bird", "snakes": "snake",
    "jaguars": "jaguar", "arms": "arm", "starfish": "starfish",
    "seastars": "seastar", "gnomes": "gnome", "houses": "house",
    "situps": "situp", "cans": "can",
}

# Objekt-Synonyme (geschlossene Klasse): pieces == slices, coins == cents...
_SYNONYMS = {
    "piece": "slice", "pieces": "slice", "slice": "slice",
    "coin": "coin", "coins": "coin", "cent": "coin", "cents": "coin",
    "item": "item", "items": "item", "total": None,
}


# Operatoren in der Frage (geschlossene Klasse)
_SUM_WORDS = ["altogether", "in all", "in total", "total", "combined",
              "all together", "sum", "both"]
_LEFT_WORDS = ["left", "remain", "remaining", "still have", "left over"]
_DIFF_WORDS = ["more than", "less than", "how many more", "difference",
               "how much more"]
_PRODUCT_WORDS = ["total cost", "how much did he spend", "how much does",
                  "how much do", "how much would"]

# Rollen-Verben: was ist die Menge des Objekts?
_ROLE_VERBS = {
    "sold": "qty", "bought": "qty", "has": "qty", "had": "qty",
    "have": "qty", "collected": "qty", "gathered": "qty", "ate": "qty",
    "drank": "qty", "made": "qty", "baked": "qty", "cooked": "qty",
    "read": "qty", "wrote": "qty", "answered": "qty", "solved": "qty",
    "earned": "qty", "spent": "qty", "paid": "qty", "cost": "price",
    "costs": "price", "save": "qty", "saved": "qty", "won": "qty",
    "lost": "qty", "gave": "qty", "donated": "qty", "lent": "qty",
    "borrowed": "qty", "received": "qty", "found": "qty", "picked": "qty",
    "caught": "qty", "planted": "qty", "watered": "qty", "walked": "qty",
    "ran": "qty", "swam": "qty", "drove": "qty", "flew": "qty",
    "traveled": "qty", "visited": "qty", "built": "qty", "painted": "qty",
    "learned": "qty", "studied": "qty", "practiced": "qty", "played": "qty",
    "watched": "qty", "listened": "qty", "worked": "qty", "needed": "qty",
    "wanted": "qty", "invited": "qty", "attended": "qty", "joined": "qty",
    "used": "qty", "purchased": "qty", "ordered": "qty", "brought": "qty",
    "took": "qty", "got": "qty", "kept": "qty", "returned": "qty",
    "sent": "qty", "mailed": "qty", "delivered": "qty", "produced": "qty",
    "created": "qty", "designed": "qty", "wrapped": "qty", "filled": "qty",
}

# Zahlwörter (geschlossene Klasse)
_WORD_NUM = {
    "half": Fraction(1, 2), "twice": Fraction(2), "double": Fraction(2),
    "triple": Fraction(3), "quarter": Fraction(1, 4),
    "a dozen": Fraction(12), "a couple": Fraction(2),
}


# ---------------------------------------------------------------------------
# 1. Mengen-Phrasen finden
# ---------------------------------------------------------------------------

_NUM = r"(?:\d+(?:\.\d+)?|\d+/\d+)"


def _find_quantities(text: str) -> List[Tuple[str, Fraction, int, int]]:
    """(Phrase, Wert, Start, Ende) — Zahlen UND Zahlwörter mit Kontext."""
    out: List[Tuple[str, Fraction, int, int]] = []
    low = text.lower()
    # Zahlwörter mit Verhältnis-Struktur: "half as many X", "3 times as many X"
    for m in re.finditer(
            r"(?:(\d+)\s+times\s+as\s+many|half\s+as\s+many|twice\s+as\s+many"
            r"|three\s+times\s+as\s+many)\s+([a-z][a-z ]+?)(?=[,.;]|\s+(?:as|"
            r"than|in|for|to|and|but)\b|\s*$)", low):
        val = (Fraction(m.group(1)) if m.group(1)
               else (Fraction(2) if "twice" in m.group(0)
                     else Fraction(1, 2)))
        if "three" in m.group(0):
            val = Fraction(3)
        out.append((m.group(0).strip(), val, m.start(), m.end()))
    # Ziffern: "48 clips", "48 of her friends", "$2", "2 days", "48%"
    for m in re.finditer(
            r"(?<![a-z])(\d+(?:\.\d+)?)\s*(%|dollars?|\$)?\s*"
            r"([a-z][a-z ]{0,20}?)?(?=[,.;]|\s+(?:of|each|per|for|in|to|"
            r"and|but|than|at|with|by|from|on)\b|\s*$)",
            low):
        val = Fraction(m.group(1))
        unit = (m.group(2) or "").replace("$", "dollars").strip()
        rest = (m.group(3) or "").strip()
        phrase = m.group(0).strip()
        out.append((phrase, val, m.start(), m.end()))
        # "48 of her friends" -> partitive: Zahl gefolgt von "of X"
    return out


def _np_head(phrase: str) -> Optional[str]:
    """Der letzte Nomen-Teil einer Phrase ("48 of her friends" -> friends)."""
    words = [w for w in re.split(r"\s+", phrase) if w]
    if not words:
        return None
    # "of her friends" -> friends
    if "of" in words:
        idx = words.index("of")
        tail = words[idx + 1:]
    else:
        tail = words
    # Determinierer abwerfen
    tail = [w for w in tail if w not in
            ("the", "a", "an", "her", "his", "their", "my", "its", "our")]
    if not tail:
        return None
    head = tail[-1]
    return head


# ---------------------------------------------------------------------------
# 2. Rollen-Bindung
# ---------------------------------------------------------------------------

# jede-Relationen: <Menge> <obj> (each) (has/contains/costs/...) <Menge> <unit>
_EACH_RE = [
    re.compile(r"(\d+)\s+([a-z]+)\s+each\s+(?:has|contains?|holds|costs?|"
               r"is\s+worth|weighs?|is)\s+(\d+(?:\.\d+)?)\s*([a-z]+)?"),
    re.compile(r"each\s+(?:of\s+)?(\d+)?\s*([a-z]+)\s+(?:has|contains?|"
               r"holds|costs?|weighs?|is)\s+(\d+(?:\.\d+)?)\s*([a-z]+)?"),
    re.compile(r"(\d+)\s+([a-z]+)\s+per\s+(\w+)"),  # rate: 12 beetles per day
]

# more/fewer-than: "M more <obj> than <base>" — base kann eine Zahl ODER
# ein referenziertes Objekt sein ("than snowflake stamps" -> 11)
_MORE_FEWER_RE = re.compile(
    r"(\d+)\s+(more|fewer|less)\s+([a-z]+(?:\s+[a-z]+){0,2})\s+than\s+"
    r"(?:(\d+)|([a-z]+(?:\s+[a-z]+){0,2}))")

# a/an <obj> has/contains M <unit>: Typ->Pro-Stück-Menge (Adjektive
# überspringen: "a large pizza has 16 slices")
_A_AN_HAS_RE = re.compile(
    r"\b(?:a|an|each)\s+(?:[a-z]+\s+){0,2}([a-z]+)\s+(?:has|contains?|"
    r"holds|costs?|weighs?)\s+(\d+(?:\.\d+)?)\s*([a-z]+)?")

# times-as-many: "X times as many <obj> as <base>" — base kann Zahl ODER
# eine Variable sein ("as Carlos memorized"); "digits of pi" erlaubt;
# "half as many ... as" = factor 1/2
_TIMES_AS_MANY_RE = re.compile(
    r"(?:(\d+)\s+times|half)\s+as\s+many\s+"
    r"([a-z]+(?:\s+of\s+[a-z]+){0,2})\s+as\s+(?:(\d+)|([a-z]+))")

# Variablen-Zuweisung: "If Mina memorized 24 digits" / "Mina memorized 24"
# — unit muss ein Nomen sein ("30 less" ist KEINE Zuweisung)
_VAR_ASSIGN_RE = re.compile(
    r"\b(?:if\s+)?([A-Z][a-z]+)\s+(?:memorized|has|had|bought|sold|ate|\
    collected|saved|earned|spent|wrote|read|answered|solved|ran|walked|\
    traveled|earned|made|built|found|planted|caught|picked|received|\
    took|got)\s+(\d+)\s*(?!less\b|more\b|than\b)([a-z]+)?")


# each-of-N: "each of the first four houses has 3 gnomes" -> 4x3
_EACH_OF_RE = re.compile(
    r"each\s+of\s+(?:the\s+)?(?:first\s+)?(\d+)\s+([a-z]+)\s+"
    r"(?:has|contains?|holds)\s+(\d+)\s*([a-z]+)?")

# with-Mengen: "7 starfish with 5 arms each and one seastar with 14
# arms" -> 7x5 + 14
_WITH_EACH_RE = re.compile(
    r"(\d+)\s+([a-z]+)\s+with\s+(\d+)\s+([a-z]+)\s+each")
_WITH_SINGLE_RE = re.compile(
    r"\b(?:one|a|an)\s+([a-z]+)\s+with\s+(\d+)\s+([a-z]+)")

# Futterketten: "Each bird eats 12 beetles per day, each snake eats 3
# birds per day, each jaguar eats 5 snakes per day. 6 jaguars..."
_EATS_RE = re.compile(
    r"each\s+([a-z]+)\s+eats?\s+(\d+)\s+([a-z]+)\s+per\s+day")


# Raten-Dauer: "For the next two hours, she collected 35 coins" -> 2x35
_FOR_DURATION_RE = re.compile(
    r"for\s+(?:the\s+)?(?:next\s+)?(\d+)\s+(hours?|days?|weeks?|\
    months?|minutes?|seconds?)\s*,\s*([a-z]+)\s+collected\s+"
    r"(\d+)\s*([a-z]+)?")

# Subtraktion: "gave 15 of them" / "ate 3 of the cookies"
_GAVE_RE = re.compile(
    r"(?:gave|donated|lent|threw\s+away|ate|used|sold)\s+(\d+)\s+"
    r"(?:of\s+)?(?:them|it|those|the\s+[a-z]+)\b")

# Dauer: "for 43 minutes" / "worked 3 hours"
_DURATION_RE = re.compile(
    r"(?:for|worked|lasting|spent)\s+(\d+)\s+"
    r"(minutes?|hours?|days?|weeks?|months?|years?)")


# more-than-twice: "five more roommates than twice as many as Bob"
# -> John = 2xBob + 5 (auf dem ORIGINAL-Text, Wort-Zahlen erlaubt)
_MORE_THAN_TWICE_RE = re.compile(
    r"(?:(\d+)|five|four|three|six|seven|eight|nine|ten|two|twenty|\
    thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\s+more\s+"
    r"([a-z]+)\s+than\s+twice\s+as\s+many\s+as\s+([A-Z][a-z]+)")


@dataclass
class Relation:
    """Bindungs-Relation zwischen Mengen: per/each/more/fewer/times."""

    kind: str                # each | more | fewer | times | assign
    source: int              # Index in quantities (Träger der Relation)
    base_value: Fraction      # Basis-Wert (bei each: Anzahl der Träger)
    factor: Fraction = Fraction(1)   # Multiplikator (each: Wert je Träger)
    unit: Optional[str] = None       # Ziel-Einheit (each: Scheiben...)
    var: Optional[str] = None        # Variablen-Name (times/assign)
    text: str = ""


# Wort-Zahlen -> Ziffern (geschlossene Klasse) — für Relations-Matching
_WORD2NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "hundred": 100,
}


def _digitize(text: str) -> str:
    """Wort-Zahlen in Ziffern wandeln (nur für Relations-Matching)."""
    out = text
    for w, n in sorted(_WORD2NUM.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(rf"\b{w}\b", str(n), out)
    return out


def _find_relations(text: str, quants: List[Quantity]
                    ) -> Tuple[List[Relation], List[Quantity]]:
    """jede/more/fewer/times-Relationen finden und die betroffenen
    Quantities durch die RELATIVEN Werte ersetzen (Bindung statt Summe
    von Rohzahlen)."""
    low = _digitize(text.lower())
    rels: List[Relation] = []
    for m in _EACH_RE[0].finditer(low):
        n, obj, m2, unit = m.group(1), m.group(2), m.group(3), m.group(4)
        # "N obj each has M unit" -> N Stück zu je M -> Gesamt M*N
        rels.append(Relation(kind="each", source=-1,
                             base_value=Fraction(n),
                             factor=Fraction(m2),
                             unit=unit or obj,
                             text=m.group(0)))
    for m in _EACH_RE[1].finditer(low):
        n, obj, m2, unit = m.group(1), m.group(2), m.group(3), m.group(4)
        cnt = Fraction(n) if n else Fraction(1)
        rels.append(Relation(kind="each", source=-1,
                             base_value=cnt, factor=Fraction(m2),
                             unit=unit or obj, text=m.group(0)))
    for m in _EACH_RE[2].finditer(low):
        # Rate: "12 beetles per day" -> nur registrieren, wenn ein
        # Zeitraum folgt (sonst Abstinenz)
        rels.append(Relation(kind="rate", source=-1,
                             base_value=Fraction(m.group(1)),
                             factor=Fraction(1), unit=m.group(2),
                             text=m.group(0)))
    for m in _MORE_FEWER_RE.finditer(low):
        if "as many" in m.group(0):
            continue   # "more X than twice as many" gehört more2x
        unit = m.group(3)
        # Kern-Nomen: "digits of pi" -> digits (of-Phrase abtrennen)
        core = unit.split(" of ")[0].split()[-1]
        var_ref = None
        if m.group(4):
            base = Fraction(m.group(4))
        else:
            # Referenz: der TYP-First-Wort ("truck stamps" -> "truck")
            # — eindeutig, während das Kern-Nomen ("stamps") mehrdeutig ist
            ref_name = m.group(5).split()[0]
            ref = [q for q in quants
                   if q.obj == _SINGULAR.get(ref_name, ref_name)
                   or (q.unit or "") == ref_name
                   or ref_name in (q.unit or "")
                   or ref_name in q.text]
            if not ref:
                # keine Quantity -> Variablen-Referenz ("than Carlos")
                var_ref = m.group(5).split()[0].lower()
                base = Fraction(0)
            else:
                # PRIORITÄT: Relation-generierte Werte (ref_obj gesetzt) sind
                # absoluter als Roh-Zahlen — "truck" muss 20 sein, nicht das
                # rohe Inkrement 9.
                rel_ref = [q for q in ref if q.ref_obj]
                base = (rel_ref[0] if rel_ref else ref[0]).value
        rels.append(Relation(kind=m.group(2), source=-1,
                             base_value=base,
                             factor=Fraction(m.group(1)),
                             unit=core, var=var_ref, text=m.group(0)))
    for m in _TIMES_AS_MANY_RE.finditer(low):
        if m.group(3):
            base_v = Fraction(m.group(3))
        else:
            # Variable: "as Carlos memorized" — wird später aufgelöst
            base_v = Fraction(0)
        factor = (Fraction(1, 2) if m.group(1) is None  # "half"
                  else Fraction(m.group(1)))
        rels.append(Relation(kind="times", source=-1,
                             base_value=base_v,
                             factor=factor,
                             unit=m.group(2),
                             var=m.group(4) if m.group(4) else None,
                             text=m.group(0)))
    for m in _VAR_ASSIGN_RE.finditer(text):
        if m.group(3) is None:
            # "Tim has 30 less/more/than ..." ist KEINE Zuweisung
            seg = text[m.end(2):m.end(2) + 8].lower()
            if re.match(r"\s*(less|more|than)\b", seg):
                continue
        rels.append(Relation(kind="assign", source=-1,
                             base_value=Fraction(m.group(2)),
                             factor=Fraction(1),
                             unit=m.group(3) or "",
                             var=m.group(1).lower(),
                             text=m.group(0)))
    for m in _WITH_EACH_RE.finditer(low):
        rels.append(Relation(kind="each", source=-1,
                             base_value=Fraction(m.group(1)),
                             factor=Fraction(m.group(3)),
                             unit=m.group(4), text=m.group(0)))
    for m in _EACH_OF_RE.finditer(low):
        rels.append(Relation(kind="each", source=-1,
                             base_value=Fraction(m.group(1)),
                             factor=Fraction(m.group(3)),
                             unit=m.group(4) or m.group(2),
                             text=m.group(0)))
    for m in _WITH_SINGLE_RE.finditer(low):
        rels.append(Relation(kind="each", source=-1,
                             base_value=Fraction(1),
                             factor=Fraction(m.group(2)),
                             unit=m.group(3), text=m.group(0)))
    # Typ->Pro-Stück: "a large pizza has 16 slices" — verknüpft mit einer
    # vorherigen Menge desselben Typs ("2 large pizzas")
    for m in _A_AN_HAS_RE.finditer(low):
        typ, per, unit = m.group(1), m.group(2), m.group(3)
        # Typ-Menge suchen: "2 large pizzas" -> unit==pizza
        typ_key = _SINGULAR.get(typ, typ)
        holder = [q for q in quants
                  if q.role == "qty"
                  and (q.obj == typ_key or (q.unit or "") == typ)]
        if holder:
            total = holder[0].value * Fraction(per)
            rels.append(Relation(kind="each", source=-1,
                                 base_value=holder[0].value,
                                 factor=Fraction(per),
                                 unit=unit or typ, text=m.group(0)))
    for m in _FOR_DURATION_RE.finditer(low):
        rels.append(Relation(kind="each", source=-1,
                             base_value=Fraction(m.group(1)),
                             factor=Fraction(m.group(4)),
                             unit=m.group(5) or m.group(3),
                             text=m.group(0)))
    for m in _GAVE_RE.finditer(low):
        rels.append(Relation(kind="subtract", source=-1,
                             base_value=Fraction(m.group(1)),
                             factor=Fraction(1),
                             unit="", text=m.group(0)))
    for m in _MORE_THAN_TWICE_RE.finditer(text):
        if m.group(1):
            factor = Fraction(m.group(1))
        else:
            factor = Fraction(_WORD2NUM.get(m.group(0).split()[0], 5))
        rels.append(Relation(kind="more2x", source=-1,
                             base_value=Fraction(0),
                             factor=factor,
                             unit=m.group(2),
                             var=m.group(3).lower(),
                             text=m.group(0)))
    for m in _DURATION_RE.finditer(low):
        rels.append(Relation(kind="duration", source=-1,
                             base_value=Fraction(m.group(1)),
                             factor=Fraction(1),
                             unit=m.group(2), text=m.group(0)))
    return rels, quants


def _effective_quantities(quants: List[Quantity]) -> List[Quantity]:
    """Relation-Mengen (ref_obj gesetzt) sind ABSOLUTE Werte; sie
    ersetzen Roh-Mengen, die Teil ihrer Text-Bindung sind — sowohl
    relative ("9 more X") als auch Pro-Stück-Werte ("16 slices" in
    "a large pizza has 16 slices"). Sonst summiert man Basis + Inkrement
    + Pro-Stück-Wert."""
    eff = [q for q in quants if q.role == "qty" and q.ref_obj]
    if not eff:
        return quants
    out = []
    for q in quants:
        if q.role == "qty" and not q.ref_obj:
            replaced = False
            for e in eff:
                if (e.unit == q.unit and q.text in e.text
                        and q.value != e.value):
                    replaced = True
                    break
            if replaced:
                continue
        out.append(q)
    return out


def _solve_variables(rels: List[Relation], question: str
                     ) -> Optional[Fraction]:
    """Variablen-Gleichungen auflösen: "Mina memorized six times as many
    digits as Carlos. If Mina memorized 24 digits, how many did Sam
    memorize?" -> Carlos=4, Sam=10.

    Gleichungsformen (alle mit Subjekt-Variable):
      assign: Subjekt = base_value
      times:  Subjekt = factor * (base_value | Var)
      more/fewer: Subjekt = (base_value | Var) ± factor
    """
    low = question.lower()
    # Subjekt einer Relation: Satzanfang der den Ausdruck enthält
    def _subject(rel: Relation) -> Optional[str]:
        # Satz, der rel.text enthält (beide Seiten digitize-normalisiert);
        # Subjekt = die KLAUSEL (Satz- oder Nebensatz-Anfang) —
        # "..., and Harry has half as many..." -> Harry, nicht Tim.
        for sent in re.split(r"(?<=[.!?])\s+", question):
            idx = _digitize(sent.lower()).find(_digitize(rel.text.lower()))
            if idx >= 0:
                before = sent[:idx]
                clause = re.split(
                    r",\s+(?:and|but|then|so|yet)\s+|;\s+|\.\s+",
                    before)[-1]
                names = re.findall(r"([A-Z][a-z]+)", clause)
                return names[0].lower() if names else None
        return None

    # Ziel: "how many digits did Sam memorize" / "does Harry have" /
    # "do THEY have in total" (Summe aller aufgelösten Personen)
    tm = re.search(r"how many [a-z ]*? (?:did|does|do|would) "
                   r"([A-Z][a-z]+|they|them)", question)
    if not tm:
        return None
    target = tm.group(1).lower()
    they_total = target in ("they", "them")

    known: Dict[str, Fraction] = {}
    eqs = []  # (subj, kind, factor, base, ref_var)
    for r in rels:
        subj = _subject(r)
        if not subj:
            continue
        if r.kind == "assign":
            # "If Mina memorized 24" — der Satz startet mit 'If', das
            # Subjekt ist die Variable selbst.
            known[r.var or subj] = r.base_value
        elif r.kind == "times" and r.var:
            eqs.append((subj, "times", r.factor, r.base_value, r.var))
        elif r.kind == "times" and not r.var:
            known[subj] = r.base_value * r.factor
        elif r.kind in ("more", "fewer", "less", "more2x") and r.var:
            eqs.append((subj, r.kind, r.factor, r.base_value, r.var))

    # Propagation (einfache Gleichungen, keine zyklischen Systeme)
    for _ in range(10):
        progressed = False
        for subj, kind, factor, base, ref in eqs:
            if subj in known and ref is not None and ref not in known:
                # RÜCKWÄRTS: Subjekt bekannt, Referenz unbekannt
                # ("Mina=24, Mina=6xCarlos" -> Carlos=4)
                if kind == "times":
                    known[ref] = known[subj] / factor
                elif kind == "more":
                    known[ref] = known[subj] - factor
                elif kind in ("fewer", "less"):
                    known[ref] = known[subj] + factor
                progressed = True
                continue
            if subj in known:
                continue
            src = known.get(ref)
            if src is None and base > 0:
                src = base
            if src is None:
                continue
            if kind == "times":
                known[subj] = src * factor
            elif kind == "more2x":
                known[subj] = src * 2 + factor
            elif kind == "more":
                known[subj] = src + factor
            elif kind in ("fewer", "less"):
                known[subj] = src - factor
            progressed = True
        if not progressed:
            break

    if they_total:
        if known:
            return sum(known.values())
        return None
    return known.get(target)


def _food_chain(question: str, rels: List[Relation],
                quants: List[Quantity]) -> Optional[Quantity]:
    """Futterkette: each X eats N Y per day; ... ; M Z (Prädator-Zahl).
    -> M x N1 x N2 x ... (Kettenmultiplikation). Ziel-Einheit = unterste
    Beute."""
    low = _digitize(question.lower())
    eats = {}   # predator -> (prey, rate)
    for m in _EATS_RE.finditer(low):
        eats[m.group(1)] = (m.group(3), Fraction(m.group(2)))
    if not eats:
        return None

    def _sing(w: str) -> str:
        return _SINGULAR.get(w, w[:-1] if w.endswith("s") else w)

    # Rate-Mengen ausschließen ("3 birds" in "each snake eats 3 birds")
    # — Prädator-Menge ist NUR die unabhängige Menge ("6 jaguars").
    rate_phrases = {f"{m.group(2)} {m.group(3)}" for m in
                    _EATS_RE.finditer(low)}

    # Prädator-Menge: "M jaguars" / "2 snakes" (letzte Stufe)
    top = [q for q in quants
           if q.role == "qty" and q.text not in rate_phrases
           and _sing(q.obj or "") in eats]
    if not top:
        return None
    # Kette absteigen: top-Predator -> ... -> unterste Beute
    total = Fraction(0)
    for start in top:
        pred, n = _sing(start.obj or ""), start.value
        v = n
        seen = set()
        while pred in eats and pred not in seen:
            seen.add(pred)
            prey, rate = eats[pred]
            v = v * rate
            pred = _sing(prey)      # "snakes" -> "snake" (eats-Key)
        total += v
        unit = pred
    if total > 0:
        return Quantity(value=total, text="Futterkette", unit=unit,
                        obj=unit, role="qty", ref_obj=unit)
    return None


def _rate_duration(question: str, rels: List[Relation],
                   quants: List[Quantity],
                   tgt: QuestionTarget) -> Optional[Fraction]:
    """Rate x Dauer: "can type 6 sentences per minute ... for 43 minutes"
    -> 6 x 43. Einheiten müssen kompatibel sein (minute <-> minutes)."""
    rates = [r for r in rels if r.kind == "rate"]
    durs = [r for r in rels if r.kind == "duration"]
    if not rates or not durs:
        return None
    out = Fraction(0)
    for r in rates:
        for d in durs:
            # Einheiten-Kompatibilität: "per minute" vs "43 minutes"
            r_unit = (r.text.rsplit("per", 1)[-1].strip()
                      if "per" in r.text else "")
            if r_unit.rstrip("s") != d.unit.rstrip("s"):
                continue
            out += r.base_value * d.base_value
    return out if out > 0 else None


def _apply_relations(rels: List[Relation], quants: List[Quantity]
                     ) -> List[Quantity]:
    """Relationen in zusätzliche gebundene Quantities umsetzen.
    Jede Relation liefert eine NEUE Menge mit Objekt+Einheit — die
    Roh-Quantities bleiben, aber Relationen sind höherwertig."""
    extra: List[Quantity] = []
    existing_texts = {q.text for q in quants}
    for r in rels:
        if r.text in existing_texts:
            continue
        if r.var:
            # Variablen-Relationen ("as Carlos") sind GLEICHUNGEN — sie
            # erzeugen keine Mengen, sie verbinden Personen-Werte.
            continue
        if r.kind == "each":
            total = r.base_value * r.factor
            extra.append(Quantity(value=total, text=r.text,
                                  unit=r.unit, obj=_SINGULAR.get(
                                      r.unit, r.unit), role="qty",
                                  ref_obj=r.unit))
        elif r.kind in ("more", "fewer"):
            v = r.base_value + (r.factor if r.kind == "more" else -r.factor)
            if v >= 0:
                extra.append(Quantity(value=v, text=r.text, unit=r.unit,
                                      obj=_SINGULAR.get(r.unit, r.unit),
                                      role="qty", ref_obj=r.unit))
        elif r.kind == "times":
            v = r.base_value * r.factor
            extra.append(Quantity(value=v, text=r.text, unit=r.unit,
                                  obj=_SINGULAR.get(r.unit, r.unit),
                                  role="qty", ref_obj=r.unit))
        elif r.kind == "subtract":
            extra.append(Quantity(value=r.base_value, text=r.text,
                                  unit=None, obj=None, role="subtract",
                                  ref_obj=""))
    return quants + extra


def _bind_roles(text: str, quants: List[Quantity]) -> List[Quantity]:
    """Jeder Menge eine Rolle geben: qty/partitive/ratio/price/duration."""
    low = text.lower()
    # ratio: "half as many X" referenziert ein vorheriges X
    for q in quants:
        if q.role == "ratio":
            continue
        if re.search(r"\b(as many|as much)\b", q.text):
            q.role = "ratio"
            # referenziertes Objekt = dasselbe NP wie im Phrasen-Rest
            q.ref_obj = q.obj
    # partitive: "N of her friends" (Ziffer direkt vor 'of' — nicht
    # "each of the first 4 houses", das eine jede-Relation ist; und
    # nicht subtract-Quantities)
    for q in quants:
        if q.role in ("ratio", "subtract"):
            continue
        if re.search(r"\d+\s+of\b", q.text):
            q.role = "partitive"
    # 1:1-Übertragung: "sold <obj> to N of her friends" -> die Menge von
    # <obj> ist N (jeder Empfänger erhält eins) — die partitive Menge wird
    # zur Objekt-Menge des Verkaufs.
    for q in quants:
        if q.role != "partitive":
            continue
        before = low[max(0, q.span[0] - 60):q.span[0]]
        m = re.search(r"(sold|gave|sent|mailed|donated|lent|distributed|\
                       (?:gave|sold)\s+out)\s+([a-z]+)\s+to\s*$", before)
        if m:
            obj = _SINGULAR.get(m.group(2), m.group(2))
            q.obj = obj          # jetzt: Menge des Verkaufsobjekts
            q.unit = obj
            q.role = "qty"
            q.text += f" (={obj}s)"
    # duration: Einheit ist Zeit und Kontext ist "for/in N days"
    for q in quants:
        if q.role in ("ratio", "partitive"):
            continue
        if q.unit in ("days", "hours", "weeks", "months", "years",
                      "minutes", "seconds"):
            before = low[max(0, q.span[0] - 6):q.span[0]]
            if re.search(r"\b(for|in|over|during)\s*$", before):
                q.role = "duration"
    # price: Dollar-Einheit oder "costs X" 
    for q in quants:
        if q.role not in ("qty",):
            continue
        if q.unit == "dollars" or (q.obj and q.obj in ("cost", "price")):
            q.role = "price"
    return quants


def _unit_and_obj(q: Quantity, text: str) -> Quantity:
    """Einheit + Objekt aus der Phrase ableiten."""
    phrase = q.text
    head = _np_head(phrase)
    if head:
        q.obj = _SINGULAR.get(head, head)
        q.unit = head
    # Einheit explizit: "48 clips" -> unit=clips obj=clips
    return q


# ---------------------------------------------------------------------------
# 3. Frageziel
# ---------------------------------------------------------------------------


def _parse_target(question: str) -> QuestionTarget:
    """'How many X ... altogether' -> Target(X, sum).
    Die FRAGE ist der letzte how-Satz (nicht "how much pizza he CAN
    eat" in der Erzählung)."""
    low = question.lower()
    t = QuestionTarget(ok=False, op="sum")
    # letzter Satz mit 'how many/much'
    sents = [s for s in re.split(r"(?<=[.!?])\s+", low)
             if re.search(r"how (?:many|much)", s)]
    frag = sents[-1] if sents else low
    m = re.search(r"how (?:many|much)\s+([a-z][a-z ]{1,24}?)\s+"
                  r"(?:did|do|does|would|are|were|is|has|have)?\s*"
                  r"([a-z ]{0,12}?)(?:altogether|in all|in total|total|"
                  r"combined|left|remain|remaining|more than|less than|"
                  r"spend|pay|cost)?", frag)
    if m:
        obj = m.group(1).strip()
        t.obj = _SYNONYMS.get(obj, _SINGULAR.get(obj, obj))
        tail = m.group(2) or ""
        if re.search(r"left|remain", low):
            t.op = "left"
        elif (re.search(r"fifth|last|the rest|remaining", low)
              or (re.search(r"(?:second|third|fourth)\b", low)
                  and not re.search(r"(?:second|third|fourth)\s+"
                                    r"(?:hour|day|week|month|year)", low))):
            t.op = "left"
        elif any(w in low for w in _DIFF_WORDS):
            t.op = "diff"
        elif any(w in low for w in _SUM_WORDS):
            t.op = "sum"
        elif any(w in low for w in _PRODUCT_WORDS):
            t.op = "product"
        t.ok = bool(t.obj)
        return t
    # "how much" ohne Objekt -> Geldziel
    if re.search(r"how much (?:did|does|do)", low):
        t.op = "product"
        t.ok = True
    return t


# ---------------------------------------------------------------------------
# 4. Resolver: Bindungs-Operationen
# ---------------------------------------------------------------------------


def _resolve(question: str) -> BindingResult:
    """Vollständige Bindung + Rechnung (erste Stufe: Summe/Ratio/Partitiv)."""
    res = BindingResult()
    low = question.lower()
    qs = _find_quantities(question)
    if not qs:
        res.reason = "keine Mengen"
        return res
    quants: List[Quantity] = []
    for phrase, val, s, e in qs:
        q = Quantity(value=val, text=phrase, unit=None, obj=None,
                     role="qty", span=(s, e))
        _unit_and_obj(q, question)
        quants.append(q)
    # Relationen (jede/more/fewer/times/Typ-pro-Stück) erzeugen ZUSÄTZLICHE
    # gebundene Mengen — sie sind höherwertig als Roh-Zahlen. ITERATIV:
    # Relationen können Relationen referenzieren ("13 fewer rose than
    # truck" mit truck=20 aus "9 more truck than snowflake").
    for _ in range(4):
        rels, quants = _find_relations(question, quants)
        before = len(quants)
        quants = _apply_relations(rels, quants)
        if len(quants) == before:
            break
    quants = _effective_quantities(quants)
    quants = _bind_roles(question, quants)
    res.quantities = quants
    res.target = _parse_target(question)

    # --- Abstinenz-Gate 0.5: Variablen-Gleichungen ("If Mina memorized
    # 24 digits, how many did Sam memorize?") — vor den Objekt-Gates,
    # weil das Ziel eine PERSON ist, kein Objekt.
    var_ans = _solve_variables(rels, question)
    if var_ans is not None:
        res.answer = _fmt(var_ans)
        res.ok = True
        res.reason = "Variablen-Propagation"
        return res

    # --- Abstinenz-Gate 0.75: Futterketten (each X eats N Y; M Z)
    fc = _food_chain(question, rels, quants)
    if fc is not None:
        # Rate-Roh-Mengen derselben Einheit ersetzen ("12 beetles" sind
        # Teil der Kette, keine eigene Menge)
        quants = [q for q in quants
                  if not (q.role == "qty" and q.text != fc.text
                          and (q.unit or "").rstrip("s") ==
                          (fc.unit or "").rstrip("s"))]
        quants.append(fc)


    # --- Abstinenz-Gate 1: Ziel-Objekt muss gebunden sein ---
    tgt = res.target
    # --- Rate x Dauer ("6 sentences per minute, for 43 minutes" -> 258)
    rate_ans = _rate_duration(question, rels, quants, tgt)
    if rate_ans is not None:
        res.answer = _fmt(rate_ans)
        res.ok = True
        res.reason = "rate x dauer"
        return res
    if not tgt.ok:
        res.reason = "kein Frageziel"
        return res

    # --- Abstinenz-Gate 2: Mengen des Zielobjekts (qty) müssen existieren ---
    target_qty = [q for q in quants if q.role == "qty" and q.obj == tgt.obj]
    ratios = [q for q in quants if q.role == "ratio" and q.obj == tgt.obj]
    subtracts = [q.value for q in quants if q.role == "subtract"]

    def _mit_subtraktion(total: Fraction) -> Fraction:
        return total - sum(subtracts) if subtracts else total

    # Fall A: "N clips ... half as many clips ... altogether"
    #   -> sum(N, N/2) — ratio referenziert die erste qty desselben Objekts
    if target_qty and ratios and tgt.op == "sum":
        base = target_qty[0].value
        total = base
        for r in ratios:
            total = total + base * r.value
        # partitive Mengen desselben Objekts addieren (Natalia-Fall)
        part = [q for q in quants if q.role == "partitive" and q.obj == tgt.obj]
        if part:
            total = total + base * part[0].value
        res.answer = _fmt(total)
        res.ok = True
        res.reason = f"sum(base={base}, ratio, partitiv)"
        return res

    # Fall B: einfache Summe: "N X, then M X ... altogether"
    if len(target_qty) >= 2 and tgt.op == "sum":
        total = _mit_subtraktion(sum(q.value for q in target_qty))
        res.answer = _fmt(total)
        res.ok = True
        res.reason = f"sum({len(target_qty)} Mengen)"
        return res

    # Fall B': eine einzige qty desselben Objekts + sum-Ziel -> die Menge
    # selbst (die Frage fragt nur nach der Gesamtheit einer Teilmenge)
    if len(target_qty) == 1 and not ratios and tgt.op == "sum":
        res.answer = _fmt(target_qty[0].value)
        res.ok = True
        res.reason = "einzelne gebundene Menge"
        return res

    # Fall C: left — "total of N, each of M groups has K, how many does
    # the rest/fifth/last have" -> total - Summe der Teile
    if tgt.op == "left":
        total_q = [q for q in target_qty
                   if re.search(r"\btotal\b|\bin all\b", q.text)
                   or re.search(r"\btotal\b|\bin all\b",
                                low[max(0, q.span[0] - 15):q.span[0]])]
        parts = [q for q in target_qty if q not in total_q]
        if total_q and parts:
            res.answer = _fmt(total_q[0].value -
                              sum(p.value for p in parts))
            res.ok = True
            res.reason = "left(total - teile)"
            return res
        if total_q and len(target_qty) == 1:
            res.answer = _fmt(total_q[0].value)
            res.ok = True
            res.reason = "left(total)"
            return res

    # Fall D: diff: "how many more X than Y"
    if len(target_qty) >= 2 and tgt.op == "diff":
        res.answer = _fmt(abs(target_qty[0].value - target_qty[1].value))
        res.ok = True
        res.reason = "diff"
        return res

    res.reason = f"Bindung unvollständig: {len(target_qty)} qty, " \
                 f"{len(ratios)} ratio, op={tgt.op}"
    return res


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def bind(question: str) -> BindingResult:
    """Bindungs-Parser-Einstieg."""
    try:
        return _resolve(question)
    except Exception:
        return BindingResult(reason="Parser-Fehler (abstinent)")


def solve(question: str) -> Optional[str]:
    """Lösen via Bindung — None wenn Bindung unvollständig."""
    r = bind(question)
    return r.answer if r.ok else None


def _fmt(f: Fraction) -> str:
    """Exakt wie math._fmt: Ganzzahl oder max-4-stelliges Dezimal."""
    if f.denominator == 1:
        return str(f.numerator)
    scaled = f * 10000
    if scaled.denominator == 1:
        s = str(scaled.numerator)
        return s[:-4] + "." + s[-4:] if len(s) > 4 else "0." + s.zfill(4)
    return str(float(f))
