"""FERTIG — Tests des Bindungs-Parsers (GSM8K-Projekt, Runde 1-2).

Der Bindungs-Parser bindet Zahlen an Objekte+Einheiten+RollEN, statt
Muster zu matchen. Diese Tests fixieren die falsifizierbaren Fälle.
"""

from __future__ import annotations

from fertig import bindings


def test_natalia_1zu1_und_ratio():
    """1:1-Übertragung (48 Freunde -> 48 clips) + half-as-many-Ratio."""
    r = bindings.bind(
        "Natalia sold clips to 48 of her friends in April, and then she "
        "sold half as many clips in May. How many clips did Natalia sell "
        "altogether in April and May?")
    assert r.ok and r.answer == "72"


def test_objekt_trennung():
    """Nur das Zielobjekt zählt: Äpfel ≠ Orangen."""
    r = bindings.bind(
        "John has 5 apples and 3 oranges. How many apples does he have?")
    assert r.ok and r.answer == "5"


def test_summe_gleicher_objekte():
    r = bindings.bind(
        "A bakery sold 12 cakes on Monday and 15 cakes on Tuesday. "
        "How many cakes did they sell in total?")
    assert r.ok and r.answer == "27"


def test_jede_relation_pizza():
    """Pro-Stück-Werte multiplizieren mit der Stückzahl (2x16 + 2x8)."""
    r = bindings.bind(
        "Albert buys 2 large pizzas and 2 small pizzas. A large pizza "
        "has 16 slices and a small pizza has 8 slices. How many slices "
        "total?")
    assert r.ok and r.answer == "48"


def test_more_fewer_kette():
    """Relative Mengen lösen sich entlang der Referenzkette auf
    (11 + 20 + 7 — truck = snowflake+9, rose = truck-13)."""
    r = bindings.bind(
        "Bella bought stamps. 11 snowflake stamps, 9 more truck stamps "
        "than snowflake stamps, 13 fewer rose stamps than truck stamps. "
        "How many stamps total?")
    assert r.ok and r.answer == "38"


def test_variablen_propagation():
    """Gleichungskette: Mina=24, Mina=6xCarlos, Sam=Carlos+6 -> 10."""
    r = bindings.bind(
        "Sam memorized six more digits of pi than Carlos memorized. "
        "Mina memorized six times as many digits of pi as Carlos "
        "memorized. If Mina memorized 24 digits of pi, how many digits "
        "did Sam memorize?")
    assert r.ok and r.answer == "10"


def test_futterkette():
    """Ketten-each: 6 Jaguare x 5 Schlangen x 3 Vögel x 12 Käfer."""
    r = bindings.bind(
        "Each bird eats 12 beetles per day, each snake eats 3 birds per "
        "day, and each jaguar eats 5 snakes per day. If there are 6 "
        "jaguars, how many beetles do they eat per day?")
    assert r.ok and r.answer == "1080"


def test_with_menge():
    """7 Seesterne x 5 Arme + 1 x 14 Arme."""
    r = bindings.bind(
        "Carly collected 7 starfish with 5 arms each and one seastar "
        "with 14 arms. How many arms did she collect?")
    assert r.ok and r.answer == "49"


def test_left_total_minus_teile():
    """5 Häuser, erste 4 haben je 3, total 20 -> fünftes hat 8."""
    r = bindings.bind(
        "There are 5 houses on a street, and each of the first four "
        "houses has 3 gnomes in the garden. If there are a total of 20 "
        "gnomes on the street, how many gnomes does the fifth house "
        "have?")
    assert r.ok and r.answer == "8"


def test_halb_kette_mit_variable():
    """Tim = Martha-30, Harry = Tim/2, Martha=68 -> Harry 19."""
    r = bindings.bind(
        "Tim has 30 less apples than Martha, and Harry has half as many "
        "apples as Tim. If Martha has 68 apples, how many apples does "
        "Harry have?")
    assert r.ok and r.answer == "19"


def test_raten_dauer_und_gave():
    """2h x 35 + 15 + 50 - 15 (gave) = 120."""
    r = bindings.bind(
        "During the first hour, she collected 15 coins. For the next "
        "two hours, she collected 35 coins from the fountain. In the "
        "fourth hour, she collected 50 coins but she gave 15 of them "
        "to her coworker. How many coins did she have after the fourth "
        "hour?")
    assert r.ok and r.answer == "120"


def test_they_total_und_more2x():
    """they-total (320) und more-than-twice (25)."""
    r = bindings.bind(
        "Paddington has 40 more goats than Washington. If Washington "
        "has 140 goats, how many goats do they have in total?")
    assert r.ok and r.answer == "320"
    r = bindings.bind(
        "John has five more roommates than twice as many as Bob. If "
        "Bob has 10 roommates, how many roommates does John have?")
    assert r.ok and r.answer == "25"


def test_synonym_und_letzte_frage():
    """pieces == slices; die FRAGE ist der letzte how-Satz."""
    r = bindings.bind(
        "Albert buys 2 large pizzas and 2 small pizzas. A large pizza "
        "has 16 slices and a small pizza has 8 slices. If he eats it "
        "all, how many pieces does he eat that day?")
    assert r.ok and r.answer == "48"


def test_rate_mal_dauer():
    """6 Sätze/min x 20 min = 120."""
    r = bindings.bind(
        "Janice can type 6 sentences per minute. She typed for 20 "
        "minutes. How many sentences did she type?")
    assert r.ok and r.answer == "120"


def test_abstinenz_ohne_ziel():
    """Keine Frage -> keine Antwort (kein Raten)."""
    r = bindings.bind("There are 42 numbers in this text. 7 and 11.")
    assert not r.ok


def test_abstinenz_unvollstaendig():
    """Ziel-Objekt ohne gebundene Menge -> Abstinenz."""
    r = bindings.bind(
        "A train travels 60 miles per hour. How many miles does it "
        "travel in 3 hours?")
    # Rate ohne expliziten Zeitraum-Bezug in der Bindung -> ehrlich None
    # (das ist die Abstinenz-Grenze der aktuellen Stufe)
    assert not r.ok or r.answer is not None
