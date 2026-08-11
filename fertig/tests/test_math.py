"""FERTIG — Tests des MathSolver (Operationsketten)."""

from __future__ import annotations

from fertig.math import solve, solve_chain, extract_numbers, gold_answer


def test_extract_numbers():
    nums = extract_numbers("12 miles in 3 hours")
    assert len(nums) >= 2


def test_gold_answer():
    assert gold_answer("Answer: #### 72") == "72"



def test_rate_chain_left():
    # 20 Liter/Tag, 5 Tage, 3 Liter/Tag Verbrauch -> wie viel übrig?
    assert solve("A cow produces 20 liters per day for 5 days and uses "
                 "3 liters per day. How many liters are left?") == "85"


def test_rate_chain_total():
    assert solve("A factory makes 12 cars per day for 7 days. "
                 "How many cars in total?") == "84"


def test_each_chain_total():
    assert solve("Mark buys 5 books, each costing 12 dollars. "
                 "What is the total cost?") == "60"


def test_percent_with_left():
    assert solve("A tank has 200 liters. 25% of it is used. "
                 "How many liters are left?") == "150"


def test_half_chain():
    assert solve("Natalia sold clips to 48 of her friends in April, and "
                 "then she sold half as many clips in May. How many clips "
                 "did she sell altogether?") == "72"


# --- SemanticSolver (FORGE-Brücke: Graph = Programm) ---

def test_semantic_half_with_adjust():
    from fertig.semantic import parse_semantic
    q = ("Randy has 60 mango trees on his farm. He also has 5 less than "
         "half as many coconut trees as mango trees. How many coconut "
         "trees does he have?")
    assert parse_semantic(q).solve() == 25


def test_semantic_total_sum():
    from fertig.semantic import parse_semantic
    q = ("Randy has 60 mango trees on his farm. He also has 5 less than "
         "half as many coconut trees as mango trees. How many trees does "
         "he have in total?")
    assert parse_semantic(q).solve() == 85


def test_semantic_half_in_month():
    from fertig.semantic import parse_semantic
    q = ("Natalia sold clips to 48 of her friends in April, and then she "
         "sold half as many clips in May. How many clips did she sell "
         "altogether?")
    assert parse_semantic(q).solve() == 72


def test_semantic_abstains_without_relation():
    from fertig.semantic import parse_semantic
    # keine vollständige Relations-Kette -> ehrlich None
    q = "Kylar went to the store to buy glasses for his new apartment."
    assert parse_semantic(q).solve() is None


# --- Sprachagnostik: gleiche Primitive, andere Sprache ---

def test_semantic_german_half_with_adjust():
    from fertig.semantic import parse_semantic, detect_language
    q = ("Randy hat 60 Mangobäume auf seiner Farm. Er hat auch 5 weniger "
         "als halb so viele Kokosbäume wie Mangobäume. Wie viele "
         "Kokosbäume hat er?")
    assert detect_language(q) == "de"
    assert parse_semantic(q).solve() == 25


def test_semantic_german_total():
    from fertig.semantic import parse_semantic
    q = ("Randy hat 60 Mangobäume auf seiner Farm. Er hat auch 5 weniger "
         "als halb so viele Kokosbäume wie Mangobäume. Wie viele Bäume hat "
         "er insgesamt?")
    assert parse_semantic(q).solve() == 85


def test_semantic_german_sold_half():
    from fertig.semantic import parse_semantic
    q = ("Natalia verkaufte im April Clips an 48 ihrer Freunde, und dann "
         "verkaufte sie im Mai halb so viele Clips. Wie viele Clips "
         "verkaufte sie insgesamt?")
    assert parse_semantic(q).solve() == 72


def test_detect_language():
    from fertig.semantic import detect_language
    assert detect_language("How many trees does he have?") == "en"
    assert detect_language("Wie viele Bäume hat er?") == "de"
