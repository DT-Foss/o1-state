"""FERTIG — Tests des automatischen Regel-Miners."""

from __future__ import annotations

from fertig.miner import parse_gold_ops, mine_rules, apply_rules


def test_parse_gold_ops():
    a = "Natalia sold 48/2 = <<48/2=24>>24 clips in May."
    ops = parse_gold_ops(a)
    assert ops == [("div", 48, 2, 24)]


def test_mine_rules_learns_per_mul():
    # "X per Y" -> mul aus dem Gold-Signal lernen
    qs = ["A machine makes 5 cars per hour for 3 hours.",
          "A baker bakes 4 cakes per day for 6 days.",
          "A pump moves 7 liters per minute for 2 minutes.",
          "A plant grows 3 cm per week for 8 weeks."]
    ans = [f"5*3 = <<5*3=15>>15", f"4*6 = <<4*6=24>>24",
           f"7*2 = <<7*2=14>>14", f"3*8 = <<3*8=24>>24"]
    rules = mine_rules(qs, ans, min_count=2)
    # "per" erscheint zwischen den Zahlen -> mul gelernt
    assert any("per" in sig and rules[sig] == "mul" for sig in rules)


def test_mine_rules_learns_each_mul():
    qs = ["5 books each costing 12 dollars.",
          "3 pens each costing 2 dollars.",
          "8 shirts each costing 15 dollars."]
    ans = [f"5*12 = <<5*12=60>>60", f"3*2 = <<3*2=6>>6",
           f"8*15 = <<8*15=120>>120"]
    rules = mine_rules(qs, ans, min_count=2)
    assert any("each" in sig and rules[sig] == "mul" for sig in rules)


def test_apply_rules_single_step():
    rules = {"per": "mul"}
    assert apply_rules("A machine makes 5 cars per hour for 3 hours.", rules) \
        in ("15", "15.0")


def test_apply_rules_abstains():
    # keine Regel passt -> ehrlich None
    assert apply_rules("The sky is blue today.", {"per": "mul"}) is None
