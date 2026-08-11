"""pytest-Tests fuer key_filter.py (Junk-Schluessel-Filter, Sprecher-
Datenbasis v2, Baustein 2).

Run:
    python3 -m pytest hsslm/data/test_key_filter.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from key_filter import is_junk_key, junk_reason, filter_pair, compute_junk_stats


def test_heading_residue_is_junk():
    assert is_junk_key("= = overview") is True
    assert is_junk_key("brick = = main") is True
    assert junk_reason("= = overview") == "heading_residue (=)"


def test_at_artifact_is_junk():
    assert is_junk_key("@-@ king ravana") is True
    assert is_junk_key("3 @.@ 5 m") is True
    assert is_junk_key("on @-@ site museum has") is True
    assert junk_reason("@-@ king ravana") == "at_artifact (@)"


def test_no_alpha_word_is_junk():
    assert is_junk_key("1941") is True
    assert is_junk_key("123 456") is True
    assert is_junk_key("...") is True
    assert junk_reason("1941") == "no_alpha_word"


def test_single_char_is_junk():
    assert is_junk_key("x") is True
    assert is_junk_key(" ") is True  # trims to empty -> len<=1
    assert junk_reason("x") == "single_char"


def test_clean_keys_are_not_junk():
    assert is_junk_key("the meeting") is False
    assert is_junk_key("modern musical instruments") is False
    assert is_junk_key("director daniel kleinman") is False
    assert junk_reason("the meeting") == "clean"


def test_first_matching_rule_wins():
    # traegt sowohl '=' als auch '@' -- Regel-Reihenfolge: '=' zuerst
    assert junk_reason("= @ =") == "heading_residue (=)"


def test_filter_pair_rejects_when_any_field_is_junk():
    pair = {"trigger": "the meeting", "outcome": "@-@ artifact"}
    assert filter_pair(pair) is False
    pair2 = {"trigger": "@-@ artifact", "outcome": "the meeting"}
    assert filter_pair(pair2) is False


def test_filter_pair_accepts_when_all_fields_clean():
    pair = {"trigger": "the meeting", "outcome": "modern instruments"}
    assert filter_pair(pair) is True


def test_filter_pair_only_checks_present_fields():
    # kein 'outcome'-Feld im Dict -> nur 'trigger' wird geprueft
    pair = {"trigger": "the meeting"}
    assert filter_pair(pair) is True
    pair2 = {"trigger": "@-@ x"}
    assert filter_pair(pair2) is False


def test_compute_junk_stats_counts_and_ratio():
    keys = ["the meeting", "@-@ x", "= = overview", "1941", "clean phrase"]
    stats = compute_junk_stats(keys)
    assert stats["total"] == 5
    assert stats["n_junk"] == 3
    assert stats["n_clean"] == 2
    assert abs(stats["junk_ratio"] - 3 / 5) < 1e-9
    assert stats["reason_counts"]["at_artifact (@)"] == 1
    assert stats["reason_counts"]["heading_residue (=)"] == 1
    assert stats["reason_counts"]["no_alpha_word"] == 1


def test_compute_junk_stats_empty_input():
    stats = compute_junk_stats([])
    assert stats["total"] == 0
    assert stats["junk_ratio"] == 0.0
