"""FERTIG — Tests der Intent/Tool/Lern/Arena-Kette."""

from __future__ import annotations

from pathlib import Path

import pytest

from fertig import intent, tools, learn, arena
from fertig.pipeline import load_graph, DEFAULT_GRAPH

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def vocab():
    return load_graph(DATA / "chained.causal")[0]


# ---------------------------------------------------------------------------
# intent: Parse
# ---------------------------------------------------------------------------

def test_intent_explain(vocab):
    it = intent.parse_command("explain how smoking affects health", vocab)
    assert it.action == "explain"
    assert it.target == "smoking"
    assert it.status == "ok"
    assert it.grounded
    assert it.confidence > 0.5
    assert it.tool == "speech"


def test_intent_prevent(vocab):
    it = intent.parse_command("how can i prevent lung damage", vocab)
    assert it.action == "prevent"
    assert it.target == "lung damage"
    assert it.tool == "prevent"


def test_intent_find(vocab):
    it = intent.parse_command("list what tar buildup causes", vocab)
    assert it.action == "find"
    assert it.target == "tar buildup"


def test_intent_unknown_action_honest(vocab):
    it = intent.parse_command("flurble the smoking", vocab)
    assert it.status == "unknown-action"
    assert it.tree.startswith("(parse: verb unbekannt")


def test_intent_deterministic(vocab):
    a = intent.parse_command("explain how smoking affects health", vocab)
    b = intent.parse_command("explain how smoking affects health", vocab)
    assert a.tree == b.tree and a.confidence == b.confidence


# ---------------------------------------------------------------------------
# intent: Which-Path / Ambiguität
# ---------------------------------------------------------------------------

def test_intent_ambiguity_flag(vocab):
    # "show" ist mehrdeutig — das System muss entweder eindeutig oder
    # ehrlich mehrdeutig sein, nie raten
    it = intent.parse_command("show me the effects of exercise", vocab)
    assert it.status in ("ok", "ambiguous")


def test_intent_ambiguity_fires_on_tie(vocab):
    # Zwei Ziel-Kandidaten mit identischem Score und KEINEM syntaktischen
    # Signal -> echtes Patt -> die Which-Path-Visibility MUSS feuern
    # (Regression: war immer 0, weil Parsimony-Scores negativ sind und die
    # alte Formel s1 > 0 verlangte)
    it = intent.parse_command("explain smoking health", vocab)
    assert it.status == "ambiguous"
    assert it.ambiguity > 0.5


def test_intent_how_pattern_resolves_topic(vocab):
    # "how X affects Y" -> X ist das Thema, kein falsches Patt
    it = intent.parse_command("explain how smoking affects health", vocab)
    assert it.target == "smoking"
    assert it.status == "ok"
    assert it.ambiguity < 0.5  # klar genug, um zu entscheiden


def test_intent_ambiguity_zero_on_clear_winner(vocab):
    it = intent.parse_command("stop smoking", vocab)
    assert it.ambiguity == 0.0
    assert it.status == "ok"


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

def test_tools_execute_explain(vocab):
    it = intent.parse_command("explain how smoking affects health", vocab)
    res = tools.execute(it, str(DATA / "chained.causal"))
    assert res.ok
    assert res.tool == "speech"
    assert "smoking" in res.text.lower()  # Fakten exakt


def test_tools_execute_prevent(vocab):
    # "exercise" hat im Graphen einen echten Reduktor (breathlessness reduces ...)
    it = intent.parse_command("how can i prevent exercise", vocab)
    res = tools.execute(it, str(DATA / "chained.causal"))
    assert res.ok
    assert res.tool == "prevent"
    assert "reduces" in res.text or "prevents" in res.text


def test_tools_prevent_honest_dead_end(vocab):
    # "lung damage" hat im Graphen keinen Reduktor -> ehrlicher Fehlschlag
    it = intent.parse_command("how can i prevent lung damage", vocab)
    res = tools.execute(it, str(DATA / "chained.causal"))
    assert not res.ok
    assert "kein Reduktor" in res.text


def test_tools_execute_not_grounded(vocab):
    it = intent.parse_command("explain how flurble works", vocab)
    res = tools.execute(it, str(DATA / "chained.causal"))
    assert not res.ok  # ehrlich: nicht ausführbar


# ---------------------------------------------------------------------------
# learn
# ---------------------------------------------------------------------------

def test_learn_grows_lexicon(tmp_path):
    text = ("the flame rises. reduce the risk. avoid the smoke. "
            "the candle burns. reduce the damage. prevent the buildup. "
            "the flame flickers. reduce the smoke.")
    lex = learn.learn(text, min_count=1)
    assert lex.tokens == len(text.split())
    assert "reduce" in lex.actions
    assert "flame" in lex.nouns
    # Roundtrip
    p = tmp_path / "lex.json"
    lex.save(p)
    loaded = learn.Lexicon.load(p)
    assert loaded.actions == lex.actions


def test_learn_extends_intent_coverage(vocab):
    # Ein gelerntes Verb erweitert die Intent-Abdeckung ohne Code-Änderung
    lex = learn.Lexicon()
    lex.actions["minimize"] = {"action": "prevent", "weight": 0.9}
    it = intent.parse_command("minimize the lung damage", vocab, lexicon=lex)
    assert it.action == "prevent"
    assert it.target == "lung damage"


# ---------------------------------------------------------------------------
# arena
# ---------------------------------------------------------------------------

def test_arena_runs_and_is_deterministic(vocab, capsys):
    r1 = arena.run_arena(verbose=False)
    assert r1.total == len(arena.EVAL_SET)
    assert r1.action_hits + r1.unknown <= r1.total
    # deterministisch: zweiter Lauf identisch
    r2 = arena.run_arena(verbose=False)
    assert (r1.action_hits, r1.target_hits, r1.full_hits,
            r1.ambiguous, r1.unknown) == \
           (r2.action_hits, r2.target_hits, r2.full_hits,
            r2.ambiguous, r2.unknown)
