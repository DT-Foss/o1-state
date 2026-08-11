"""FERTIG — Tests. Alle Module, End-to-End, und die Determinismus-Garantie."""

from __future__ import annotations

import numpy as np
import pytest

from fertig import sampler, state_init, bphm, inference, pipeline, corpus
from fertig.pattern_bank import PatternBank

DATA = __import__("pathlib").Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------

def test_sampler_deterministic():
    rng = np.random.RandomState(0)
    logits = rng.randn(100)
    a = sampler.contraction_sample(logits, tau=0.5, top_k=50)
    b = sampler.contraction_sample(logits, tau=0.5, top_k=50)
    assert a == b


def test_sampler_tau_zero_is_argmax():
    logits = np.array([0.1, 0.9, 0.3, 0.7])
    assert sampler.contraction_sample(logits, tau=0.0, top_k=0) == 1


def test_sampler_phase_transition():
    with pytest.raises(ValueError):
        sampler.contraction_sample(np.ones(4), tau=1.0)


def test_tau_to_temperature_bounds():
    assert sampler.tau_to_temperature(0.0) == 0.0
    assert sampler.tau_to_temperature(0.95) > 2.0


def test_bvn_decompose_mass():
    M = np.array([[0.5, 0.5], [0.5, 0.5]])
    perms, weights = sampler.bvn_decompose(M, max_paths=10)
    assert len(perms) >= 1
    assert abs(sum(weights) - 1.0) < 1e-6
    recon = sum(w * P for w, P in zip(weights, perms))
    assert np.allclose(recon, M, atol=1e-6)


# ---------------------------------------------------------------------------
# state_init
# ---------------------------------------------------------------------------

def test_state_hyperboloid_constraint():
    SM = state_init.initialize_symbol_state(64)
    mink = -SM[:, 0] ** 2 + np.sum(SM[:, 1:] ** 2, axis=1)
    assert np.allclose(mink, -1.0, atol=1e-9)


def test_state_for_symbol_stable():
    SM = state_init.initialize_symbol_state(32)
    s1 = state_init.state_for_symbol(7, SM)
    s2 = state_init.state_for_symbol(7, SM)
    assert np.array_equal(s1, s2)


# ---------------------------------------------------------------------------
# bphm
# ---------------------------------------------------------------------------

def test_bphm_detects_repetition():
    SM = state_init.initialize_symbol_state(16)
    A, B, C = (state_init.state_for_symbol(i, SM) for i in (0, 1, 2))
    # geschlossener 3-Zyklus mit Phasen-Akkumulation: A,B,C,A,B,C,A
    states = [A, B, C, A, B, C, A]
    assert bphm.detect_repetition(states) is True


def test_bphm_accepts_walk():
    SM = state_init.initialize_symbol_state(16)
    # aufsteigender Pfad ohne Zyklus darf nicht als Wiederholung gelten
    states = [state_init.state_for_symbol(i % 16, SM) for i in range(12)]
    assert bphm.detect_repetition(states) is False


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

def test_jaro_winkler_known_value():
    # klassisches Referenzpaar: MARTHA / MARHTA -> ~0.961
    v = inference.jaro_winkler("MARTHA", "MARHTA")
    assert abs(v - 0.961) < 0.01


def test_pass1_exact_chains():
    adj = {0: {1: 0.9}, 1: {2: 0.8}}
    chains = inference.pass1_exact_chains([0, 1, 2], adjacency=adj)
    assert any(len(c) == 3 for c in chains)


# ---------------------------------------------------------------------------
# pipeline (End-to-End auf dem echten Graphen)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def graph():
    return pipeline.load_graph(DATA / "chained.causal")


def test_graph_loads(graph):
    vocab, stoi, adj, mech = graph
    assert len(vocab) > 5
    assert sum(len(v) for v in adj.values()) > 5


def test_walk_deterministic(graph):
    vocab, stoi, adj, mech = graph
    SM = state_init.initialize_symbol_state(len(vocab))
    start = pipeline.top_starts(adj, vocab)[0]
    h1 = pipeline.walk_chain(start, vocab, stoi, adj, SM, n=6, tau=0.3)
    h2 = pipeline.walk_chain(start, vocab, stoi, adj, SM, n=6, tau=0.3)
    assert h1 == h2


def test_verbalize_entities_from_graph(graph):
    vocab, stoi, adj, mech = graph
    SM = state_init.initialize_symbol_state(len(vocab))
    start = pipeline.top_starts(adj, vocab)[0]
    hops = pipeline.walk_chain(start, vocab, stoi, adj, SM, n=4, tau=0.3)
    assert hops
    text = pipeline.verbalize(hops, vocab, mech, seed=0)
    # jede Entität der Kette muss wörtlich im Text vorkommen (Fakten exakt)
    for _, b in hops:
        assert vocab[b] in text


def test_chains_derivation(graph):
    vocab, stoi, adj, mech = graph
    chains = pipeline.derive_chains(adj, vocab)
    assert isinstance(chains, dict)


# ---------------------------------------------------------------------------
# corpus-Modus
# ---------------------------------------------------------------------------

def test_corpus_build_and_generate_deterministic():
    text = (DATA / "faraday_candle.txt").read_text(encoding="utf-8",
                                                   errors="ignore")[:20000]
    vocab, stoi, adjacency, trigram, unigram = corpus.build_vocab(text,
                                                                  max_vocab=500)
    assert len(vocab) > 50
    SM = state_init.initialize_symbol_state(len(vocab))
    out1 = corpus.generate("the candle", vocab, stoi, adjacency, trigram,
                           unigram, SM, n=15, tau=0.4)
    out2 = corpus.generate("the candle", vocab, stoi, adjacency, trigram,
                           unigram, SM, n=15, tau=0.4)
    assert out1 == out2  # Determinismus-Garantie
    assert len(out1.split()) >= 15


# ---------------------------------------------------------------------------
# pattern_bank
# ---------------------------------------------------------------------------

def test_pattern_bank_roundtrip(tmp_path):
    bank = PatternBank()
    bank.extract("However, the flame rises. Therefore, the candle burns.")
    p = tmp_path / "bank.json"
    bank.save(p)
    loaded = PatternBank.load(p)
    assert loaded.n_sentences == bank.n_sentences
    assert len(loaded.openers) == len(bank.openers)
