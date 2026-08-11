"""pytest-Tests fuer vocab.py (parameterisiertes Vokabular, Sprecher-
Datenbasis v2, Baustein 4).

Run:
    python3 -m pytest hsslm/data/test_vocab.py -q
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vocab import (
    build_vocab_n, build_extended_vocab_n, tokenize_n,
    save_vocab, load_vocab, compute_oov_stats,
    FACT_TOKEN, SAY_TOKEN,
)


SAMPLE_TEXT = "the cat sat on the mat . the dog ran in the park . " * 3 + \
              "a rare zebra appeared once ."


def test_build_vocab_n_respects_size_limit():
    vocab, stoi, unk, mask = build_vocab_n(SAMPLE_TEXT, 3)
    assert len(vocab) == 3
    assert unk == 3
    assert mask == 4


def test_build_vocab_n_frequency_order():
    vocab, stoi, unk, mask = build_vocab_n(SAMPLE_TEXT, 10)
    # 'the' is by far the most frequent word in the sample
    assert vocab[0] == "the"
    assert stoi["the"] == 0


def test_build_vocab_n_larger_n_yields_more_words():
    _, _, unk5, _ = build_vocab_n(SAMPLE_TEXT, 5)
    _, _, unk_full, _ = build_vocab_n(SAMPLE_TEXT, 100)
    assert unk_full >= unk5  # more slots -> covers at least as many distinct words


def test_build_extended_vocab_n_id_scheme():
    stoi, unk, mask, fact_id, say_id, total = build_extended_vocab_n(SAMPLE_TEXT, 5)
    assert unk == 5
    assert mask == 6
    assert fact_id == 7
    assert say_id == 8
    assert total == 9
    assert stoi[FACT_TOKEN] == fact_id
    assert stoi[SAY_TOKEN] == say_id


def test_tokenize_n_uses_unk_for_oov():
    vocab, stoi, unk, mask = build_vocab_n("apple banana", 1)
    ids = tokenize_n("apple banana cherry", stoi, unk)
    assert len(ids) == 3
    assert unk in ids  # at least one OOV word


def test_save_and_load_vocab_roundtrip():
    vocab, stoi, unk, mask = build_vocab_n(SAMPLE_TEXT, 8)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        path = f.name
    try:
        save_vocab(path, vocab)
        loaded_vocab, loaded_stoi, loaded_unk, loaded_mask, fact_id, say_id, total = load_vocab(path)
        assert loaded_vocab == vocab
        assert loaded_unk == unk
        assert loaded_mask == mask
        assert total == mask + 3  # unk, mask, fact, say beyond len(vocab)
        for w in vocab:
            assert loaded_stoi[w] == stoi[w]
    finally:
        os.unlink(path)


def test_load_vocab_id_scheme_matches_build_extended_vocab_n():
    """Ein persistiertes Vokabular muss EXAKT dasselbe ID-Schema
    reproduzieren wie build_extended_vocab_n auf demselben Text --
    sonst waere ein --vocab-file-Lauf nicht reproduzierbar gegenueber
    einem frisch gebauten Vokabular derselben Groesse."""
    n = 12
    stoi_built, unk_b, mask_b, fact_b, say_b, total_b = build_extended_vocab_n(SAMPLE_TEXT, n)
    vocab_words, _, _, _ = build_vocab_n(SAMPLE_TEXT, n)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        path = f.name
    try:
        save_vocab(path, vocab_words)
        _, stoi_loaded, unk_l, mask_l, fact_l, say_l, total_l = load_vocab(path)
        assert unk_b == unk_l
        assert mask_b == mask_l
        assert fact_b == fact_l
        assert say_b == say_l
        assert total_b == total_l
    finally:
        os.unlink(path)


def test_compute_oov_stats_all_in_vocab():
    vocab, stoi, unk, mask = build_vocab_n("apple banana apple banana", 2)
    stats = compute_oov_stats(["apple banana apple"], stoi, unk)
    assert stats["mean_oov_rate"] == 0.0
    assert stats["pct_texts_zero_oov"] == 1.0


def test_compute_oov_stats_all_oov():
    vocab, stoi, unk, mask = build_vocab_n("apple banana", 2)
    stats = compute_oov_stats(["zebra quokka"], stoi, unk)
    assert stats["mean_oov_rate"] == 1.0


def test_compute_oov_stats_empty_input():
    vocab, stoi, unk, mask = build_vocab_n("apple", 1)
    stats = compute_oov_stats([], stoi, unk)
    assert stats["n_texts"] == 0
