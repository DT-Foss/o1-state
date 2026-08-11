"""FERTIG — Tests des Benchmark-Runners (offline, ohne Netz)."""

from __future__ import annotations

from fertig.bench import TrigramLM, _first_content_word


def test_trigram_lm_prefers_fluent():
    text = ("the cat sat on the mat. the cat sat on the mat. "
            "the dog ran through the park. the cat sat on the mat. "
            "the cat sat on the mat. the cat sat on the mat. ")
    lm = TrigramLM(text, max_vocab=200)
    good = lm.sentence_logprob("the cat sat on the mat")
    bad = lm.sentence_logprob("mat the sat cat the on")
    assert good > bad


def test_trigram_lm_deterministic():
    text = "a b c. a b c. a b c. a b c. "
    lm1 = TrigramLM(text)
    lm2 = TrigramLM(text)
    assert lm1.sentence_logprob("a b c") == lm2.sentence_logprob("a b c")


def test_trigram_lm_oov_sentence():
    lm = TrigramLM("the cat sat. the cat sat. ")
    assert lm.sentence_logprob("zzz zzz zzz") == float("-inf")


def test_first_content_word():
    assert _first_content_word("please play some music") == "play"
    assert _first_content_word("the weather in berlin") == "weather"
    assert _first_content_word("the") is None
