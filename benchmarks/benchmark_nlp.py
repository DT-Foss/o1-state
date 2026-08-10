#!/usr/bin/env python3
"""
Benchmark: NLP Pipeline — Sentiment, Classification, TextUtils
================================================================
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return 1 if passed else 0


def run():
    print("=" * 60)
    print("BENCHMARK: NLP Pipeline")
    print("=" * 60)
    total = 0
    passed = 0

    # === SENTIMENT ===
    print("\n=== Sentiment Analysis ===")
    from core.sentiment import SentimentAnalyzer, NaiveBayesClassifier, IntentDetector, UrgencyDetector

    sa = SentimentAnalyzer()

    # 1. Positive
    total += 1
    r = sa.analyze("This is absolutely amazing and wonderful!")
    ok = r['label'] == 'positive' and r['score'] > 0.5
    passed += test("Positive sentiment", ok, f"score={r['score']}, label={r['label']}")

    # 2. Negative
    total += 1
    r = sa.analyze("This is terrible and horrible, worst experience ever.")
    ok = r['label'] == 'negative' and r['score'] < -0.5
    passed += test("Negative sentiment", ok, f"score={r['score']}, label={r['label']}")

    # 3. Neutral
    total += 1
    r = sa.analyze("The meeting is scheduled for tomorrow at three.")
    ok = r['label'] == 'neutral'
    passed += test("Neutral sentiment", ok, f"score={r['score']}, label={r['label']}")

    # 4. Negation handling
    total += 1
    r = sa.analyze("This is not good at all.")
    ok = r['score'] < 0  # "not good" should be negative
    passed += test("Negation handling", ok, f"score={r['score']}")

    # 5. Intensifier
    total += 1
    r1 = sa.analyze("good")
    r2 = sa.analyze("extremely good")
    ok = r2['score'] > r1['score']
    passed += test("Intensifier amplification", ok,
                   f"good={r1['score']}, extremely good={r2['score']}")

    # 6. Emotion detection
    total += 1
    r = sa.analyze("I am so happy and excited about this!")
    ok = 'joy' in r['emotions']
    passed += test("Emotion detection (joy)", ok, f"emotions={r['emotions']}")

    # 7. Naive Bayes classifier
    print("\n=== Text Classification ===")
    total += 1
    clf = NaiveBayesClassifier()
    clf.train([
        ("I love this product, it's amazing", "positive"),
        ("Great service, very helpful staff", "positive"),
        ("Best purchase I've ever made", "positive"),
        ("Wonderful experience, highly recommend", "positive"),
        ("This is terrible, waste of money", "negative"),
        ("Awful customer service, very rude", "negative"),
        ("Product broke after one day", "negative"),
        ("Worst experience, never again", "negative"),
    ])
    r = clf.classify("This is really great and helpful")
    ok = r['label'] == 'positive' and r['confidence'] > 0.5
    passed += test("NaiveBayes positive", ok, f"label={r['label']}, conf={r['confidence']}")

    # 8. NaiveBayes negative
    total += 1
    r = clf.classify("Terrible product, completely broken")
    ok = r['label'] == 'negative'
    passed += test("NaiveBayes negative", ok, f"label={r['label']}, conf={r['confidence']}")

    # 9. Intent detection
    print("\n=== Intent Detection ===")
    total += 1
    det = IntentDetector()
    r = det.detect("What is the capital of France?")
    ok = r['intent'] == 'question'
    passed += test("Question intent", ok, f"intent={r['intent']}")

    # 10. Command intent
    total += 1
    r = det.detect("Create a new file called test.py")
    ok = r['intent'] == 'command'
    passed += test("Command intent", ok, f"intent={r['intent']}")

    # 11. Greeting
    total += 1
    r = det.detect("Hello, how are you?")
    ok = 'greeting' in r['all_intents'] or 'question' in r['all_intents']
    passed += test("Greeting detection", ok, f"intents={r['all_intents']}")

    # 12. Urgency
    total += 1
    urg = UrgencyDetector()
    r = urg.detect("URGENT: Server is down, fix immediately!")
    ok = r['level'] in ('critical', 'high')
    passed += test("Urgency detection (critical)", ok, f"level={r['level']}, score={r['score']}")

    # 13. Low urgency
    total += 1
    r = urg.detect("When you get a chance, could you review this?")
    ok = r['level'] in ('normal', 'low')
    passed += test("Urgency detection (low)", ok, f"level={r['level']}, score={r['score']}")

    # === TEXT UTILITIES ===
    print("\n=== Text Utilities ===")
    from core.textutils import (
        word_tokenize, sentence_tokenize, ngrams, stem,
        levenshtein, damerau_levenshtein, jaccard_similarity,
        fuzzy_match, flesch_kincaid, type_token_ratio,
        syllable_count, word_frequency,
    )

    # 14. Word tokenization
    total += 1
    tokens = word_tokenize("I can't believe it's not butter!")
    ok = len(tokens) >= 6
    passed += test("Word tokenization", ok, f"tokens={tokens}")

    # 15. Sentence tokenization
    total += 1
    sents = sentence_tokenize("Dr. Smith went to Washington. He arrived at 3 p.m. It was raining.")
    ok = len(sents) == 3
    passed += test("Sentence tokenization", ok, f"{len(sents)} sentences")

    # 16. N-grams
    total += 1
    bg = ngrams(['the', 'cat', 'sat', 'on', 'the', 'mat'], 2)
    ok = len(bg) == 5 and bg[0] == ('the', 'cat')
    passed += test("Bigram generation", ok, f"{len(bg)} bigrams")

    # 17. Stemming
    total += 1
    stems = [stem(w) for w in ['running', 'flies', 'played', 'happiness', 'studies']]
    ok = stem('running') != 'running' and stem('happiness') != 'happiness'
    passed += test("Stemming", ok, f"stems={stems}")

    # 18. Levenshtein distance
    total += 1
    d = levenshtein('kitten', 'sitting')
    ok = d == 3
    passed += test("Levenshtein distance", ok, f"kitten→sitting={d}")

    # 19. Damerau-Levenshtein
    total += 1
    d1 = damerau_levenshtein('ab', 'ba')  # Transposition = 1
    d2 = levenshtein('ab', 'ba')  # No transposition = 2
    ok = d1 < d2
    passed += test("Damerau-Levenshtein (transposition)", ok, f"DL={d1}, L={d2}")

    # 20. Jaccard similarity
    total += 1
    sim = jaccard_similarity('python', 'pythin')
    ok = sim > 0.3  # 3/7 bigrams shared
    passed += test("Jaccard similarity", ok, f"python~pythin={sim:.3f}")

    # 21. Fuzzy matching
    total += 1
    candidates = ['python', 'javascript', 'typescript', 'rust', 'java']
    matches = fuzzy_match('pythn', candidates, threshold=0.3)
    ok = len(matches) > 0 and matches[0][0] == 'python'
    passed += test("Fuzzy matching", ok, f"matches={matches}")

    # 22. Flesch-Kincaid
    total += 1
    text = "The cat sat on the mat. The dog ran in the park. It was a sunny day."
    fk = flesch_kincaid(text)
    ok = fk['reading_ease'] > 50 and fk['grade_level'] < 10
    passed += test("Flesch-Kincaid readability", ok,
                   f"ease={fk['reading_ease']}, grade={fk['grade_level']}")

    # 23. Type-token ratio
    total += 1
    ttr = type_token_ratio("the cat sat on the mat and the dog sat on the rug")
    ok = 0 < ttr < 1
    passed += test("Type-token ratio", ok, f"TTR={ttr:.3f}")

    # 24. Syllable count
    total += 1
    ok = syllable_count('beautiful') >= 3 and syllable_count('cat') == 1
    passed += test("Syllable counting", ok,
                   f"beautiful={syllable_count('beautiful')}, cat={syllable_count('cat')}")

    # 25. Word frequency
    total += 1
    freq = word_frequency("the cat the dog the bird the cat the cat", top_n=3)
    ok = freq[0][0] == 'the' and freq[1][0] == 'cat'
    passed += test("Word frequency", ok, f"top={freq}")

    # Summary
    print("\n" + "=" * 60)
    pct = (passed / total * 100) if total else 0
    print(f"RESULT: {passed}/{total} ({pct:.0f}%)")
    print("=" * 60)
    return passed, total


if __name__ == '__main__':
    run()
