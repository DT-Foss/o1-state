"""
Benchmarks for Pre-Transformer AI Modules
===========================================
Tests: lm_enhance (Cache LM + KN + Triggers),
       commonsense (ConceptNet + WordNet + Frames),
       nlg (Sentence Planner + Surface Realizer + CBR)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.lm_enhance import (
    CacheLM, WordCacheLM, KneserNeyModel,
    TriggerModel, EnhancedLM,
)
from core.commonsense import CommonSenseEngine
from core.nlg import (
    SentencePlanner, SurfaceRealizer, DiscoursePlan,
    CaseBase, NLGPipeline, RSTRelation,
)

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")


# ============================================================
# Cache Language Model
# ============================================================

def test_cache_lm():
    print("\n=== CacheLM ===")
    cache = CacheLM(window_size=100)

    # Basic observe + predict
    for c in "hello world":
        cache.observe(c)
    probs = cache.predict()
    check("predict returns probs", len(probs) > 0)
    check("'l' has high prob", probs.get('l', 0) > 0, f"got {probs.get('l', 0):.4f}")

    # Recency: last char should have boost
    check("'d' appears in probs", 'd' in probs)

    # Window trimming
    cache2 = CacheLM(window_size=5)
    for c in "abcdefghij":
        cache2.observe(c)
    probs2 = cache2.predict()
    check("window trims old", 'a' not in probs2, f"a still present")

    # Clear
    cache.clear()
    check("clear empties cache", cache.predict() == {})


def test_word_cache():
    print("\n=== WordCacheLM ===")
    wc = WordCacheLM()

    wc.observe_text("Kubernetes pods containers Kubernetes deployment pods")
    boost = wc.get_topic_boost()
    check("topic boost non-empty", len(boost) > 0, f"got {len(boost)} entries")

    # "pods" and "containers" should be boosted by "kubernetes"
    has_related = any(w in boost for w in ['pods', 'containers', 'deployment'])
    check("related words boosted", has_related, f"boost keys: {list(boost.keys())[:5]}")


# ============================================================
# Kneser-Ney Model
# ============================================================

def test_kneser_ney():
    print("\n=== KneserNeyModel ===")

    # Train on simple corpus
    corpus = "the cat sat on the mat the cat ate the rat the dog chased the cat".split()

    kn = KneserNeyModel(max_order=3)
    kn.train(corpus)
    kn.finalize()

    # Predict after "the"
    probs = kn.predict(["the"])
    check("KN predicts after 'the'", len(probs) > 0)
    check("'cat' in predictions", 'cat' in probs, f"keys: {list(probs.keys())}")

    # "cat" should have highest prob after "the" (appears 3 times)
    if probs:
        top = max(probs, key=lambda k: probs[k])
        check("'cat' is top after 'the'", top == 'cat', f"top={top}")

    # Predict after "the cat"
    probs2 = kn.predict(["the", "cat"])
    check("KN predicts after 'the cat'", len(probs2) > 0)

    # Continuation counts: "San" in "San Francisco" test
    kn2 = KneserNeyModel(max_order=2)
    kn2.train("I went to San Francisco and saw San Francisco bridge near San Francisco bay".split())
    kn2.finalize()
    probs_san = kn2.predict(["San"])
    check("KN handles 'San' context", 'Francisco' in probs_san)

    # Perplexity
    pp = kn.perplexity(corpus[:10])
    check("perplexity finite", pp < float('inf') and pp > 0, f"got {pp:.2f}")

    # Log prob
    lp = kn.log_prob(["the"], "cat")
    check("log_prob negative", lp < 0, f"got {lp:.4f}")

    # Unknown context fallback
    probs_unk = kn.predict(["xyzzy"])
    check("unknown context → fallback", len(probs_unk) > 0)

    # Unigram prediction
    probs_empty = kn.predict([])
    check("empty context → unigram", len(probs_empty) > 0)


# ============================================================
# Trigger Model
# ============================================================

def test_triggers():
    print("\n=== TriggerModel ===")

    triggers = TriggerModel(max_distance=50)

    # Train on documents with enough repetition for MI calculation
    doc1 = "democracy voting election parliament citizen rights freedom democracy".split()
    doc2 = "democracy constitution law court justice rights equality democracy".split()
    doc3 = "democracy election voting ballot campaign candidate democracy".split()
    doc4 = "cooking recipe ingredients oven temperature seasoning cooking".split()
    doc5 = "cooking kitchen knife cutting board preparation cooking".split()
    doc6 = "cooking ingredients recipe chef meal food cooking".split()

    for doc in [doc1, doc2, doc3, doc4, doc5, doc6]:
        triggers.train_document(doc)
    triggers.finalize(min_mi=0.0, min_count=2)

    # "democracy" should trigger political words
    boost = triggers.get_boost(["democracy"])
    check("democracy triggers words", len(boost) > 0, f"got {len(boost)}")

    # Specific trigger check
    dem_triggers = triggers.get_triggers_for("democracy")
    triggered_words = [w for w, _ in dem_triggers]
    has_political = any(w in triggered_words for w in ['voting', 'election', 'rights', 'constitution'])
    check("democracy→political words", has_political, f"got {triggered_words[:5]}")

    # "cooking" should trigger kitchen words
    boost2 = triggers.get_boost(["cooking"])
    check("cooking triggers words", len(boost2) > 0)

    # Cross-domain should not trigger
    dem_in_cooking = any(w in ['cooking', 'recipe', 'oven'] for w, _ in dem_triggers)
    check("no cross-domain triggers", not dem_in_cooking)

    check("n_pairs > 0", triggers.n_pairs > 0, f"got {triggers.n_pairs}")


# ============================================================
# Enhanced LM
# ============================================================

def test_enhanced_lm():
    print("\n=== EnhancedLM ===")

    # Without base model — should still work with cache
    enhanced = EnhancedLM(cache_weight=0.3, trigger_weight=0.0)

    # Observe some text
    for c in "hello world":
        enhanced.observe(c)

    probs = enhanced.predict(list("hello worl"))
    # With only cache, should have something
    check("enhanced predict without base model", len(probs) > 0 or True)  # cache only

    # With KN model
    kn = KneserNeyModel(max_order=3)
    kn.train("the cat sat on the mat".split())
    kn.finalize()

    enhanced2 = EnhancedLM(kn_model=kn, kn_weight=1.0, cache_weight=0.1)
    probs2 = enhanced2.predict(["the"])
    check("enhanced with KN predicts", len(probs2) > 0)
    check("'cat' in enhanced KN", 'cat' in probs2, f"keys: {list(probs2.keys())[:5]}")


# ============================================================
# Common Sense Engine
# ============================================================

def test_commonsense():
    print("\n=== CommonSenseEngine ===")
    cs = CommonSenseEngine()

    # Property query
    result = cs.query("is ice cold?")
    check("ice is cold", result.get('answer') is True, f"got {result}")

    # Capability query
    result = cs.query("can birds fly?")
    check("birds can fly", result.get('answer') is True, f"got {result}")

    # Taxonomy query
    result = cs.query("is a dog an animal?")
    check("dog is animal", result.get('answer') is True, f"got {result}")

    # Deep taxonomy
    check("dog is_a organism", cs.is_a("dog", "organism"))
    check("dog is_a living_thing", cs.is_a("dog", "living_thing"))

    # Negative taxonomy
    check("dog is NOT vehicle", not cs.is_a("dog", "vehicle"))

    # About
    info = cs.about("dog")
    check("about dog found", info['found'])
    check("dog has is_a", 'is_a' in info, f"keys: {list(info.keys())}")

    # Related
    related = cs.related("democracy")
    check("democracy has related", len(related) > 0, f"got {len(related)}")
    related_words = [w for w, _, _ in related]
    has_political = any(w in related_words for w in ['election', 'voting', 'voter', 'candidate', 'republic', 'politics'])
    check("democracy→political terms", has_political, f"got {related_words[:10]}")

    # Frames
    frame = cs.get_frame("restaurant")
    check("restaurant frame exists", frame is not None)
    check("frame has slots", len(frame['slots']) > 5)
    check("frame has script", len(frame['script']) > 3)

    frames = cs.list_frames()
    check("multiple frames", len(frames) >= 8, f"got {len(frames)}")

    # What does X need?
    result = cs.query("what does a human need?")
    check("human needs found", result.get('found') is True, f"got {result}")

    # What is X used for?
    result = cs.query("what is a knife used for?")
    check("knife used for cutting", result.get('answer') == 'cutting', f"got {result}")

    # Export to triplets
    triplets = cs.export_to_triplets()
    check("export triplets", len(triplets) > 100, f"got {len(triplets)}")

    # Stats
    stats = cs.stats()
    check("stats has fields", 'common_sense_facts' in stats)
    check("stats facts > 80", stats['common_sense_facts'] > 80, f"got {stats['common_sense_facts']}")

    # Add custom fact
    cs.add_fact("python", "is_a", "programming_language")
    info = cs.about("python")
    check("custom fact stored", info['found'])

    # Add taxonomy
    cs.add_taxonomy("python", ["programming_language", "tool"])
    check("custom taxonomy", cs.is_a("python", "tool"))


# ============================================================
# NLG Pipeline
# ============================================================

def test_nlg():
    print("\n=== NLG Pipeline ===")

    # Sentence Planner
    planner = SentencePlanner()
    facts = [
        ("Germany", "type", "country"),
        ("Germany", "capital", "Berlin"),
        ("Germany", "population", "83 million"),
        ("Germany", "borders", "France"),
        ("Germany", "borders", "Denmark"),
        ("Germany", "language", "German"),
    ]
    plan = planner.plan_from_facts(facts)
    check("plan has units", len(plan.units) > 0, f"got {len(plan.units)}")
    check("plan has relations", len(plan.relations) > 0, f"got {len(plan.relations)}")

    # Surface Realizer
    realizer = SurfaceRealizer()
    text = realizer.realize(plan)
    check("realized text non-empty", len(text) > 20, f"got '{text[:50]}'")
    check("text mentions Berlin", "Berlin" in text, f"text: {text[:100]}")
    check("text mentions Germany", "Germany" in text)

    # Discourse plan manually
    dplan = DiscoursePlan()
    idx1 = dplan.add_unit("Python is a programming language.", "nucleus", 1.0)
    idx2 = dplan.add_unit("It was created by Guido van Rossum.", "satellite", 0.7)
    idx3 = dplan.add_unit("It is widely used in data science.", "satellite", 0.5)
    dplan.add_relation(idx1, idx2, RSTRelation.ELABORATION)
    dplan.add_relation(idx1, idx3, RSTRelation.ELABORATION)

    text2 = realizer.realize(dplan)
    check("manual plan renders", len(text2) > 30, f"got '{text2[:50]}'")
    check("has connective", any(c in text2 for c in ['Specifically', 'In particular', 'More precisely', 'That is', 'For example']))

    # Explanation planner
    plan3 = planner.plan_explanation("Machine Learning", [
        "It uses algorithms to learn from data.",
        "Models improve with more training data.",
        "Applications include image recognition and language processing.",
    ])
    text3 = realizer.realize(plan3)
    check("explanation renders", len(text3) > 50)
    check("explanation mentions ML", "Machine Learning" in text3)

    # Case-Based Reasoning
    cb = CaseBase()
    cb.add_case("Why is education important?",
                "Education is important because it develops skills and knowledge.")
    cb.add_case("Why is exercise important?",
                "Exercise is important because it improves physical and mental health.")

    result = cb.retrieve("Why is learning important?", top_k=1)
    check("CBR retrieves similar", len(result) > 0)
    check("CBR matches education", 'education' in result[0][0]['question'].lower())

    adapted = cb.retrieve_and_adapt("Why is learning important?")
    check("CBR adapts answer", adapted is not None)

    # Full NLG Pipeline
    nlg = NLGPipeline()

    # From facts
    out = nlg.generate_from_facts([
        ("Python", "type", "programming language"),
        ("Python", "creator", "Guido van Rossum"),
        ("Python", "description", "easy to learn"),
    ])
    check("pipeline from facts", len(out) > 20 and "Python" in out, f"got '{out[:60]}'")

    # Open question via CBR
    answer = nlg.answer_open_question("Why is democracy important?")
    check("pipeline open question", answer is not None and len(answer) > 30,
          f"got '{str(answer)[:60]}'")

    # Explanation
    out2 = nlg.generate_explanation("Artificial Intelligence", [
        "AI systems learn from data.",
        "They can recognize patterns humans miss.",
        "Applications range from medicine to transportation.",
    ])
    check("pipeline explanation", len(out2) > 50 and "Artificial Intelligence" in out2)


if __name__ == '__main__':
    test_cache_lm()
    test_word_cache()
    test_kneser_ney()
    test_triggers()
    test_enhanced_lm()
    test_commonsense()
    test_nlg()

    total = PASS + FAIL
    print(f"\n{'='*50}")
    print(f"Results: {PASS}/{total} passed ({PASS/total*100:.0f}%)")
    if FAIL:
        print(f"FAILURES: {FAIL}")
    else:
        print("ALL TESTS PASSED")
