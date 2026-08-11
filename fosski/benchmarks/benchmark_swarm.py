"""
Benchmark: Swarm Architecture (S14-S18)
=========================================
Tests:
  1. Brain Snapshot: save/load/clone fidelity
  2. Swarm Spawn: agent creation and registration
  3. Cross-Domain Query: agents help each other
  4. Federated Merge: brain combination
  5. Superposition Retrieval: ambiguity resolution
  6. Anti-Hallucination: survives all operations
"""

import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.knowledge import KnowledgeStore
from core.language import PPMModel
from core.brain import BrainSnapshot, merge_brains
from core.swarm import SwarmAgent, MetaAgent, SuperpositionRetrieval


def test_brain_snapshot():
    """Test 1: Brain save/load/clone fidelity."""
    print("Test 1: Brain Snapshot")

    ks = KnowledgeStore(dim=64)
    facts = [('Germany','capital','Berlin'), ('France','capital','Paris'),
             ('Japan','capital','Tokyo'), ('Water','formula','H2O')]
    for s, r, o in facts:
        ks.store_fact(s, r, o)

    ppm = PPMModel(max_order=5)
    ppm.train(list("Germany is a country. Its capital is Berlin."))

    # Capture
    brain = BrainSnapshot.capture_from_parts(
        knowledge_store=ks, ppm_model=ppm,
        domain='test', agent_id='test-001'
    )

    # Save compressed
    path = '/tmp/bench_brain.brain'
    brain.save(path)
    size = os.path.getsize(path)

    # Load
    loaded = BrainSnapshot.load(path)

    # Verify
    assert brain.fingerprint() == loaded.fingerprint(), "Fingerprint mismatch!"
    assert loaded.n_facts == len(facts), f"Fact count: {loaded.n_facts} != {len(facts)}"
    assert loaded.domain == 'test', f"Domain: {loaded.domain}"

    # Restore and query
    parts = loaded.restore_parts()
    ks2 = parts['knowledge']
    for s, r, _ in facts:
        result = ks2.query(s, r)
        assert result['confidence_level'] == 'HIGH', f"Query {s},{r} = {result['confidence_level']}"

    # Anti-H survives
    narnia = ks2.query('Narnia', 'capital')
    assert narnia['confidence_level'] == 'REJECTED', f"Narnia: {narnia['confidence_level']}"

    # PPM clone
    ppm2 = parts['language']
    ctx = list("Germany is")
    p1 = ppm.predict(ctx)
    p2 = ppm2.predict(ctx)
    assert p1 == p2, "PPM predictions differ!"

    os.unlink(path)
    print(f"  PASS: {len(facts)} facts, {size} bytes, fingerprints match, PPM identical")
    return True


def test_swarm_local():
    """Test 2: Swarm spawn and local queries."""
    print("Test 2: Swarm Local Queries")

    geo = SwarmAgent('geo', domain='geography', port=8001)
    for s, r, o in [('Germany','capital','Berlin'), ('France','capital','Paris')]:
        geo.knowledge.store_fact(s, r, o)

    results = []
    for q, expected in [
        ('What is the capital of Germany?', 'Berlin'),
        ('What is the capital of France?', 'Paris'),
    ]:
        r = geo.query(q)
        ok = r['answer'] == expected
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: {q} → {r['answer']}")

    # Anti-H
    r = geo.query('What is the capital of Narnia?')
    anti_h = 'information' in str(r['answer']).lower() or r['confidence_level'] == 'REJECTED'
    results.append(anti_h)
    print(f"  {'PASS' if anti_h else 'FAIL'}: Narnia → {r['confidence_level']}")

    return all(results)


def test_cross_domain():
    """Test 3: Cross-domain swarm queries over UDP."""
    print("Test 3: Cross-Domain Swarm Query")

    meta = MetaAgent(port=8100)
    geo = SwarmAgent('geo', domain='geography', port=8101, meta_addr=('127.0.0.1', 8100))
    sci = SwarmAgent('sci', domain='science', port=8102, meta_addr=('127.0.0.1', 8100))

    for s, r, o in [('Germany','capital','Berlin'), ('Japan','capital','Tokyo')]:
        geo.knowledge.store_fact(s, r, o)
    for s, r, o in [('Water','formula','H2O'), ('Einstein','birthplace','Ulm')]:
        sci.knowledge.store_fact(s, r, o)

    meta.start()
    geo.start()
    sci.start()
    time.sleep(0.5)

    results = []

    # Geo asks Sci question
    r = geo.swarm_query('What is the formula of Water?', timeout=1.5)
    ok = r.get('answer') == 'H2O'
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: Geo asks Water → {r.get('answer')} [{r.get('source','?')}]")

    # Sci asks Geo question
    r = sci.swarm_query('What is the capital of Japan?', timeout=1.5)
    ok = r.get('answer') == 'Tokyo'
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: Sci asks Japan → {r.get('answer')} [{r.get('source','?')}]")

    # Nobody knows Atlantis
    r = geo.swarm_query('What is the capital of Atlantis?', timeout=1.0)
    ok = r['confidence_level'] in ('REJECTED', 'UNKNOWN')
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: Atlantis → {r['confidence_level']}")

    meta.stop(); geo.stop(); sci.stop()
    return all(results)


def test_brain_merge():
    """Test 4: Federated brain merge."""
    print("Test 4: Federated Brain Merge")

    ks1 = KnowledgeStore(dim=64)
    for s, r, o in [('Germany','capital','Berlin'), ('France','capital','Paris')]:
        ks1.store_fact(s, r, o)

    ks2 = KnowledgeStore(dim=64)
    for s, r, o in [('Water','formula','H2O'), ('Germany','capital','Berlin')]:  # Overlap!
        ks2.store_fact(s, r, o)

    brain1 = BrainSnapshot.capture_from_parts(knowledge_store=ks1, domain='geo')
    brain2 = BrainSnapshot.capture_from_parts(knowledge_store=ks2, domain='sci')

    # Union merge
    merged = merge_brains([brain1, brain2], strategy='union')
    parts = merged.restore_parts()
    ks_m = parts['knowledge']

    results = []

    # Dedup: Germany/capital/Berlin appears in both → should be 3 unique
    ok = merged.n_facts == 3
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: Dedup: 2+2(1 overlap) → {merged.n_facts} facts")

    # All facts queryable
    for s, r, expected in [('Germany','capital','Berlin'), ('France','capital','Paris'), ('Water','formula','H2O')]:
        result = ks_m.query(s, r)
        ok = result['fact'] and result['fact'][2] == expected
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: {s}/{r} → {result['fact'][2] if result['fact'] else '?'}")

    # Anti-H survives merge
    r = ks_m.query('Narnia', 'capital')
    ok = r['confidence_level'] == 'REJECTED'
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: Anti-H post-merge: {r['confidence_level']}")

    return all(results)


def test_superposition():
    """Test 5: Superposition retrieval with context collapse."""
    print("Test 5: Superposition Retrieval")

    ks = KnowledgeStore(dim=64)
    for s, r, o in [
        ('Bank', 'type', 'financial institution'),
        ('Bank', 'type', 'park bench'),
        ('Bank', 'type', 'river bank'),
    ]:
        ks.store_fact(s, r, o)

    sp = SuperpositionRetrieval(ks, top_k=3)
    results = []

    # Initial superposition should be ambiguous
    sup = sp.query_superposition('Bank', 'type')
    ok = len(sup['matches']) == 3
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: 3 matches in superposition")

    # Entropy should be high (near-uniform)
    ok = sup['entropy'] > 1.0
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: High entropy: {sup['entropy']:.2f}")

    # Context collapse tests
    test_cases = [
        ('I need to deposit money', 'financial institution'),
        ('sitting on a bench in the park', 'park bench'),
        ('along the river bank', 'river bank'),
    ]

    for context, expected in test_cases:
        sp.reset()
        sp.query_superposition('Bank', 'type')
        result = sp.collapse(context)
        match_obj = result['match'][2] if result['match'] else '?'
        ok = match_obj == expected
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: \"{context}\" → {match_obj} (expected: {expected})")

    return all(results)


def test_meta_routing():
    """Test 6: Meta-agent domain routing."""
    print("Test 6: Meta-Agent Routing")

    meta = MetaAgent(port=8200)
    results = []

    cases = [
        ('What is the capital of Germany?', 'geography'),
        ('What is the formula of water?', 'science'),
        ('How do I define a class in Python?', 'code'),
        ('What are the symptoms of flu?', 'medicine'),
        ('Is theft a crime?', 'law'),
    ]

    for query, expected_domain in cases:
        domains = meta.route_to_specialist(query)
        ok = expected_domain in domains
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: \"{query[:40]}\" → {domains} (expected: {expected_domain})")

    return all(results)


if __name__ == '__main__':
    print("=" * 60)
    print("SWARM ARCHITECTURE BENCHMARK (S14-S18)")
    print("=" * 60)
    print()

    tests = [
        test_brain_snapshot,
        test_swarm_local,
        test_cross_domain,
        test_brain_merge,
        test_superposition,
        test_meta_routing,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            ok = test()
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
        print()

    print("=" * 60)
    total = passed + failed
    print(f"RESULTS: {passed}/{total} tests passed")
    if failed == 0:
        print("ALL TESTS PASSED")
    print("=" * 60)
