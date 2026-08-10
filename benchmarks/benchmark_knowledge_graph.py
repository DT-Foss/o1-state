#!/usr/bin/env python3
"""
S2 — Real Knowledge Graph Benchmark (FB15k-237)
=================================================
Tests KnowledgeStore on real Freebase triplets, not synthetic data.

FB15k-237: 272K train, 17.5K valid, 20.5K test
14,505 entities, 237 relations

Tasks:
  1. Store N real triplets, measure time and memory
  2. Retrieval accuracy: can we find stored facts?
  3. Anti-hallucination: reject queries about unstored entities
  4. Link prediction: given (S, R, ?), is the correct O in top-K?
  5. Scale analysis: how does accuracy degrade with N?
"""

import numpy as np
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.knowledge import KnowledgeStore


def load_fb15k237(split='train', max_lines=None):
    """Load FB15k-237 triplets."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'fb15k237', f'{split}.txt')
    triplets = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            parts = line.strip().split('\t')
            if len(parts) == 3:
                triplets.append(tuple(parts))
    return triplets


def test_retrieval(ks, facts, n_queries=200, seed=42):
    """Test retrieval accuracy on stored facts."""
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(facts), min(n_queries, len(facts)), replace=False)

    correct = 0
    high = 0
    times = []

    for idx in indices:
        subj, rel, obj = facts[idx]
        t0 = time.time()
        result = ks.query(subj, rel)
        times.append(time.time() - t0)

        if result['fact'] is not None and result['fact'][2] == obj:
            correct += 1
        if result['confidence_level'] == 'HIGH':
            high += 1

    return {
        'accuracy': correct / len(indices),
        'high_confidence': high / len(indices),
        'avg_ms': np.mean(times) * 1000,
        'p95_ms': np.percentile(times, 95) * 1000,
    }


def test_anti_hallucination(ks, stored_entities, n_queries=100, seed=77):
    """Test rejection of fake entities not in the store."""
    rng = np.random.RandomState(seed)
    rejected = 0

    for i in range(n_queries):
        # Generate fake entity that's NOT in stored set
        fake = f"/m/{rng.randint(9999999):07d}_FAKE"
        while fake in stored_entities:
            fake = f"/m/{rng.randint(9999999):07d}_FAKE"

        rel = "/location/country/form_of_government"  # real relation
        result = ks.query(fake, rel)
        if result['confidence_level'] in ('REJECTED', 'UNKNOWN'):
            rejected += 1

    return rejected / n_queries


def test_link_prediction(ks, test_facts, n_queries=100, seed=55):
    """Link prediction: given (S, R), check if correct O is returned."""
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(test_facts), min(n_queries, len(test_facts)), replace=False)

    hits = 0
    for idx in indices:
        subj, rel, expected_obj = test_facts[idx]
        result = ks.query(subj, rel)
        if result['fact'] is not None and result['fact'][2] == expected_obj:
            hits += 1

    return hits / len(indices)


def main():
    print("=" * 70)
    print("S2 — REAL KNOWLEDGE GRAPH BENCHMARK (FB15k-237)")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════
    # TEST 1: Scale analysis — store increasing N
    # ══════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("[1] Scale Analysis — Store N real Freebase triplets")
    print(f"{'━' * 70}")

    n_values = [500, 1000, 2000]
    all_train = load_fb15k237('train')

    print(f"\n  FB15k-237: {len(all_train)} train triplets available")
    print(f"\n  {'N':>6} {'Store(s)':>8} {'Acc':>8} {'Anti-H':>8} {'Avg ms':>8} {'P95 ms':>8}")
    print(f"  {'─' * 50}")

    best_ks = None
    best_facts = None

    for n in n_values:
        facts = all_train[:n]
        entities = set()
        for s, r, o in facts:
            entities.add(s)
            entities.add(o)

        ks = KnowledgeStore(dim=128)
        t0 = time.time()
        ks.store_facts(facts)
        t_store = time.time() - t0

        ret = test_retrieval(ks, facts, n_queries=min(100, n))
        anti_h = test_anti_hallucination(ks, entities, n_queries=50)

        print(f"  {n:>6} {t_store:>7.1f} {ret['accuracy']:>7.1%} {anti_h:>7.1%} "
              f"{ret['avg_ms']:>7.1f} {ret['p95_ms']:>7.1f}")

        if n == n_values[-1]:
            best_ks = ks
            best_facts = facts

    # ══════════════════════════════════════════════════════════
    # TEST 2: Link prediction on held-out test set
    # ══════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("[2] Link Prediction — Test set queries against stored train")
    print(f"{'━' * 70}")

    if best_ks is not None:
        test_facts = load_fb15k237('test', max_lines=1000)

        # Filter test facts to those whose (S, R) pair exists in train
        train_sr = set((s, r) for s, r, o in best_facts)
        matching_test = [(s, r, o) for s, r, o in test_facts if (s, r) in train_sr]

        print(f"\n  Test facts with matching (S,R) in train: {len(matching_test)}/{len(test_facts)}")

        if matching_test:
            hits = test_link_prediction(best_ks, matching_test,
                                        n_queries=min(100, len(matching_test)))
            print(f"  Link prediction accuracy (exact match): {hits:.1%}")
            print(f"  Note: FB15k-237 has many-to-many relations,")
            print(f"  so exact match is the HARDEST metric.")
        else:
            print("  No matching (S,R) pairs found in test subset.")

    # ══════════════════════════════════════════════════════════
    # TEST 3: Relation-specific analysis
    # ══════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("[3] Relation-Specific Accuracy")
    print(f"{'━' * 70}")

    if best_ks is not None:
        # Group facts by relation
        rel_groups = {}
        for s, r, o in best_facts:
            rel_groups.setdefault(r, []).append((s, r, o))

        # Test top-5 most common relations
        top_rels = sorted(rel_groups.items(), key=lambda x: -len(x[1]))[:8]

        print(f"\n  {'Relation':<55} {'N':>5} {'Acc':>6}")
        print(f"  {'─' * 70}")

        for rel, facts_for_rel in top_rels:
            ret = test_retrieval(best_ks, facts_for_rel,
                                n_queries=min(20, len(facts_for_rel)))
            rel_short = rel[-50:] if len(rel) > 50 else rel
            print(f"  {rel_short:<55} {len(facts_for_rel):>5} {ret['accuracy']:>5.0%}")

    # ══════════════════════════════════════════════════════════
    # TEST 4: Dimension scaling on real data
    # ══════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("[4] Dimension Scaling — N=2000 real triplets")
    print(f"{'━' * 70}")

    facts_2k = all_train[:2000]
    entities_2k = set()
    for s, r, o in facts_2k:
        entities_2k.add(s)
        entities_2k.add(o)

    print(f"\n  {'Dim':>6} {'Acc':>8} {'Anti-H':>8} {'Avg ms':>8}")
    print(f"  {'─' * 35}")

    for dim in [64, 128, 256]:
        ks = KnowledgeStore(dim=dim)
        ks.store_facts(facts_2k)
        ret = test_retrieval(ks, facts_2k, n_queries=100)
        anti_h = test_anti_hallucination(ks, entities_2k, n_queries=50)
        print(f"  {dim:>6} {ret['accuracy']:>7.1%} {anti_h:>7.1%} {ret['avg_ms']:>7.1f}")
        del ks

    # ══════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("S2 SUMMARY — REAL KNOWLEDGE GRAPH PERFORMANCE")
    print(f"{'═' * 70}")
    print(f"""
  FB15k-237 is the standard benchmark for Knowledge Graph Completion.
  Unlike synthetic data, entities are Freebase MIDs (/m/XXXXXX) and
  relations are hierarchical paths — a harder test for our n-gram encoder.

  Key questions answered:
    1. Does retrieval accuracy hold on real (non-synthetic) triplets?
    2. Does anti-hallucination still work with Freebase entity IDs?
    3. How does dimension affect accuracy on real data?
    4. Can we do link prediction (test set queries)?
""")


if __name__ == "__main__":
    main()
