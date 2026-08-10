#!/usr/bin/env python3
"""
T91 — Hopfield Knowledge Store Scaling Test
=============================================
Tests: How does retrieval accuracy and speed scale with
number of stored facts × embedding dimension?

Current baseline: 15 facts, 128d, 100% accuracy, ~4ms.
Target: 100K facts at >90% accuracy.

Approach:
1. Generate synthetic (S, R, O) triplets with controlled similarity
2. Store 100, 1K, 10K, 50K, 100K facts
3. Measure retrieval accuracy and time per query
4. Test across dimensions: 128, 256, 512, 1024
5. Find the dimension needed for 100K at >90%

Theory: Modern Hopfield capacity scales as exp(d/2) patterns.
For d=128: exp(64) >> any practical N.
But PhraseEncoder is NOT a random embedding — similar phrases
produce similar vectors, so effective capacity is lower.
"""

import numpy as np
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.knowledge import KnowledgeStore, PhraseEncoder, ModernHopfield


# ── Synthetic data generation ──

SUBJECTS = [
    "Country", "City", "Element", "Animal", "Planet", "River",
    "Mountain", "Ocean", "Language", "Currency", "Company", "Person",
    "Food", "Sport", "Color", "Metal", "Mineral", "Gas",
    "Flower", "Tree", "Bird", "Fish", "Insect", "Star",
]

RELATIONS = [
    "capital", "population", "area", "symbol", "formula",
    "color", "weight", "height", "speed", "temperature",
    "creator", "founded", "located_in", "part_of", "type",
]

def generate_facts(n, seed=42):
    """Generate n unique synthetic (S, R, O) triplets."""
    rng = np.random.RandomState(seed)
    facts = set()
    i = 0
    while len(facts) < n:
        # Generate subject: Category_XXXX
        cat = SUBJECTS[rng.randint(len(SUBJECTS))]
        subj = f"{cat}_{i:06d}"
        rel = RELATIONS[rng.randint(len(RELATIONS))]
        # Object: semi-random word
        obj = f"Value_{rng.randint(1000000):06d}"
        facts.add((subj, rel, obj))
        i += 1
    return list(facts)


def test_retrieval(ks, facts, n_queries=200, seed=99):
    """Test retrieval accuracy on random subset of stored facts."""
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(facts), min(n_queries, len(facts)), replace=False)

    correct = 0
    times = []

    for idx in indices:
        subj, rel, expected_obj = facts[idx]
        t0 = time.time()
        result = ks.query(subj, rel)
        t_q = time.time() - t0
        times.append(t_q)

        if result['fact'] is not None:
            if result['fact'][2] == expected_obj:
                correct += 1

    accuracy = correct / len(indices)
    avg_time = np.mean(times) * 1000  # ms
    p95_time = np.percentile(times, 95) * 1000
    return accuracy, avg_time, p95_time


def test_anti_hallucination(ks, n_queries=50, seed=77):
    """Test that random queries are rejected."""
    rng = np.random.RandomState(seed)
    rejected = 0

    for i in range(n_queries):
        fake_subj = f"FakeEntity_{rng.randint(9999999):07d}"
        result = ks.query(fake_subj, "capital")
        if result['confidence_level'] in ('REJECTED', 'UNKNOWN'):
            rejected += 1
        elif result['confidence'] < 0.55:
            rejected += 1  # Low confidence = effectively rejected

    return rejected / n_queries


def main():
    print("=" * 70)
    print("T91 — HOPFIELD KNOWLEDGE STORE SCALING")
    print("=" * 70)

    # ── Test 1: Scaling with number of facts at fixed dim ──
    print(f"\n{'━' * 70}")
    print("[1] Scaling: N facts × fixed dim=128")
    print(f"{'━' * 70}")

    n_values = [100, 500, 1000, 2000]
    dim = 128

    print(f"\n  {'N facts':>8} {'Accuracy':>10} {'Avg ms':>8} {'P95 ms':>8} {'Anti-H':>8} {'Store s':>8}")
    print(f"  {'─' * 56}")

    for n in n_values:
        facts = generate_facts(n)

        t0 = time.time()
        ks = KnowledgeStore(dim=dim)
        ks.store_facts(facts)
        t_store = time.time() - t0

        acc, avg_ms, p95_ms = test_retrieval(ks, facts, n_queries=min(50, n))
        anti_h = test_anti_hallucination(ks, n_queries=20)

        print(f"  {n:>8} {acc:>9.1%} {avg_ms:>7.2f} {p95_ms:>7.2f} {anti_h:>7.1%} {t_store:>7.1f}")

        # Memory cleanup
        del ks

    # ── Test 2: Scaling with dimension at fixed N ──
    print(f"\n{'━' * 70}")
    print("[2] Scaling: fixed N=1000 × dim")
    print(f"{'━' * 70}")

    dims = [64, 128, 256, 512]
    n_fixed = 1000
    facts = generate_facts(n_fixed)

    print(f"\n  {'Dim':>6} {'Accuracy':>10} {'Avg ms':>8} {'P95 ms':>8} {'Anti-H':>8} {'Memory MB':>10}")
    print(f"  {'─' * 56}")

    for d in dims:
        ks = KnowledgeStore(dim=d)
        ks.store_facts(facts)

        acc, avg_ms, p95_ms = test_retrieval(ks, facts)
        anti_h = test_anti_hallucination(ks)

        # Estimate memory: pattern matrix is (dim*3, n_facts) float64
        mem_mb = n_fixed * d * 3 * 8 / (1024 * 1024)

        print(f"  {d:>6} {acc:>9.1%} {avg_ms:>7.2f} {p95_ms:>7.2f} {anti_h:>7.1%} {mem_mb:>9.1f}")

        del ks

    # ── Test 3: Large scale with higher dim ──
    print(f"\n{'━' * 70}")
    print("[3] Large-Scale: Finding the sweet spot")
    print(f"{'━' * 70}")

    configs = [
        (1000, 128),
        (1000, 256),
        (2000, 128),
        (2000, 256),
    ]

    print(f"\n  {'N':>6} {'Dim':>5} {'Accuracy':>10} {'Avg ms':>8} {'Anti-H':>8} {'Mem MB':>8}")
    print(f"  {'─' * 50}")

    for n, d in configs:
        facts = generate_facts(n)
        ks = KnowledgeStore(dim=d)

        t0 = time.time()
        ks.store_facts(facts)
        t_store = time.time() - t0

        acc, avg_ms, p95_ms = test_retrieval(ks, facts, n_queries=min(50, n))
        anti_h = test_anti_hallucination(ks, n_queries=20)
        mem_mb = n * d * 3 * 8 / (1024 * 1024)

        print(f"  {n:>6} {d:>5} {acc:>9.1%} {avg_ms:>7.2f} {anti_h:>7.1%} {mem_mb:>7.1f}")
        del ks

    # ── Test 4: Similarity confusion test ──
    print(f"\n{'━' * 70}")
    print("[4] Similarity Confusion (hardest case)")
    print(f"{'━' * 70}")

    # Store facts with SIMILAR subjects
    ks = KnowledgeStore(dim=128)
    confusing_facts = []
    for i in range(100):
        # All subjects are "Country_XXX" — very similar encoding
        confusing_facts.append((f"Country_{i:03d}", "capital", f"Capital_{i:03d}"))
    ks.store_facts(confusing_facts)

    acc, avg_ms, _ = test_retrieval(ks, confusing_facts, n_queries=100)
    print(f"\n  100 similar subjects (Country_000..099):")
    print(f"  Accuracy: {acc:.1%}, Avg time: {avg_ms:.2f}ms")

    # Now with 500 similar
    ks2 = KnowledgeStore(dim=128)
    confusing_facts2 = []
    for i in range(500):
        confusing_facts2.append((f"Country_{i:04d}", "capital", f"Capital_{i:04d}"))
    ks2.store_facts(confusing_facts2)

    acc2, avg_ms2, _ = test_retrieval(ks2, confusing_facts2, n_queries=200)
    print(f"  500 similar subjects (Country_0000..0499):")
    print(f"  Accuracy: {acc2:.1%}, Avg time: {avg_ms2:.2f}ms")

    # Same with higher dim
    ks3 = KnowledgeStore(dim=256)
    ks3.store_facts(confusing_facts2)
    acc3, avg_ms3, _ = test_retrieval(ks3, confusing_facts2, n_queries=200)
    print(f"  500 similar subjects, dim=256:")
    print(f"  Accuracy: {acc3:.1%}, Avg time: {avg_ms3:.2f}ms")

    # ── Test 5: Query time scaling ──
    print(f"\n{'━' * 70}")
    print("[5] Query Time Scaling")
    print(f"{'━' * 70}")

    print(f"\n  Query time = O(N × d) — linear in both.")
    print(f"  At 100K facts × 512d:")
    n_est = 100000
    d_est = 512
    # Extrapolate from measured data
    # Time ∝ N × d, baseline from N=1000, d=128
    mem_est = n_est * d_est * 3 * 8 / (1024 * 1024)
    print(f"  Memory estimate: {mem_est:.0f} MB")
    print(f"  This is dense matrix × dense vector — no index structure.")
    print(f"  For >10K facts, need approximate nearest neighbor (FAISS/Annoy)")
    print(f"  or hierarchical Hopfield (multi-layer retrieval).")

    # ── Summary ──
    print(f"\n{'═' * 70}")
    print("T91 SUMMARY")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
