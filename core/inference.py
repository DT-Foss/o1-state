"""
3-Pass Inference Engine — Ported from FORGE KnowledgeEngine
=============================================================
Amplifies explicit knowledge by inferring new connections:

Pass 1: Exact Chaining
    If (A, R1, B) and (B, R2, C) exist, infer (A, chains_to, C)
    Example: (Python, creator, Guido) + (Guido, nationality, Dutch)
           → (Python, created_by_national, Dutch)

Pass 2: Semantic Direction Propagation
    Classify relations as positive/negative/neutral, chain directions.
    positive + positive = positive (confidence * gamma)

Pass 3: Fuzzy Entity Bridging
    Entities with similar names share knowledge.
    "United Kingdom" ~ "UK" → bridge facts between them.
    Uses Jaro-Winkler similarity (no external deps, stdlib fallback).

Adapted from FORGE's .causal inference (which uses trigger/mechanism/outcome)
to work with FOSS-KI's (subject, relation, object) Dict-Index.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# Relation direction classification (adapted from FORGE's POSITIVE/NEGATIVE_MECHANISMS)
POSITIVE_RELATIONS = {
    'capital', 'language', 'currency', 'creator', 'founder', 'author',
    'type', 'location', 'country', 'continent', 'population', 'founded',
    'born', 'nationality', 'occupation', 'industry', 'known_as',
    'part_of', 'contains', 'uses', 'produces', 'enables',
}

NEGATIVE_RELATIONS = {
    'not', 'lacks', 'prevents', 'conflicts_with', 'opposite',
    'antonym', 'excludes', 'blocks',
}

# Direction chaining rules (from FORGE)
DIRECTION_CHAINS = {
    ('positive', 'positive'): ('positive', 0.80),
    ('negative', 'negative'): ('positive', 0.75),
    ('positive', 'negative'): ('negative', 0.75),
    ('negative', 'positive'): ('negative', 0.75),
    ('neutral', 'neutral'): ('neutral', 0.70),
}


def _classify_relation(relation):
    """Classify relation as positive, negative, or neutral."""
    r_low = relation.lower()
    if r_low in POSITIVE_RELATIONS:
        return 'positive'
    if r_low in NEGATIVE_RELATIONS:
        return 'negative'
    return 'neutral'


def _jaro_winkler(s1, s2):
    """Jaro-Winkler similarity — pure Python, no external deps."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_dist = max(len1, len2) // 2 - 1
    if match_dist < 0:
        match_dist = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 +
            (matches - transpositions / 2) / matches) / 3

    # Winkler bonus for common prefix (up to 4 chars)
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1 - jaro)


class InferenceEngine:
    """
    3-Pass Inference over FOSS-KI's KnowledgeStore.

    Works directly with the Dict-Index for O(1) lookups.
    Inferred facts are stored back into the KnowledgeStore
    with is_inferred=True marker (via metadata).
    """

    def __init__(self, knowledge_store, max_pass1=5000, max_pass2=3000,
                 max_pass3=2000, min_confidence=0.30, fuzzy_threshold=0.93):
        self.knowledge = knowledge_store
        self.max_pass1 = max_pass1
        self.max_pass2 = max_pass2
        self.max_pass3 = max_pass3
        self.min_confidence = min_confidence
        self.fuzzy_threshold = fuzzy_threshold
        self.stats = {'pass1': 0, 'pass2': 0, 'pass3': 0}

    def run(self, store_inferred=True):
        """
        Run full 3-pass inference. Returns list of inferred (S, R, O) triplets.
        If store_inferred=True, also stores them in the KnowledgeStore.
        """
        seen = set(self.knowledge._fact_keys)
        all_inferred = []

        # Pass 1: Exact chaining
        p1 = self._pass1_exact_chain(seen)
        all_inferred.extend(p1)
        for s, r, o in p1:
            seen.add((s.lower(), r.lower(), o.lower()))
        self.stats['pass1'] = len(p1)

        # Pass 2: Semantic direction
        p2 = self._pass2_semantic_direction(seen)
        all_inferred.extend(p2)
        for s, r, o in p2:
            seen.add((s.lower(), r.lower(), o.lower()))
        self.stats['pass2'] = len(p2)

        # Pass 3: Fuzzy entity bridging
        p3 = self._pass3_fuzzy_bridge(seen)
        all_inferred.extend(p3)
        self.stats['pass3'] = len(p3)

        if store_inferred:
            self.knowledge.store_facts_fast(all_inferred)

        return all_inferred

    def _pass1_exact_chain(self, seen):
        """
        Pass 1: Exact keyword chaining.
        If (A, R1, B) and (B, R2, C), infer (A, chains_to, C).
        B is the bridge entity appearing as object in one fact and subject in another.
        """
        inferred = []
        facts = self.knowledge.facts

        # Find bridge entities: entities that appear as both subject AND object
        subjects = set(self.knowledge._entity_subject_index.keys())
        objects = set(self.knowledge._entity_object_index.keys())
        bridges = subjects & objects

        for bridge in bridges:
            # incoming: facts where bridge is the object (... → bridge)
            incoming_indices = self.knowledge._entity_object_index.get(bridge, [])
            # outgoing: facts where bridge is the subject (bridge → ...)
            outgoing_indices = self.knowledge._entity_subject_index.get(bridge, [])

            if not incoming_indices or not outgoing_indices:
                continue

            # Limit combinatorial explosion
            for i_idx in incoming_indices[:20]:
                s1, r1, o1 = facts[i_idx]
                for o_idx in outgoing_indices[:20]:
                    s2, r2, o2 = facts[o_idx]

                    # Cycle detection
                    if s1.lower() == o2.lower():
                        continue

                    key = (s1.lower(), 'chains_to', o2.lower())
                    if key in seen:
                        continue

                    inferred.append((s1, 'chains_to', o2))
                    seen.add(key)

                    if len(inferred) >= self.max_pass1:
                        return inferred

        return inferred

    def _pass2_semantic_direction(self, seen):
        """
        Pass 2: Semantic direction propagation.
        Chain relation directions: positive+positive=positive, etc.
        """
        inferred = []
        facts = self.knowledge.facts

        subjects = set(self.knowledge._entity_subject_index.keys())
        objects = set(self.knowledge._entity_object_index.keys())
        bridges = subjects & objects

        for bridge in bridges:
            incoming_indices = self.knowledge._entity_object_index.get(bridge, [])
            outgoing_indices = self.knowledge._entity_subject_index.get(bridge, [])

            if not incoming_indices or not outgoing_indices:
                continue

            for i_idx in incoming_indices[:15]:
                s1, r1, o1 = facts[i_idx]
                dir1 = _classify_relation(r1)

                for o_idx in outgoing_indices[:15]:
                    s2, r2, o2 = facts[o_idx]

                    if s1.lower() == o2.lower():
                        continue

                    dir2 = _classify_relation(r2)
                    chain_key = (dir1, dir2)
                    if chain_key not in DIRECTION_CHAINS:
                        chain_key = ('neutral', 'neutral')

                    result_dir, _ = DIRECTION_CHAINS[chain_key]

                    if result_dir == 'positive':
                        mech = f"indirectly_{r2}"
                    elif result_dir == 'negative':
                        mech = f"inversely_{r2}"
                    else:
                        mech = "relates_to"

                    key = (s1.lower(), mech.lower(), o2.lower())
                    if key in seen:
                        continue

                    inferred.append((s1, mech, o2))
                    seen.add(key)

                    if len(inferred) >= self.max_pass2:
                        return inferred

        return inferred

    def _pass3_fuzzy_bridge(self, seen):
        """
        Pass 3: Fuzzy entity bridging via Jaro-Winkler.
        Entities with similar names share their facts.
        """
        inferred = []
        facts = self.knowledge.facts

        entities = list(self.knowledge._entity_subject_index.keys())

        # Bucket by first 3 chars for O(N) instead of O(N²)
        buckets = defaultdict(list)
        for e in entities:
            if len(e) >= 3:
                buckets[e[:3].lower()].append(e)

        for prefix, members in buckets.items():
            if len(members) < 2:
                continue

            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    e1, e2 = members[i], members[j]
                    if e1 == e2:
                        continue

                    sim = _jaro_winkler(e1.lower(), e2.lower())
                    if sim < self.fuzzy_threshold:
                        continue

                    # Bridge e1's facts to e2
                    for idx in self.knowledge._entity_subject_index.get(e1, [])[:10]:
                        s, r, o = facts[idx]
                        key = (e2.lower(), r.lower(), o.lower())
                        if key not in seen:
                            inferred.append((e2, r, o))
                            seen.add(key)
                            if len(inferred) >= self.max_pass3:
                                return inferred

                    # Bridge e2's facts to e1
                    for idx in self.knowledge._entity_subject_index.get(e2, [])[:10]:
                        s, r, o = facts[idx]
                        key = (e1.lower(), r.lower(), o.lower())
                        if key not in seen:
                            inferred.append((e1, r, o))
                            seen.add(key)
                            if len(inferred) >= self.max_pass3:
                                return inferred

        return inferred

    def summary(self):
        """Return inference statistics."""
        total = sum(self.stats.values())
        explicit = len(self.knowledge.facts) - total
        amp = (total / max(explicit, 1)) * 100
        return (f"Inference: {total} inferred from {explicit} explicit "
                f"({amp:.0f}% amplification)\n"
                f"  Pass 1 (exact chain): {self.stats['pass1']}\n"
                f"  Pass 2 (semantic dir): {self.stats['pass2']}\n"
                f"  Pass 3 (fuzzy bridge): {self.stats['pass3']}")
