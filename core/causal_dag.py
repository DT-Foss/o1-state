"""
Causal DAG — Directed Acyclic Graph for Interventional Reasoning
=================================================================
Builds a causal graph from KB + ConceptNet relations:
  causes, leads_to, needs, enables, prevents

Supports:
  1. Forward propagation: X causes Y causes Z → chain
  2. Interventional: do(X=0) → what breaks? (remove node, trace effects)
  3. Counterfactual: what if X had been different?
  4. Common cause: X ← Z → Y (confounders)

This is Pearl's do-calculus simplified for a knowledge graph.
No probabilities — just reachability and path analysis.
"""

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional


# Relations that imply causal direction
CAUSAL_FORWARD = frozenset({
    'causes', 'leads_to', 'enables', 'produces', 'creates',
    'results_in', 'triggers', 'generates', 'makes',
    'HasSubevent', 'Causes', 'MannerOf',
})

CAUSAL_BACKWARD = frozenset({
    'needs', 'requires', 'depends_on', 'caused_by',
    'HasPrerequisite', 'HasFirstSubevent',
})

CAUSAL_PREVENTS = frozenset({
    'prevents', 'blocks', 'inhibits', 'stops',
})


class CausalDAG:
    """Directed causal graph built from knowledge base relations."""

    def __init__(self):
        # Adjacency lists: node → [(target, relation, weight)]
        self.forward = defaultdict(list)   # cause → effect
        self.backward = defaultdict(list)  # effect → cause
        self.prevents = defaultdict(list)  # X prevents Y
        self.nodes = set()

    def add_edge(self, cause: str, effect: str, relation: str,
                 weight: float = 1.0):
        """Add a causal edge: cause → effect."""
        c, e = cause.lower().strip(), effect.lower().strip()
        if not c or not e or c == e:
            return
        self.forward[c].append((e, relation, weight))
        self.backward[e].append((c, relation, weight))
        self.nodes.add(c)
        self.nodes.add(e)

    def add_prevention(self, blocker: str, target: str, relation: str,
                       weight: float = 1.0):
        """Add a prevention edge: blocker -|→ target."""
        b, t = blocker.lower().strip(), target.lower().strip()
        self.prevents[b].append((t, relation, weight))
        self.nodes.add(b)
        self.nodes.add(t)

    def build_from_kb(self, knowledge_store):
        """Extract causal edges from KnowledgeStore facts."""
        for s, r, o in knowledge_store.facts:
            r_lower = r.lower().replace(' ', '_')
            s_str = str(s).lower().strip()
            o_str = str(o).lower().strip()

            if r_lower in CAUSAL_FORWARD or r in CAUSAL_FORWARD:
                self.add_edge(s_str, o_str, r_lower)
            elif r_lower in CAUSAL_BACKWARD or r in CAUSAL_BACKWARD:
                self.add_edge(o_str, s_str, r_lower)  # Reverse direction
            elif r_lower in CAUSAL_PREVENTS or r in CAUSAL_PREVENTS:
                self.add_prevention(s_str, o_str, r_lower)

    def build_from_commonsense(self, commonsense_engine, concepts=None):
        """Extract causal edges from ConceptNet via CommonSenseEngine.

        Args:
            commonsense_engine: CommonSenseEngine with cn_lookup
            concepts: optional list of concepts to query (default: use engine's subjects)
        """
        if concepts is None:
            concepts = list(commonsense_engine._by_subject.keys())[:2000]

        for concept in concepts:
            facts = commonsense_engine.cn_lookup(concept, max_results=20)
            for rel, obj, weight in facts:
                if weight < 1.0:
                    continue
                if rel in CAUSAL_FORWARD:
                    self.add_edge(concept, obj, rel, weight)
                elif rel in CAUSAL_BACKWARD:
                    self.add_edge(obj, concept, rel, weight)

    # ================================================================
    # Queries
    # ================================================================

    def effects_of(self, concept: str, max_depth: int = 3) -> List[Tuple[str, int, List]]:
        """Forward propagation: what does X cause?

        Returns: [(effect, depth, path_from_concept), ...]
        """
        c = concept.lower().strip()
        results = []
        visited = set()
        queue = deque([(c, 0, [])])

        while queue:
            node, depth, path = queue.popleft()
            if node in visited or depth > max_depth:
                continue
            visited.add(node)

            for target, rel, weight in self.forward.get(node, []):
                if target not in visited:
                    new_path = path + [(node, rel, target)]
                    results.append((target, depth + 1, new_path))
                    queue.append((target, depth + 1, new_path))

        return sorted(results, key=lambda x: x[1])

    def causes_of(self, concept: str, max_depth: int = 3) -> List[Tuple[str, int, List]]:
        """Backward propagation: what causes X?

        Returns: [(cause, depth, path_to_concept), ...]
        """
        c = concept.lower().strip()
        results = []
        visited = set()
        queue = deque([(c, 0, [])])

        while queue:
            node, depth, path = queue.popleft()
            if node in visited or depth > max_depth:
                continue
            visited.add(node)

            for source, rel, weight in self.backward.get(node, []):
                if source not in visited:
                    new_path = [(source, rel, node)] + path
                    results.append((source, depth + 1, new_path))
                    queue.append((source, depth + 1, new_path))

        return sorted(results, key=lambda x: x[1])

    def intervene(self, removed: str, max_depth: int = 3) -> Dict[str, any]:
        """Interventional reasoning: do(X=0) — what breaks if X is removed?

        Pearl's do-calculus simplified:
        1. Remove all incoming edges to X (intervention = fixing X)
        2. Propagate: everything downstream of X is affected
        3. Check: what else needed X? Those are broken too.

        Returns:
            {
                'removed': str,
                'directly_affected': [(effect, relation)],
                'cascade': [(effect, depth, path)],
                'dependencies_broken': [(dependent, relation)],
                'prevented_effects': [(target, relation)],
            }
        """
        c = removed.lower().strip()

        # Direct effects (downstream)
        directly_affected = [(t, r) for t, r, w in self.forward.get(c, [])]

        # Cascade: full downstream propagation
        cascade = self.effects_of(c, max_depth=max_depth)

        # What depended on X? (things that need X are broken)
        dependencies_broken = []
        for node in self.nodes:
            for source, rel, weight in self.backward.get(node, []):
                if source == c:
                    dependencies_broken.append((node, rel))

        # What was X preventing? (removal = those things now happen)
        prevented_effects = [(t, r) for t, r, w in self.prevents.get(c, [])]

        return {
            'removed': c,
            'directly_affected': directly_affected,
            'cascade': cascade,
            'dependencies_broken': dependencies_broken,
            'prevented_effects': prevented_effects,
        }

    def common_causes(self, a: str, b: str,
                      max_depth: int = 2) -> List[str]:
        """Find common causes (confounders) of A and B."""
        causes_a = {c for c, _, _ in self.causes_of(a, max_depth)}
        causes_b = {c for c, _, _ in self.causes_of(b, max_depth)}
        return list(causes_a & causes_b)

    def causal_path(self, source: str, target: str,
                    max_depth: int = 5) -> Optional[List[Tuple[str, str, str]]]:
        """Find a causal path from source to target (BFS)."""
        s = source.lower().strip()
        t = target.lower().strip()
        visited = set()
        queue = deque([(s, [])])

        while queue:
            node, path = queue.popleft()
            if node == t:
                return path
            if node in visited or len(path) >= max_depth:
                continue
            visited.add(node)

            for next_node, rel, weight in self.forward.get(node, []):
                if next_node not in visited:
                    queue.append((next_node, path + [(node, rel, next_node)]))

        return None

    def answer_intervention(self, question: str) -> Optional[str]:
        """Answer 'what would happen if X disappeared/stopped/was removed?'

        Parses the question, runs interventional analysis, builds answer.
        """
        import re
        q = question.lower().strip().rstrip('?').strip()

        # Pattern: "what would happen if X disappeared/stopped/was removed"
        m = re.search(
            r'what\s+(?:would|will)\s+happen\s+if\s+(?:the\s+)?'
            r'(\w+(?:\s+\w+)?)\s+'
            r'(?:disappeared|stopped|was\s+removed|went\s+away|ceased|ended)',
            q)
        if not m:
            # Pattern: "what if there was no X"
            m = re.search(r'what\s+if\s+(?:there\s+was\s+)?no\s+(\w+)', q)
        if not m:
            return None

        concept = m.group(1).strip()
        result = self.intervene(concept)

        if not result['directly_affected'] and not result['dependencies_broken']:
            return None

        parts = []
        if result['directly_affected']:
            effects = [f"{e}" for e, r in result['directly_affected'][:3]]
            parts.append(f"Without {concept}, {', '.join(effects)} would be affected")

        if result['dependencies_broken']:
            deps = [f"{d}" for d, r in result['dependencies_broken'][:3]]
            parts.append(f"{', '.join(deps)} depend on {concept} and would fail")

        if result['prevented_effects']:
            freed = [f"{t}" for t, r in result['prevented_effects'][:2]]
            parts.append(f"also, {', '.join(freed)} would no longer be prevented")

        if result['cascade'] and len(result['cascade']) > len(result['directly_affected']):
            n_cascade = len(result['cascade'])
            parts.append(f"in total, {n_cascade} things would be affected downstream")

        return '. '.join(p.capitalize() for p in parts) + '.'

    def stats(self) -> Dict[str, int]:
        """Return DAG statistics."""
        n_edges = sum(len(v) for v in self.forward.values())
        n_prevent = sum(len(v) for v in self.prevents.values())
        return {
            'nodes': len(self.nodes),
            'causal_edges': n_edges,
            'prevention_edges': n_prevent,
        }
