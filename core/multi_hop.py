"""
Multi-Hop Reasoning — Chain Facts to Answer Complex Questions
===============================================================
Forward-chaining over CommonSense relations to answer:
  - Syllogisms: "If all cats are animals and animals need food, do cats need food?"
  - Hypothetical: "What would happen if the moon disappeared?"
  - Explanation chains: "Why do plants need sunlight?" → need sunlight → for photosynthesis → produces food

No external dependencies.
"""

import re
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import deque


class MultiHopReasoner:
    """Chain facts across multiple hops to answer questions."""

    def __init__(self, commonsense=None, knowledge=None):
        self.cs = commonsense  # CommonSenseEngine instance
        self.kb = knowledge    # KnowledgeStore instance (graph traversal)

    def set_engine(self, engine):
        """Set the CommonSenseEngine to reason over."""
        self.cs = engine

    def reason(self, question: str) -> Dict[str, Any]:
        """
        Attempt multi-hop reasoning for a question.

        Returns:
            {
                'answered': bool,
                'answer': str,
                'chain': [(subject, relation, object), ...],
                'hops': int,
                'method': str,
            }
        """
        q = question.lower().strip().rstrip('?').strip()

        # Try syllogism detection
        result = self._try_syllogism(q)
        if result['answered']:
            return result

        # Try hypothetical ("what would happen if X")
        result = self._try_hypothetical(q)
        if result['answered']:
            return result

        # Try causal chain ("why does X Y")
        result = self._try_causal_chain(q)
        if result['answered']:
            return result

        # Try property inheritance
        result = self._try_inheritance(q)
        if result['answered']:
            return result

        # Try KB graph path finding
        result = self._try_kb_path(q, question)
        if result['answered']:
            return result

        return {'answered': False, 'answer': '', 'chain': [], 'hops': 0, 'method': 'none'}

    def _try_syllogism(self, q: str) -> Dict[str, Any]:
        """
        Detect and resolve syllogisms.
        "if all X are Y and all Y Z, do X Z?"
        "if cats are animals and animals need food, do cats need food?"
        """
        # Pattern: "if (all) X are Y and (all) Y R Z, do X R Z"
        m = re.search(
            r'if\s+(?:all\s+)?(\w+)\s+(?:are|is)\s+(\w+)\s+and\s+'
            r'(?:all\s+)?(\w+)\s+(\w+)\s+(\w+),?\s*'
            r'(?:do|does|can|will)\s+(\w+)\s+(\w+)\s+(\w+)',
            q
        )
        if m:
            premise1_subj = m.group(1)  # cats
            premise1_obj = m.group(2)   # animals
            premise2_subj = m.group(3)  # animals
            premise2_rel = m.group(4)   # need
            premise2_obj = m.group(5)   # food
            query_subj = m.group(6)     # cats
            query_rel = m.group(7)      # need
            query_obj = m.group(8)      # food

            # Check if the chain holds:
            # premise1: X is_a Y
            # premise2: Y rel Z
            # conclusion: X rel Z (by inheritance)
            if (premise1_subj == query_subj and
                premise1_obj == premise2_subj and
                premise2_rel == query_rel and
                premise2_obj == query_obj):

                chain = [
                    (premise1_subj, 'is_a', premise1_obj),
                    (premise2_subj, premise2_rel, premise2_obj),
                    (query_subj, query_rel, query_obj),
                ]
                return {
                    'answered': True,
                    'answer': f"Yes, {query_subj} {query_rel} {query_obj}.",
                    'chain': chain,
                    'hops': 2,
                    'method': 'syllogism',
                }

        # Simpler pattern: "do X need Y" → check inheritance
        m = re.match(r'(?:do|does|can)\s+(\w+)\s+(\w+)\s+(\w+)', q)
        if m and self.cs:
            subj, rel, obj = m.group(1), m.group(2), m.group(3)
            return self._check_inherited(subj, rel, obj)

        return {'answered': False, 'answer': '', 'chain': [], 'hops': 0, 'method': 'none'}

    def _try_hypothetical(self, q: str) -> Dict[str, Any]:
        """
        Answer hypothetical questions using causal chains.
        "what would happen if X disappeared/stopped/etc."
        """
        if not self.cs:
            return {'answered': False, 'answer': '', 'chain': [], 'hops': 0, 'method': 'none'}

        # "what would happen if X disappeared/stopped"
        m = re.search(r'what\s+(?:would|will)\s+happen\s+if\s+(?:the\s+)?(\w+)\s+(\w+)', q)
        if not m:
            return {'answered': False, 'answer': '', 'chain': [], 'hops': 0, 'method': 'none'}

        subject = m.group(1)
        action = m.group(2)  # disappeared, stopped, etc.

        # Find what the subject causes
        chain = []
        effects = self._find_effects(subject, max_hops=3)

        if effects:
            chain = effects
            # Build answer from effects
            effect_strs = []
            for s, r, o in effects:
                if r == 'causes':
                    effect_strs.append(f"{o.replace('_', ' ')}")
                elif r == 'needs':
                    effect_strs.append(f"things that need {s.replace('_', ' ')} would be affected")

            if effect_strs:
                answer = f"If {subject} {action}, several things would be affected. "
                answer += f"{subject.title()} is known to cause {', '.join(effect_strs[:3])}. "
                answer += f"Without it, these effects would cease or be disrupted."
                return {
                    'answered': True,
                    'answer': answer,
                    'chain': chain,
                    'hops': len(chain),
                    'method': 'hypothetical',
                }

        # Check what needs the subject
        dependents = self._find_dependents(subject)
        if dependents:
            dep_names = [d.replace('_', ' ') for d, _, _ in dependents[:5]]
            answer = f"If {subject} {action}, it would affect {', '.join(dep_names)}. "
            answer += f"These all depend on {subject} in some way."
            return {
                'answered': True,
                'answer': answer,
                'chain': dependents,
                'hops': len(dependents),
                'method': 'hypothetical',
            }

        return {'answered': False, 'answer': '', 'chain': [], 'hops': 0, 'method': 'none'}

    def _try_causal_chain(self, q: str) -> Dict[str, Any]:
        """
        Build causal explanation chains.
        "why do plants need sunlight" → chain from cause to effect
        """
        if not self.cs:
            return {'answered': False, 'answer': '', 'chain': [], 'hops': 0, 'method': 'none'}

        # "why do/does X Y" or "why is X Y"
        m = re.match(r'why\s+(?:do|does|is|are)\s+(?:the\s+)?(\w+)\s+(\w+)(?:\s+(\w+))?', q)
        if not m:
            return {'answered': False, 'answer': '', 'chain': [], 'hops': 0, 'method': 'none'}

        subject = m.group(1)
        predicate = m.group(2)
        obj = m.group(3) or ''

        # Try to find a causal chain
        chain = self._find_causal_chain(subject, predicate, obj)
        if chain:
            # Build natural answer
            sentences = []
            for s, r, o in chain:
                s_clean = s.replace('_', ' ')
                o_clean = o.replace('_', ' ')
                if r == 'causes':
                    sentences.append(f"{s_clean} causes {o_clean}")
                elif r == 'needs':
                    sentences.append(f"{s_clean} needs {o_clean}")
                elif r == 'is_a':
                    sentences.append(f"{s_clean} is a {o_clean}")
                else:
                    sentences.append(f"{s_clean} {r.replace('_', ' ')} {o_clean}")

            answer = '. '.join(s.capitalize() for s in sentences) + '.'
            return {
                'answered': True,
                'answer': answer,
                'chain': chain,
                'hops': len(chain),
                'method': 'causal_chain',
            }

        return {'answered': False, 'answer': '', 'chain': [], 'hops': 0, 'method': 'none'}

    def _try_inheritance(self, q: str) -> Dict[str, Any]:
        """
        Answer through property inheritance.
        "do cats need food?" → cat is_a animal, animal needs food → yes
        """
        if not self.cs:
            return {'answered': False, 'answer': '', 'chain': [], 'hops': 0, 'method': 'none'}

        m = re.match(r'(?:do|does|can)\s+(?:a\s+)?(\w+?)s?\s+(\w+)\s+(\w+)', q)
        if m:
            return self._check_inherited(m.group(1), m.group(2), m.group(3))

        return {'answered': False, 'answer': '', 'chain': [], 'hops': 0, 'method': 'none'}

    def _check_inherited(self, subject: str, relation: str, obj: str) -> Dict[str, Any]:
        """Check if subject has relation to obj, directly or through inheritance."""
        if not self.cs:
            return {'answered': False, 'answer': '', 'chain': [], 'hops': 0, 'method': 'none'}

        s = subject.lower()
        # Handle singular/plural relation matching (need/needs, cause/causes)
        rel_variants = {relation, relation + 's', relation.rstrip('s')}

        # Direct check
        for r, o, w in self.cs._by_subject.get(s, []):
            if r in rel_variants and o == obj:
                return {
                    'answered': True,
                    'answer': f"Yes, {subject} {relation} {obj}.",
                    'chain': [(s, relation, obj)],
                    'hops': 1,
                    'method': 'direct',
                }

        # Inheritance check through taxonomy
        ancestors = self.cs._get_ancestors(s)
        for ancestor in ancestors:
            for r, o, w in self.cs._by_subject.get(ancestor, []):
                if r in rel_variants and o == obj:
                    chain = [
                        (s, 'is_a', ancestor),
                        (ancestor, relation, obj),
                        (s, relation, obj),
                    ]
                    return {
                        'answered': True,
                        'answer': f"Yes, {subject} {relation} {obj}, because {subject} is a {ancestor} and {ancestor} {relation} {obj}.",
                        'chain': chain,
                        'hops': 2,
                        'method': 'inheritance',
                    }

        return {'answered': False, 'answer': '', 'chain': [], 'hops': 0, 'method': 'none'}

    def _try_kb_path(self, q: str, original: str) -> Dict[str, Any]:
        """
        Use KB graph BFS to find multi-hop connections between entities.

        Extracts entities from the question, finds paths between them
        in the knowledge graph, and builds a natural answer from the path.
        """
        if not self.kb or not hasattr(self.kb, 'find_path'):
            return {'answered': False, 'answer': '', 'chain': [], 'hops': 0, 'method': 'none'}

        # Extract capitalized entities from original question (filter question words)
        _SKIP_CAPS = {'How', 'What', 'Why', 'When', 'Where', 'Who', 'Which',
                      'Does', 'Do', 'Did', 'Is', 'Are', 'Was', 'Were', 'Can',
                      'Could', 'Would', 'Should', 'The', 'If', 'And', 'But', 'Or'}
        entities = [e for e in re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', original)
                    if e not in _SKIP_CAPS]
        # Also try significant lowercase words against KB
        if len(entities) < 2:
            words = [w for w in q.split() if len(w) > 3
                     and w not in ('what', 'does', 'have', 'with', 'this', 'that',
                                   'from', 'about', 'between', 'which', 'where')]
            for w in words:
                if w not in [e.lower() for e in entities]:
                    # Check if KB knows this entity
                    if self.kb._entity_subject_index.get(w) or self.kb._entity_object_index.get(w):
                        entities.append(w)

        if len(entities) < 2:
            return {'answered': False, 'answer': '', 'chain': [], 'hops': 0, 'method': 'none'}

        # Try to find path between all entity pairs
        best_path = None
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                path = self.kb.find_path(entities[i], entities[j], max_hops=3)
                if path and (best_path is None or len(path) < len(best_path)):
                    best_path = path

        if best_path:
            # Build natural answer from path using relation templates
            from .naturalizer import RELATION_TEMPLATES
            sentences = []
            for s, r, o in best_path:
                s_title = s.replace('_', ' ').title()
                o_title = o.replace('_', ' ').title()
                template = RELATION_TEMPLATES.get(r)
                if template:
                    sentences.append(template.format(subject=s_title, object=o_title))
                else:
                    sentences.append(f"{s_title} {r.replace('_', ' ')} {o_title}.")

            answer = ' '.join(sentences)
            return {
                'answered': True,
                'answer': answer,
                'chain': best_path,
                'hops': len(best_path),
                'method': 'kb_graph_path',
            }

        return {'answered': False, 'answer': '', 'chain': [], 'hops': 0, 'method': 'none'}

    def _find_effects(self, concept: str, max_hops: int = 3) -> List[Tuple[str, str, str]]:
        """Find what a concept causes (forward chaining)."""
        if not self.cs:
            return []

        effects = []
        visited: Set[str] = set()
        queue: deque = deque([(concept.lower(), 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_hops or current in visited:
                continue
            visited.add(current)

            for r, o, w in self.cs._by_subject.get(current, []):
                if r == 'causes':
                    effects.append((current, 'causes', o))
                    queue.append((o, depth + 1))

        return effects

    def _find_dependents(self, concept: str) -> List[Tuple[str, str, str]]:
        """Find what depends on / needs a concept."""
        if not self.cs:
            return []

        dependents = []
        c = concept.lower()
        for r, s, w in self.cs._by_object.get(c, []):
            if r == 'needs':
                dependents.append((s, 'needs', c))

        return dependents

    def _find_causal_chain(self, subject: str, predicate: str,
                           obj: str) -> List[Tuple[str, str, str]]:
        """Try to build a causal chain explaining why subject predicate obj."""
        if not self.cs:
            return []

        s = subject.lower()
        chain = []

        # Check direct cause
        for r, o, w in self.cs._by_subject.get(s, []):
            if r == 'causes' or r == predicate:
                chain.append((s, r, o))

        # Check what causes the subject's behavior
        for r, src, w in self.cs._by_object.get(s, []):
            if r == 'causes':
                chain.insert(0, (src, 'causes', s))

        return chain
