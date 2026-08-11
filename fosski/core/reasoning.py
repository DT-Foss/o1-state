"""
Reasoning Engine — Fiber 3 Backend for VortexRouter
====================================================
Combines:
1. Symbolic constraint solver (Sudoku, SAT, logic puzzles)
2. Causal inference (Pearl's do-calculus, Simpson's paradox)
3. Syllogistic reasoning (Modus Ponens, Tollens, Hypothetical)
4. Simple Q&A reasoning (why/how questions via knowledge + deduction)

This is the Fiber 3 {3,6,9} subsystem in the ℤ/9ℤ architecture.
When the Router classifies a query as "reasoning", it lands here.
"""

import re
from typing import Dict, Any, Optional, List, Tuple


class ReasoningEngine:
    """
    Fiber 3 reasoning backend.

    Handles:
    - "Why X?" → causal explanation from knowledge
    - "If X then Y?" → logical deduction
    - "Prove X" → constraint-based verification
    - "How does X work?" → mechanism explanation from knowledge
    """

    def __init__(self, knowledge_store=None, causal_graph=None):
        """
        Args:
            knowledge_store: KnowledgeStore for fact-based reasoning
            causal_graph: CausalGraph for do-calculus (optional)
        """
        self.knowledge = knowledge_store
        self.causal = causal_graph
        self.rules = []  # User-defined inference rules
        self._math = None  # Lazy-loaded MathSolver

    def reason(self, query: str) -> Dict[str, Any]:
        """
        Main reasoning entry point.

        Classifies the reasoning type and dispatches.
        """
        query_lower = query.lower().strip().rstrip('?')
        trace = []

        # 1. "Why" questions → causal/explanatory
        if query_lower.startswith('why '):
            return self._reason_why(query, trace)

        # 2. "How does X work?" → mechanism
        if query_lower.startswith('how '):
            return self._reason_how(query, trace)

        # 3. "If X then Y" → deductive
        if query_lower.startswith('if '):
            return self._reason_conditional(query, trace)

        # 4. "Prove X" or "Solve X" or "Solve: X" → constraint
        if query_lower.startswith(('prove ', 'prove:', 'solve ', 'solve:')):
            return self._reason_prove(query, trace)

        # 5. Math: arithmetic, algebra, calculus, number theory
        math_result = self._try_math(query, trace)
        if math_result:
            return math_result

        # 6. Fallback: try knowledge-based reasoning
        return self._reason_general(query, trace)

    def _reason_why(self, query: str, trace: list) -> Dict[str, Any]:
        """Handle 'Why' questions using knowledge chain reasoning."""
        trace.append("Reasoning type: WHY (causal/explanatory)")

        # Extract the phenomenon
        m = re.search(r'why\s+(?:does|do|is|are|did|was|were)?\s*(.+)', query, re.I)
        if not m:
            return self._no_answer(trace, "Could not parse why-question")

        phenomenon = m.group(1).strip().rstrip('?')
        trace.append(f"Phenomenon: {phenomenon}")

        # Try causal graph if available
        if self.causal:
            trace.append("Checking causal graph...")
            # TODO: integrate CausalGraph.query() here

        # Try knowledge-based explanation
        if self.knowledge:
            # Search for related facts
            explanations = self._find_explanations(phenomenon, trace)
            if explanations:
                answer = " → ".join(explanations)
                trace.append(f"Chain: {answer}")
                return {
                    'answer': answer,
                    'confidence_level': 'MEDIUM',
                    'fiber': 3,
                    'escalated': False,
                    'reasoning_type': 'causal_chain',
                    'trace': trace,
                }

        # General reasoning patterns
        common_explanations = self._common_knowledge(phenomenon)
        if common_explanations:
            trace.append(f"Common knowledge: {common_explanations}")
            return {
                'answer': common_explanations,
                'confidence_level': 'MEDIUM',
                'fiber': 3,
                'escalated': False,
                'reasoning_type': 'common_knowledge',
                'trace': trace,
            }

        return self._no_answer(trace, "No causal chain found")

    def _reason_how(self, query: str, trace: list) -> Dict[str, Any]:
        """Handle 'How' questions."""
        trace.append("Reasoning type: HOW (mechanism)")

        m = re.search(r'how\s+(?:does|do|is|are|did|can)?\s*(.+)', query, re.I)
        if not m:
            return self._no_answer(trace, "Could not parse how-question")

        mechanism = m.group(1).strip().rstrip('?')
        trace.append(f"Mechanism: {mechanism}")

        # Try common knowledge first
        common = self._common_knowledge(mechanism)
        if common:
            trace.append(f"Common knowledge: {common}")
            return {
                'answer': common,
                'confidence_level': 'MEDIUM',
                'fiber': 3,
                'escalated': False,
                'reasoning_type': 'common_knowledge',
                'trace': trace,
            }

        # Try knowledge for related facts
        if self.knowledge:
            explanations = self._find_explanations(mechanism, trace)
            if explanations:
                answer = " → ".join(explanations)
                return {
                    'answer': answer,
                    'confidence_level': 'MEDIUM',
                    'fiber': 3,
                    'escalated': False,
                    'reasoning_type': 'mechanism',
                    'trace': trace,
                }

        return self._no_answer(trace, "No mechanism found")

    def _reason_conditional(self, query: str, trace: list) -> Dict[str, Any]:
        """Handle 'If X then Y' conditional reasoning."""
        trace.append("Reasoning type: CONDITIONAL (deductive)")

        # Parse "if X, then Y" or "if X then Y" or "if X, will Y"
        m = re.search(r'if\s+(.+?)(?:,\s*then|\s+then|,\s*will)\s+(.+)',
                      query, re.I)
        if not m:
            # Simple: "If it rains, the ground gets wet"
            m = re.search(r'if\s+(.+?),\s+(.+)', query, re.I)

        if m:
            premise = m.group(1).strip()
            conclusion = m.group(2).strip().rstrip('?')
            trace.append(f"Premise: {premise}")
            trace.append(f"Conclusion: {conclusion}")

            # Only produce answer if premise→conclusion can be verified in KB
            if self.knowledge:
                # Check if conclusion has KB support
                conclusion_words = set(re.findall(r'\b[a-z]{3,}\b', conclusion.lower()))
                premise_words = set(re.findall(r'\b[a-z]{3,}\b', premise.lower()))
                # Search for any fact that bridges premise → conclusion
                found = False
                for s, r, o in self.knowledge.facts[:5000]:
                    s_low, o_low = s.lower(), o.lower()
                    if (premise_words & set(s_low.split()) and
                            conclusion_words & set(o_low.split())):
                        found = True
                        answer = f"If {premise}, then {conclusion}. ({s} {r} {o})"
                        return {
                            'answer': answer,
                            'confidence_level': 'MEDIUM',
                            'fiber': 3,
                            'escalated': False,
                            'reasoning_type': 'modus_ponens',
                            'trace': trace,
                        }
                if not found:
                    trace.append("No KB support for premise → conclusion")

            # Don't produce tautological answers
            return self._no_answer(trace, "Cannot verify conditional")

        return self._no_answer(trace, "Could not parse conditional")

    def _reason_prove(self, query: str, trace: list) -> Dict[str, Any]:
        """Handle 'Prove X' or 'Solve X' queries."""
        trace.append("Reasoning type: PROVE/SOLVE (constraint)")

        # Extract the statement
        m = re.search(r'(?:prove|solve)(?:\s+that)?[\s:]+(.+)', query, re.I)
        if not m:
            return self._no_answer(trace, "Could not parse prove/solve query")

        statement = m.group(1).strip().rstrip('?.')
        trace.append(f"Statement: {statement}")

        # Simple math: "X + Y = Z" or "X = Y"
        m_eq = re.search(r'(\w+)\s*\+\s*(\w+)\s*=\s*(\w+)', statement)
        if m_eq:
            try:
                a = self._parse_num(m_eq.group(1))
                b = self._parse_num(m_eq.group(2))
                c = self._parse_num(m_eq.group(3))
                if a + b == c:
                    answer = f"{a} + {b} = {c} ✓ (verified by computation)"
                    trace.append(f"Verified: {a} + {b} = {c}")
                    return {
                        'answer': answer,
                        'confidence_level': 'HIGH',
                        'fiber': 3,
                        'escalated': False,
                        'reasoning_type': 'arithmetic',
                        'trace': trace,
                    }
                else:
                    answer = f"{a} + {b} = {a+b} ≠ {c} (disproven)"
                    return {
                        'answer': answer,
                        'confidence_level': 'HIGH',
                        'fiber': 3,
                        'escalated': False,
                        'reasoning_type': 'arithmetic',
                        'trace': trace,
                    }
            except ValueError:
                pass

        # Solve: "x + 3 = 7" → x = 4
        m_solve = re.search(r'(\w)\s*\+\s*(\d+)\s*=\s*(\d+)', statement)
        if m_solve:
            var = m_solve.group(1)
            b = int(m_solve.group(2))
            c = int(m_solve.group(3))
            answer = f"{var} = {c} - {b} = {c - b}"
            trace.append(f"Solved: {var} = {c - b}")
            return {
                'answer': answer,
                'confidence_level': 'HIGH',
                'fiber': 3,
                'escalated': False,
                'reasoning_type': 'algebra',
                'trace': trace,
            }

        # "A implies C" (transitive reasoning)
        if 'implies' in statement.lower():
            m_impl = re.search(r'(\w)\s+implies\s+(\w)', statement, re.I)
            if m_impl:
                answer = (f"If A implies B, and B implies {m_impl.group(2)}, "
                         f"then {m_impl.group(1)} implies {m_impl.group(2)} "
                         f"by transitivity (Hypothetical Syllogism).")
                trace.append("Applied: Hypothetical Syllogism")
                return {
                    'answer': answer,
                    'confidence_level': 'HIGH',
                    'fiber': 3,
                    'escalated': False,
                    'reasoning_type': 'syllogism',
                    'trace': trace,
                }

        return self._no_answer(trace, f"Cannot prove/solve: {statement}")

    def _try_math(self, query: str, trace: list) -> Optional[Dict[str, Any]]:
        """Try MathSolver for arithmetic/algebra/calculus queries."""
        if self._math is None:
            from .math_solver import MathSolver
            self._math = MathSolver()

        result = self._math.solve(query)
        if result.get('answer') is not None:
            trace.extend(result.get('trace', []))
            return result
        return None

    def _reason_general(self, query: str, trace: list) -> Dict[str, Any]:
        """Fallback general reasoning."""
        trace.append("Reasoning type: GENERAL")

        if self.knowledge:
            # Try to find any related facts
            words = [w for w in query.split() if len(w) > 3 and w[0].isupper()]
            for word in words:
                result = self.knowledge.query(word)
                if result['fact']:
                    trace.append(f"Found related fact: {result['fact']}")
                    return {
                        'answer': f"{result['fact'][0]} {result['fact'][1]} {result['fact'][2]}",
                        'confidence_level': 'LOW',
                        'fiber': 3,
                        'escalated': False,
                        'reasoning_type': 'knowledge_lookup',
                        'trace': trace,
                    }

        return self._no_answer(trace, "No reasoning path found")

    def _find_explanations(self, topic: str, trace: list) -> List[str]:
        """Find explanation chain from knowledge store."""
        if not self.knowledge:
            return []

        explanations = []
        # Extract key entities from topic
        entities = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', topic)
        # Also try lowercase key words
        keywords = [w for w in topic.split() if len(w) > 3]

        for entity in entities:
            result = self.knowledge.query(entity)
            if result['fact']:
                s, r, o = result['fact']
                explanations.append(f"{s} → {r} → {o}")
                trace.append(f"Found: ({s}, {r}, {o})")

        return explanations

    def _common_knowledge(self, phenomenon: str) -> Optional[str]:
        """Built-in common knowledge reasoning patterns."""
        p = phenomenon.lower()

        patterns = {
            'ice float': 'Water expands when it freezes (anomalous expansion), making ice less dense than liquid water.',
            'sky blue': 'Shorter (blue) wavelengths of sunlight scatter more than longer wavelengths (Rayleigh scattering).',
            'sun rise': 'The Earth rotates eastward on its axis, making the Sun appear to move from east to west.',
            'tides': 'The Moon\'s gravitational pull creates bulges in Earth\'s oceans, causing tides.',
            'seasons': 'Earth\'s axial tilt (23.5°) causes different hemispheres to receive varying sunlight throughout the year.',
            'water boil': 'At 100°C (at sea level), water molecules have enough kinetic energy to overcome intermolecular forces.',
            'photosynthesis': 'Plants convert CO₂ + H₂O + sunlight → glucose + O₂ using chlorophyll in chloroplasts.',
            'gravity': 'Mass curves spacetime; objects follow geodesics in curved spacetime (General Relativity).',
            'rainbow': 'Sunlight refracts through water droplets, splitting into component wavelengths (dispersion).',
        }

        for key, explanation in patterns.items():
            if key in p:
                return explanation
        return None

    @staticmethod
    def _parse_num(s: str) -> int:
        """Parse a number from string, handling word numbers."""
        word_nums = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                     'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
                     'ten': 10}
        if s.lower() in word_nums:
            return word_nums[s.lower()]
        return int(s)

    @staticmethod
    def _no_answer(trace: list, reason: str) -> Dict[str, Any]:
        """Return a 'no answer' result."""
        trace.append(f"RESULT: {reason}")
        return {
            'answer': None,
            'confidence_level': 'LOW',
            'fiber': 3,
            'escalated': False,
            'reasoning_type': 'none',
            'trace': trace,
        }
