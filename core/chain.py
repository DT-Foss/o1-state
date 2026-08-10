"""
Chain-of-Thought Engine — Multi-Step Reasoning Decomposition
=============================================================
Complex queries get decomposed into steps. Each step queries
the router/swarm, and intermediate results feed the next step.

Example:
  "In which country was the physicist who discovered radium born?"
  → Step 1: Who discovered radium? → Marie Curie
  → Step 2: Where was Marie Curie born? → Warsaw
  → Step 3: What country is Warsaw in? → Poland
  → Final: Poland

This is NOT a prompt-engineered "think step by step".
It's a structured query decomposition engine that uses
pattern matching + knowledge graph traversal.
"""

import re
from typing import Dict, Any, List, Optional, Tuple


# Relation synonyms for smarter decomposition
_RELATION_SYNONYMS = {
    'country': ['nationality', 'country', 'birthplace'],
    'city': ['birthplace', 'location', 'city'],
    'language': ['language', 'official_language'],
}


def _chain_of_decompose(m):
    """Smart decomposition for 'X of Y of Z' chains.

    Tries multiple phrasings for the intermediate relation
    since knowledge stores may use different relation names.
    """
    outer = m.group(1)
    inner = m.group(2)
    entity = m.group(3).rstrip('?').strip()

    # If inner relation has synonyms, try the most common phrasing first
    # "country of Einstein" → try birthplace first, then nationality
    if inner.lower() in _RELATION_SYNONYMS:
        # Person→location chain needs 3 hops:
        # birthplace → city, city → country, country → {outer}
        return [
            f"Where was {entity} born?",
            "What country is {prev} in?",
            "What is the {outer} of {{prev}}?".format(outer=outer),
        ]

    return [
        f"What is the {inner} of {entity}?",
        "What is the {outer} of {{prev}}?".format(outer=outer),
    ]


class ChainOfThought:
    """
    Decomposes complex queries into executable sub-steps.

    Each step produces an intermediate result that can be
    referenced by subsequent steps via {prev} or {step_N}.
    """

    # Patterns that indicate multi-hop queries
    MULTI_HOP_PATTERNS = [
        # "X of the Y who/that Z"
        (r'(?:what|where)\s+is\s+the\s+(\w+)\s+of\s+the\s+(\w+)\s+(?:who|that|which)\s+(.+)',
         'nested_relation'),
        # "In which X was/is the Y who Z [verb]?"
        # Group 3 = inner clause, Group 4 = trailing verb (born/located/founded)
        (r'in\s+which\s+(\w+)\s+(?:was|is)\s+the\s+(\w+)\s+(?:who|that|which)\s+(.+?)\s+(born|located|founded|raised|buried|died|living|based)\b',
         'nested_location_verb'),
        # Simpler: "In which X was/is the Y who Z" (no trailing verb)
        (r'in\s+which\s+(\w+)\s+(?:was|is)\s+the\s+(\w+)\s+(?:who|that|which)\s+(.+)',
         'nested_location'),
        # "Who X the Y that Z"
        (r'who\s+(\w+)\s+the\s+(\w+)\s+(?:that|which)\s+(.+)',
         'nested_agent'),
        # "What is X of Y of Z"
        (r'(?:what|where)\s+is\s+(?:the\s+)?(\w+)\s+of\s+(?:the\s+)?(\w+)\s+of\s+(.+)',
         'chain_of'),
        # "X and Y" compound questions
        (r'(.+?)\s+and\s+(?:also|what is|where is|who is)\s+(.+)',
         'compound'),
    ]

    # Decomposition templates for different patterns
    DECOMPOSITIONS = {
        'nested_relation': [
            # "What is the capital of the country that discovered radium?"
            # → Step 1: What {inner_verb} {inner_object}? → entity
            # → Step 2: What is the {outer_relation} of {entity}?
            lambda m: [
                f"Who {m.group(3).rstrip('?')}?",
                "What is the {outer} of {{prev}}?".format(outer=m.group(1)),
            ],
        ],
        'nested_location_verb': [
            # "In which country was the physicist who discovered radium born?"
            # g1=country, g2=physicist, g3=discovered radium, g4=born
            # → Step 1: Who {g3}? → Marie Curie
            # → Step 2: Where was {prev} {g4}? → Warsaw
            # → Step 3: What {g1} is {prev} in? → Poland
            lambda m: [
                f"Who {m.group(3).strip().rstrip('?')}?",
                "Where was {{prev}} {verb}?".format(verb=m.group(4)),
                "What {outer} is {{prev}} in?".format(outer=m.group(1)),
            ],
        ],
        'nested_location': [
            # Simpler version without trailing verb
            lambda m: [
                f"Who {m.group(3).rstrip('?')}?",
                "Where was {{prev}} born?",
                "What {outer} is {{prev}} in?".format(outer=m.group(1)),
            ],
        ],
        'nested_agent': [
            lambda m: [
                f"What {m.group(2)} {m.group(3).rstrip('?')}?",
                "Who {verb} {{prev}}?".format(verb=m.group(1)),
            ],
        ],
        'chain_of': [
            # "What is the capital of the country of Einstein?"
            # → Step 1: What is the {inner} of {entity}?
            # → Step 2: What is the {outer} of {step1}?
            lambda m: _chain_of_decompose(m),
        ],
        'compound': [
            lambda m: [
                m.group(1).strip().rstrip('?') + '?',
                m.group(2).strip().rstrip('?') + '?',
            ],
        ],
    }

    def __init__(self, router=None, swarm_agent=None):
        """
        Args:
            router: VortexRouter instance for local queries
            swarm_agent: SwarmAgent for distributed queries (optional)
        """
        self.router = router
        self.swarm = swarm_agent
        self.max_hops = 5  # Safety limit

    def think(self, query: str) -> Dict[str, Any]:
        """
        Main entry: decompose and execute a query chain.

        Returns:
            dict with:
                answer: final answer
                chain: list of (step_query, step_answer) tuples
                n_hops: number of steps executed
                confidence_level: aggregate confidence
                trace: detailed execution log
        """
        trace = []

        # Step 1: Try to decompose
        steps = self.decompose(query)

        if not steps:
            # Single-hop query — just route normally
            trace.append("Single-hop query, routing directly")
            result = self._execute_step(query, trace)
            return {
                'answer': result.get('answer'),
                'chain': [(query, result.get('answer'))],
                'n_hops': 1,
                'confidence_level': result.get('confidence_level', 'UNKNOWN'),
                'trace': trace,
            }

        trace.append(f"Decomposed into {len(steps)} steps")

        # Step 2: Execute chain
        chain = []
        prev_answer = None
        step_answers = {}
        aggregate_confidence = 'HIGH'

        for i, step_template in enumerate(steps):
            # Substitute previous answers
            step_query = self._substitute(step_template, prev_answer, step_answers)
            trace.append(f"Step {i+1}: {step_query}")

            result = self._execute_step(step_query, trace)
            answer = result.get('answer')
            confidence = result.get('confidence_level', 'UNKNOWN')

            chain.append((step_query, answer))
            step_answers[f'step_{i+1}'] = answer
            prev_answer = answer

            trace.append(f"  → {answer} [{confidence}]")

            # Downgrade aggregate confidence
            if confidence in ('REJECTED', 'UNKNOWN'):
                aggregate_confidence = 'LOW'
                trace.append(f"  Chain broken at step {i+1}")
                break
            elif confidence == 'LOW':
                aggregate_confidence = 'LOW'
            elif confidence == 'MEDIUM' and aggregate_confidence == 'HIGH':
                aggregate_confidence = 'MEDIUM'

            if i >= self.max_hops - 1:
                trace.append(f"  Max hops ({self.max_hops}) reached")
                break

        # Step 3: Final answer is the last successful result
        final_answer = prev_answer
        if final_answer and 'information' in str(final_answer).lower():
            # The chain ended with "I don't have information" — go back one step
            for q, a in reversed(chain[:-1]):
                if a and 'information' not in str(a).lower():
                    final_answer = a
                    break

        return {
            'answer': final_answer,
            'chain': chain,
            'n_hops': len(chain),
            'confidence_level': aggregate_confidence,
            'trace': trace,
        }

    def decompose(self, query: str) -> Optional[List[str]]:
        """
        Try to decompose a complex query into sub-steps.

        Returns list of step templates, or None if single-hop.
        Templates can contain {prev} and {step_N} placeholders.
        """
        query_clean = query.strip()

        for pattern, pattern_type in self.MULTI_HOP_PATTERNS:
            m = re.search(pattern, query_clean, re.I)
            if m:
                # Get decomposition functions for this pattern type
                decomp_fns = self.DECOMPOSITIONS.get(pattern_type, [])
                for fn in decomp_fns:
                    steps = fn(m)
                    if steps:
                        return steps

        # Try heuristic decomposition for "X of Y" chains
        steps = self._heuristic_decompose(query_clean)
        if steps:
            return steps

        return None

    def _heuristic_decompose(self, query: str) -> Optional[List[str]]:
        """
        Heuristic decomposition for patterns not caught by regex.

        Detects embedded clauses like:
        - "born in the city where X happened"
        - "the author of the book that Y wrote"
        """
        q = query.lower().rstrip('?')

        # Count nested references
        nested_markers = ['who ', 'that ', 'which ', 'where ', 'whose ']
        nesting = sum(1 for m in nested_markers if m in q)

        if nesting < 1:
            return None

        # Try relative clause extraction
        # "... the [entity] who/that [did something]"
        m = re.search(
            r'(.+?)\s+the\s+(\w+)\s+(who|that|which)\s+(.+?)(?:\s+(?:born|located|founded|discovered|invented|wrote|created))',
            query, re.I
        )
        if m:
            inner_query = f"Who {m.group(4).strip().rstrip('?')}?"
            # Reconstruct outer query with placeholder
            outer = re.sub(
                r'the\s+\w+\s+(?:who|that|which)\s+.+',
                '{prev}',
                query,
                flags=re.I
            ).rstrip('?') + '?'
            return [inner_query, outer]

        return None

    def _substitute(self, template: str, prev_answer: Optional[str],
                    step_answers: Dict[str, str]) -> str:
        """Replace {prev} and {step_N} placeholders with actual answers."""
        result = template

        if prev_answer and '{prev}' in result:
            result = result.replace('{prev}', str(prev_answer))

        for key, val in step_answers.items():
            placeholder = '{' + key + '}'
            if placeholder in result and val:
                result = result.replace(placeholder, str(val))

        return result

    def _execute_step(self, query: str, trace: list) -> Dict[str, Any]:
        """Execute a single step query via router or swarm."""
        # Prefer swarm if available (can reach more knowledge)
        if self.swarm:
            result = self.swarm.swarm_query(query, timeout=2.0)
            trace.append(f"  (via swarm, source={result.get('source', 'local')})")
            return result

        if self.router:
            result = self.router.route(query)
            trace.append(f"  (via router, fiber={result.get('fiber', '?')})")
            return result

        return {'answer': None, 'confidence_level': 'UNKNOWN'}

    def explain(self, query: str) -> str:
        """
        Human-readable explanation of how a chain would be decomposed.
        Useful for debugging and transparency.
        """
        steps = self.decompose(query)
        if not steps:
            return f"Single-hop query: {query}"

        lines = [f"Query: {query}", f"Decomposed into {len(steps)} steps:"]
        for i, step in enumerate(steps):
            lines.append(f"  Step {i+1}: {step}")
        return '\n'.join(lines)


class ChainBuilder:
    """
    Programmatic chain construction for complex reasoning tasks.

    Allows building multi-step chains explicitly:
        chain = ChainBuilder(router)
        chain.ask("Who discovered radium?")
        chain.then("Where was {prev} born?")
        chain.then("What country is {prev} in?")
        result = chain.execute()
    """

    def __init__(self, router=None, swarm_agent=None):
        self.router = router
        self.swarm = swarm_agent
        self._steps = []

    def ask(self, query: str) -> 'ChainBuilder':
        """Add the first step."""
        self._steps = [query]
        return self

    def then(self, query_template: str) -> 'ChainBuilder':
        """Add a follow-up step with {prev}/{step_N} placeholders."""
        self._steps.append(query_template)
        return self

    def execute(self) -> Dict[str, Any]:
        """Execute the chain."""
        cot = ChainOfThought(router=self.router, swarm_agent=self.swarm)
        trace = [f"Explicit chain with {len(self._steps)} steps"]

        chain = []
        prev_answer = None
        step_answers = {}

        for i, step_template in enumerate(self._steps):
            step_query = cot._substitute(step_template, prev_answer, step_answers)
            trace.append(f"Step {i+1}: {step_query}")

            result = cot._execute_step(step_query, trace)
            answer = result.get('answer')
            chain.append((step_query, answer))
            step_answers[f'step_{i+1}'] = answer
            prev_answer = answer
            trace.append(f"  → {answer}")

        return {
            'answer': prev_answer,
            'chain': chain,
            'n_hops': len(chain),
            'trace': trace,
        }

    def reset(self) -> 'ChainBuilder':
        """Clear all steps."""
        self._steps = []
        return self
