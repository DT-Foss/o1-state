"""
Vortex System Router — ℤ/9ℤ Fiber Structure as Architecture Layer
===================================================================
T89 showed: Vortex components HURT when stuffed inside the PPM.
David's insight: Vortex belongs ABOVE the components, not inside.

Three fibers from ℤ/9ℤ → ℤ/3ℤ:
  Fiber 1 {1,4,7}: Language subsystem (PPM)
  Fiber 2 {2,5,8}: Knowledge subsystem (Hopfield)
  Fiber 3 {3,6,9}: Reasoning subsystem (Logic Solver + Causal)

The ×3 bridge is the escalation moment:
  A fact question that PPM can't answer → escalate to Hopfield.
  A Hopfield answer that needs verification → escalate to Reasoning.

Digital root routing: input → digital root → fiber → subsystem.
This is the natural level for fiber structure: system routing.
"""

import re
import numpy as np
from typing import Optional, Dict, Any


class VortexRouter:
    """
    Routes queries to the appropriate subsystem using ℤ/9ℤ fiber structure.

    Classification happens by analyzing the query structure:
    - Factual queries → Fiber 2 (Hopfield Knowledge Store)
    - Generative queries → Fiber 1 (PPM Language Model)
    - Logical/causal queries → Fiber 3 (Reasoning Engine)

    The ×3 bridge enables cross-fiber escalation when one
    subsystem's confidence is too low.
    """

    # Query type indicators
    FACT_WORDS = {
        'what', 'who', 'where', 'when', 'which', 'whose',
        'capital', 'population', 'formula', 'symbol', 'language',
        'founder', 'inventor', 'author', 'president', 'leader',
        'born', 'located', 'founded', 'discovered',
    }

    REASONING_WORDS = {
        'why', 'how', 'prove', 'because', 'therefore', 'implies',
        'if', 'then', 'cause', 'effect', 'conclude', 'follows',
        'logically', 'deduce', 'infer', 'reason', 'solve',
        'calculate', 'puzzle', 'sudoku', 'satisfy', 'constraint',
    }

    GENERATE_WORDS = {
        'write', 'generate', 'create', 'compose', 'continue',
        'complete', 'suggest', 'describe', 'tell', 'story',
        'poem', 'summarize', 'explain', 'rephrase', 'translate',
    }

    def __init__(self, knowledge_store=None, language_model=None,
                 reasoning_engine=None, confidence_threshold=0.5):
        """
        Args:
            knowledge_store: KnowledgeStore instance (Fiber 2)
            language_model: PPM or VortexLanguageModel (Fiber 1)
            reasoning_engine: Logic solver / causal engine (Fiber 3)
            confidence_threshold: below this, escalate to next fiber
        """
        self.knowledge = knowledge_store
        self.language = language_model
        self.reasoning = reasoning_engine
        self.confidence_threshold = confidence_threshold
        self._composer = None  # Lazy-initialized ResponseComposer
        self._formulierer = None  # Lazy-initialized Formulierer

        # Routing statistics
        self.stats = {
            'fiber1_calls': 0,  # PPM
            'fiber2_calls': 0,  # Hopfield
            'fiber3_calls': 0,  # Reasoning
            'escalations': 0,   # ×3 bridge crossings
            'total_queries': 0,
        }

    def classify_query(self, query: str) -> int:
        """
        Classify query into fiber using lexical analysis.

        Returns: 1 (Language), 2 (Knowledge), 3 (Reasoning)
        """
        words = set(query.lower().split())
        # Also check for question patterns
        query_lower = query.lower().strip()

        # Scoring
        fact_score = len(words & self.FACT_WORDS)
        reason_score = len(words & self.REASONING_WORDS)
        gen_score = len(words & self.GENERATE_WORDS)

        # Question word patterns boost fact score
        if query_lower.startswith(('what is', 'who is', 'where is', 'when was',
                                    'what are', 'who was', 'where was')):
            fact_score += 2
        if '?' in query and any(w in words for w in ('capital', 'formula', 'symbol', 'population')):
            fact_score += 2

        # "why" and "how" strongly indicate reasoning
        if query_lower.startswith(('why ', 'how does', 'how can', 'if ', 'prove', 'solve')):
            reason_score += 2

        # "tell me about" / "describe" / "compare" are knowledge queries, not generation
        if any(query_lower.startswith(p) for p in (
            'tell me about', 'what do you know about', 'describe ',
            'compare ', 'what are the ',
        )):
            fact_score += 3

        # Imperative verbs indicate generation (but NOT "tell me about")
        elif query_lower.startswith(('write ', 'generate ', 'create ', 'tell ', 'describe ')):
            gen_score += 2

        # "of X" pattern strongly indicates knowledge lookup
        if re.search(r'\bof\s+[A-Z]', query):
            fact_score += 1

        # Route
        if reason_score > fact_score and reason_score > gen_score:
            return 3
        if fact_score > gen_score:
            return 2
        if gen_score > 0:
            return 1
        # Default: knowledge lookup for questions, language for statements
        if '?' in query:
            return 2
        return 1

    def route(self, query: str) -> Dict[str, Any]:
        """
        Route a query through the Vortex fiber structure.

        1. Classify → primary fiber
        2. Execute on primary subsystem
        3. If confidence low → ×3 bridge → escalate

        Returns:
            dict with:
                answer: the response
                fiber: which fiber handled it (1/2/3)
                confidence: confidence level
                escalated: whether ×3 bridge was used
                trace: routing decisions
        """
        self.stats['total_queries'] += 1
        trace = []

        # Step 1: Classify
        primary_fiber = self.classify_query(query)
        trace.append(f"Classified as Fiber {primary_fiber} "
                    f"({'Language' if primary_fiber == 1 else 'Knowledge' if primary_fiber == 2 else 'Reasoning'})")

        # Step 2: Execute on primary fiber
        result = self._execute_fiber(primary_fiber, query, trace)

        # Step 3: ×3 Bridge — escalate if low confidence
        if result['confidence_level'] in ('REJECTED', 'UNKNOWN', 'LOW'):
            # Escalate: Fiber 1→2→3→1 (cyclic, mod 3)
            next_fiber = (primary_fiber % 3) + 1
            trace.append(f"×3 BRIDGE: Low confidence → escalate to Fiber {next_fiber}")
            self.stats['escalations'] += 1

            result2 = self._execute_fiber(next_fiber, query, trace)

            if result2['confidence_level'] not in ('REJECTED', 'UNKNOWN', 'LOW'):
                result = result2
                result['escalated'] = True
                result['fiber'] = next_fiber
            else:
                # Second escalation
                final_fiber = (next_fiber % 3) + 1
                if final_fiber != primary_fiber:
                    trace.append(f"×3 BRIDGE (2nd): → Fiber {final_fiber}")
                    result3 = self._execute_fiber(final_fiber, query, trace)
                    if result3['confidence_level'] not in ('REJECTED', 'UNKNOWN', 'LOW'):
                        result = result3
                        result['escalated'] = True
                        result['fiber'] = final_fiber

        # If ALL fibers failed → honest "I don't know"
        if result.get('answer') is None or result['confidence_level'] in ('REJECTED', 'UNKNOWN'):
            result['answer'] = "I don't have information about that topic."

        result['trace'] = trace
        return result

    def _execute_fiber(self, fiber: int, query: str, trace: list) -> Dict[str, Any]:
        """Execute query on a specific fiber's subsystem."""
        if fiber == 1:
            return self._execute_language(query, trace)
        elif fiber == 2:
            return self._execute_knowledge(query, trace)
        elif fiber == 3:
            return self._execute_reasoning(query, trace)
        return {'answer': None, 'confidence_level': 'UNKNOWN', 'fiber': fiber, 'escalated': False}

    def _execute_language(self, query: str, trace: list) -> Dict[str, Any]:
        """Fiber 1: PPM Language Model.

        CRITICAL RULE (David's directive):
        PPM is for FORMULATION of existing facts, NOT for inventing knowledge.
        If no facts exist about a topic → honestly say so.
        PPM NEVER fills knowledge gaps. Anti-hallucination guarantee dies
        the second PPM starts filling gaps.
        """
        self.stats['fiber1_calls'] += 1
        trace.append("  Fiber 1 (PPM): Checking for facts to reformulate...")

        # PPM only reformulates existing knowledge, never invents
        # If we landed here via escalation (no facts found), be honest
        return {'answer': "I don't have information about that topic.",
                'confidence_level': 'LOW', 'fiber': 1, 'escalated': False}

    def _execute_knowledge(self, query: str, trace: list) -> Dict[str, Any]:
        """Fiber 2: Hopfield Knowledge Store."""
        self.stats['fiber2_calls'] += 1

        if self.knowledge is None:
            return {'answer': None, 'confidence_level': 'UNKNOWN',
                    'fiber': 2, 'escalated': False}

        # Check if this is a "tell me about" / "describe" query → use Composer
        query_lower = query.lower().strip()
        is_about_query = any(query_lower.startswith(p) for p in (
            'tell me about', 'what do you know about', 'describe ',
            'compare ', 'what are the ',
        ))

        if is_about_query and self._composer is None:
            from .composer import ResponseComposer
            self._composer = ResponseComposer(knowledge_store=self.knowledge)

        if is_about_query and self._composer:
            trace.append("  Fiber 2 (Composer): Multi-fact response...")
            result = self._composer.compose(query)
            if result['answer']:
                trace.append(f"  Fiber 2: Composed {len(result['facts_used'])} facts")
                # Formulierer: PPM reformulates template output into fluent prose
                answer = self._reformulate(result, query, trace)
                return {'answer': answer,
                        'confidence_level': result['confidence_level'],
                        'fiber': 2, 'escalated': False}

        # Standard query: Dict-Index FIRST, Hopfield as fuzzy fallback
        subject, relation = self._parse_query_for_knowledge(query)
        trace.append(f"  Fiber 2: subject={subject}, relation={relation}")

        # PRIMARY: Dict-Index (O(1), exact, no hallucination)
        answer = None
        confidence = 'REJECTED'
        if subject:
            entity_facts = self.knowledge.find_by_entity(subject)
            if entity_facts:
                trace.append(f"  Fiber 2 (Dict-Index): {len(entity_facts)} facts for '{subject}'")
                # For identity queries, compose a multi-fact answer
                if relation == 'identity':
                    subj = entity_facts[0][0]
                    # Filter out meta-type facts that pollute answers
                    _META_TYPES = {'abbreviation_used_by_user', 'code_template',
                                   'grammar_pattern', 'user_vocabulary'}
                    parts = []
                    seen_parts = set()  # dedup by (relation, object_lower)
                    for s, r, o in entity_facts[:10]:
                        if r == 'type' and o in _META_TYPES:
                            continue
                        # Dedup: skip if we've already seen this (r, o)
                        # Also dedup by object value for is_a/type (same semantics)
                        dedup_key = (r.lower(), o.lower().strip())
                        if dedup_key in seen_parts:
                            continue
                        seen_parts.add(dedup_key)
                        # Cross-relation dedup for is_a and type (both mean "X is a Y")
                        if r.lower() in ('is_a', 'type'):
                            o_low = o.lower().strip()
                            obj_key = ('_isa_type', o_low)
                            if obj_key in seen_parts:
                                continue
                            # Also skip if o is a substring of an existing type
                            # (e.g., "molecule" is subsumed by "complex molecule")
                            skip = False
                            for k in list(seen_parts):
                                if k[0] == '_isa_type':
                                    if o_low in k[1] or k[1] in o_low:
                                        skip = True
                                        break
                            if skip:
                                continue
                            seen_parts.add(obj_key)
                        # Skip reflexive (DNA is_a DNA)
                        if o.lower().strip() == subj.lower().strip():
                            continue
                        if r == 'is_a':
                            parts.insert(0, f"{subj} is a {o}")
                        elif r in ('known_for', 'notable_for'):
                            parts.append(f"known for {o}")
                        elif r == 'creator':
                            parts.append(f"created by {o}")
                        elif r == 'birthplace':
                            parts.append(f"born in {o}")
                        elif r == 'born':
                            parts.append(f"born {o}")
                        elif r == 'definition':
                            parts.insert(0, f"{subj}: {o}")
                        elif r == 'type':
                            parts.insert(0, f"{subj} is a {o}")
                        else:
                            parts.append(f"{r}: {o}")
                    if parts:
                        answer = '. '.join(parts) + '.'
                        confidence = 'HIGH'
                else:
                    # Specific relation query
                    rel_lower = (relation or '').lower()
                    for s, r, o in entity_facts:
                        if r.lower() == rel_lower:
                            answer = o
                            confidence = 'HIGH'
                            trace.append(f"  Fiber 2 (exact rel): ({s}, {r}, {o})")
                            break
                    if not answer:
                        for s, r, o in entity_facts:
                            if rel_lower in r.lower() or r.lower() in rel_lower:
                                answer = o
                                confidence = 'MEDIUM'
                                trace.append(f"  Fiber 2 (partial rel): ({s}, {r}, {o})")
                                break

        # FALLBACK: Hopfield attractor (fuzzy, only if Dict-Index found nothing)
        if answer is None:
            trace.append(f"  Fiber 2 (Hopfield fallback): query({subject}, {relation})")
            result = self.knowledge.query(subject, relation)
            if result['fact']:
                # Anti-hallucination: verify the Hopfield answer has word overlap with query
                query_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
                fact_words = set(re.findall(r'\b[a-z]{3,}\b',
                                            f"{result['fact'][0]} {result['fact'][2]}".lower()))
                if query_words & fact_words:
                    answer = result['fact'][2]
                    confidence = result['confidence_level']
                    trace.append(f"  Fiber 2 (Hopfield): ({result['fact'][0]}, {result['fact'][1]}, {result['fact'][2]})")
                else:
                    trace.append(f"  Fiber 2 (Hopfield): REJECTED — no word overlap with query")

        return {'answer': answer, 'confidence_level': confidence,
                'fiber': 2, 'escalated': False}

    def _execute_reasoning(self, query: str, trace: list) -> Dict[str, Any]:
        """Fiber 3: Reasoning Engine."""
        self.stats['fiber3_calls'] += 1
        trace.append("  Fiber 3 (Reasoning): Analyzing...")

        if self.reasoning is None:
            # Auto-create reasoning engine with knowledge store if available
            from .reasoning import ReasoningEngine
            self.reasoning = ReasoningEngine(knowledge_store=self.knowledge)

        result = self.reasoning.reason(query)
        trace.extend(result.get('trace', []))
        return result

    def _parse_query_for_knowledge(self, query: str):
        """Extract subject and relation from a natural language query."""
        query_clean = query.rstrip('?').strip()

        # "What is the capital of France?"
        m = re.search(r'(?:what|who)\s+is\s+the\s+(\w+(?:\s+\w+)?)\s+of\s+(.+)', query_clean, re.I)
        if m:
            return m.group(2).strip(), m.group(1).strip()

        # "What is X known for?" → (X, known_for)
        m = re.search(r'what\s+is\s+(.+?)\s+known\s+for', query_clean, re.I)
        if m:
            return m.group(1).strip(), 'known_for'

        # "When did X start/begin?" → (X, start_year)
        m = re.search(r'when\s+did\s+(.+?)\s+(?:start|begin)', query_clean, re.I)
        if m:
            return m.group(1).strip(), 'start_year'

        # "When did X end/finish?" → (X, end_year)
        m = re.search(r'when\s+did\s+(.+?)\s+(?:end|finish)', query_clean, re.I)
        if m:
            return m.group(1).strip(), 'end_year'

        # "When was the X?" → (X, start_year) — events like "the Moon Landing"
        m = re.search(r'when\s+was\s+(?:the\s+)?(.+)', query_clean, re.I)
        if m:
            entity = m.group(1).strip()
            # Avoid matching "When was X born?" (handled below)
            if 'born' not in entity.lower():
                return entity, 'start_year'

        # "When was X discovered?" → (X, start_year)
        m = re.search(r'when\s+was\s+(.+?)\s+discovered', query_clean, re.I)
        if m:
            return m.group(1).strip(), 'start_year'

        # "Where is X located?"
        m = re.search(r'where\s+is\s+(.+?)(?:\s+located)?$', query_clean, re.I)
        if m:
            return m.group(1).strip(), 'location'

        # "Where was X born?" → (X, birthplace)
        m = re.search(r'where\s+was\s+(.+?)(?:\s+born)', query_clean, re.I)
        if m:
            return m.group(1).strip(), 'birthplace'

        # "When was X born?" → (X, born)
        m = re.search(r'when\s+was\s+(.+?)(?:\s+born)?$', query_clean, re.I)
        if m:
            return m.group(1).strip(), 'born'

        # "Who founded/invented/discovered/wrote/created X?"
        m = re.search(r'who\s+(?:founded|invented|discovered|wrote|created)\s+(.+)', query_clean, re.I)
        if m:
            verb = re.search(r'who\s+(\w+)', query_clean, re.I).group(1)
            return m.group(1).strip(), verb

        # "What did X invent/create/discover/write?" → reverse lookup
        m = re.search(r'what\s+did\s+(.+?)\s+(invent|create|discover|write|compose)', query_clean, re.I)
        if m:
            return m.group(1).strip(), 'known_for'

        # "How many X does Y have?" → (Y, X)
        m = re.search(r'how\s+many\s+(\w+)\s+does\s+(.+?)\s+have', query_clean, re.I)
        if m:
            return m.group(2).strip(), m.group(1).strip()

        # "How long does X live?" → (X, lifespan)
        m = re.search(r'how\s+long\s+does\s+(?:a(?:n)?\s+)?(.+?)\s+live', query_clean, re.I)
        if m:
            return m.group(1).strip(), 'lifespan'

        # "What is the diet of X?" / "What is the habitat of X?" already covered by "what is the X of Y"

        # "Is X a Y?" → (X, is_a) — taxonomy check (anchored to start)
        m = re.match(r'^is\s+(?:a(?:n)?\s+)?(.+?)\s+(?:a(?:n)?\s+)?(\w+)$', query_clean, re.I)
        if m:
            return m.group(1).strip(), 'is_a'

        # "What sport uses/has the X?" → (X, sport)
        m = re.search(r'what\s+sport\s+(?:uses|has)\s+(?:the\s+)?(.+)', query_clean, re.I)
        if m:
            return m.group(1).strip(), 'sport'

        # "What country/city is X in?" → (X, country/city)
        m = re.search(r'what\s+(\w+)\s+is\s+(.+?)\s+in$', query_clean, re.I)
        if m:
            return m.group(2).strip(), m.group(1).strip()

        # "What/Who is X?" (identity/type query)
        m = re.search(r'(?:what|who)\s+is\s+(?:a(?:n)?\s+)?(.+)', query_clean, re.I)
        if m:
            return m.group(1).strip(), 'identity'

        # Fallback: first capitalized word as subject
        # Skip leading verbs that are capitalized only because they start the sentence
        _SKIP_VERBS = {'tell', 'name', 'list', 'describe', 'write', 'create',
                       'explain', 'show', 'give', 'find', 'say', 'identify'}
        m = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', query_clean)
        if m:
            candidate = m.group(1)
            if candidate.lower() in _SKIP_VERBS:
                # Try next capitalized word instead
                rest = query_clean[m.end():]
                m2 = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', rest)
                if m2:
                    return m2.group(1), None
            else:
                return candidate, None

        return query_clean, None

    def _reformulate(self, composer_result: Dict, query: str, trace: list) -> str:
        """Apply Formulierer (PPM reformulation) to Composer output."""
        if self.language is None:
            # No PPM loaded → return template output as-is
            return composer_result['answer']

        if self._formulierer is None:
            from .formulierer import Formulierer
            self._formulierer = Formulierer(lm_model=self.language)

        result = self._formulierer.reformulate_composer_output(
            composer_result, query=query
        )
        trace.append(f"  Fiber 1 (Formulierer): {result.get('method', '?')}, "
                    f"verified={result.get('verified', '?')}")
        return result.get('answer', composer_result['answer'])

    def get_stats(self) -> Dict[str, Any]:
        """Return routing statistics."""
        total = max(self.stats['total_queries'], 1)
        return {
            **self.stats,
            'fiber1_pct': self.stats['fiber1_calls'] / total * 100,
            'fiber2_pct': self.stats['fiber2_calls'] / total * 100,
            'fiber3_pct': self.stats['fiber3_calls'] / total * 100,
            'escalation_rate': self.stats['escalations'] / total * 100,
        }
