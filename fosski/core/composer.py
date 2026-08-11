"""
Response Composer — Hierarchische Antwort-Generierung
=====================================================
Das fehlende Stück: Vom Einzelfakt zur kohärenten Antwort.

Pipeline:
  1. Multi-Fact Retrieval: Alle relevanten Fakten zu einer Entity sammeln
  2. Fact Ranking: Relevanteste Fakten zuerst (basierend auf Query)
  3. Template Selection: Passende Satzstruktur wählen
  4. Composition: Fakten zu natürlichem Text zusammenbauen

Beispiel:
  Query: "Tell me about Germany"
  Fakten: [(Germany, capital, Berlin), (Germany, borders, France),
           (Germany, borders, Denmark), (Germany, population, 83M)]
  Output: "Germany is a country with its capital in Berlin. It has a
           population of 83 million. Germany borders France and Denmark."
"""

import re
from typing import Dict, Any, List, Tuple, Optional


class ResponseComposer:
    """
    Composes multi-fact natural language responses.

    Not a language model — a fact-to-text engine.
    Every sentence is grounded in a stored fact.
    No hallucination possible by construction.
    """

    # Relation → sentence template
    # {s} = subject, {o} = object, {r} = relation
    TEMPLATES = {
        'capital': "{o} is the capital of {s}.",
        'population': "{s} has a population of {o}.",
        'language': "The official language of {s} is {o}.",
        'borders': "{s} borders {o}.",
        'location': "{s} is located in {o}.",
        'birthplace': "{s} was born in {o}.",
        'founder': "{s} was founded by {o}.",
        'creator': "{s} was created by {o}.",
        'discoverer': "{s} was discovered by {o}.",
        'inventor': "{s} was invented by {o}.",
        'author': "{s} was written by {o}.",
        'type': "{s} is a {o}.",
        'identity': "{s} is a {o}.",
        'known_as': "{s} is also known as {o}.",
        'description': "{s} is {o}.",
        'founded': "{s} was founded in {o}.",
        'formula': "The formula of {s} is {o}.",
        'symbol': "The symbol of {s} is {o}.",
        'part_of': "{s} is part of {o}.",
        'currency': "The currency of {s} is {o}.",
        'born': "{s} was born in {o}.",
        'died': "{s} died in {o}.",
        'nationality': "{s} is {o}.",
        'occupation': "{s} is a {o}.",
        'country': "{s} is in {o}.",
        'subclass_of': "{s} is a type of {o}.",
        'capital_of': "{s} is the capital of {o}.",
        'deathplace': "{s} died in {o}.",
        'leader': "The leader of {s} is {o}.",
    }

    # Default template for unknown relations
    DEFAULT_TEMPLATE = "The {r} of {s} is {o}."

    # Relations to prioritize in "tell me about" queries
    PRIORITY_ORDER = [
        'type', 'identity', 'occupation', 'capital', 'population', 'language',
        'location', 'birthplace', 'born', 'nationality', 'founded', 'borders',
        'creator', 'discoverer', 'inventor', 'author',
        'formula', 'symbol', 'known_as', 'description',
        'part_of', 'currency', 'founder', 'leader',
    ]

    def __init__(self, knowledge_store=None, reasoning_engine=None):
        self.knowledge = knowledge_store
        self.reasoning = reasoning_engine

    def compose(self, query: str) -> Dict[str, Any]:
        """
        Generate a multi-fact response for a query.

        Returns:
            dict with:
                answer: composed text
                facts_used: list of (S, R, O) tuples
                confidence_level: overall confidence
                trace: composition steps
        """
        trace = []

        # Determine query type
        query_type, entity, relation = self._parse_query(query)
        trace.append(f"Query type: {query_type}, entity: {entity}, relation: {relation}")

        if query_type == 'specific':
            return self._compose_specific(entity, relation, trace)
        elif query_type == 'about':
            return self._compose_about(entity, trace)
        elif query_type == 'compare':
            entities = entity.split(' and ')
            if len(entities) == 2:
                return self._compose_compare(entities[0].strip(),
                                            entities[1].strip(), trace)
        elif query_type == 'list':
            return self._compose_list(entity, relation, trace)

        # Fallback: try as specific, then about
        result = self._compose_specific(entity, relation, trace)
        if result['facts_used']:
            return result
        return self._compose_about(entity, trace)

    def _parse_query(self, query: str):
        """Classify query and extract entity/relation."""
        q = query.lower().strip().rstrip('?.')

        # "Tell me about X" / "What do you know about X"
        m = re.search(r'(?:tell\s+me\s+about|what\s+(?:do\s+you\s+)?know\s+about|'
                      r'describe|explain)\s+(.+)', q)
        if m:
            return 'about', m.group(1).strip().title(), None

        # "Compare X and Y"
        m = re.search(r'compare\s+(.+?)\s+and\s+(.+)', q)
        if m:
            a = m.group(1).strip().title()
            b = m.group(2).strip().title()
            return 'compare', f"{a} and {b}", None

        # "What is the capital of X?"
        m = re.search(r'what\s+is\s+the\s+(\w+)\s+of\s+(.+)', q)
        if m:
            return 'specific', m.group(2).strip().title(), m.group(1).strip()

        # "Who invented X?" / "Who discovered X?"
        m = re.search(r'who\s+(\w+)\s+(.+)', q)
        if m:
            verb = m.group(1).strip()
            entity = m.group(2).strip().title()
            # Map verb to relation
            verb_map = {
                'invented': 'inventor', 'discovered': 'discoverer',
                'created': 'creator', 'wrote': 'author',
                'founded': 'founder',
            }
            return 'specific', entity, verb_map.get(verb, verb)

        # "Where is X?"
        m = re.search(r'where\s+is\s+(.+)', q)
        if m:
            return 'specific', m.group(1).strip().title(), 'location'

        # "List all X of Y" or "What are the borders of X"
        m = re.search(r'(?:list|what\s+are)\s+(?:the\s+)?(\w+)\s+of\s+(.+)', q)
        if m:
            return 'list', m.group(2).strip().title(), m.group(1).strip()

        # Fallback: treat whole query as entity
        # Extract first capitalized phrase
        m = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', query)
        if m:
            return 'about', m.group(1), None

        return 'about', query.strip().title(), None

    def _compose_specific(self, entity: str, relation: str,
                         trace: list) -> Dict[str, Any]:
        """Answer a specific fact query."""
        if not self.knowledge:
            return self._empty_result(trace, "No knowledge store")

        result = self.knowledge.query(entity, relation)
        trace.append(f"Query: ({entity}, {relation}) → {result['confidence_level']}")

        if result['fact']:
            s, r, o = result['fact']
            sentence = self._fact_to_sentence(s, r, o)
            return {
                'answer': sentence,
                'facts_used': [result['fact']],
                'confidence_level': result['confidence_level'],
                'trace': trace,
            }

        # Try reverse lookup: entity might be the object
        if relation:
            entity_lower = entity.lower()
            for s, r, o in self.knowledge.facts:
                if o.lower() == entity_lower and r == relation:
                    sentence = self._fact_to_sentence(s, r, o)
                    trace.append(f"Found via reverse lookup: ({s}, {r}, {o})")
                    return {
                        'answer': sentence,
                        'facts_used': [(s, r, o)],
                        'confidence_level': 'MEDIUM',
                        'trace': trace,
                    }

        return self._empty_result(trace, f"No fact found for ({entity}, {relation})")

    def _compose_about(self, entity: str, trace: list) -> Dict[str, Any]:
        """Compose a multi-fact response about an entity."""
        if not self.knowledge:
            return self._empty_result(trace, "No knowledge store")

        trace.append(f"Collecting all facts about: {entity}")

        # Collect all facts where entity is subject
        facts = self._find_all_facts(entity)
        trace.append(f"Found {len(facts)} related facts")

        if not facts:
            # Try case-insensitive
            facts = self._find_all_facts(entity.lower())
            if not facts:
                return self._empty_result(trace, f"No facts about {entity}")

        # Rank by priority
        ranked = self._rank_facts(facts, entity)
        trace.append(f"Ranked: {[f'{s},{r},{o}' for s,r,o in ranked[:5]]}")

        # Compose sentences
        sentences = []
        used_relations = set()
        facts_used = []

        # Group type/identity/description as one category
        type_rels = {'type', 'identity', 'description'}

        # Track which (relation, value) pairs we've seen to skip redundant reverses
        # e.g., if we have (Denmark, borders, Germany), skip (Germany, borders, Denmark)
        covered_pairs = set()

        used_objects = set()

        for s, r, o in ranked:
            # Skip duplicate relations (e.g., multiple "type" entries)
            if r in type_rels:
                if type_rels & used_relations:
                    continue
                # Skip trivial type facts (e.g., "Germany is German")
                if len(o.split()) <= 2 and o.lower()[:4] == s.lower()[:4]:
                    continue
            elif r in used_relations and r not in ('borders', 'language'):
                continue

            # Skip if same object already used via a different relation
            # (e.g., creator=X and artist=X are redundant)
            o_key = o.lower().strip()
            if o_key in used_objects and r not in type_rels:
                continue
            used_objects.add(o_key)

            # Skip redundant reverse facts
            # If (Denmark, borders, Germany) is already covered,
            # skip (Germany, borders, Denmark)
            pair_key = tuple(sorted([s.lower(), o.lower()])) + (r,)
            if pair_key in covered_pairs:
                continue
            covered_pairs.add(pair_key)

            used_relations.add(r)

            sentence = self._fact_to_sentence(s, r, o)
            sentences.append(sentence)
            facts_used.append((s, r, o))

            # Limit to reasonable length
            if len(sentences) >= 8:
                break

        if not sentences:
            return self._empty_result(trace, "Could not compose sentences")

        # Merge border facts: "Germany borders France, Denmark, and Poland."
        sentences = self._merge_list_facts(sentences, facts_used)

        # Final dedup: remove duplicate sentences (case-insensitive)
        seen_sentences = set()
        unique_sentences = []
        for s in sentences:
            key = s.lower().strip()
            if key not in seen_sentences:
                seen_sentences.add(key)
                unique_sentences.append(s)
        sentences = unique_sentences

        answer = " ".join(sentences)
        trace.append(f"Composed {len(sentences)} sentences")

        return {
            'answer': answer,
            'facts_used': facts_used,
            'confidence_level': 'HIGH' if len(facts_used) >= 3 else 'MEDIUM',
            'trace': trace,
        }

    def _compose_compare(self, entity_a: str, entity_b: str,
                        trace: list) -> Dict[str, Any]:
        """Compare two entities by their shared relations."""
        if not self.knowledge:
            return self._empty_result(trace, "No knowledge store")

        facts_a = self._find_all_facts(entity_a)
        facts_b = self._find_all_facts(entity_b)

        trace.append(f"Comparing: {entity_a} ({len(facts_a)} facts) vs "
                    f"{entity_b} ({len(facts_b)} facts)")

        # Find shared relations
        rels_a = {r: (s, o) for s, r, o in facts_a}
        rels_b = {r: (s, o) for s, r, o in facts_b}
        shared = set(rels_a.keys()) & set(rels_b.keys())

        sentences = []
        facts_used = []

        for rel in self.PRIORITY_ORDER:
            if rel in shared:
                sa, oa = rels_a[rel]
                sb, ob = rels_b[rel]
                template = self.TEMPLATES.get(rel, self.DEFAULT_TEMPLATE)
                sent_a = template.format(s=entity_a, r=rel, o=oa)
                sent_b = template.format(s=entity_b, r=rel, o=ob)

                if oa == ob:
                    sentences.append(f"Both {entity_a} and {entity_b}: {sent_a}")
                else:
                    sentences.append(f"{sent_a} {sent_b}")

                facts_used.append((sa, rel, oa))
                facts_used.append((sb, rel, ob))

                if len(sentences) >= 5:
                    break

        if not sentences:
            return self._empty_result(trace, "No shared relations to compare")

        answer = " ".join(sentences)
        return {
            'answer': answer,
            'facts_used': facts_used,
            'confidence_level': 'MEDIUM',
            'trace': trace,
        }

    def _compose_list(self, entity: str, relation: str,
                     trace: list) -> Dict[str, Any]:
        """List all values for a specific relation."""
        if not self.knowledge:
            return self._empty_result(trace, "No knowledge store")

        # Find all facts with this entity and relation
        matches = []
        entity_lower = entity.lower()
        rel_lower = relation.lower() if relation else None

        for s, r, o in self.knowledge.facts:
            if s.lower() == entity_lower:
                if rel_lower is None or r.lower() == rel_lower:
                    matches.append((s, r, o))

        if not matches:
            return self._empty_result(trace, f"No {relation} facts for {entity}")

        objects = [o for _, _, o in matches]
        if len(objects) == 1:
            answer = f"The {relation} of {entity} is {objects[0]}."
        else:
            last = objects[-1]
            rest = ", ".join(objects[:-1])
            answer = f"The {relation}s of {entity} are {rest}, and {last}."

        return {
            'answer': answer,
            'facts_used': matches,
            'confidence_level': 'HIGH',
            'trace': trace,
        }

    def _find_all_facts(self, entity: str) -> List[Tuple[str, str, str]]:
        """Find all facts related to an entity (as subject or object).

        Uses fuzzy matching: "Einstein" matches "Albert Einstein",
        "Shakespeare" matches "William Shakespeare", etc.
        Exact match is preferred; partial match only on word boundaries.
        """
        if not self.knowledge:
            return []

        entity_lower = entity.lower()
        facts = []
        seen = set()

        # Phase 1: exact match (preferred)
        for s, r, o in self.knowledge.facts:
            if s.lower() == entity_lower:
                key = (s.lower(), r, o.lower())
                if key not in seen:
                    seen.add(key)
                    facts.append((s, r, o))
            elif o.lower() == entity_lower:
                key = (s.lower(), r, o.lower())
                if key not in seen:
                    seen.add(key)
                    facts.append((s, r, o))

        if facts:
            return facts

        # Phase 2: fuzzy match — entity appears as a complete word in subject
        # "Einstein" matches "Albert Einstein" but NOT "Einsteinium"
        import re
        pattern = re.compile(r'\b' + re.escape(entity_lower) + r'\b', re.I)
        for s, r, o in self.knowledge.facts:
            if pattern.search(s):
                key = (s.lower(), r, o.lower())
                if key not in seen:
                    seen.add(key)
                    facts.append((s, r, o))
            elif pattern.search(o):
                key = (s.lower(), r, o.lower())
                if key not in seen:
                    seen.add(key)
                    facts.append((s, r, o))

        return facts

    def _rank_facts(self, facts: List[Tuple], entity: str) -> List[Tuple]:
        """Rank facts by importance/priority."""
        def priority(fact):
            s, r, o = fact
            try:
                idx = self.PRIORITY_ORDER.index(r)
            except ValueError:
                idx = len(self.PRIORITY_ORDER)
            # Boost facts where entity is subject
            if s.lower() == entity.lower():
                idx -= 100
            return idx

        return sorted(facts, key=priority)

    def _fact_to_sentence(self, s: str, r: str, o: str) -> str:
        """Convert a single fact to a natural language sentence."""
        template = self.TEMPLATES.get(r, self.DEFAULT_TEMPLATE)
        return template.format(s=s, r=r, o=o)

    def _merge_list_facts(self, sentences: List[str],
                         facts: List[Tuple]) -> List[str]:
        """Merge facts with same relation into lists.

        "Germany borders France. Germany borders Denmark."
        → "Germany borders France and Denmark."
        """
        # Group by (subject, relation)
        groups = {}
        for i, (s, r, o) in enumerate(facts):
            key = (s.lower(), r)
            if key not in groups:
                groups[key] = []
            groups[key].append((i, o))

        # Find groups with multiple entries
        merged = []
        skip = set()
        for key, entries in groups.items():
            if len(entries) > 1:
                s_lower, r = key
                objects = [o for _, o in entries]
                # Find actual subject from first fact
                s = facts[entries[0][0]][0]
                if len(objects) == 2:
                    obj_str = f"{objects[0]} and {objects[1]}"
                else:
                    obj_str = ", ".join(objects[:-1]) + f", and {objects[-1]}"

                template = self.TEMPLATES.get(r, self.DEFAULT_TEMPLATE)
                merged_sent = template.format(s=s, r=r, o=obj_str)
                merged.append(merged_sent)
                for idx, _ in entries:
                    skip.add(idx)

        # Add non-merged sentences
        result = []
        for i, sent in enumerate(sentences):
            if i not in skip:
                result.append(sent)
        result.extend(merged)

        return result

    @staticmethod
    def _empty_result(trace: list, reason: str) -> Dict[str, Any]:
        trace.append(f"EMPTY: {reason}")
        return {
            'answer': None,
            'facts_used': [],
            'confidence_level': 'UNKNOWN',
            'trace': trace,
        }
