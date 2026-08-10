"""
Answer Naturalizer — Transform Robotic Answers to Natural Language
===================================================================
Converts structured/robotic responses into human-readable text.

Examples:
  "ice has_property cold" → "Ice is cold."
  "(France, capital, Paris)" → "The capital of France is Paris."
  Properties list → "France is a country in Europe. Its capital is Paris."

This is the FINAL stage before output — everything passes through here.
"""

import re
from typing import Dict, Any, List, Tuple, Optional


# Relation → natural phrasing templates
RELATION_TEMPLATES = {
    'has_property': '{subject} is {object}.',
    'is_a': '{subject} is a {object}.',
    'capital': 'The capital of {subject} is {object}.',
    'language': 'The main language of {subject} is {object}.',
    'continent': '{subject} is located in {object}.',
    'currency': 'The currency of {subject} is {object}.',
    'type': '{subject} is a {object}.',
    'creator': '{subject} was created by {object}.',
    'discoverer': '{subject} was discovered/invented by {object}.',
    'year': '{subject} dates from {object}.',
    'part_of': '{subject} is part of {object}.',
    'used_for': '{subject} is used for {object}.',
    'can': '{subject} can {object}.',
    'causes': '{subject} causes {object}.',
    'needs': '{subject} needs {object}.',
    'found_in': '{subject} is found in {object}.',
    'made_of': '{subject} is made of {object}.',
    'opposite_of': '{subject} is the opposite of {object}.',
    'synonym': '{subject} is similar to {object}.',
    'symbol': 'The symbol of {subject} is {object}.',
    'atomic_number': '{subject} has atomic number {object}.',
    'population': '{subject} has a population of {object}.',
    'order_from_sun': '{subject} is planet number {object} from the Sun.',
    'complexity': 'The complexity of {subject} is {object}.',
    'description': '{subject} is {object}.',
    'has_part': '{subject} has {object}.',
    'requires': '{subject} requires {object}.',
    'happens_before': '{subject} comes before {object}.',
    'related_to': '{subject} is related to {object}.',
    'location': '{subject} is located in {object}.',
    'comparison': '{subject} is comparable to {object}.',
}

# Relations to prioritize in multi-property answers
PRIORITY_RELATIONS = [
    'type', 'is_a', 'description', 'capital', 'creator', 'discoverer',
    'language', 'continent', 'currency', 'year', 'has_property',
    'used_for', 'part_of', 'symbol', 'atomic_number',
]


class AnswerNaturalizer:
    """Transform structured answers into natural language."""

    def naturalize_properties(self, concept: str,
                               properties: List[Tuple[str, str]],
                               max_sentences: int = 5) -> str:
        """
        Convert a property list into natural text.

        Args:
            concept: The subject entity
            properties: List of (relation, object) tuples
            max_sentences: Maximum sentences to generate
        """
        if not properties:
            return f"I know about {concept}, but don't have detailed information."

        concept_title = concept.replace('_', ' ').title()

        # Sort properties by priority
        sorted_props = sorted(
            properties,
            key=lambda p: PRIORITY_RELATIONS.index(p[0]) if p[0] in PRIORITY_RELATIONS else 99
        )

        sentences = []
        used_relations = set()
        seen_facts = set()

        for relation, obj in sorted_props[:max_sentences * 2]:  # over-fetch for dedup
            if len(sentences) >= max_sentences:
                break
            # Dedup by (relation, object) and skip reflexive
            fact_key = (relation.lower(), obj.lower().strip())
            if fact_key in seen_facts:
                continue
            seen_facts.add(fact_key)
            # Skip reflexive assertions (X is_a X)
            if obj.lower().strip() == concept.lower().strip():
                continue
            if relation in used_relations:
                continue
            used_relations.add(relation)

            obj_clean = obj.replace('_', ' ')
            # Capitalize proper nouns (names, places, languages)
            proper_relations = {'capital', 'language', 'continent', 'currency',
                                'creator', 'discoverer', 'symbol'}
            if relation in proper_relations:
                obj_clean = obj_clean.title()

            template = RELATION_TEMPLATES.get(relation)
            if template:
                sentence = template.format(
                    subject=concept_title,
                    object=obj_clean
                )
            else:
                sentence = f"{concept_title} {relation.replace('_', ' ')} {obj_clean}."

            sentences.append(sentence)

        if not sentences:
            return f"I know about {concept_title}."

        # Combine: First sentence standalone, rest connected
        text = sentences[0]
        for i, sent in enumerate(sentences[1:], 1):
            # Use pronouns for subsequent sentences about same subject
            if sent.startswith(concept_title):
                # Replace subject with pronoun for natural flow
                sent = sent.replace(concept_title + ' is ', 'It is ', 1)
                sent = sent.replace(concept_title + ' was ', 'It was ', 1)
                sent = sent.replace(concept_title + ' has ', 'It has ', 1)
                sent = sent.replace(concept_title + ' can ', 'It can ', 1)
                sent = sent.replace(f'The capital of {concept_title}',
                                    'Its capital', 1)
                sent = sent.replace(f'The currency of {concept_title}',
                                    'Its currency', 1)
                sent = sent.replace(f'The main language of {concept_title}',
                                    'Its main language', 1)
            text += ' ' + sent

        return text

    def naturalize_cs_answer(self, cs_result: Dict[str, Any],
                             question: str = '') -> str:
        """
        Naturalize a CommonSenseEngine query result.

        Handles all result types: boolean, property lookup, about.
        If question is provided, tries to answer specifically.
        """
        if not cs_result.get('found'):
            return ''

        answer = cs_result.get('answer')

        # Boolean answers (is ice cold? / can birds fly?)
        if isinstance(answer, bool):
            explanation = cs_result.get('explanation', '')
            if answer:
                if explanation:
                    return self._naturalize_explanation(explanation, positive=True)
                return "Yes."
            else:
                if explanation:
                    return self._naturalize_explanation(explanation, positive=False)
                return "No."

        # String answer (what is X used for? → "cutting")
        if isinstance(answer, str) and answer:
            return answer

        # Property list (about query) — try to match specific question
        if cs_result.get('properties'):
            concept = cs_result.get('concept', '?')
            props = cs_result['properties']

            # If we have a specific question, try to extract the relevant property
            if question:
                specific = self._answer_specific_property(concept, props, question)
                if specific:
                    return specific

            return self.naturalize_properties(concept, props)

        # Fallback
        if answer is not None:
            return str(answer)

        return ''

    def naturalize_fact(self, subject: str, relation: str, obj: str) -> str:
        """Naturalize a single (S, R, O) triple."""
        subj = subject.replace('_', ' ').title()
        obj_clean = obj.replace('_', ' ').title()

        template = RELATION_TEMPLATES.get(relation)
        if template:
            return template.format(subject=subj, object=obj_clean)
        return f"{subj} {relation.replace('_', ' ')} {obj_clean}."

    def naturalize_math(self, query: str, result: str) -> str:
        """Naturalize a math answer."""
        result_str = str(result)

        # "is X prime?" → "Yes, X is prime." / "No, X is not prime."
        if 'prime' in query.lower():
            if 'is prime' in result_str.lower():
                return f"Yes, {result_str}."
            elif 'is not prime' in result_str.lower():
                return f"No, {result_str}."

        # "GCD of X and Y" → "The GCD of X and Y is Z."
        m = re.search(r'(?:gcd|GCD)\s+(?:of\s+)?(\d+)\s+(?:and\s+)?(\d+)', query)
        if m:
            return f"The greatest common divisor of {m.group(1)} and {m.group(2)} is {result_str}."

        # "Calculate X" → "The result is Z."
        if re.match(r'(?:calculate|compute|what is)\s+', query, re.I):
            return f"{result_str}"

        # "Solve ..." → keep as-is (usually already formatted)
        if 'solve' in query.lower():
            return result_str

        return result_str

    def _naturalize_explanation(self, explanation: str, positive: bool) -> str:
        """Convert an explanation like 'ice has_property cold' to natural text."""
        # "ice has_property cold" → "Yes, ice is cold."
        # "dog is a type of animal" → "Yes, a dog is a type of animal."
        # "dog cannot fly" → "No, dogs cannot fly."

        exp = explanation.strip()

        if 'has_property' in exp:
            m = re.match(r'(\w+)\s+has_property\s+(.+)', exp)
            if m:
                subj, prop = m.group(1), m.group(2)
                prefix = "Yes" if positive else "No"
                return f"{prefix}, {subj} is {prop}."

        if 'is a type of' in exp:
            prefix = "Yes" if positive else "No"
            return f"{prefix}, {exp.replace('is a type of', 'is a type of')}."

        if 'cannot' in exp:
            return f"No, {exp}."

        if ' can ' in exp:
            prefix = "Yes" if positive else "No"
            # Clean up "(as a bird)" patterns
            exp = re.sub(r'\s*\(as a \w+\)\s*', ' ', exp)
            return f"{prefix}, {exp.strip()}."

        prefix = "Yes" if positive else "No"
        return f"{prefix}. {exp}"

    def _answer_specific_property(self, concept: str, properties: List[Tuple[str, str]],
                                   question: str) -> str:
        """Try to answer a specific question from a property list."""
        q = question.lower()
        concept_title = concept.replace('_', ' ').title()

        # Map question keywords to relation types
        keyword_to_relation = {
            'capital': 'capital',
            'language': 'language',
            'continent': 'continent',
            'located': 'continent',
            'currency': 'currency',
            'created': 'creator',
            'invented': 'discoverer',
            'discovered': 'discoverer',
            'made of': 'made_of',
            'used for': 'used_for',
            'type': 'type',
            'symbol': 'symbol',
            'population': 'population',
        }

        for keyword, relation in keyword_to_relation.items():
            if keyword in q:
                for rel, obj in properties:
                    if rel == relation:
                        obj_clean = obj.replace('_', ' ').title()
                        template = RELATION_TEMPLATES.get(relation)
                        if template:
                            return template.format(subject=concept_title, object=obj_clean)
                        return f"{concept_title} {relation.replace('_', ' ')} {obj_clean}."
        return ''

    def naturalize_multi_hop(self, chain: List[Tuple[str, str, str]]) -> str:
        """
        Naturalize a reasoning chain.

        chain: [(subject, relation, object), ...]
        """
        if not chain:
            return ''

        sentences = []
        for s, r, o in chain:
            sentences.append(self.naturalize_fact(s, r, o))

        if len(sentences) == 1:
            return sentences[0]

        # Connect with "Therefore" for the conclusion
        text = sentences[0]
        for i, sent in enumerate(sentences[1:]):
            if i == len(sentences) - 2:
                text += f" Therefore, {sent[0].lower()}{sent[1:]}"
            else:
                text += f" {sent}"
        return text
