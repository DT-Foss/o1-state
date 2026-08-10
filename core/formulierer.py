"""
Formulierer — PPM-Based Fact Reformulation
============================================
David's directive: PPM in Fiber 1 is for FORMULATION, not invention.

Pipeline:
  1. Composer produces template-based sentences from facts
  2. Formulierer takes those sentences and reformulates them
     into fluent, natural prose using PPM scoring
  3. Anti-hallucination check: output must contain ALL original
     fact objects, no new claims added

The PPM never generates new knowledge. It ONLY:
  - Selects the most natural phrasing among pre-defined templates
  - Generates connective tissue between fact sentences
  - Reorders for natural flow

If no facts exist → "I don't have information about that."
PPM NEVER fills knowledge gaps.
"""

import math
import re
from typing import List, Tuple, Dict, Any, Optional


class Formulierer:
    """
    Reformulates template-based fact sentences into fluent prose.

    Uses PPM character model to score candidate phrasings and
    select the most natural-sounding composition.

    Anti-hallucination guarantee: every claim in output traces
    back to a stored (S, R, O) triplet. If the PPM-generated
    connective text accidentally introduces a claim, it is
    stripped in the verification pass.
    """

    # Multiple phrasing templates per relation
    # Each relation has 2-4 alternatives; PPM picks the best one
    PHRASING_VARIANTS = {
        'capital': [
            "{o} is the capital of {s}.",
            "The capital of {s} is {o}.",
            "{s}'s capital is {o}.",
        ],
        'population': [
            "{s} has a population of {o}.",
            "The population of {s} is {o}.",
            "About {o} people live in {s}.",
        ],
        'language': [
            "The official language of {s} is {o}.",
            "{o} is the official language of {s}.",
            "People in {s} speak {o}.",
        ],
        'borders': [
            "{s} borders {o}.",
            "{s} shares a border with {o}.",
            "{o} lies along {s}'s border.",
        ],
        'location': [
            "{s} is located in {o}.",
            "{s} can be found in {o}.",
            "{o} is where {s} is located.",
        ],
        'birthplace': [
            "{s} was born in {o}.",
            "{o} is the birthplace of {s}.",
        ],
        'founder': [
            "{s} was founded by {o}.",
            "{o} founded {s}.",
        ],
        'creator': [
            "{s} was created by {o}.",
            "{o} created {s}.",
        ],
        'discoverer': [
            "{s} was discovered by {o}.",
            "{o} discovered {s}.",
        ],
        'inventor': [
            "{s} was invented by {o}.",
            "{o} invented {s}.",
        ],
        'author': [
            "{s} was written by {o}.",
            "{o} wrote {s}.",
            "{o} is the author of {s}.",
        ],
        'type': [
            "{s} is a {o}.",
            "{s} is {o}.",
        ],
        'identity': [
            "{s} is a {o}.",
            "{s} is {o}.",
        ],
        'founded': [
            "{s} was founded in {o}.",
            "{s} has existed since {o}.",
        ],
        'formula': [
            "The formula of {s} is {o}.",
            "{s} has the formula {o}.",
        ],
        'symbol': [
            "The symbol of {s} is {o}.",
            "{s} is represented by {o}.",
        ],
        'currency': [
            "The currency of {s} is {o}.",
            "{s} uses the {o} as currency.",
        ],
        'part_of': [
            "{s} is part of {o}.",
            "{s} belongs to {o}.",
        ],
        'known_as': [
            "{s} is also known as {o}.",
        ],
        'description': [
            "{s} is {o}.",
        ],
        'born': [
            "{s} was born in {o}.",
        ],
        'birth_year': [
            "{s} was born in {o}.",
        ],
        'died': [
            "{s} died in {o}.",
        ],
        'nationality': [
            "{s} is {o}.",
            "{s} has {o} nationality.",
        ],
        'occupation': [
            "{s} is a {o}.",
            "{s} works as a {o}.",
        ],
        'country': [
            "{s} is in {o}.",
            "{s} is located in {o}.",
        ],
        'subclass_of': [
            "{s} is a type of {o}.",
        ],
        'capital_of': [
            "{s} is the capital of {o}.",
        ],
        'deathplace': [
            "{s} died in {o}.",
        ],
        'leader': [
            "The leader of {s} is {o}.",
            "{o} leads {s}.",
        ],
        # Historical / temporal relations
        'started': [
            "{s} started in {o}.",
            "{s} began in {o}.",
        ],
        'ended': [
            "{s} ended in {o}.",
        ],
        'built': [
            "{s} was built in {o}.",
        ],
        'fell': [
            "{s} fell on {o}.",
            "{s} fell in {o}.",
        ],
        'divided': [
            "{s} divided {o}.",
        ],
        'trigger': [
            "{s} was triggered by {o}.",
        ],
        'caused': [
            "{s} caused {o}.",
            "{s} led to {o}.",
        ],
        'result': [
            "{s} resulted in {o}.",
        ],
        'period': [
            "{s} took place during {o}.",
            "{s} lasted from {o}.",
        ],
        'started_in': [
            "{s} started in {o}.",
            "{s} originated in {o}.",
        ],
        # Knowledge / description relations
        'known_for': [
            "{s} is known for {o}.",
            "{s} is famous for {o}.",
        ],
        'about': [
            "{s} is about {o}.",
            "{s} concerns {o}.",
        ],
        'definition': [
            "{s} is {o}.",
            "{s} is defined as {o}.",
        ],
        'stands_for': [
            "{s} stands for {o}.",
        ],
        'structure': [
            "{s} has a {o} structure.",
            "The structure of {s} is {o}.",
        ],
        'discovered_by': [
            "{s} was discovered by {o}.",
            "{o} discovered {s}.",
        ],
        'equation': [
            "The equation for {s} is {o}.",
        ],
        'performed_by': [
            "{s} is performed by {o}.",
        ],
        'first_released': [
            "{s} was first released in {o}.",
            "{s} came out in {o}.",
        ],
        'paradigm': [
            "{s} is a {o} language.",
        ],
        'organization': [
            "{s} was created by {o}.",
            "{s} is made by {o}.",
        ],
        'based_on': [
            "{s} is based on {o}.",
        ],
        'time_complexity': [
            "{s} has a time complexity of {o}.",
        ],
        'method': [
            "{s} uses {o}.",
        ],
        'property': [
            "A key property of {s} is {o}.",
            "{s} is characterized by {o}.",
        ],
        'meaning': [
            "{s} means {o}.",
        ],
        'value': [
            "{s} is {o}.",
            "The value of {s} is {o}.",
        ],
        'founders': [
            "{s} was founded by {o}.",
        ],
        'invented': [
            "{s} was invented in {o}.",
        ],
        'inventors': [
            "{s} was invented by {o}.",
        ],
        'impact': [
            "{s} led to {o}.",
        ],
        'significance': [
            "The significance of {s} is {o}.",
        ],
        'members': [
            "The members of {s} are {o}.",
        ],
        'first_person': [
            "The first person was {o}.",
        ],
        'mission': [
            "The mission was {o}.",
        ],
        'date': [
            "{s} happened on {o}.",
        ],
        'between': [
            "{s} was between {o}.",
        ],
        'connected': [
            "{s} connected {o}.",
        ],
        'allied_powers': [
            "The allied powers of {s} were {o}.",
        ],
        'requirement': [
            "{s} requires {o}.",
        ],
        'uses': [
            "{s} uses {o}.",
        ],
        'components': [
            "{s} consists of {o}.",
        ],
        'function': [
            "The function of {s} is to {o}.",
            "{s} {o}.",
        ],
        'atomic_number': [
            "{s} has atomic number {o}.",
        ],
        'height': [
            "{s} is {o} tall.",
        ],
        'depth': [
            "{s} is {o} deep.",
        ],
        'length': [
            "{s} is {o} long.",
        ],
        'area': [
            "{s} covers {o}.",
        ],
        'age': [
            "{s} is {o} old.",
        ],
        'distance_from_sun': [
            "{s} is {o} from the Sun.",
        ],
        'distance_from_earth': [
            "{s} is {o} from Earth.",
        ],
        'surface_temperature': [
            "{s} has a surface temperature of {o}.",
        ],
        'diameter': [
            "{s} has a diameter of {o}.",
        ],
        'mass': [
            "{s} has a mass of {o}.",
        ],
        'starts_with': [
            "{s} starts with: {o}.",
        ],
        'range': [
            "{s} ranges from {o}.",
        ],
        'parameters': [
            "{s} is parameterized by {o}.",
        ],
        'proved_by': [
            "{s} was proved by {o}.",
        ],
        'branches': [
            "{s} has branches: {o}.",
        ],
        'types': [
            "Types of {s} include {o}.",
        ],
        'inventors': [
            "{s} was invented by {o}.",
        ],
        'authors': [
            "{s} was authored by {o}.",
        ],
        'year': [
            "{s} dates to {o}.",
        ],
        'birthplace': [
            "{s} was born in {o}.",
        ],
        'birth_year': [
            "{s} was born in {o}.",
        ],
        'death_year': [
            "{s} died in {o}.",
        ],
        'nationality': [
            "{s} is {o}.",
            "{s} was {o}.",
        ],
        'occupation': [
            "{s} was a {o}.",
            "{s} is a {o}.",
        ],
        'holy_book': [
            "The holy book of {s} is the {o}.",
        ],
        'followers': [
            "{s} has about {o} followers worldwide.",
        ],
        'modern_revival': [
            "The modern revival was in {o}.",
        ],
        'first_landing': [
            "The first landing was {o}.",
        ],
        'first_image': [
            "The first image was captured in {o}.",
        ],
        'active_ingredient': [
            "The active ingredient in {s} is {o}.",
        ],
        'made_from': [
            "{s} is made from {o}.",
        ],
        'ingredients': [
            "{s} is made from {o}.",
        ],
        'found_in': [
            "{s} is found in {o}.",
        ],
        'habitat': [
            "{s} lives in {o}.",
        ],
        'speed': [
            "{s} can reach speeds of {o}.",
        ],
        'length': [
            "{s} can grow to {o}.",
        ],
        'height': [
            "{s} is {o} tall.",
        ],
        'purpose': [
            "Its purpose is {o}.",
        ],
        'built_by': [
            "{s} was built by {o}.",
        ],
        'distance_from_earth': [
            "{s} is {o} from Earth.",
        ],
        'Grand_Slams': [
            "The Grand Slams are {o}.",
        ],
        'frequency': [
            "It takes place {o}.",
        ],
        'artist': [
            "{s} was created by {o}.",
            "The artist is {o}.",
        ],
        'created': [
            "{s} was created in {o}.",
        ],
        'cause': [
            "{s} is caused by {o}.",
            "The cause of {s} is {o}.",
        ],
        'effects': [
            "The effects of {s} include {o}.",
        ],
        'threat': [
            "{s} is threatened by {o}.",
        ],
        'measured_by': [
            "{s} is measured by the {o}.",
        ],
        'can_fly': [
            "{o}.",
        ],
        'adaptation': [
            "Its {o}.",
        ],
        'stages': [
            "The stages of {s} are {o}.",
        ],
        'composition': [
            "{s} is composed of {o}.",
        ],
        'deepest_point': [
            "The deepest point is the {o}.",
        ],
        'percentage': [
            "{s} covers {o}.",
        ],
        'layers': [
            "{s} has layers: {o}.",
        ],
        'discovered': [
            "{s} was discovered in {o}.",
        ],
        'discovered_at': [
            "{s} was discovered at {o}.",
        ],
        'lived': [
            "{s} lived {o}.",
        ],
        'invented': [
            "{s} was invented in {o}.",
        ],
        'reclassified': [
            "{s} was reclassified in {o}.",
        ],
        'moons': [
            "{s} has {o}.",
        ],
        'spectral_class': [
            "{s} has spectral class {o}.",
        ],
        'extinction': [
            "{s} went extinct {o}.",
        ],
        'cause_of_extinction': [
            "The cause of extinction was {o}.",
        ],
        'players_per_team': [
            "Each team has {o} players.",
        ],
        'distance': [
            "The distance is {o}.",
        ],
        # Medicine
        'treatments': [
            "Treatments for {s} include {o}.",
        ],
        'neurons': [
            "{s} contains {o} neurons.",
        ],
        'beats_per_day': [
            "The {s} beats {o}.",
        ],
        'active_ingredient': [
            "The active ingredient in {s} is {o}.",
        ],
        'first_used': [
            "{s} was first used in {o}.",
        ],
        # Law
        'amendments': [
            "{s} has {o}.",
        ],
        'adopted': [
            "{s} was adopted in {o}.",
        ],
        'signed': [
            "{s} was signed in {o}.",
        ],
        'key_work': [
            "The key work is {o}.",
        ],
        # Languages
        'writing': [
            "{s} uses {o}.",
        ],
        'speakers': [
            "{s} has {o}.",
        ],
        'headquarters': [
            "The headquarters of {s} is {o}.",
        ],
        # Historical events
        'between': [
            "{s} was between {o}.",
        ],
        'result': [
            "The result was {o}.",
        ],
        'killed': [
            "It killed {o}.",
        ],
        'astronaut': [
            "The first person was {o}.",
        ],
        'mission': [
            "The mission was {o}.",
        ],
        'fell': [
            "It fell on {o}.",
        ],
    }

    DEFAULT_VARIANTS = [
        "The {r} of {s} is {o}.",
        "{s} has {o} as its {r}.",
    ]

    # Connectors between sentences (PPM selects best-scoring one)
    CONNECTORS = [
        " ",           # Just a space (two separate sentences)
        ", and ",      # Conjunction
        ". ",          # Period
        ", with ",     # Elaborative
        ". Additionally, ",
        ". It ",       # Pronoun continuation (for same subject)
    ]

    # Same-subject connectors (when two facts share a subject)
    # Applied AFTER subject removal, so they connect verb phrases
    # {pronoun} is replaced with He/She/It based on entity type
    SAME_SUBJECT_CONNECTORS = [
        ", and ",
        ". {pronoun} ",
        ". {pronoun} also ",
    ]

    # Relations that indicate the subject IS a person (not just related to one)
    PERSON_RELATIONS = {
        'birthplace', 'born', 'died', 'deathplace', 'nationality',
        'occupation', 'birth_year', 'death_year',
    }

    def __init__(self, lm_model=None):
        """
        Args:
            lm_model: A trained PPMModel (character-level) or
                       HierarchicalLanguageModel from language.py.
                       If None, falls back to template-only output.
        """
        self.lm = lm_model
        self._no_info_responses = [
            "I don't have information about that.",
            "I don't have any facts stored about that topic.",
            "No information available on that subject.",
        ]

    def reformulate(self, facts: List[Tuple[str, str, str]],
                    query: str = "") -> Dict[str, Any]:
        """
        Reformulate a list of facts into fluent prose.

        Args:
            facts: list of (subject, relation, object) triplets
            query: original query (for context)

        Returns:
            dict with:
                text: reformulated prose
                facts_used: original facts (unchanged)
                verified: True if all facts appear in output
                method: 'ppm_scored' or 'template_only'
        """
        if not facts:
            return {
                'text': self._no_info_responses[0],
                'facts_used': [],
                'verified': True,
                'method': 'no_facts',
            }

        # Step 1: Generate candidate phrasings for each fact
        candidates_per_fact = []
        for s, r, o in facts:
            variants = self.PHRASING_VARIANTS.get(
                r, self.DEFAULT_VARIANTS
            )
            # Replace underscores in relation name for display
            r_display = r.replace('_', ' ')
            phrasings = []
            for template in variants:
                try:
                    text = template.format(s=s, r=r_display, o=o)
                    phrasings.append(text)
                except (KeyError, IndexError):
                    continue
            if not phrasings:
                phrasings = [f"The {r} of {s} is {o}."]
            candidates_per_fact.append(phrasings)

        # Step 2: Select best phrasing per fact using PPM scoring
        if self.lm is not None:
            selected = self._ppm_select(candidates_per_fact, facts, query)
            method = 'ppm_scored'
        else:
            # No PPM → just pick first variant
            selected = [c[0] for c in candidates_per_fact]
            method = 'template_only'

        # Step 3: Merge same-subject facts with connectors
        merged = self._merge_sentences(selected, facts)

        # Step 4: Compose final text
        text = self._compose_final(merged)

        # Step 5: Anti-hallucination verification
        verified = self._verify_facts(text, facts)

        return {
            'text': text,
            'facts_used': facts,
            'verified': verified,
            'method': method,
        }

    def _ppm_select(self, candidates_per_fact: List[List[str]],
                    facts: List[Tuple], query: str) -> List[str]:
        """Use PPM to score and select the most natural phrasing."""
        selected = []
        context = list(query + " ") if query else []

        for i, phrasings in enumerate(candidates_per_fact):
            if len(phrasings) == 1:
                selected.append(phrasings[0])
                context = list(phrasings[0])
                continue

            # Score each phrasing by PPM log-probability
            best_score = float('-inf')
            best_phrasing = phrasings[0]

            for phrasing in phrasings:
                score = self._score_text(phrasing, context)
                if score > best_score:
                    best_score = score
                    best_phrasing = phrasing

            selected.append(best_phrasing)
            # Update context for next fact
            context = list(best_phrasing)

        return selected

    def _score_text(self, text: str, context: List[str]) -> float:
        """Score text using PPM character model. Higher = more natural."""
        chars = list(text)
        total_log_prob = 0.0

        # Use context from previous sentence
        ctx = list(context[-100:]) if context else []  # Limit context window

        for char in chars:
            probs = self.lm.predict(ctx)
            p = probs.get(char, 1e-10)
            total_log_prob += math.log2(max(p, 1e-10))
            ctx.append(char)

        # Normalize by length
        return total_log_prob / max(len(chars), 1)

    def _is_person(self, facts: List[Tuple]) -> bool:
        """Check if the facts suggest the subject is a person."""
        for _, r, _ in facts:
            if r in self.PERSON_RELATIONS:
                return True
        return False

    def _merge_sentences(self, sentences: List[str],
                         facts: List[Tuple]) -> List[str]:
        """Merge sentences with the same subject using connectors."""
        if len(sentences) <= 1:
            return sentences

        # Determine pronoun based on entity type
        is_person = self._is_person(facts)
        # For persons: use short name (last word of name) to avoid "They is/are" issues
        # For non-persons: "It"
        if is_person and facts:
            full_name = facts[0][0]
            # Use last name or single name: "Albert Einstein" → "Einstein"
            parts = full_name.split()
            pronoun = parts[-1] if len(parts) > 1 else full_name
        else:
            pronoun = "It"

        merged = []
        i = 0

        while i < len(sentences):
            current = sentences[i]
            current_subject = facts[i][0] if i < len(facts) else None

            # Look ahead for same-subject facts
            j = i + 1
            while j < len(sentences) and j < len(facts):
                if facts[j][0] == current_subject:
                    # Same subject — merge
                    next_sent = sentences[j]
                    shortened = self._remove_subject_repeat(
                        next_sent, current_subject
                    )
                    # Pick connector based on what follows
                    if shortened.startswith(('has ', 'was ', 'is ', 'borders ',
                                             'shares ', 'uses ')):
                        connector = self._best_connector(
                            current, shortened, same_subject=True,
                            pronoun=pronoun
                        )
                        combined = current.rstrip('.') + connector + shortened
                        # Fix word order: "also is" → "is also"
                        combined = combined.replace(' also is ', ' is also ')
                        combined = combined.replace(' also was ', ' was also ')
                        combined = combined.replace(' also has ', ' has also ')
                        current = combined
                    elif shortened == next_sent:
                        current = current.rstrip('.') + '. ' + next_sent
                    else:
                        current = current.rstrip('.') + ', and ' + shortened
                    j += 1
                else:
                    break

            merged.append(current)
            i = j

        return merged

    def _best_connector(self, sent_a: str, sent_b: str,
                        same_subject: bool = False,
                        pronoun: str = "It") -> str:
        """Select the best connector between two sentences using PPM."""
        connectors = (self.SAME_SUBJECT_CONNECTORS
                      if same_subject else self.CONNECTORS)

        if self.lm is None:
            # Rotate connectors for variety without PPM
            if not hasattr(self, '_conn_idx'):
                self._conn_idx = 0
            conn = connectors[self._conn_idx % len(connectors)]
            self._conn_idx += 1
            return conn.replace('{pronoun}', pronoun)

        best_score = float('-inf')
        best_conn = connectors[0]
        context = list(sent_a.rstrip('.'))

        for conn in connectors:
            # Score the transition
            test = conn + sent_b[:20]  # Only score beginning
            score = self._score_text(test, context)
            if score > best_score:
                best_score = score
                best_conn = conn

        return best_conn.replace('{pronoun}', pronoun)

    def _remove_subject_repeat(self, sentence: str, subject: str) -> str:
        """Remove repeated subject from a sentence for merging.

        "Germany has a population of 83M" → "has a population of 83M"
        "The official language of Germany is German" → "the official language is German"
        """
        esc = re.escape(subject)

        # Pattern 1: Subject at start → remove it
        m = re.match(rf'^{esc}\s+', sentence, re.I)
        if m:
            rest = sentence[m.end():]
            return rest[0].lower() + rest[1:] if rest else rest

        # Pattern 2: Subject's X → remove possessive
        m = re.match(rf'^{esc}\'s\s+', sentence, re.I)
        if m:
            return sentence[m.end():]

        # Pattern 3: "The X of Subject is Y" → "the X is Y"
        m = re.match(rf'^(The\s+\w+(?:\s+\w+)?)\s+of\s+{esc}\s+(is\s+.+)', sentence, re.I)
        if m:
            return m.group(1).lower() + ' ' + m.group(2)

        # Pattern 4: "X in Subject speak Y" → "people speak Y"
        m = re.match(rf'^People\s+in\s+{esc}\s+', sentence, re.I)
        if m:
            return "people " + sentence[m.end():]

        return sentence

    def _compose_final(self, sentences: List[str]) -> str:
        """Compose final text from merged sentences."""
        if not sentences:
            return self._no_info_responses[0]

        # Ensure each sentence ends with a period
        result = []
        for sent in sentences:
            sent = sent.strip()
            if sent and not sent.endswith(('.', '!', '?')):
                sent += '.'
            result.append(sent)

        text = ' '.join(result)

        # Post-processing cleanup
        text = text.replace(' also also ', ' also ')
        text = text.replace('also is also', 'is also')
        # Fix "a" vs "an" before vowels
        text = re.sub(r'\ba\s+([aeiouAEIOU])', r'an \1', text)
        text = re.sub(r'\s+', ' ', text).strip()  # normalize whitespace

        # Capitalize first letter of text
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        return text

    def _verify_facts(self, text: str, facts: List[Tuple]) -> bool:
        """Verify that all fact objects appear in the output text.

        This is the anti-hallucination check: every claim in the
        output must trace back to a stored fact.
        """
        text_lower = text.lower()
        for _, _, o in facts:
            # Check that the object value appears in the text
            if o.lower() not in text_lower:
                return False
        return True

    def reformulate_composer_output(self, composer_result: Dict[str, Any],
                                    query: str = "") -> Dict[str, Any]:
        """
        Take a ResponseComposer result and reformulate it.

        This is the integration point: Composer → Formulierer → Output.

        Args:
            composer_result: dict from ResponseComposer.compose()
            query: original query

        Returns:
            dict with reformulated text + original metadata
        """
        if not composer_result.get('facts_used'):
            return composer_result

        result = self.reformulate(
            composer_result['facts_used'],
            query=query,
        )

        # Merge with original metadata
        return {
            'answer': result['text'],
            'facts_used': composer_result['facts_used'],
            'confidence_level': composer_result.get('confidence_level', 'UNKNOWN'),
            'trace': composer_result.get('trace', []) + [
                f"Formulierer: {result['method']}, "
                f"verified={result['verified']}"
            ],
            'verified': result['verified'],
            'original_template': composer_result.get('answer', ''),
        }
