"""
Multi-Turn Dialogue System — FLM Context + Hopfield Reference Resolution
=========================================================================
Tracks conversation state across turns using:
  1. FLM context: each turn is fed into the language model for context
  2. Entity tracking: Hopfield KnowledgeStore resolves references
  3. Confidence-gated responses: REJECTED → "I don't know" instead of guessing

No NLU, no intent classifier, no transformer.
Architecture: FLM (language) + Hopfield (memory) + Constraint Solver (reasoning).

Example:
    dialog = DialogSystem()
    dialog.load_knowledge([("France", "capital", "Paris"), ...])
    r1 = dialog.turn("What is the capital of France?")
    # → {"answer": "Paris", "confidence": "HIGH"}
    r2 = dialog.turn("And what about Germany?")
    # → resolves "Germany" from context, answers "Berlin"
    r3 = dialog.turn("What is the capital of Narnia?")
    # → {"answer": None, "confidence": "REJECTED", "response": "I don't know."}
"""

import re
import numpy as np


class EntityTracker:
    """
    Tracks entities mentioned across conversation turns.

    Maintains a recency-weighted list of entities. When a pronoun
    or ellipsis is detected, resolves to the most recent matching entity.
    """

    def __init__(self, max_history=20):
        self.max_history = max_history
        self.history = []  # [(entity, relation, turn_idx), ...]
        self.turn_count = 0

    def mention(self, entity, relation=None):
        """Record an entity mention."""
        self.history.append((entity, relation, self.turn_count))
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def advance_turn(self):
        self.turn_count += 1

    def resolve_reference(self, text, available_relations=None):
        """
        Try to resolve a reference in the text.

        Handles:
          - "it", "that", "this" → most recent entity
          - "And what about X?" → X with same relation as previous query
          - "What about its Y?" → most recent entity with relation Y
        """
        if not self.history:
            return None, None

        text_lower = text.lower().strip()

        # Pronoun reference: "it", "its", "that"
        words = [w.strip('?.,!;:') for w in text_lower.split()]

        # "What is its X?" → last entity with NEW relation X
        # But NOT "has X as its Y" (possessive determiner, not pronoun reference)
        if 'its' in words and 'as' not in words:
            its_idx = words.index('its')
            if its_idx + 1 < len(words):
                new_relation = words[its_idx + 1]
                if new_relation not in ('a', 'an', 'the', 'is', 'was', 'are'):
                    return self.history[-1][0], new_relation

        # "How far/old/big/fast is it?" → extract implied relation
        how_map = {
            'far': 'distance_from_earth',
            'old': 'age',
            'big': 'size',
            'fast': 'speed',
            'tall': 'height',
            'heavy': 'mass',
            'long': 'length',
            'deep': 'depth',
            'hot': 'temperature',
            'large': 'size',
            'many': 'population',
        }
        if 'how' in words:
            how_idx = words.index('how')
            if how_idx + 1 < len(words):
                adj = words[how_idx + 1]
                if adj in how_map:
                    return self.history[-1][0], how_map[adj]

        # Simple pronoun: "it", "that", "this", "the same"
        pronouns = ['it', 'that', 'this', 'the same']
        for p in pronouns:
            if p in words:
                entity = self.history[-1][0]
                # Try to extract a NEW relation from the current query
                cleaned = re.sub(r'\b(?:it|that|this|the same)\b', entity, text,
                                 count=1, flags=re.I).strip()
                _, new_rel = self._parse_for_relation(cleaned)
                if new_rel:
                    return entity, new_rel
                return entity, self.history[-1][1]

        # Location reference: "there" → last entity, keep relation=None
        # (let the parser extract the relation from the rest of the query)
        if 'there' in words:
            # Try to extract a relation from the current query (minus "there")
            # e.g., "What language do they speak there?" → relation=language
            cleaned = re.sub(r'\bthere\b', '', text, flags=re.I).strip()
            if cleaned:
                # Re-parse without "there" to get the relation
                _, rel = self._parse_for_relation(cleaned)
                if rel:
                    return self.history[-1][0], rel
            return self.history[-1][0], None

        # Ellipsis: "And what about Germany?" / "What about Japan?" / "And Germany?"
        ellipsis_patterns = [
            r'(?:and\s+)?what\s+about\s+(\w+)',   # "and what about X" / "what about X"
            r'(?:and\s+)?how\s+about\s+(\w+)',     # "and how about X" / "how about X"
            r'and\s+(?:for\s+)?(\w+)\s*\??$',      # "and X?" / "and for X?"
        ]
        for pattern in ellipsis_patterns:
            m = re.search(pattern, text_lower)
            if m:
                new_entity = m.group(1).capitalize()
                # Skip if captured a stop word
                if new_entity.lower() in ('the', 'a', 'an', 'is', 'was', 'are'):
                    continue
                last_rel = self.history[-1][1] if self.history else None
                return new_entity, last_rel

        return None, None

    def _parse_for_relation(self, text):
        """Extract just the relation from a query (ignoring subject)."""
        text_lower = text.lower().strip()
        # Common relation keywords
        rel_keywords = {
            'language': 'language', 'speak': 'language', 'spoken': 'language',
            'capital': 'capital', 'population': 'population', 'people': 'population',
            'live': 'population', 'currency': 'currency', 'money': 'currency',
            'continent': 'location', 'located': 'location', 'where': 'location',
            'created': 'creator', 'invented': 'inventor', 'founded': 'founded',
            'born': 'birthplace', 'borders': 'borders',
            'type': 'type', 'kind': 'type', 'category': 'type',
            'big': 'population', 'size': 'population', 'large': 'population',
        }
        for keyword, rel in rel_keywords.items():
            if keyword in text_lower:
                return None, rel
        return None, None

    def last_entity(self):
        if self.history:
            return self.history[-1][0]
        return None

    def last_relation(self):
        if self.history:
            return self.history[-1][1]
        return None


class QueryParser:
    """
    Parse natural language questions into (Subject, Relation) queries.

    Rule-based, no ML. Handles common question patterns:
      - "What is the X of Y?" → (Y, X)
      - "Who created X?" → (X, creator)
      - "Where is X located?" → (X, location)
      - "What is the symbol for X?" → (X, symbol)
    """

    PATTERNS = [
        # "What X has Y as its Z?" → reverse lookup (MUST be before generic "has")
        (r'what\s+(\w+)\s+has\s+(.+?)\s+as\s+(?:its|their|the)\s+(\w+)[\?\.]?$',
         lambda m: (m.group(2).strip().capitalize(), f'reverse_{m.group(3).strip().lower()}')),

        # "What/Which country uses/has X?" → reverse currency (generic)
        (r'(?:what|which)\s+(?:country|nation|place)\s+(?:uses?)\s+(?:the\s+)?(.+?)[\?\.]?$',
         lambda m: (m.group(1).strip().capitalize(), 'reverse_currency')),
        (r'(?:what|which)\s+(?:country|nation)\s+borders?\s+(.+?)[\?\.]?$',
         lambda m: (m.group(1).strip().capitalize(), 'reverse_borders')),

        # "Who made/built X?" → creator
        (r'who\s+(?:made|built)\s+(.+?)[\?\.]?$',
         lambda m: (m.group(1).strip().title(), 'creator')),

        # "When did X come out/start/launch?" → founded
        (r'when\s+did\s+(.+?)\s+(?:come\s+out|start|launch|appear|release)[\?\.]?$',
         lambda m: (m.group(1).strip().title(), 'founded')),

        # "What country does X belong to?" / "What country is X part of?"
        (r'what\s+country\s+(?:does|do|is)\s+(.+?)\s+(?:belong\s+to|part\s+of|located\s+in|in)[\?\.]?$',
         lambda m: (m.group(1).strip().title(), 'country')),

        # "What language do they speak in X?" / "What language is spoken in X?"
        (r'what\s+language\s+(?:do\s+they\s+speak|is\s+spoken)\s+in\s+(.+?)[\?\.]?$',
         lambda m: (m.group(1).strip().capitalize(), 'language')),

        # "Which continent is X on/in?" / "What continent is X on/in?"
        (r'(?:which|what)\s+continent\s+is\s+(.+?)\s+(?:on|in)[\?\.]?$',
         lambda m: (m.group(1).strip().capitalize(), 'continent')),

        # "What currency does X use?" / "What currency is used in X?"
        (r'what\s+currency\s+(?:does|do)\s+(.+?)\s+use[\?\.]?$',
         lambda m: (m.group(1).strip().capitalize(), 'currency')),
        (r'what\s+currency\s+is\s+used\s+in\s+(.+?)[\?\.]?$',
         lambda m: (m.group(1).strip().capitalize(), 'currency')),

        # "What {relation} does X have/use?" — generic relation extraction
        (r'what\s+(\w+)\s+(?:does|do)\s+(.+?)\s+(?:have|use)[\?\.]?$',
         lambda m: (m.group(2).strip().capitalize(), m.group(1).strip().lower())),

        # "What does X use?" / "What does X use for Y?"
        (r'what\s+does\s+(.+?)\s+use(?:\s+for\s+.+?)?[\?\.]?$',
         lambda m: (m.group(1).strip().title(), 'uses')),

        # "Compare X and Y" / "differences between X and Y" / "X versus Y"
        (r'compare\s+(.+?)\s+and\s+(.+?)[\?\.]?$',
         lambda m: (f"{m.group(1).strip().title()} and {m.group(2).strip().title()}", None)),
        (r'(?:differences?\s+between|(?:what\s+is\s+)?(?:bigger|larger|smaller|taller|faster|older|newer)\s*,?\s+)(.+?)\s+(?:and|or|vs\.?|versus)\s+(.+?)[\?\.]?$',
         lambda m: (f"{m.group(1).strip().title()} and {m.group(2).strip().title()}", None)),
        (r'(.+?)\s+(?:vs\.?|versus)\s+(.+?)[\?\.]?$',
         lambda m: (f"{m.group(1).strip().title()} and {m.group(2).strip().title()}", None)),

        # "Tell me about France" / "What do you know about France"
        (r'(?:tell\s+me\s+about|what\s+(?:do\s+you\s+)?know\s+about|describe|explain)\s+(.+?)[\?\.]?$',
         lambda m: (m.group(1).strip().title(), None)),

        # "What is/are the capital/types of/for France/tea?"
        (r'what\s+(?:is|are)\s+the\s+(\w+)\s+(?:of|for)\s+(.+?)[\?\.]?$',
         lambda m: (m.group(2).strip().capitalize(), m.group(1).strip().lower())),

        # "What is France's capital?"
        (r'what\s+is\s+(.+?)\'s\s+(\w+)[\?\.]?$',
         lambda m: (m.group(1).strip().capitalize(), m.group(2).strip().lower())),

        # "Who is X?" / "Who was X?" → treat as "about X" (relation=None)
        (r'who\s+(?:is|was|are|were)\s+(.+?)[\?\.]?$',
         lambda m: (m.group(1).strip().title(), None)),

        # "Who created Python?" / "Who discovered X?" / "Who wrote X?"
        (r'who\s+(\w+)\s+(.+?)[\?\.]?$',
         lambda m: (m.group(2).strip().title(),
                    {'invented': 'inventor', 'discovered': 'discoverer',
                     'created': 'creator', 'wrote': 'author',
                     'founded': 'founder'}.get(m.group(1).strip().lower(),
                                                m.group(1).strip().lower()))),

        # "Where is X?" / "Where is X located?"
        (r'where\s+is\s+(.+?)(?:\s+located)?[\?\.]?$',
         lambda m: (m.group(1).strip().capitalize(), 'location')),

        # "What country is X in?" / "What X is Y in?"
        (r'what\s+(\w+)\s+is\s+(?:the\s+)?(.+?)\s+in[\?\.]?$',
         lambda m: (m.group(2).strip().title(), m.group(1).strip().lower())),

        # "What causes X?" / "What caused X?"
        (r'what\s+(?:causes|caused|triggers)\s+(.+?)[\?\.]?$',
         lambda m: (m.group(1).strip().title(), 'cause')),

        # "How does X work?" / "How do X work?"
        (r'how\s+(?:does|do|did)\s+(.+?)\s+work[\?\.]?$',
         lambda m: (m.group(1).strip().title(), None)),

        # "Can X Y?" — yes/no questions → treat as about X
        (r'can\s+(.+?)\s+(\w+)[\?\.]?$',
         lambda m: (m.group(1).strip().title(), None)),

        # "When was X founded/signed/built/created?"
        (r'when\s+was\s+(?:the\s+)?(.+?)\s+(\w+)[\?\.]?$',
         lambda m: (m.group(1).strip().title(),
                    {'founded': 'founded', 'signed': 'signed',
                     'built': 'founded', 'created': 'founded',
                     'discovered': 'founded', 'invented': 'founded',
                     'adopted': 'adopted', 'established': 'founded',
                     'born': 'birth_year'}.get(m.group(2).strip().lower(),
                                                m.group(2).strip().lower()))),

        # "When did X start/begin/end/happen?"
        (r'when\s+did\s+(?:the\s+)?(.+?)\s+(\w+)[\?\.]?$',
         lambda m: (m.group(1).strip().title(),
                    {'start': 'started', 'begin': 'started', 'end': 'ended',
                     'happen': 'date', 'fall': 'fell', 'die': 'died',
                     'occur': 'date'}.get(m.group(2).strip().lower(),
                                           m.group(2).strip().lower()))),

        # "How many X does Y have?" / "How many speakers does Mandarin have?"
        (r'how\s+many\s+(\w+)\s+does\s+(?:the\s+)?(.+?)\s+have[\?\.]?$',
         lambda m: (m.group(2).strip().title(), m.group(1).strip().lower())),

        # "I want to know about X" / "Give me facts about X"
        (r'(?:i\s+want\s+to\s+know\s+about|give\s+me\s+facts?\s+about|'
         r'tell\s+me\s+(?:something|more)\s+about)\s+(.+?)[\?\.]?$',
         lambda m: (m.group(1).strip().title(), None)),

        # "What money/currency do they use in X?" / "What money is used in X?"
        (r'what\s+(?:money|currency)\s+(?:do\s+they\s+use|is\s+used)\s+in\s+(.+?)[\?\.]?$',
         lambda m: (m.group(1).strip().capitalize(), 'currency')),

        # "Does X border Y?" → yes/no check
        (r'(?:does|do)\s+(.+?)\s+border\s+(.+?)[\?\.]?$',
         lambda m: (m.group(1).strip().capitalize(), f'yesno_borders_{m.group(2).strip().capitalize()}')),
        # "Is X a Y?" → yes/no type check
        (r'is\s+(.+?)\s+(?:a|an)\s+(.+?)[\?\.]?$',
         lambda m: (m.group(1).strip().capitalize(), f'yesno_type_{m.group(2).strip().lower()}')),

        # (reverse lookup "What X has Y as its Z?" is handled at top of PATTERNS)

        # "What is DNA?" / "What is a hash table?" / "What is the Fibonacci sequence?"
        # Generic "what is X" → treat as "tell me about X"
        (r'what\s+(?:is|are)\s+(?:a\s+|an\s+|the\s+)?(.+?)[\?\.]?$',
         lambda m: (m.group(1).strip().title(), None)),

        # "capital of France"
        (r'(\w+)\s+of\s+(.+?)[\?\.]?$',
         lambda m: (m.group(2).strip().capitalize(), m.group(1).strip().lower())),

        # "France capital" (bare query)
        (r'^(\w+)\s+(\w+)$',
         lambda m: (m.group(1).strip().capitalize(), m.group(2).strip().lower())),
    ]

    @staticmethod
    def _smart_cap(s):
        """Capitalize first letter but preserve rest (PS-Lifted stays PS-Lifted)."""
        s = s.strip()
        if not s:
            return s
        # If already has uppercase beyond first char, preserve as-is
        if any(c.isupper() for c in s[1:]):
            return s
        # If contains hyphen or special chars, preserve
        if '-' in s:
            return s[0].upper() + s[1:]
        return s[0].upper() + s[1:]

    @classmethod
    def parse(cls, text):
        """Parse a question into (subject, relation) or None."""
        text_clean = text.strip()

        for pattern, extractor in cls.PATTERNS:
            m = re.search(pattern, text_clean, re.IGNORECASE)
            if m:
                subject, relation = extractor(m)
                # Fix capitalization: preserve original case from input when possible
                if subject:
                    # Find the subject text in the original input to preserve case
                    subj_lower = subject.lower()
                    idx = text_clean.lower().find(subj_lower)
                    if idx >= 0:
                        subject = text_clean[idx:idx + len(subject)]
                        # Ensure first char is uppercase
                        subject = subject[0].upper() + subject[1:] if subject else subject
                    else:
                        subject = cls._smart_cap(subject)
                return subject, relation

        return None, None


class DialogSystem:
    """
    Multi-turn dialogue system using FLM + Hopfield + Constraints.

    Each turn:
      1. Parse the question (rule-based QueryParser)
      2. Resolve references (EntityTracker)
      3. Query KnowledgeStore (Hopfield with anti-hallucination)
      4. Gap-Fill if rejected (Metacognition, optional)
      5. Compose multi-fact response (Composer, optional)
      6. Reformulate via FLM (Formulierer, optional)
      7. Update conversation state

    No transformer, no NLU model, no intent classifier.
    """

    def __init__(self, knowledge_dim=128):
        from .knowledge import KnowledgeStore
        from .normalizer import EntityNormalizer
        self.knowledge = KnowledgeStore(dim=knowledge_dim,
                                        normalizer=EntityNormalizer())
        self.entity_tracker = EntityTracker()
        self.parser = QueryParser()
        self.turns = []  # conversation history
        self.context_entities = set()  # all entities mentioned
        self._metacognition = None
        self._composer = None
        self._formulierer = None

    def load_knowledge(self, facts):
        """Load (Subject, Relation, Object) facts into the knowledge store."""
        self.knowledge.store_facts(facts)

    def enable_metacognition(self, meta_engine=None):
        """Enable self-improving gap-fill on rejected queries."""
        if meta_engine is None:
            from .metacognition import MetacognitionEngine
            meta_engine = MetacognitionEngine(self.knowledge)
        self._metacognition = meta_engine

    def enable_composer(self, composer=None):
        """Enable multi-fact response composition."""
        if composer is None:
            from .composer import ResponseComposer
            composer = ResponseComposer(knowledge_store=self.knowledge)
        self._composer = composer

    def enable_formulierer(self, formulierer=None):
        """Enable FLM-based reformulation."""
        if formulierer is None:
            from .formulierer import Formulierer
            formulierer = Formulierer()
        self._formulierer = formulierer

    def turn(self, user_input):
        """
        Process one conversation turn.

        Returns:
            dict with:
                response: str (natural language response)
                answer: str or None (the extracted answer)
                confidence: str (HIGH/MEDIUM/REJECTED/UNKNOWN)
                subject: str (resolved subject)
                relation: str (resolved relation)
                source: str (how the answer was found)
        """
        try:
            return self._turn_inner(user_input)
        except Exception:
            return {
                'response': "Something went wrong processing your question.",
                'answer': None,
                'confidence': 'UNKNOWN',
                'subject': None,
                'relation': None,
                'source': 'error_fallback',
                'turn': len(self.turns),
            }

    def _turn_inner(self, user_input):
        """Internal turn logic (extracted for graceful error handling)."""
        self.entity_tracker.advance_turn()

        # Step 1: Try reference resolution FIRST (ellipsis, pronouns)
        ref_entity, ref_relation = self.entity_tracker.resolve_reference(user_input)

        if ref_entity is not None:
            subject = ref_entity
            relation = ref_relation
        else:
            # Step 2: Full parse
            subject, relation = self.parser.parse(user_input)

        # Step 3: If we still have nothing, try bare entity lookup
        if subject is None:
            words = [w.strip('?.,!') for w in user_input.split() if len(w) > 2]
            for word in reversed(words):
                if self.knowledge.knows(word.capitalize()):
                    subject = word.capitalize()
                    break

        # Relation aliases: map common query relations to stored relations
        RELATION_ALIASES = {
            'continent': 'location',
            'region': 'location',
            'area': 'location',
            'spoken_language': 'language',
            'native_language': 'language',
            'city': 'capital',
            'money': 'currency',
            'people': 'population',
            'inhabitants': 'population',
            'birth': 'birthplace',
            'born': 'birthplace',
            'made_by': 'creator',
            'built_by': 'creator',
            'designed_by': 'creator',
        }
        if relation and relation.lower() in RELATION_ALIASES:
            relation = RELATION_ALIASES[relation.lower()]

        # Build result
        result = {
            'response': '',
            'answer': None,
            'confidence': 'UNKNOWN',
            'subject': subject,
            'relation': relation,
            'source': 'none',
            'turn': len(self.turns),
        }

        if subject is None:
            # Try to find any entity-like words for suggestions
            words = [w.strip('?.,!') for w in user_input.split() if len(w) > 2]
            suggestions = []
            for w in words:
                suggestions.extend(self._suggest_similar(w.capitalize()))
            suggestions = list(dict.fromkeys(suggestions))[:3]  # dedupe, max 3
            if suggestions:
                hint = ", ".join(suggestions)
                result['response'] = f"I'm not sure what you're asking. Did you mean: {hint}?"
            else:
                result['response'] = "I don't understand the question."
            result['source'] = 'parse_failure'
            result['suggestions'] = suggestions
        else:
            # Detect if this is an "about" query (wants overview) vs "what is" (wants type)
            _is_about_query = bool(re.search(
                r'(?:tell\s+me\s+about|what\s+(?:do\s+you\s+)?know\s+about|describe)\s+',
                user_input, re.I))

            # Step 4a: For "what is X?" queries (no relation), try type/identity FIRST
            # Skip this for "tell me about X" — those want a full overview
            if relation is None and not _is_about_query:
                for try_rel in ('type', 'identity'):
                    qr = self.knowledge.query_smart(subject, try_rel)
                    if qr['confidence_level'] in ('HIGH', 'MEDIUM'):
                        fact = qr['fact']
                        if fact[0].lower() == subject.lower():
                            result['answer'] = fact[2]
                            result['confidence'] = qr['confidence_level']
                            result['response'] = self._format_single_fact(fact)
                            result['source'] = 'knowledge_store'
                            self.entity_tracker.mention(fact[0], fact[1])
                            self.context_entities.add(fact[0])
                            self.turns.append({'input': user_input, 'result': result})
                            return result

            # Step 4b: For "about" queries, build structured overview
            if relation is None:
                # Collect ALL facts about this entity
                entity_facts = [(s, r, o) for s, r, o in self.knowledge.facts
                                if s.lower() == subject.lower()]
                if entity_facts:
                    result['confidence'] = 'HIGH'
                    result['source'] = 'knowledge_store'
                    result['facts_used'] = entity_facts
                    result['response'] = self._format_overview(subject, entity_facts)
                    self.entity_tracker.mention(subject, None)
                    self.context_entities.add(subject)
                    self.turns.append({'input': user_input, 'result': result})
                    return result

            # Step 4c-yesno: Yes/No questions (e.g., "Does France border Germany?")
            if relation and relation.startswith('yesno_'):
                parts = relation.split('_', 2)  # ['yesno', 'borders', 'Germany']
                real_rel = parts[1]
                target = parts[2] if len(parts) > 2 else ''
                found = False
                for s, r, o in self.knowledge.facts:
                    if real_rel == 'borders':
                        if ((s.lower() == subject.lower() and o.lower() == target.lower()) or
                                (o.lower() == subject.lower() and s.lower() == target.lower())):
                            found = True
                            break
                    elif real_rel == 'type':
                        if s.lower() == subject.lower() and target.lower() in o.lower():
                            found = True
                            break
                if found:
                    result['response'] = f"Yes, {subject} {real_rel} {target}." if real_rel == 'borders' \
                        else f"Yes, {subject} is a {target}."
                else:
                    result['response'] = f"No, I don't have information that {subject} {real_rel} {target}." if real_rel == 'borders' \
                        else f"I don't have information that {subject} is a {target}."
                result['confidence'] = 'HIGH' if found else 'LOW'
                result['source'] = 'knowledge_store'
                self.entity_tracker.mention(subject, real_rel)
                self.turns.append({'input': user_input, 'result': result})
                return result

            # Step 4c-pre: Reverse lookup (e.g., "What country has Berlin as its capital?")
            if relation and relation.startswith('reverse_'):
                real_rel = relation[8:]  # strip 'reverse_' prefix
                for s, r, o in self.knowledge.facts:
                    if r.lower() == real_rel and o.lower() == subject.lower():
                        result['answer'] = s
                        result['confidence'] = 'HIGH'
                        result['response'] = f"{s} has {subject} as its {real_rel}."
                        result['source'] = 'knowledge_store'
                        self.entity_tracker.mention(s, real_rel)
                        self.turns.append({'input': user_input, 'result': result})
                        return result
                # Not found
                result['response'] = f"I don't know which entity has {subject} as its {real_rel}."
                result['source'] = 'knowledge_store'
                self.turns.append({'input': user_input, 'result': result})
                return result

            # Step 4c: Direct knowledge query
            if relation is None:
                # Try any relation
                query_result = self.knowledge.query_smart(subject, None)
            else:
                query_result = self.knowledge.query_smart(subject, relation)
                # Relation fallback chain: location→country, founder→creator
                # Trigger if: rejected/unknown OR wrong subject OR wrong relation
                fact = query_result.get('fact')
                wrong_subject = (fact and fact[0].lower() != subject.lower())
                wrong_relation = (fact and relation
                                  and fact[1].lower() != relation.lower())
                if (query_result['confidence_level'] in ('REJECTED', 'UNKNOWN')
                        or wrong_subject or wrong_relation):
                    fallbacks = {
                        'location': ['country', 'part_of'],
                        'founder': ['creator'],
                        'creator': ['founder', 'inventor'],
                        'inventor': ['creator', 'discoverer'],
                    }
                    for alt_rel in fallbacks.get(relation, []):
                        alt_result = self.knowledge.query_smart(subject, alt_rel)
                        if (alt_result['confidence_level'] not in ('REJECTED', 'UNKNOWN')
                                and alt_result.get('fact')
                                and alt_result['fact'][0].lower() == subject.lower()):
                            query_result = alt_result
                            break

            # Multi-hop reasoning: try chains_to inferred facts
            if query_result['confidence_level'] in ('REJECTED', 'UNKNOWN'):
                chain_result = self.knowledge.query_smart(subject, 'chains_to')
                if chain_result['confidence_level'] in ('HIGH', 'MEDIUM'):
                    # Found a chain — try to answer via the chained entity
                    chained_entity = chain_result['fact'][2]
                    if relation:
                        hop2 = self.knowledge.query_smart(chained_entity, relation)
                        if hop2['confidence_level'] in ('HIGH', 'MEDIUM'):
                            query_result = hop2
                            result['source'] = 'multi_hop'

            # Gap-fill if rejected and metacognition is enabled
            if (query_result['confidence_level'] in ('REJECTED', 'UNKNOWN')
                    and self._metacognition is not None):
                meta_result = self._metacognition.query(subject, relation)
                if meta_result['confidence_level'] in ('HIGH', 'MEDIUM'):
                    query_result = meta_result
                    result['source'] = 'gap_filled'

            result['confidence'] = query_result['confidence_level']

            if query_result['confidence_level'] in ('HIGH', 'MEDIUM'):
                fact = query_result['fact']
                result['answer'] = fact[2]
                result['response'] = self._format_single_fact(fact)
                if result['source'] == 'none':
                    result['source'] = 'knowledge_store'

                self.entity_tracker.mention(fact[0], fact[1])
                self.context_entities.add(fact[0])

            elif query_result['confidence_level'] == 'REJECTED':
                # Fallback 1: if specific query rejected AND no specific relation
                # was asked, try Composer "about". If a specific relation was
                # requested (e.g. "king of Mars") and rejected, DON'T dump all
                # facts about the entity — that's not what was asked.
                entity_known = any(
                    s.lower() == subject.lower() or o.lower() == subject.lower()
                    for s, _, o in self.knowledge.facts
                )
                if self._composer and entity_known:
                    about_query = f"Tell me about {subject}"
                    comp = self._composer.compose(about_query)
                    if comp['facts_used']:
                        fact = comp['facts_used'][0]
                        result['answer'] = fact[2]
                        result['confidence'] = comp['confidence_level']
                        if self._formulierer:
                            form = self._formulierer.reformulate_composer_output(
                                comp, query=about_query)
                            result['response'] = form['answer']
                            result['source'] = 'composer_fallback+formulierer'
                        else:
                            result['response'] = comp['answer']
                            result['source'] = 'composer_fallback'
                        result['facts_used'] = comp['facts_used']
                        self.entity_tracker.mention(fact[0], fact[1])
                        self.context_entities.add(fact[0])
                        self.turns.append({'input': user_input, 'result': result})
                        return result

                # Fallback 2: try multi-word phrases from input as entities
                # "What is the speed of light?" → try "speed of light" as entity
                # Pick LONGEST matching entity (avoid "Moon" when "Moon landing" exists)
                if self._composer and not entity_known:
                    input_lower = user_input.lower()
                    candidates = set()
                    for s, _, _ in self.knowledge.facts:
                        s_lower = s.lower()
                        # Match stored subject in input OR input words in stored subject
                        if s_lower in input_lower and len(s) > 3:
                            candidates.add(s)
                        elif subject and subject.lower() in s_lower and len(subject) > 3:
                            candidates.add(s)
                    # Sort by length descending — longest match first
                    for candidate in sorted(candidates, key=len, reverse=True):
                        comp = self._composer.compose(f"Tell me about {candidate}")
                        if comp['facts_used']:
                            fact = comp['facts_used'][0]
                            result['answer'] = fact[2]
                            result['confidence'] = comp['confidence_level']
                            if self._formulierer:
                                form = self._formulierer.reformulate_composer_output(
                                    comp, query=user_input)
                                result['response'] = form['answer']
                                result['source'] = 'phrase_match+formulierer'
                            else:
                                result['response'] = comp['answer']
                                result['source'] = 'phrase_match'
                            result['facts_used'] = comp['facts_used']
                            self.entity_tracker.mention(fact[0], fact[1])
                            self.turns.append({'input': user_input, 'result': result})
                            return result

                # Fallback 3: superlative/keyword search in known_for/known_as
                # "tallest mountain" → search for "tallest" in known_for/known_as
                subj_lower = subject.lower() if subject else ''
                superlative_words = [w for w in subj_lower.split()
                                     if w in ('tallest', 'longest', 'largest', 'smallest',
                                              'fastest', 'biggest', 'oldest', 'deepest',
                                              'highest', 'hottest', 'coldest')]
                # Also extract noun context (e.g. "mountain" from "tallest mountain")
                context_nouns = [w for w in subj_lower.split()
                                 if w not in superlative_words and len(w) > 2]
                if superlative_words:
                    for s, r, o in self.knowledge.facts:
                        if r.lower() in ('known_for', 'known_as') and \
                           any(sw in o.lower() for sw in superlative_words) and \
                           (not context_nouns or
                            any(n in o.lower() or n in s.lower()
                                for n in context_nouns)):
                            if self._composer:
                                comp = self._composer.compose(f"Tell me about {s}")
                                if comp['facts_used']:
                                    if self._formulierer:
                                        form = self._formulierer.reformulate_composer_output(
                                            comp, query=user_input)
                                        result['response'] = form['answer']
                                        result['source'] = 'superlative_match+formulierer'
                                    else:
                                        result['response'] = comp['answer']
                                        result['source'] = 'superlative_match'
                                    result['facts_used'] = comp['facts_used']
                                    self.entity_tracker.mention(s, r)
                                    self.turns.append({'input': user_input, 'result': result})
                                    return result

                suggestions = self._suggest_similar(subject)
                if suggestions:
                    hint = ", ".join(suggestions[:3])
                    result['response'] = f"I don't have reliable information about {subject}. Did you mean: {hint}?"
                else:
                    result['response'] = f"I don't have reliable information about {subject}."
                result['source'] = 'anti_hallucination'
                result['suggestions'] = suggestions
                self.entity_tracker.mention(subject, relation)

            else:
                result['response'] = "I don't know."
                result['source'] = 'unknown'

        # Step 7: Record turn
        self.turns.append({
            'input': user_input,
            'result': result,
        })

        return result

    def _suggest_similar(self, subject, query_context=None):
        """
        5-strategy entity matching with PS-Lifted Consensus.

        Strategies:
          1. Lexical: Edit distance, prefix, substring (catches typos)
          2. Phonetic: Soundex (catches pronunciation errors)
          3. Semantic: FLM perplexity scoring in query context
          4. Embedding: Character n-gram cosine similarity
          5. KB-Relation: Knowledge structure overlap

        Returns top-3 candidates ranked by consensus score.
        Threshold: no suggestion if best score < 0.15 (anti-hallucination).
        """
        if not subject or len(subject) < 2:
            return []

        # Collect all known entities (min 2 chars, no single letters)
        all_entities = list(set(s for s, _, _ in self.knowledge.facts if len(s) > 1))
        if not all_entities:
            return []

        subj_lower = subject.lower()

        # Levenshtein edit distance (proper, handles insertions/deletions)
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            prev = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                curr = [i + 1]
                for j, c2 in enumerate(s2):
                    curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1 != c2)))
                prev = curr
            return prev[-1]

        # Pre-filter: only score entities with meaningful surface similarity
        prefilter = []
        for ent in all_entities:
            e_lower = ent.lower()
            if e_lower == subj_lower:
                continue  # exact match = no suggestion needed

            same_first = (e_lower[0] == subj_lower[0]) if e_lower and subj_lower else False
            max_len = max(len(subj_lower), len(e_lower))

            # Levenshtein on full string
            lev = levenshtein(subj_lower, e_lower)
            lev_ratio = lev / max(max_len, 1)

            # Check individual words of multi-word entities
            word_lev_ratio = 1.0
            for w in e_lower.split():
                if len(w) >= 3:
                    wlev = levenshtein(subj_lower, w)
                    wr = wlev / max(len(subj_lower), len(w), 1)
                    word_lev_ratio = min(word_lev_ratio, wr)

            # Substring check
            is_substr = subj_lower in e_lower or e_lower in subj_lower

            if same_first and lev_ratio < 0.5:
                prefilter.append(ent)
            elif same_first and word_lev_ratio < 0.35:
                prefilter.append(ent)
            elif is_substr:
                prefilter.append(ent)
            elif lev_ratio < 0.3:
                prefilter.append(ent)
            elif word_lev_ratio < 0.3:
                prefilter.append(ent)
        if not prefilter:
            return []

        candidates = sorted(prefilter)[:30]  # cap for performance

        # ── Strategy 1: Lexical (Levenshtein + prefix + substring) ──
        def lexical_score(candidate):
            c_lower = candidate.lower()
            score = 0.0
            # Substring
            if subj_lower in c_lower or c_lower in subj_lower:
                score = max(score, 0.8)
            # Prefix
            prefix_len = min(3, len(subj_lower), len(c_lower))
            if prefix_len > 0 and c_lower[:prefix_len] == subj_lower[:prefix_len]:
                score = max(score, 0.5 + 0.1 * prefix_len)
            # Levenshtein (full string)
            max_len = max(len(subj_lower), len(c_lower))
            if max_len > 0:
                lev = levenshtein(subj_lower, c_lower)
                ed_score = max(0, 1.0 - lev / max_len)
                score = max(score, ed_score)
            # Multi-word: check individual words of candidate
            for word in c_lower.split():
                if len(word) < 3:
                    continue
                wmax = max(len(subj_lower), len(word))
                wlev = levenshtein(subj_lower, word)
                wscore = max(0, 1.0 - wlev / wmax) if wmax > 0 else 0
                score = max(score, wscore)
            return score

        # ── Strategy 2: Phonetic (Soundex) ──
        def soundex(name):
            """Compute Soundex code (Robert C. Russell, 1918)."""
            if not name:
                return '0000'
            name = name.upper()
            code = name[0]
            mapping = {
                'B': '1', 'F': '1', 'P': '1', 'V': '1',
                'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2',
                'S': '2', 'X': '2', 'Z': '2',
                'D': '3', 'T': '3',
                'L': '4',
                'M': '5', 'N': '5',
                'R': '6',
            }
            prev = mapping.get(name[0], '0')
            for ch in name[1:]:
                digit = mapping.get(ch, '0')
                if digit != '0' and digit != prev:
                    code += digit
                prev = digit if digit != '0' else prev
            return (code + '0000')[:4]

        def phonetic_score(candidate):
            s1 = soundex(subject)
            s2 = soundex(candidate)
            if s1 == s2:
                return 1.0
            # Partial match: same first letter + some digits
            match = sum(a == b for a, b in zip(s1, s2))
            return match / 4.0

        # ── Strategy 3: Semantic (FLM perplexity in context) ──
        def semantic_scores(candidates_list):
            """Score all candidates via FLM — batch for efficiency."""
            scores = {}
            try:
                lm = getattr(self, '_lm', None)
                if lm is None:
                    return {c: 0.5 for c in candidates_list}
                # Build context template from query
                ctx = query_context or f"information about {subject}"
                base_ppl = lm.perplexity(list(ctx.lower()))
                for c in candidates_list:
                    test = ctx.replace(subject, c) if subject in ctx else f"information about {c}"
                    ppl = lm.perplexity(list(test.lower()))
                    # Lower perplexity = better fit
                    if base_ppl > 0 and ppl > 0:
                        ratio = base_ppl / ppl
                        scores[c] = min(ratio, 1.0)
                    else:
                        scores[c] = 0.5
            except Exception:
                return {c: 0.5 for c in candidates_list}
            return scores

        # ── Strategy 4: Embedding (character n-gram cosine similarity) ──
        def char_ngram_vec(text, n=3):
            """Character n-gram frequency vector."""
            t = text.lower()
            grams = {}
            for i in range(len(t) - n + 1):
                g = t[i:i+n]
                grams[g] = grams.get(g, 0) + 1
            return grams

        def cosine_sim(v1, v2):
            keys = set(v1) | set(v2)
            if not keys:
                return 0.0
            dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in keys)
            n1 = sum(v ** 2 for v in v1.values()) ** 0.5
            n2 = sum(v ** 2 for v in v2.values()) ** 0.5
            if n1 == 0 or n2 == 0:
                return 0.0
            return dot / (n1 * n2)

        def embedding_score(candidate):
            v1 = char_ngram_vec(subject)
            v2 = char_ngram_vec(candidate)
            return cosine_sim(v1, v2)

        # ── Strategy 5: KB-Relation Overlap ──
        def kb_overlap_score(candidate):
            """How many relations does the candidate share with what's being asked?"""
            cand_rels = set(r for s, r, o in self.knowledge.facts
                           if s.lower() == candidate.lower())
            if not cand_rels:
                return 0.0
            # Common relations for the entity type we're querying
            common_rels = {'capital', 'language', 'population', 'location',
                           'currency', 'type', 'creator', 'founded',
                           'born', 'died', 'nationality', 'occupation',
                           'known_for', 'birthplace', 'author', 'origin'}
            overlap = len(cand_rels & common_rels)
            return min(overlap / 4.0, 1.0)

        # ── Score all candidates across all strategies ──
        sem_scores = semantic_scores(candidates)

        scored = []
        for c in candidates:
            s1 = lexical_score(c)
            s2 = phonetic_score(c)
            s3 = sem_scores.get(c, 0.5)
            s4 = embedding_score(c)
            s5 = kb_overlap_score(c)
            scores = [s1, s2, s3, s4, s5]

            # PS-Lifted Consensus: topology-aware weighted average
            # Use barbell topology (5 nodes, strategies as endpoints)
            try:
                from .consensus import FossConsensus
                A = np.array([
                    [0, 1, 0, 1, 0],  # lexical ↔ phonetic, embedding
                    [1, 0, 1, 0, 0],  # phonetic ↔ lexical, semantic
                    [0, 1, 0, 1, 1],  # semantic ↔ phonetic, embedding, kb
                    [1, 0, 1, 0, 1],  # embedding ↔ lexical, semantic, kb
                    [0, 0, 1, 1, 0],  # kb ↔ semantic, embedding
                ], dtype=float)
                consensus = FossConsensus(A)
                vals = np.array(scores)
                converged, _, _ = consensus.average_consensus(vals)
                final_score = float(converged[0])
            except Exception:
                # Fallback: simple weighted average
                weights = [0.30, 0.15, 0.25, 0.15, 0.15]
                final_score = sum(w * s for w, s in zip(weights, scores))

            scored.append((c, final_score))

        # Sort by consensus score, descending
        scored.sort(key=lambda x: -x[1])

        # Anti-hallucination threshold: no suggestion if best < 0.15
        result = [c for c, score in scored if score >= 0.15]
        return result[:3]

    @staticmethod
    def _article(word):
        """Return 'an' if word starts with a vowel sound, else 'a'."""
        return 'an' if word and word[0].lower() in 'aeiou' else 'a'

    def _format_overview(self, entity, facts):
        """Format a structured overview of an entity (Haiku-style: answer first, clean sentences)."""
        # Priority order for relations
        priority = ['type', 'identity', 'capital', 'language', 'population',
                     'location', 'currency', 'creator', 'inventor', 'founder',
                     'founded', 'borders', 'paradigm', 'first_released']

        # Group facts by relation
        by_rel = {}
        for s, r, o in facts:
            by_rel.setdefault(r.lower(), []).append((s, r, o))

        sentences = []
        used_rels = set()

        # 1. Type/identity first (if exists)
        for try_rel in ('type', 'identity'):
            if try_rel in by_rel and try_rel not in used_rels:
                s, r, o = by_rel[try_rel][0]
                sentences.append(f"{entity} is {self._article(o)} {o}." if try_rel == 'type'
                                 else f"{entity} is {o}.")
                used_rels.add(try_rel)

        # 2. Key facts in priority order (skip duplicates like founded+first_released)
        for rel in priority:
            if rel == 'first_released' and 'founded' in used_rels:
                used_rels.add(rel)
                continue
            if rel in by_rel and rel not in used_rels:
                items = by_rel[rel]
                if rel == 'borders':
                    neighbors = [o for _, _, o in items]
                    if neighbors:
                        sentences.append(f"It borders {', '.join(neighbors)}.")
                elif rel == 'capital':
                    sentences.append(f"The capital is {items[0][2]}.")
                elif rel == 'language':
                    sentences.append(f"The official language is {items[0][2]}.")
                elif rel == 'population':
                    sentences.append(f"It has a population of {items[0][2]}.")
                elif rel == 'location':
                    sentences.append(f"It is located in {items[0][2]}.")
                elif rel == 'currency':
                    sentences.append(f"The currency is the {items[0][2]}.")
                elif rel in ('creator', 'inventor', 'founder'):
                    sentences.append(f"It was created by {items[0][2]}.")
                elif rel == 'founded':
                    sentences.append(f"It was founded in {items[0][2]}.")
                else:
                    sentences.append(self._format_single_fact(items[0]))
                used_rels.add(rel)

        # 3. Remaining facts (not in priority list, max 3 more)
        remaining = 0
        for rel, items in by_rel.items():
            if rel not in used_rels and remaining < 3:
                sentences.append(self._format_single_fact(items[0]))
                remaining += 1

        return ' '.join(sentences) if sentences else f"I know about {entity} but have no detailed facts."

    def _format_single_fact(self, fact):
        """Format a single fact as a response sentence using Formulierer."""
        s, r, o = fact
        if hasattr(self, '_formulierer') and self._formulierer:
            result = self._formulierer.reformulate([(s, r, o)])
            text = result.get('text', '') if isinstance(result, dict) else str(result)
            fallback = f"The {r.replace('_', ' ')} of {s} is {o}."
            if text and text != fallback:
                return text
        # Natural formatting for common relations
        r_lower = r.lower()
        if r_lower == 'type':
            return f"{s} is {self._article(o)} {o}."
        if r_lower == 'identity':
            return f"{s} is {o}."
        if r_lower == 'uses':
            return f"{s} uses {o}."
        if r_lower == 'based_on':
            return f"{s} is based on {o}."
        if r_lower == 'platform':
            return f"{s} runs on {o}."
        if r_lower in ('creator', 'inventor', 'discoverer', 'author', 'founder'):
            return f"{o} is the {r_lower} of {s}."
        if r_lower == 'birthplace':
            return f"{s} was born in {o}."
        if r_lower == 'location':
            return f"{s} is located in {o}."
        if r_lower == 'part_of':
            return f"{s} is part of {o}."
        if r_lower == 'currency':
            return f"{s} uses the {o} as its currency."
        if r_lower == 'population':
            return f"{s} has a population of {o}."
        if r_lower == 'language':
            return f"The official language of {s} is {o}."
        if r_lower == 'capital':
            return f"{o} is the capital of {s}."
        if r_lower == 'borders':
            return f"{s} borders {o}."
        if r_lower == 'founded':
            # Check if entity is an invention → use "invented"
            for fs, fr, fo in self.knowledge.facts:
                if fs.lower() == s.lower() and fr.lower() == 'type':
                    if fo.lower() == 'invention':
                        return f"{s} was invented in {o}."
                    break
            return f"{s} was founded in {o}."
        if r_lower == 'country':
            return f"{s} is in {o}."
        if r_lower == 'industry':
            return f"{s} operates in the {o} industry."
        if r_lower == 'nationality':
            return f"{s} is {o}."
        if r_lower == 'occupation':
            return f"{s} is {self._article(o)} {o}."
        if r_lower == 'born':
            return f"{s} was born in {o}."
        if r_lower == 'died':
            return f"{s} died in {o}."
        if r_lower in ('language_family', 'writing_system'):
            r_display = r_lower.replace('_', ' ')
            return f"The {r_display} of {s} is {o}."
        if r_lower == 'symbol':
            return f"The symbol for {s} is {o}."
        if r_lower == 'formula':
            return f"The chemical formula of {s} is {o}."
        if r_lower == 'known_as':
            return f"{s} is also known as {o}."
        if r_lower == 'description':
            return f"{s} is {o}."
        r_display = r.replace('_', ' ')
        return f"The {r_display} of {s} is {o}."

    def reset(self):
        """Reset conversation state (keeps knowledge)."""
        self.entity_tracker = EntityTracker()
        self.turns = []
        self.context_entities = set()

    @property
    def n_turns(self):
        return len(self.turns)
