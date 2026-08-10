"""
Context Manager — Multi-Turn Conversation Tracking
=====================================================
Tracks entities, topics, and references across turns.
Resolves pronouns and anaphoric references.

"What is the capital of France?" → Paris
"How many people live there?" → there = France

No external dependencies.
"""

import re
from typing import Dict, Any, List, Optional, Tuple


class Entity:
    """A tracked entity across conversation turns."""

    __slots__ = ('name', 'type', 'aliases', 'properties', 'last_mentioned',
                 'mention_count')

    def __init__(self, name: str, entity_type: str = 'unknown'):
        self.name = name
        self.type = entity_type
        self.aliases: List[str] = []
        self.properties: Dict[str, str] = {}
        self.last_mentioned = 0
        self.mention_count = 1


class ContextManager:
    """Track conversation context across turns."""

    def __init__(self, max_entities: int = 50):
        self.entities: Dict[str, Entity] = {}
        self.topic_stack: List[str] = []
        self.turn_count = 0
        self.last_subject: Optional[str] = None
        self.last_object: Optional[str] = None
        self.last_answer_entity: Optional[str] = None
        self.max_entities = max_entities
        self._pronoun_refs: Dict[str, str] = {}

    def update(self, text: str, role: str = 'user'):
        """Update context from a new utterance."""
        self.turn_count += 1

        # Extract entities
        entities = self._extract_entities(text)
        for name, etype in entities:
            self._track_entity(name, etype, from_user=(role == 'user'))

        # Track topic (only from user, not system responses)
        if role == 'user':
            topic = self._detect_topic(text)
            if topic:
                if not self.topic_stack or self.topic_stack[-1] != topic:
                    self.topic_stack.append(topic)
                if len(self.topic_stack) > 20:
                    self.topic_stack = self.topic_stack[-20:]

        # Update pronoun references (only from user turns)
        if role == 'user':
            self._update_pronoun_refs(text)

        # Trim old entities
        if len(self.entities) > self.max_entities:
            self._prune_entities()

    def resolve_references(self, text: str) -> str:
        """
        Resolve pronouns and references in text.

        Strategy: pronouns refer to the most recent salient entity.
        - If system just answered with an entity (e.g. "Paris"), pronouns → answer entity
        - Otherwise, pronouns → last user subject

        "What is the capital of France?" → "Paris"
        "How big is it?" → "How big is Paris?" (answer entity)
        "What is its population?" → "What is Paris's population?"
        """
        resolved = text

        # Determine the best referent: answer entity (from last response) > last subject
        referent = getattr(self, 'last_answer_entity', None) or self.last_subject

        if not referent:
            return resolved

        # "its X?" → "Paris's X?"
        resolved = re.sub(
            r'\bits\b',
            referent + "'s",
            resolved, count=1, flags=re.IGNORECASE
        )

        # "it" (but not "it is a..." which may be definitional)
        resolved = re.sub(
            r'\bit\b(?!\s+is\s+(?:a|an|the)\b)',
            referent,
            resolved, count=1, flags=re.IGNORECASE
        )

        # "there" / "they" (for locations)
        loc_ref = referent if self._is_location(referent) else self.last_subject
        if loc_ref and self._is_location(loc_ref):
            if re.search(r'\bthere\b', resolved, re.I):
                resolved = re.sub(
                    r'\bthere\b', loc_ref, resolved, count=1, flags=re.IGNORECASE
                )
            elif re.search(r'\bthey\b', resolved, re.I):
                resolved = re.sub(
                    r'\bthey\b', f'people in {loc_ref}', resolved, count=1, flags=re.IGNORECASE
                )

        # "this" / "that" at sentence start
        resolved = re.sub(
            r'^(this|that)\b', referent, resolved, count=1, flags=re.IGNORECASE
        )

        # "the same" → referent
        resolved = re.sub(
            r'\bthe same\b', referent,
            resolved, count=1, flags=re.IGNORECASE
        )

        # "he"/"she" → last person entity
        for pronoun in ['he', 'she']:
            if re.search(r'\b' + pronoun + r'\b', resolved, re.I):
                person = self._find_last_person()
                if person:
                    resolved = re.sub(
                        r'\b' + pronoun + r'\b', person,
                        resolved, count=1, flags=re.IGNORECASE
                    )

        return resolved

    def _find_last_person(self) -> Optional[str]:
        """Find the most recently mentioned person entity."""
        best = None
        best_turn = -1
        for e in self.entities.values():
            if e.type == 'entity' and e.last_mentioned > best_turn:
                # Heuristic: multi-word capitalized names are likely people
                if ' ' in e.name or e.name[0].isupper():
                    best = e.name
                    best_turn = e.last_mentioned
        return best

    def get_current_topic(self) -> Optional[str]:
        """Get the current conversation topic."""
        return self.topic_stack[-1] if self.topic_stack else None

    def get_context_summary(self, max_entities: int = 5) -> Dict[str, Any]:
        """Get a summary of current context."""
        # Sort entities by recency
        recent = sorted(
            self.entities.values(),
            key=lambda e: e.last_mentioned,
            reverse=True
        )[:max_entities]

        return {
            'turn': self.turn_count,
            'topic': self.get_current_topic(),
            'last_subject': self.last_subject,
            'last_object': self.last_object,
            'entities': [
                {
                    'name': e.name,
                    'type': e.type,
                    'mentions': e.mention_count,
                    'properties': dict(e.properties),
                }
                for e in recent
            ],
            'topic_history': self.topic_stack[-5:],
        }

    def add_fact(self, subject: str, relation: str, obj: str):
        """Add a fact to entity properties."""
        subj_lower = subject.lower()
        if subj_lower in self.entities:
            self.entities[subj_lower].properties[relation] = obj

        # Update tracking
        self.last_subject = subject
        self.last_object = obj

    def clear(self):
        """Reset all context."""
        self.entities.clear()
        self.topic_stack.clear()
        self.turn_count = 0
        self.last_subject = None
        self.last_object = None
        self.last_answer_entity = None
        self._pronoun_refs.clear()

    # === Internal ===

    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities from text."""
        entities = []

        # Capitalized multi-word names (e.g., "New York", "Albert Einstein")
        for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text):
            name = m.group(1)
            # Skip common sentence starters
            if m.start() == 0 and len(text.split()) > 1:
                continue
            etype = self._guess_entity_type(name)
            entities.append((name, etype))

        # Detect concepts from "what is X" patterns
        m = re.search(r'(?:what|who)\s+(?:is|are)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\?|$)', text, re.I)
        if m:
            concept = m.group(1).strip().rstrip('?')
            if concept and not concept[0].isupper():
                entities.append((concept, 'concept'))

        # "What is the X of Y?" → Y is the main entity
        m = re.search(r'(?:what|who)\s+(?:is|are)\s+the\s+\w+(?:\s+\w+)?\s+(?:of|for|in)\s+(.+?)(?:\?|$)', text, re.I)
        if m:
            entity = m.group(1).strip().rstrip('?')
            if entity:
                etype = self._guess_entity_type(entity)
                entities.append((entity, etype))

        # "Tell me about X" / "What about X?"
        m = re.search(r'(?:tell\s+me\s+about|what\s+about)\s+(.+?)(?:\?|\.|$)', text, re.I)
        if m:
            entity = m.group(1).strip().rstrip('?.!')
            if entity:
                etype = self._guess_entity_type(entity)
                entities.append((entity, etype))

        return entities

    def _track_entity(self, name: str, entity_type: str, from_user: bool = True):
        """Track or update an entity."""
        key = name.lower()
        if key in self.entities:
            e = self.entities[key]
            e.last_mentioned = self.turn_count
            e.mention_count += 1
            if entity_type != 'unknown':
                e.type = entity_type
        else:
            e = Entity(name, entity_type)
            e.last_mentioned = self.turn_count
            self.entities[key] = e

        clean_name = name.rstrip('?.!,;:')
        if from_user:
            # User's question determines the topic/subject
            self.last_subject = clean_name
        else:
            # System answer entity becomes the answer referent.
            # "it" in follow-ups refers to the answer, not the question subject.
            # E.g., "What is the capital of France?" → "Paris"
            # "How big is it?" → "it" = Paris (the answer), not France
            self.last_answer_entity = clean_name
            # Update pronoun refs to point to the answer entity
            self._pronoun_refs['it'] = clean_name
            self._pronoun_refs['its'] = clean_name
            self._pronoun_refs['this'] = clean_name
            self._pronoun_refs['that'] = clean_name
            if self._is_location(clean_name):
                self._pronoun_refs['there'] = clean_name

    def _detect_topic(self, text: str) -> Optional[str]:
        """Detect the main topic of an utterance."""
        # "about X"
        m = re.search(r'\babout\s+(\w+(?:\s+\w+)?)', text, re.I)
        if m:
            return m.group(1).lower()

        # "what is X" / "who is X"
        m = re.search(r'(?:what|who)\s+(?:is|are)\s+(?:the\s+)?(\w+)', text, re.I)
        if m:
            return m.group(1).lower()

        # Last capitalized entity
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        if caps:
            return caps[-1].lower()

        return None

    def _update_pronoun_refs(self, text: str):
        """Update pronoun → entity mappings."""
        # After processing, the most recently mentioned entity
        # becomes the default referent for pronouns
        if self.last_subject:
            self._pronoun_refs['it'] = self.last_subject
            self._pronoun_refs['its'] = self.last_subject
            self._pronoun_refs['this'] = self.last_subject
            self._pronoun_refs['that'] = self.last_subject

    def _guess_entity_type(self, name: str) -> str:
        """Guess entity type from name."""
        lower = name.lower()
        # Countries and cities
        locations = {
            'france', 'germany', 'japan', 'china', 'india', 'brazil', 'russia',
            'australia', 'canada', 'italy', 'spain', 'mexico', 'egypt', 'turkey',
            'norway', 'sweden', 'finland', 'denmark', 'austria', 'greece',
            'portugal', 'ireland', 'poland', 'switzerland', 'netherlands',
            'new york', 'london', 'paris', 'berlin', 'tokyo', 'beijing',
            'moscow', 'rome', 'madrid', 'cairo', 'sydney', 'toronto',
            'africa', 'europe', 'asia', 'america', 'antarctica', 'arctic',
        }
        if lower in locations or any(t in lower for t in ['city', 'york']):
            return 'location'
        if any(t in lower for t in ['university', 'institute', 'company', 'corporation']):
            return 'organization'
        return 'entity'

    def _is_location(self, name: str) -> bool:
        """Check if entity is a location."""
        key = name.lower()
        if key in self.entities and self.entities[key].type == 'location':
            return True
        # Check against the locations set in _guess_entity_type
        return self._guess_entity_type(name) == 'location'

    def _prune_entities(self):
        """Remove oldest entities to stay under limit."""
        entities = sorted(
            self.entities.items(),
            key=lambda x: x[1].last_mentioned
        )
        # Keep the most recent half
        keep = entities[len(entities) // 2:]
        self.entities = dict(keep)
