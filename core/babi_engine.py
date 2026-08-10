"""
bAbI Story Comprehension Engine
================================
Parses bAbI-format stories, tracks world state, answers questions.
Covers all 20 bAbI tasks:
  1-3: Location tracking (1/2/3 supporting facts)
  4: Two-argument relations (spatial)
  5: Three-argument relations (giving)
  6: Yes/No questions
  7: Counting
  8: Lists/Sets
  9: Simple negation
  10: Indefinite knowledge
  11: Basic coreference
  12: Conjunction
  13: Compound coreference
  14: Time reasoning
  15: Basic deduction
  16: Basic induction
  17: Positional reasoning
  18: Size reasoning
  19: Path finding
  20: Agent motivations
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


# Time ordering for Task 14
TIME_ORDER = {
    'yesterday': 0,
    'this morning': 1,
    'this afternoon': 2,
    'this evening': 3,
}

# Number words for Task 7
NUM_WORDS = {
    0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
    5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
}

# Irregular plurals
IRREGULAR_PLURAL = {
    'mice': 'mouse', 'wolves': 'wolf', 'sheep': 'sheep',
    'cats': 'cat', 'dogs': 'dog', 'bears': 'bear',
}

def singularize(word: str) -> str:
    """Convert plural to singular."""
    if word in IRREGULAR_PLURAL:
        return IRREGULAR_PLURAL[word]
    if word.endswith('ves'):
        return word[:-3] + 'f'
    if word.endswith('ies'):
        return word[:-3] + 'y'
    if word.endswith('es') and len(word) > 3:
        return word[:-2]
    if word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word

# Motivation rules for Task 20
MOTIVATION_RULES = {
    'hungry': ('kitchen', ['apple', 'milk']),
    'thirsty': ('kitchen', ['milk', 'water']),
    'bored': ('garden', ['football']),
    'tired': ('bedroom', []),
}

# Direction opposites for Task 17/19
DIR_OPPOSITES = {
    'north': 'south', 'south': 'north',
    'east': 'west', 'west': 'east',
    'left': 'right', 'right': 'left',
    'above': 'below', 'below': 'above',
}


class WorldState:
    """Tracks the world state across a bAbI story."""

    def __init__(self):
        self.locations: Dict[str, str] = {}          # entity → location
        self.possessions: Dict[str, Set[str]] = defaultdict(set)  # entity → {items}
        self.spatial: Dict[str, Dict[str, str]] = defaultdict(dict)  # direction → {entity → entity}
        self.gave: Dict[str, Dict[str, str]] = defaultdict(dict)  # giver → {recipient → item}
        self.received: Dict[str, Dict[str, str]] = defaultdict(dict)  # recipient → {giver → item}
        self.give_log: List[Tuple[str, str, str]] = []  # ordered log: (giver, item, recipient)
        self.is_a: Dict[str, str] = {}               # entity → type
        self.properties: Dict[str, Dict[str, str]] = defaultdict(dict)  # entity → {prop → value}
        self.rules: Dict[str, Dict[str, str]] = defaultdict(dict)  # type → {relation → target_type}
        self.motivations: Dict[str, str] = {}         # entity → motivation
        self.size_order: List[Tuple[str, str]] = []   # (bigger, smaller) pairs
        self.fits_inside: List[Tuple[str, str]] = []  # (smaller, bigger) pairs
        self.indefinite: Dict[str, Set[str]] = {}     # entity → {possible locations}
        self.negated_locations: Dict[str, Set[str]] = defaultdict(set)  # entity → {NOT here}

        # Temporal tracking for Task 14
        self.timed_locations: Dict[str, List[Tuple[int, str]]] = defaultdict(list)  # entity → [(time_idx, location)]

        # Visit sequence for "before" questions (Task 3, 14)
        self.visit_sequence: Dict[str, List[str]] = defaultdict(list)  # entity → [loc1, loc2, ...]

        # 2D positions for Task 17 (positional reasoning)
        self.positions: Dict[str, Tuple[int, int]] = {}  # entity → (x, y)
        self.pos_constraints: List[Tuple[str, str, str]] = []  # (entity1, direction, entity2)

        # Coreference
        self.last_mentioned: List[str] = []  # stack of recently mentioned entities

    def reset(self):
        """Reset for a new story."""
        self.__init__()


class BabiEngine:
    """Parse and solve bAbI stories."""

    def __init__(self):
        self.world = WorldState()

    def solve_story(self, lines: List[str]) -> List[Tuple[str, str, str]]:
        """
        Process a bAbI story and return (question, predicted_answer, gold_answer) tuples.
        """
        self.world.reset()
        results = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse line number
            m = re.match(r'^(\d+)\s+(.+)$', line)
            if not m:
                continue

            line_num = int(m.group(1))
            content = m.group(2)

            # Reset state on new story (line_num == 1)
            if line_num == 1:
                self.world.reset()

            # Is this a question? (contains tab → answer)
            if '\t' in content:
                parts = content.split('\t')
                question = parts[0].strip()
                gold_answer = parts[1].strip()
                # Solve
                predicted = self._answer_question(question)
                results.append((question, predicted, gold_answer))
            else:
                # Statement — update world state
                self._process_statement(content)

        return results

    def _process_statement(self, stmt: str):
        """Update world state from a statement."""
        stmt = stmt.strip().rstrip('.')
        # Track visit sequence for "before" questions (Task 14)
        # Each entity gets an ordered list of locations visited

        # === COREFERENCE (Tasks 11, 13) ===
        # "After that he/she went to X" / "Then they went to X"
        # "Following that he/she/they went to X"
        m = re.match(
            r'(?:after that|then|following that|afterwards)\s+'
            r'(he|she|they|it)\s+'
            r'(?:went\s+(?:back\s+)?to|journeyed\s+to|moved\s+to|travelled\s+to)\s+'
            r'the\s+(.+)',
            stmt, re.I
        )
        if m:
            pronoun = m.group(1).lower()
            location = m.group(2).strip()
            if pronoun == 'they':
                for entity in self.world.last_mentioned:
                    self._move_entity(entity, location, stmt)
                self.world.last_mentioned = list(self.world.last_mentioned)  # preserve group
            else:
                if self.world.last_mentioned:
                    entity = self.world.last_mentioned[0]
                    self._move_entity(entity, location, stmt)
            return

        # === CONJUNCTION MOVEMENT (Task 12, 13) ===
        # "John and Mary travelled to the hallway"
        m = re.match(
            r'(\w+)\s+and\s+(\w+)\s+'
            r'(?:went\s+(?:back\s+)?to|journeyed\s+to|moved\s+(?:back\s+)?to|travelled\s+to)\s+'
            r'the\s+(.+)',
            stmt, re.I
        )
        if m:
            e1 = m.group(1)
            e2 = m.group(2)
            location = m.group(3).strip()
            self._move_entity(e1, location, stmt)
            self._move_entity(e2, location, stmt)
            self.world.last_mentioned = [e1.lower(), e2.lower()]
            return

        # === MOVEMENT (Tasks 1-3, 9, 14) ===
        # "Mary went to the bathroom" / "John journeyed to the hallway"
        # Time prefix: "This morning Mary moved to the kitchen"
        # Time suffix: "Bill journeyed to the office this morning"
        m = re.match(
            r'(?:(yesterday|this\s+morning|this\s+afternoon|this\s+evening)\s+)?'
            r'(\w+)\s+'
            r'(?:went\s+(?:back\s+)?to|journeyed\s+to|moved\s+(?:back\s+)?to|'
            r'travelled\s+to|traveled\s+to)\s+'
            r'the\s+(.+)',
            stmt, re.I
        )
        if m:
            time_str = m.group(1)
            entity = m.group(2)
            location = m.group(3).strip()
            # Check for time suffix: "the office this morning"
            suffix_m = re.search(r'\s+(yesterday|this\s+morning|this\s+afternoon|this\s+evening)$', location, re.I)
            if suffix_m:
                time_str = suffix_m.group(1)
                location = location[:suffix_m.start()].strip()
            self._move_entity(entity, location, stmt, time_str)
            return

        # === LOCATION STATE (Tasks 1, 9) ===
        # "John is in the hallway"
        m = re.match(r'(\w+)\s+is\s+in\s+the\s+(.+)', stmt, re.I)
        if m:
            entity = m.group(1)
            location = m.group(2).strip()
            self._move_entity(entity, location, stmt)
            return

        # === NEGATION (Task 9) ===
        # "Sandra is no longer in the bedroom"
        m = re.match(r'(\w+)\s+is\s+no\s+longer\s+in\s+the\s+(.+)', stmt, re.I)
        if m:
            entity = m.group(1)
            location = m.group(2).strip()
            # Remove from that location
            if self.world.locations.get(entity.lower()) == location:
                del self.world.locations[entity.lower()]
            self.world.negated_locations[entity.lower()].add(location)
            self.world.last_mentioned = [entity.lower()]
            return

        # "Mary is not in the bathroom"
        m = re.match(r'(\w+)\s+is\s+not\s+in\s+the\s+(.+)', stmt, re.I)
        if m:
            entity = m.group(1)
            location = m.group(2).strip()
            if self.world.locations.get(entity.lower()) == location:
                del self.world.locations[entity.lower()]
            self.world.negated_locations[entity.lower()].add(location)
            return

        # === INDEFINITE (Task 10) ===
        # "Bill is either in the school or the office"
        m = re.match(r'(\w+)\s+is\s+either\s+in\s+the\s+(\w+)\s+or\s+the\s+(\w+)', stmt, re.I)
        if m:
            entity = m.group(1)
            loc1 = m.group(2)
            loc2 = m.group(3)
            self.world.indefinite[entity.lower()] = {loc1, loc2}
            # Clear definite location since it's now indefinite
            self.world.locations.pop(entity.lower(), None)
            return

        # === PICKING UP / GETTING (Tasks 2, 7, 8, 20) ===
        # "Daniel picked up the football"
        m = re.match(r'(\w+)\s+(?:picked\s+up|grabbed|got|took)\s+the\s+(.+?)(?:\s+there)?$', stmt, re.I)
        if m:
            entity = m.group(1)
            item = m.group(2).strip()
            self.world.possessions[entity.lower()].add(item)
            self.world.last_mentioned = [entity.lower()]
            return

        # === DROPPING (Task 3) ===
        # "John dropped the apple" / "John put down the apple there" / "John left the apple"
        m = re.match(r'(\w+)\s+(?:dropped|put\s+down|left|discarded)\s+the\s+(.+?)(?:\s+there)?$', stmt, re.I)
        if m:
            entity = m.group(1)
            item = m.group(2).strip()
            self.world.possessions[entity.lower()].discard(item)
            # Item stays at current location of holder
            loc = self.world.locations.get(entity.lower())
            if loc:
                self.world.locations[item.lower()] = loc
                # Don't add to visit_sequence here — dropping doesn't change item location
            self.world.last_mentioned = [entity.lower()]
            return

        # === GIVING (Task 5) ===
        # "Bill gave the football to Fred"
        m = re.match(r'(\w+)\s+(?:gave|passed|handed)\s+the\s+(.+?)\s+to\s+(\w+)', stmt, re.I)
        if m:
            giver = m.group(1)
            item = m.group(2).strip()
            recipient = m.group(3)
            self.world.gave[giver.lower()][recipient.lower()] = item
            self.world.received[recipient.lower()][giver.lower()] = item
            self.world.give_log.append((giver.lower(), item.lower(), recipient.lower()))
            self.world.possessions[giver.lower()].discard(item)
            self.world.possessions[recipient.lower()].add(item)
            self.world.last_mentioned = [giver.lower(), recipient.lower()]
            return

        # === SPATIAL RELATIONS (Task 4, 17) ===
        # "The office is north of the bedroom"
        # "The pink rectangle is to the left of the triangle"
        m = re.match(
            r'the\s+(.+?)\s+is\s+'
            r'(?:to\s+the\s+)?(north|south|east|west|left|right|above|below)\s+'
            r'(?:of\s+)?the\s+(.+)',
            stmt, re.I
        )
        if m:
            entity1 = m.group(1).strip()
            direction = m.group(2).lower()
            entity2 = m.group(3).strip()
            e1 = entity1.lower()
            e2 = entity2.lower()
            self.world.spatial[direction][e2] = e1
            if direction in DIR_OPPOSITES:
                opp = DIR_OPPOSITES[direction]
                self.world.spatial[opp][e1] = e2
            # Compute 2D positions
            self.world.pos_constraints.append((e1, direction, e2))
            self._resolve_positions()
            return

        # === SIZE RELATIONS (Task 18) ===
        # "The suitcase fits inside the box"
        m = re.match(r'the\s+(.+?)\s+fits\s+inside\s+the\s+(.+)', stmt, re.I)
        if m:
            smaller = m.group(1).strip().lower()
            bigger = m.group(2).strip().lower()
            self.world.fits_inside.append((smaller, bigger))
            self.world.size_order.append((bigger, smaller))
            return

        # "The container is bigger than the suitcase"
        m = re.match(r'the\s+(.+?)\s+is\s+bigger\s+than\s+the\s+(.+)', stmt, re.I)
        if m:
            bigger = m.group(1).strip().lower()
            smaller = m.group(2).strip().lower()
            self.world.size_order.append((bigger, smaller))
            return

        # === TYPE ASSIGNMENT (Tasks 15, 16, 20) ===
        # "Winona is a sheep" / "Emily is a cat"
        m = re.match(r'(\w+)\s+is\s+a\s+(\w+)', stmt, re.I)
        if m:
            entity = m.group(1)
            etype = m.group(2)
            self.world.is_a[entity.lower()] = etype.lower()
            self.world.last_mentioned = [entity.lower()]
            return

        # === RULES (Task 15) ===
        # "Wolves are afraid of mice"
        m = re.match(r'(\w+)\s+are\s+afraid\s+of\s+(\w+)', stmt, re.I)
        if m:
            entity_type = m.group(1).lower()
            fear_type = m.group(2).lower()
            self.world.rules[entity_type]['afraid_of'] = fear_type
            return

        # === PROPERTY (Task 16) ===
        # "Bernhard is white"
        m = re.match(r'(\w+)\s+is\s+(white|gray|grey|yellow|green|blue|red|black|brown)', stmt, re.I)
        if m:
            entity = m.group(1)
            color = m.group(2)
            self.world.properties[entity.lower()]['color'] = color.lower()
            self.world.last_mentioned = [entity.lower()]
            return

        # === MOTIVATION (Task 20) ===
        # "Jason is thirsty" / "Antoine is bored"
        m = re.match(r'(\w+)\s+is\s+(hungry|thirsty|bored|tired)', stmt, re.I)
        if m:
            entity = m.group(1)
            motivation = m.group(2).lower()
            self.world.motivations[entity.lower()] = motivation
            self.world.last_mentioned = [entity.lower()]
            return

    def _move_entity(self, entity: str, location: str, stmt: str = '', time_str: str = None):
        """Move an entity to a new location, tracking visits and co-traveling items."""
        entity = entity.lower()
        self.world.locations[entity] = location
        self.world.visit_sequence[entity].append(location)
        self._add_timed_location(entity, location, stmt, time_str)
        self.world.indefinite.pop(entity, None)
        self.world.last_mentioned = [entity]
        # Co-travel: items this entity is carrying also move
        for item in self.world.possessions.get(entity, set()):
            item_lower = item.lower()
            self.world.locations[item_lower] = location
            self.world.visit_sequence[item_lower].append(location)

    def _add_timed_location(self, entity: str, location: str, stmt: str = '', time_str: str = None):
        """Add a timed location entry for Task 14."""
        time_idx = -1
        if time_str:
            time_str_clean = time_str.strip().lower()
            time_idx = TIME_ORDER.get(time_str_clean, -1)
        else:
            # Try to detect time from statement
            s_lower = stmt.lower()
            for tkey, tidx in TIME_ORDER.items():
                if tkey in s_lower:
                    time_idx = tidx
                    break

        if time_idx >= 0:
            self.world.timed_locations[entity].append((time_idx, location))
            self.world.timed_locations[entity].sort(key=lambda x: x[0])

    def _answer_question(self, question: str) -> str:
        """Answer a bAbI question from current world state."""
        q = question.strip().rstrip('?')
        q_lower = q.lower()

        # === WHERE IS [THE] X? (Tasks 1-3, 6, 9-14) ===
        m = re.match(r'where\s+is\s+(?:the\s+)?(.+)', q, re.I)
        if m:
            entity = m.group(1).strip().lower()
            # Direct location
            if entity in self.world.locations:
                return self.world.locations[entity]
            # Check possessions — item follows its holder
            for person, items in self.world.possessions.items():
                if entity in {i.lower() for i in items}:
                    if person in self.world.locations:
                        return self.world.locations[person]
            return 'unknown'

        # === WHERE WAS X BEFORE Y? (Task 3, 14) ===
        m = re.match(r'where\s+was\s+(?:the\s+)?(\w+)\s+before\s+the\s+(.+)', q, re.I)
        if m:
            entity = m.group(1).lower()
            before_loc = m.group(2).strip().lower()
            # First try timed locations (Task 14) — search from END
            timed = self.world.timed_locations.get(entity, [])
            for i in range(len(timed) - 1, 0, -1):
                if timed[i][1].lower() == before_loc:
                    return timed[i-1][1]
            # Fallback to visit sequence (Task 3) — search from END (last visit)
            visits = self.world.visit_sequence.get(entity, [])
            for i in range(len(visits) - 1, 0, -1):
                if visits[i].lower() == before_loc:
                    return visits[i-1]
            return 'unknown'

        # === WHERE WAS X IN THE MORNING/AFTERNOON? (Task 14) ===
        m = re.match(r'where\s+was\s+(\w+)\s+(?:in\s+the\s+)?(yesterday|this\s+morning|this\s+afternoon|this\s+evening|morning|afternoon|evening)', q, re.I)
        if m:
            entity = m.group(1).lower()
            time_str = m.group(2).lower()
            # Normalize
            if time_str in ('morning',):
                time_str = 'this morning'
            elif time_str in ('afternoon',):
                time_str = 'this afternoon'
            elif time_str in ('evening',):
                time_str = 'this evening'
            time_idx = TIME_ORDER.get(time_str, -1)
            timed = self.world.timed_locations.get(entity, [])
            for tidx, loc in timed:
                if tidx == time_idx:
                    return loc
            return 'unknown'

        # === IS X IN THE Y? (Tasks 6, 9, 10) ===
        m = re.match(r'is\s+(\w+)\s+in\s+the\s+(.+)', q, re.I)
        if m:
            entity = m.group(1).lower()
            location = m.group(2).strip().lower()
            # Definite location known
            if entity in self.world.locations:
                return 'yes' if self.world.locations[entity].lower() == location else 'no'
            # Indefinite location → "maybe" if location is one of the options
            if entity in self.world.indefinite:
                if location in {l.lower() for l in self.world.indefinite[entity]}:
                    return 'maybe'
                return 'no'
            # Negated
            if location in self.world.negated_locations.get(entity, set()):
                return 'no'
            return 'no'

        # === WHAT IS NORTH/SOUTH/... OF X? (Task 4) ===
        m = re.match(r'what\s+is\s+(?:to\s+the\s+)?(north|south|east|west|left|right|above|below)\s+of\s+the\s+(.+)', q, re.I)
        if m:
            direction = m.group(1).lower()
            entity = m.group(2).strip().lower()
            result = self.world.spatial.get(direction, {}).get(entity)
            return result if result else 'unknown'

        # === WHAT IS THE X [direction] OF? (Task 4 reversed) ===
        # "What is the bathroom east of?"
        m = re.match(r'what\s+is\s+the\s+(.+?)\s+(north|south|east|west|left|right|above|below)\s+of', q, re.I)
        if m:
            entity = m.group(1).strip().lower()
            direction = m.group(2).lower()
            # entity is [direction] of what? → find X where spatial[direction][X] == entity
            for base, target in self.world.spatial.get(direction, {}).items():
                if target.lower() == entity:
                    return base
            return 'unknown'

        # === IS X TO THE LEFT/RIGHT/ABOVE/BELOW OF Y? (Task 17) ===
        m = re.match(
            r'is\s+the\s+(.+?)\s+(?:to\s+the\s+)?(left|right|above|below|north|south|east|west)\s+(?:of\s+)?the\s+(.+)',
            q, re.I
        )
        if m:
            entity1 = m.group(1).strip().lower()
            direction = m.group(2).lower()
            entity2 = m.group(3).strip().lower()
            # Use 2D positions if available
            pos1 = self.world.positions.get(entity1)
            pos2 = self.world.positions.get(entity2)
            if pos1 and pos2:
                if direction in ('left', 'west'):
                    return 'yes' if pos1[0] < pos2[0] else 'no'
                elif direction in ('right', 'east'):
                    return 'yes' if pos1[0] > pos2[0] else 'no'
                elif direction in ('above', 'north'):
                    return 'yes' if pos1[1] > pos2[1] else 'no'
                elif direction in ('below', 'south'):
                    return 'yes' if pos1[1] < pos2[1] else 'no'
            # Fallback to direct/transitive check
            if self.world.spatial.get(direction, {}).get(entity2, '').lower() == entity1:
                return 'yes'
            if self._check_transitive_spatial(entity1, direction, entity2):
                return 'yes'
            if direction in DIR_OPPOSITES:
                opp = DIR_OPPOSITES[direction]
                if self.world.spatial.get(opp, {}).get(entity2, '').lower() == entity1:
                    return 'no'
                if self._check_transitive_spatial(entity1, opp, entity2):
                    return 'no'
            return 'no'

        # === WHAT DID X GIVE TO Y? (Task 5) ===
        m = re.match(r'what\s+did\s+(\w+)\s+give\s+to\s+(\w+)', q, re.I)
        if m:
            giver_q = m.group(1).lower()
            recipient_q = m.group(2).lower()
            for giver, given_item, recv in reversed(self.world.give_log):
                if giver == giver_q and recv == recipient_q:
                    return given_item
            return 'unknown'

        # === WHO DID X GIVE THE Y TO? (Task 5) ===
        m = re.match(r'who\s+did\s+(\w+)\s+give\s+the\s+(.+?)\s+to', q, re.I)
        if m:
            giver_q = m.group(1).lower()
            item = m.group(2).strip().lower()
            for giver, given_item, recv in reversed(self.world.give_log):
                if giver == giver_q and given_item == item:
                    return recv
            return 'unknown'

        # === WHO GAVE X THE Y? (Task 5) ===
        m = re.match(r'who\s+gave\s+(\w+)\s+the\s+(.+)', q, re.I)
        if m:
            recipient = m.group(1).lower()
            item = m.group(2).strip().lower()
            for giver, given_item, recv in reversed(self.world.give_log):
                if recv == recipient and given_item == item:
                    return giver
            return 'unknown'

        # === WHO GAVE THE X TO Y? (Task 5) ===
        m = re.match(r'who\s+gave\s+the\s+(.+?)\s+to\s+(\w+)', q, re.I)
        if m:
            item = m.group(1).strip().lower()
            recipient = m.group(2).lower()
            for giver, given_item, recv in reversed(self.world.give_log):
                if given_item == item and recv == recipient:
                    return giver
            return 'unknown'

        # === WHO GAVE THE X? (Task 5 — no recipient specified) ===
        m = re.match(r'who\s+gave\s+the\s+(.+)', q, re.I)
        if m:
            item = m.group(1).strip().lower()
            # Return the LAST giver of this item
            for giver, given_item, recipient in reversed(self.world.give_log):
                if given_item == item:
                    return giver
            return 'unknown'

        # === WHO RECEIVED THE X? ===
        m = re.match(r'who\s+received\s+the\s+(.+)', q, re.I)
        if m:
            item = m.group(1).strip().lower()
            # Return the LAST person to receive this item
            for giver, given_item, recipient in reversed(self.world.give_log):
                if given_item == item:
                    return recipient
            return 'unknown'

        # === HOW MANY OBJECTS IS X CARRYING? (Task 7) ===
        m = re.match(r'how\s+many\s+objects?\s+is\s+(\w+)\s+(?:carrying|holding)', q, re.I)
        if m:
            entity = m.group(1).lower()
            count = len(self.world.possessions.get(entity, set()))
            return NUM_WORDS.get(count, str(count))

        # === WHAT IS X CARRYING? (Task 8) ===
        m = re.match(r'what\s+is\s+(\w+)\s+(?:carrying|holding)', q, re.I)
        if m:
            entity = m.group(1).lower()
            items = self.world.possessions.get(entity, set())
            if items:
                return ','.join(sorted(items))
            return 'nothing'

        # === WHAT IS X AFRAID OF? (Task 15 — deduction) ===
        m = re.match(r'what\s+is\s+(\w+)\s+afraid\s+of', q, re.I)
        if m:
            entity = m.group(1).lower()
            etype = self.world.is_a.get(entity)
            if etype:
                # Try plural form (rules stored as "wolves are afraid of mice")
                plural = etype + 's' if not etype.endswith('s') else etype
                # Also try irregular plurals
                irregular = {'mouse': 'mice', 'wolf': 'wolves', 'sheep': 'sheep',
                             'cat': 'cats', 'dog': 'dogs'}
                plurals_to_try = [plural]
                if etype in irregular:
                    plurals_to_try.append(irregular[etype])
                for p in plurals_to_try:
                    fear = self.world.rules.get(p, {}).get('afraid_of')
                    if fear:
                        return singularize(fear)
                # Try singular
                fear = self.world.rules.get(etype, {}).get('afraid_of')
                if fear:
                    return singularize(fear)
            return 'unknown'

        # === WHAT COLOR IS X? (Task 16 — induction) ===
        m = re.match(r'what\s+colou?r\s+is\s+(\w+)', q, re.I)
        if m:
            entity = m.group(1).lower()
            # Direct property
            color = self.world.properties.get(entity, {}).get('color')
            if color:
                return color
            # Induction: same type → use MOST RECENTLY mentioned same-type entity's color
            etype = self.world.is_a.get(entity)
            if etype:
                last_color = None
                for other_entity, other_type in self.world.is_a.items():
                    if other_type == etype and other_entity != entity:
                        other_color = self.world.properties.get(other_entity, {}).get('color')
                        if other_color:
                            last_color = other_color  # dict preserves insertion order
                if last_color:
                    return last_color
            return 'unknown'

        # === DOES X FIT IN Y? / IS X BIGGER THAN Y? (Task 18) ===
        m = re.match(r'does\s+the\s+(.+?)\s+fit\s+in\s+the\s+(.+)', q, re.I)
        if m:
            item = m.group(1).strip().lower()
            container = m.group(2).strip().lower()
            if self._is_smaller(item, container):
                return 'yes'
            if self._is_smaller(container, item):
                return 'no'
            return 'no'

        m = re.match(r'is\s+the\s+(.+?)\s+bigger\s+than\s+the\s+(.+)', q, re.I)
        if m:
            a = m.group(1).strip().lower()
            b = m.group(2).strip().lower()
            if self._is_smaller(b, a):
                return 'yes'
            if self._is_smaller(a, b):
                return 'no'
            return 'no'

        # === PATH FINDING (Task 19) ===
        # "How do you go from the bathroom to the hallway?"
        m = re.match(r'how\s+do\s+you\s+go\s+from\s+the\s+(.+?)\s+to\s+the\s+(.+)', q, re.I)
        if m:
            start = m.group(1).strip().lower()
            end = m.group(2).strip().lower()
            path = self._find_path(start, end)
            return path if path else 'unknown'

        # === AGENT MOTIVATION (Task 20) ===
        # "Where will jason go?"
        m = re.match(r'where\s+will\s+(\w+)\s+go', q, re.I)
        if m:
            entity = m.group(1).lower()
            motivation = self.world.motivations.get(entity)
            if motivation and motivation in MOTIVATION_RULES:
                return MOTIVATION_RULES[motivation][0]
            return 'unknown'

        # "Why did jason go to the kitchen?"
        m = re.match(r'why\s+did\s+(\w+)\s+(?:go|travel|journey|move)\s+to\s+the\s+(.+)', q, re.I)
        if m:
            entity = m.group(1).lower()
            motivation = self.world.motivations.get(entity)
            if motivation:
                return motivation
            return 'unknown'

        # "Why did jason get the milk?"
        m = re.match(r'why\s+did\s+(\w+)\s+(?:get|grab|pick\s+up|take)\s+the\s+(.+)', q, re.I)
        if m:
            entity = m.group(1).lower()
            motivation = self.world.motivations.get(entity)
            if motivation:
                return motivation
            return 'unknown'

        return 'unknown'

    def _resolve_positions(self):
        """Resolve 2D positions from constraints."""
        # Direction deltas: entity1 is [direction] of entity2
        # e.g., "A is left of B" → A.x = B.x - 1
        dx = {'left': -1, 'right': 1, 'east': 1, 'west': -1, 'north': 0, 'south': 0, 'above': 0, 'below': 0}
        dy = {'left': 0, 'right': 0, 'east': 0, 'west': 0, 'north': 1, 'south': -1, 'above': 1, 'below': -1}

        # Iterative constraint propagation
        positions = self.world.positions
        for _ in range(10):  # max iterations
            changed = False
            for e1, direction, e2 in self.world.pos_constraints:
                ddx = dx.get(direction, 0)
                ddy = dy.get(direction, 0)
                if e2 in positions and e1 not in positions:
                    positions[e1] = (positions[e2][0] + ddx, positions[e2][1] + ddy)
                    changed = True
                elif e1 in positions and e2 not in positions:
                    positions[e2] = (positions[e1][0] - ddx, positions[e1][1] - ddy)
                    changed = True
                elif e1 not in positions and e2 not in positions:
                    positions[e2] = (0, 0)
                    positions[e1] = (ddx, ddy)
                    changed = True
            if not changed:
                break

    def _check_transitive_spatial(self, entity1: str, direction: str, entity2: str, visited: Set[str] = None) -> bool:
        """Check if entity1 is transitively in direction of entity2."""
        if visited is None:
            visited = set()
        if entity1 in visited:
            return False
        visited.add(entity1)

        dir_map = self.world.spatial.get(direction, {})
        # entity1 is [direction] of entity2 means dir_map[entity2] == entity1
        if dir_map.get(entity2, '').lower() == entity1:
            return True

        # Transitive: find intermediate nodes
        # If A is left of B, and B is left of C, then A is left of C
        for base, target in dir_map.items():
            if target.lower() == entity1:
                # entity1 is [direction] of base
                # Check if base is [direction] of entity2
                if self._check_transitive_spatial(base, direction, entity2, visited):
                    return True

        return False

    def _is_smaller(self, a: str, b: str, visited: Set[str] = None) -> bool:
        """Check if a is smaller than b (transitively)."""
        if visited is None:
            visited = set()
        if a in visited:
            return False
        visited.add(a)

        # Direct evidence
        for smaller, bigger in self.world.fits_inside:
            if smaller == a and bigger == b:
                return True
        for bigger, smaller in self.world.size_order:
            if smaller == a and bigger == b:
                return True

        # Transitive
        for smaller, bigger in self.world.fits_inside:
            if smaller == a:
                if self._is_smaller(bigger, b, visited.copy()) or bigger == b:
                    return True
                # a fits in bigger, if bigger <= b then a < b
        for bigger_item, smaller_item in self.world.size_order:
            if smaller_item == a:
                if bigger_item == b or self._is_smaller(bigger_item, b, visited.copy()):
                    return True

        return False

    def _find_path(self, start: str, end: str) -> Optional[str]:
        """BFS to find path from start to end using spatial relations."""
        # Build adjacency graph from spatial relations
        # For each direction, spatial[dir][entity2] = entity1 means entity1 is [dir] of entity2
        # So from entity2, going [dir] leads to entity1
        # But we want: from entity2, which direction to go to reach entity1?
        # If entity1 is north of entity2, then from entity2 go north to reach entity1

        graph: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # node → [(neighbor, direction)]

        for direction, mapping in self.world.spatial.items():
            for base, target in mapping.items():
                # target is [direction] of base
                # From base, go [direction] to reach target
                graph[base].append((target, direction))

        # BFS
        from collections import deque
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            current, path = queue.popleft()
            if current == end:
                return ','.join(path) if path else ''

            for neighbor, direction in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [direction[0]]))  # n/s/e/w

        return None


def parse_babi_file(filepath: str) -> List[List[str]]:
    """Parse a bAbI file into individual stories."""
    stories = []
    current_story = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^(\d+)\s+', line)
            if m and int(m.group(1)) == 1 and current_story:
                stories.append(current_story)
                current_story = []
            current_story.append(line)

    if current_story:
        stories.append(current_story)

    return stories


def evaluate_task(engine: BabiEngine, filepath: str, max_stories: int = 0) -> Tuple[int, int, List[str]]:
    """Evaluate engine on a bAbI task file. Returns (correct, total, errors)."""
    stories = parse_babi_file(filepath)
    if max_stories > 0:
        stories = stories[:max_stories]

    correct = 0
    total = 0
    errors = []

    for story in stories:
        results = engine.solve_story(story)
        for question, predicted, gold in results:
            total += 1
            pred_clean = predicted.lower().strip() if predicted else ''
            gold_clean = gold.lower().strip()

            # Handle multi-word answers and comma-separated lists
            if pred_clean == gold_clean:
                correct += 1
            elif set(pred_clean.split(',')) == set(gold_clean.split(',')):
                correct += 1
            else:
                if len(errors) < 5:  # Keep first 5 errors per task
                    errors.append(f"Q: {question} | Got: {predicted} | Expected: {gold}")

    return correct, total, errors
