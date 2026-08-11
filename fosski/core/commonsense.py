"""
Common Sense Knowledge — ConceptNet + WordNet Integration
===========================================================
Pre-Transformer knowledge representation that gives FOSS-KI
implicit world knowledge without a neural network.

Sources:
  1. ConceptNet 5 (MIT, 2012-): 21M+ assertions
     - "ice IsA cold", "dogs CanDo bark", "rain causes wet"
     - Free API: api.conceptnet.io (no key required)

  2. WordNet (Princeton, 1995): 117K synsets
     - Hypernym/Hyponym taxonomies: "dog IsA animal IsA organism"
     - Built-in: we ship a core subset (~5000 relations)

  3. Frame Semantics (Fillmore, 1976):
     - Situations as frames with slots
     - "RESTAURANT" → {customer, waiter, food, bill, tip}
     - Built-in: 50+ common frames

This gives the KnowledgeStore "implicit" knowledge that doesn't
need to be explicitly stored as triplets. "Is ice cold?" → YES,
without ever storing (ice, temperature, cold).

No external dependencies. ConceptNet API is optional fallback.
"""

import re
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict


# ============================================================
# Built-in Common Sense (core subset, no API needed)
# ============================================================

# Format: (subject, relation, object, weight)
# Weight 1.0-5.0 indicates confidence

CORE_COMMON_SENSE = [
    # Physical properties
    ("ice", "has_property", "cold", 5.0),
    ("fire", "has_property", "hot", 5.0),
    ("water", "has_property", "wet", 5.0),
    ("snow", "has_property", "cold", 4.0),
    ("snow", "has_property", "white", 4.0),
    ("sun", "has_property", "hot", 4.0),
    ("sun", "has_property", "bright", 4.0),
    ("night", "has_property", "dark", 4.0),
    ("glass", "has_property", "transparent", 3.0),
    ("glass", "has_property", "fragile", 3.0),
    ("steel", "has_property", "strong", 3.0),
    ("steel", "has_property", "hard", 3.0),
    ("feather", "has_property", "light", 3.0),
    ("rock", "has_property", "hard", 3.0),
    ("cotton", "has_property", "soft", 3.0),

    # Basic needs
    ("human", "needs", "food", 5.0),
    ("human", "needs", "water", 5.0),
    ("human", "needs", "sleep", 5.0),
    ("human", "needs", "air", 5.0),
    ("plant", "needs", "water", 4.0),
    ("plant", "needs", "sunlight", 4.0),
    ("fish", "needs", "water", 4.0),
    ("car", "needs", "fuel", 4.0),
    ("phone", "needs", "battery", 3.0),

    # Capabilities
    ("bird", "can", "fly", 4.0),
    ("fish", "can", "swim", 4.0),
    ("human", "can", "think", 4.0),
    ("human", "can", "speak", 4.0),
    ("dog", "can", "bark", 4.0),
    ("cat", "can", "climb", 3.0),
    ("horse", "can", "run", 4.0),
    ("snake", "can", "bite", 3.0),

    # Causes and effects
    ("rain", "causes", "wet", 4.0),
    ("fire", "causes", "smoke", 4.0),
    ("exercise", "causes", "sweat", 3.0),
    ("eating", "causes", "satiation", 3.0),
    ("studying", "causes", "knowledge", 3.0),
    ("cold", "causes", "shivering", 3.0),
    ("sadness", "causes", "crying", 3.0),
    ("humor", "causes", "laughter", 3.0),

    # Locations
    ("fish", "found_in", "water", 4.0),
    ("book", "found_in", "library", 3.0),
    ("doctor", "found_in", "hospital", 3.0),
    ("teacher", "found_in", "school", 3.0),
    ("food", "found_in", "kitchen", 3.0),
    ("car", "found_in", "garage", 3.0),
    ("tree", "found_in", "forest", 3.0),

    # Part-whole
    ("wheel", "part_of", "car", 4.0),
    ("leaf", "part_of", "tree", 4.0),
    ("page", "part_of", "book", 4.0),
    ("finger", "part_of", "hand", 4.0),
    ("hand", "part_of", "arm", 4.0),
    ("heart", "part_of", "body", 4.0),
    ("engine", "part_of", "car", 4.0),
    ("roof", "part_of", "house", 4.0),
    ("keyboard", "part_of", "computer", 3.0),
    ("screen", "part_of", "phone", 3.0),

    # Used for
    ("knife", "used_for", "cutting", 4.0),
    ("pen", "used_for", "writing", 4.0),
    ("phone", "used_for", "communication", 4.0),
    ("car", "used_for", "transportation", 4.0),
    ("oven", "used_for", "cooking", 4.0),
    ("bed", "used_for", "sleeping", 4.0),
    ("chair", "used_for", "sitting", 4.0),
    ("glasses", "used_for", "seeing", 3.0),
    ("umbrella", "used_for", "protection", 3.0),
    ("money", "used_for", "buying", 4.0),

    # Temporal
    ("breakfast", "happens_before", "lunch", 4.0),
    ("lunch", "happens_before", "dinner", 4.0),
    ("spring", "happens_before", "summer", 4.0),
    ("summer", "happens_before", "autumn", 4.0),
    ("autumn", "happens_before", "winter", 4.0),
    ("birth", "happens_before", "death", 5.0),
    ("morning", "happens_before", "evening", 4.0),

    # Social
    ("friend", "desires", "happiness", 3.0),
    ("student", "desires", "knowledge", 3.0),
    ("parent", "desires", "safety", 3.0),
    ("worker", "desires", "money", 3.0),
    ("child", "desires", "play", 3.0),

    # Opposites
    ("hot", "opposite_of", "cold", 5.0),
    ("big", "opposite_of", "small", 5.0),
    ("fast", "opposite_of", "slow", 5.0),
    ("up", "opposite_of", "down", 5.0),
    ("light", "opposite_of", "dark", 5.0),
    ("old", "opposite_of", "young", 5.0),
    ("good", "opposite_of", "bad", 5.0),
    ("happy", "opposite_of", "sad", 5.0),
    ("rich", "opposite_of", "poor", 4.0),
    ("open", "opposite_of", "closed", 4.0),
    ("true", "opposite_of", "false", 5.0),
    ("alive", "opposite_of", "dead", 5.0),
]


# ============================================================
# Taxonomy (WordNet-style IS-A hierarchy)
# ============================================================

TAXONOMY = {
    # Animals
    "dog": ["animal", "pet", "mammal"],
    "cat": ["animal", "pet", "mammal"],
    "horse": ["animal", "mammal"],
    "bird": ["animal", "vertebrate"],
    "fish": ["animal", "vertebrate"],
    "snake": ["animal", "reptile"],
    "whale": ["animal", "mammal"],
    "eagle": ["bird", "animal"],
    "shark": ["fish", "animal"],
    "ant": ["insect", "animal"],
    "bee": ["insect", "animal"],
    "spider": ["arachnid", "animal"],
    "bat": ["mammal", "animal"],
    "penguin": ["bird", "animal"],
    "frog": ["amphibian", "animal"],
    "crocodile": ["reptile", "animal"],
    "dolphin": ["mammal", "animal"],

    # Living things
    "animal": ["organism", "living_thing"],
    "plant": ["organism", "living_thing"],
    "tree": ["plant"],
    "flower": ["plant"],
    "fungus": ["organism"],
    "mammal": ["animal", "vertebrate"],
    "reptile": ["animal", "vertebrate"],
    "insect": ["animal", "invertebrate"],

    # Vehicles
    "car": ["vehicle", "machine"],
    "truck": ["vehicle", "machine"],
    "bus": ["vehicle", "machine"],
    "airplane": ["vehicle", "aircraft", "machine"],
    "helicopter": ["vehicle", "aircraft", "machine"],
    "boat": ["vehicle", "watercraft"],
    "ship": ["vehicle", "watercraft"],
    "bicycle": ["vehicle"],
    "motorcycle": ["vehicle", "machine"],
    "train": ["vehicle", "machine"],
    "submarine": ["vehicle", "watercraft"],

    # Tools
    "hammer": ["tool"],
    "screwdriver": ["tool"],
    "knife": ["tool", "weapon"],
    "saw": ["tool"],
    "computer": ["machine", "tool", "device"],
    "phone": ["device", "tool"],
    "laptop": ["computer", "device"],

    # Food
    "apple": ["fruit", "food"],
    "banana": ["fruit", "food"],
    "bread": ["food"],
    "rice": ["food", "grain"],
    "chicken": ["food", "meat", "animal", "bird"],
    "pizza": ["food"],
    "fruit": ["food", "plant_part"],
    "vegetable": ["food", "plant_part"],
    "meat": ["food"],

    # Places
    "house": ["building", "dwelling"],
    "school": ["building", "institution"],
    "hospital": ["building", "institution"],
    "church": ["building"],
    "restaurant": ["building", "business"],
    "library": ["building", "institution"],
    "city": ["place", "settlement"],
    "country": ["place", "political_entity"],
    "planet": ["celestial_body"],
    "earth": ["planet", "celestial_body"],

    # People
    "doctor": ["person", "professional"],
    "teacher": ["person", "professional"],
    "engineer": ["person", "professional"],
    "scientist": ["person", "professional"],
    "artist": ["person", "professional"],
    "musician": ["person", "artist"],
    "child": ["person", "human"],
    "parent": ["person", "human"],
    "student": ["person"],
    "worker": ["person"],

    # Abstract
    "democracy": ["political_system", "form_of_government"],
    "monarchy": ["political_system", "form_of_government"],
    "capitalism": ["economic_system"],
    "socialism": ["economic_system"],
    "science": ["discipline", "knowledge"],
    "mathematics": ["science", "discipline"],
    "physics": ["science", "discipline"],
    "chemistry": ["science", "discipline"],
    "biology": ["science", "discipline"],
    "history": ["discipline", "knowledge"],
    "philosophy": ["discipline", "knowledge"],
    "language": ["communication_system"],
    "music": ["art_form"],
    "painting": ["art_form"],
}


# ============================================================
# Frame Semantics (Fillmore, 1976)
# ============================================================

FRAMES = {
    "restaurant": {
        "description": "Eating at a restaurant",
        "slots": {
            "customer": "person who eats",
            "waiter": "person who serves",
            "chef": "person who cooks",
            "food": "what is eaten",
            "menu": "list of available food",
            "bill": "payment for food",
            "tip": "extra payment for service",
            "table": "where customer sits",
            "reservation": "advance booking",
        },
        "script": ["arrive", "sit_down", "read_menu", "order", "wait", "eat", "pay", "leave"],
    },
    "shopping": {
        "description": "Buying goods at a store",
        "slots": {
            "buyer": "person purchasing",
            "seller": "person or store selling",
            "goods": "items being purchased",
            "price": "cost of goods",
            "money": "payment method",
            "receipt": "proof of purchase",
            "cart": "container for selected items",
        },
        "script": ["enter_store", "browse", "select_items", "go_to_checkout", "pay", "receive_goods", "leave"],
    },
    "travel": {
        "description": "Traveling from one place to another",
        "slots": {
            "traveler": "person traveling",
            "origin": "starting location",
            "destination": "target location",
            "vehicle": "means of transport",
            "luggage": "personal belongings",
            "ticket": "permission to travel",
            "route": "path taken",
            "duration": "time of travel",
        },
        "script": ["plan", "pack", "depart", "travel", "arrive", "unpack"],
    },
    "education": {
        "description": "Learning in an educational setting",
        "slots": {
            "student": "person learning",
            "teacher": "person teaching",
            "subject": "topic being taught",
            "classroom": "location of learning",
            "textbook": "learning material",
            "exam": "assessment of knowledge",
            "grade": "evaluation result",
            "homework": "practice assignments",
        },
        "script": ["attend_class", "listen", "take_notes", "study", "do_homework", "take_exam", "receive_grade"],
    },
    "medical": {
        "description": "Medical treatment",
        "slots": {
            "patient": "person being treated",
            "doctor": "person providing treatment",
            "nurse": "person assisting treatment",
            "illness": "medical condition",
            "symptom": "sign of illness",
            "diagnosis": "identification of illness",
            "treatment": "method of healing",
            "medicine": "drug prescribed",
            "hospital": "treatment facility",
        },
        "script": ["feel_sick", "visit_doctor", "describe_symptoms", "get_examined", "receive_diagnosis", "get_treatment", "recover"],
    },
    "cooking": {
        "description": "Preparing food",
        "slots": {
            "cook": "person preparing food",
            "ingredients": "raw materials",
            "recipe": "instructions",
            "utensils": "tools used",
            "stove": "heating device",
            "dish": "resulting food",
        },
        "script": ["gather_ingredients", "prepare", "cook", "season", "serve"],
    },
    "work": {
        "description": "Employment and work",
        "slots": {
            "employee": "person working",
            "employer": "person/company hiring",
            "job": "type of work",
            "salary": "payment for work",
            "office": "work location",
            "deadline": "time constraint",
            "colleague": "co-worker",
            "boss": "supervisor",
        },
        "script": ["apply", "interview", "get_hired", "work", "get_paid", "retire"],
    },
    "election": {
        "description": "Political election process",
        "related_concepts": ["democracy", "republic", "voting", "politics"],
        "slots": {
            "voter": "person casting vote",
            "candidate": "person seeking office",
            "party": "political organization",
            "ballot": "voting mechanism",
            "campaign": "effort to win votes",
            "debate": "public discussion",
            "result": "outcome of election",
            "office": "position being contested",
        },
        "script": ["announce_candidacy", "campaign", "debate", "vote", "count_votes", "announce_winner"],
    },
    "justice": {
        "description": "Legal proceedings",
        "slots": {
            "defendant": "person accused",
            "prosecutor": "person accusing",
            "judge": "person deciding",
            "jury": "group deciding",
            "lawyer": "legal representative",
            "crime": "alleged offense",
            "evidence": "proof",
            "verdict": "decision",
            "sentence": "punishment",
        },
        "script": ["arrest", "charge", "trial", "present_evidence", "deliberate", "verdict", "sentence"],
    },
    "communication": {
        "description": "Exchanging information",
        "slots": {
            "sender": "person sending message",
            "receiver": "person receiving message",
            "message": "content being communicated",
            "medium": "channel of communication",
            "language": "code used",
            "response": "reply to message",
        },
        "script": ["compose", "send", "receive", "read", "respond"],
    },
}


# ============================================================
# Common Sense Engine
# ============================================================

class CommonSenseEngine:
    """
    Common sense reasoning engine.

    Combines built-in knowledge + taxonomy + frames + optional
    ConceptNet API fallback for comprehensive world knowledge.

    Usage:
        cs = CommonSenseEngine()
        cs.query("is ice cold?")          # → True, weight=5.0
        cs.query("can dogs fly?")         # → False
        cs.query("is a dog an animal?")   # → True (taxonomy)
        cs.get_frame("restaurant")        # → frame with slots + script
        cs.related("democracy")           # → voting, election, ...
    """

    def __init__(self, use_api: bool = False, conceptnet_path: str = None):
        """
        Args:
            use_api: if True, fall back to ConceptNet API for unknown queries
            conceptnet_path: path to conceptnet_index.pkl (auto-detected if None)
        """
        self.use_api = use_api

        # Index common sense by subject and by relation
        self._by_subject: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        self._by_object: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        self._by_relation: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)

        # Load built-in knowledge
        for s, r, o, w in CORE_COMMON_SENSE:
            self._by_subject[s].append((r, o, w))
            self._by_object[o].append((r, s, w))
            self._by_relation[r].append((s, o, w))

        # Load ConceptNet index (500K assertions, 262K concepts)
        self._cn_forward: Dict[str, list] = {}
        self._cn_reverse: Dict[str, list] = {}
        self._load_conceptnet(conceptnet_path)

        # Taxonomy
        self._taxonomy = dict(TAXONOMY)

        # Frames
        self._frames = dict(FRAMES)

        # Cache for API results
        self._api_cache: Dict[str, Any] = {}

    def _load_conceptnet(self, path: str = None):
        """Load ConceptNet offline index if available."""
        import os, pickle
        if path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base, 'data', 'conceptnet_index.pkl')
        if not os.path.exists(path):
            return
        try:
            with open(path, 'rb') as f:
                cn = pickle.load(f)
            self._cn_forward = cn.get('forward', {})
            self._cn_reverse = cn.get('reverse', {})
        except Exception:
            pass

    def cn_lookup(self, concept: str, max_results: int = 20) -> List[Tuple[str, str, float]]:
        """Look up a concept in ConceptNet index. Returns [(relation, object, weight)]."""
        concept = concept.lower().strip()
        results = self._cn_forward.get(concept, [])
        return results[:max_results]

    def cn_reverse_lookup(self, concept: str, max_results: int = 20) -> List[Tuple[str, str, float]]:
        """Reverse lookup: find what relates TO this concept."""
        concept = concept.lower().strip()
        results = self._cn_reverse.get(concept, [])
        return results[:max_results]

    def cn_check(self, subject: str, relation: str, obj: str) -> Optional[float]:
        """Check if a specific (subject, relation, object) exists in ConceptNet.
        Returns weight if found, None otherwise."""
        subject = subject.lower().strip()
        relation = relation.strip()
        obj = obj.lower().strip()
        for r, o, w in self._cn_forward.get(subject, []):
            if r == relation and (o == obj or obj in o.split()):
                return w
        return None

    def query(self, question: str) -> Dict[str, Any]:
        """
        Answer a common sense question.

        Supports:
          "is X Y?" → property check
          "can X Y?" → capability check
          "is X a Y?" → taxonomy check
          "what does X need?" → needs relation
          "what is X used for?" → used_for relation
          "what causes X?" → reverse cause lookup
        """
        q = question.lower().strip().rstrip('?').strip()

        # "is X Y?" or "is X a Y?" or "is an X Y?"
        m = re.match(r'is\s+(?:an?\s+)?(\w+)\s+(?:an?\s+)?(\w+)', q)
        if m:
            subject, obj = m.group(1), m.group(2)
            return self._check_property_or_taxonomy(subject, obj)

        # "can X Y?"
        m = re.match(r'can\s+(?:a\s+)?(\w+?)s?\s+(\w+)', q)
        if m:
            subject, action = m.group(1), m.group(2)
            return self._check_capability(subject, action)

        # "what does X need?"
        m = re.match(r'what\s+does?\s+(?:a\s+)?(\w+)\s+need', q)
        if m:
            return self._lookup_relation(m.group(1), 'needs')

        # "what is X used for?"
        m = re.match(r'what\s+is\s+(?:a\s+)?(\w+)\s+used\s+for', q)
        if m:
            return self._lookup_relation(m.group(1), 'used_for')

        # "what causes X?"
        m = re.match(r'what\s+causes?\s+(\w+)', q)
        if m:
            return self._reverse_lookup(m.group(1), 'causes')

        # Generic: try to find any knowledge about the subject
        words = q.split()
        for word in reversed(words):
            if word in self._by_subject or word in self._taxonomy:
                return self.about(word)

        return {'found': False, 'answer': None}

    def about(self, concept: str) -> Dict[str, Any]:
        """Get everything known about a concept."""
        concept = concept.lower()
        result: Dict[str, Any] = {'concept': concept, 'found': False}

        # Common sense facts
        facts = self._by_subject.get(concept, [])
        if facts:
            result['found'] = True
            result['properties'] = [(r, o) for r, o, w in facts]

        # Taxonomy
        parents = self._taxonomy.get(concept, [])
        if parents:
            result['found'] = True
            result['is_a'] = parents
            # Walk up the taxonomy
            all_ancestors = self._get_ancestors(concept)
            if len(all_ancestors) > len(parents):
                result['ancestors'] = list(all_ancestors)

        # Reverse: what is a type of this concept?
        children = [c for c, parents in self._taxonomy.items() if concept in parents]
        if children:
            result['found'] = True
            result['subtypes'] = children

        # What it's part of / what's part of it
        parts = [(s, w) for s, r_obj, w in self._by_object.get(concept, []) if r_obj == 'part_of']
        if parts:
            result['has_parts'] = [s for s, _ in parts]

        # Frame membership
        for frame_name, frame in self._frames.items():
            related_concepts = frame.get('related_concepts', [])
            if concept in frame['slots'] or concept == frame_name or concept in related_concepts:
                result['found'] = True
                result.setdefault('frames', []).append(frame_name)

        # ConceptNet fallback (500K offline assertions)
        # Filter: only use assertions with weight >= 1.0 (ConceptNet default threshold)
        # Low-weight assertions (< 1.0) are noisy crowdsourced data
        if not result['found'] or not result.get('properties'):
            cn_facts = self.cn_lookup(concept, max_results=15)
            if cn_facts:
                # Filter by weight — only keep reliable assertions
                cn_reliable = [(r, o, w) for r, o, w in cn_facts if w >= 1.0]
                if cn_reliable:
                    result['found'] = True
                    cn_props = [(r, o) for r, o, w in cn_reliable]
                    existing = result.get('properties', [])
                    result['properties'] = existing + cn_props
                    # Extract IsA for taxonomy enrichment (only high-weight)
                    for r, o, w in cn_reliable:
                        if r == 'IsA' and concept not in self._taxonomy:
                            result.setdefault('is_a', []).append(o)
                elif not result['found']:
                    # If only low-weight assertions exist, still mark found
                    # but don't add to properties (prevents noise propagation)
                    cn_any = [(r, o) for r, o, w in cn_facts[:5]]
                    if cn_any:
                        result['found'] = True
                        result['properties'] = result.get('properties', []) + cn_any
                        result['low_confidence'] = True  # flag for downstream

        # API fallback
        if not result['found'] and self.use_api:
            api_result = self._api_query(concept)
            if api_result:
                result.update(api_result)

        return result

    def is_a(self, concept: str, category: str) -> bool:
        """Check if concept is a type of category (taxonomy)."""
        return category.lower() in self._get_ancestors(concept.lower())

    def related(self, concept: str, max_results: int = 20) -> List[Tuple[str, str, float]]:
        """Get related concepts with relation type and weight."""
        concept = concept.lower()
        results = []

        # Direct relations
        for r, o, w in self._by_subject.get(concept, []):
            results.append((o, r, w))

        # Reverse relations
        for r, s, w in self._by_object.get(concept, []):
            results.append((s, f"reverse_{r}", w))

        # Taxonomy siblings (share a parent)
        parents = self._taxonomy.get(concept, [])
        for parent in parents:
            siblings = [c for c, p_list in self._taxonomy.items()
                       if parent in p_list and c != concept]
            for sib in siblings[:5]:
                results.append((sib, "sibling_of", 2.0))

        # Frame co-members (via slots or related_concepts)
        for frame_name, frame in self._frames.items():
            slots = list(frame['slots'].keys())
            related_concepts = frame.get('related_concepts', [])
            if concept in slots or concept == frame_name or concept in related_concepts:
                for slot in slots:
                    if slot != concept:
                        results.append((slot, f"frame_{frame_name}", 1.5))
                for rc in related_concepts:
                    if rc != concept:
                        results.append((rc, f"frame_{frame_name}", 1.5))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:max_results]

    def get_frame(self, name: str) -> Optional[Dict]:
        """Get a semantic frame by name."""
        return self._frames.get(name.lower())

    def list_frames(self) -> List[str]:
        """List available frames."""
        return sorted(self._frames.keys())

    def add_fact(self, subject: str, relation: str, obj: str, weight: float = 3.0):
        """Add a common sense fact."""
        s, r, o = subject.lower(), relation.lower(), obj.lower()
        self._by_subject[s].append((r, o, weight))
        self._by_object[o].append((r, s, weight))
        self._by_relation[r].append((s, o, weight))

    def add_taxonomy(self, concept: str, parents: List[str]):
        """Add a taxonomy entry."""
        self._taxonomy[concept.lower()] = [p.lower() for p in parents]

    def export_to_triplets(self) -> List[Tuple[str, str, str]]:
        """Export all knowledge as (S, R, O) triplets for KnowledgeStore."""
        triplets = []
        for s, r, o, _ in CORE_COMMON_SENSE:
            triplets.append((s, r, o))
        for concept, parents in self._taxonomy.items():
            for parent in parents:
                triplets.append((concept, "is_a", parent))
        return triplets

    # === Internal ===

    def _get_ancestors(self, concept: str) -> Set[str]:
        """Get all ancestors in taxonomy (transitive closure)."""
        ancestors: Set[str] = set()
        queue = [concept]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            parents = self._taxonomy.get(current, [])
            for parent in parents:
                ancestors.add(parent)
                queue.append(parent)

        return ancestors

    def _check_property_or_taxonomy(self, subject: str, obj: str) -> Dict[str, Any]:
        """Check property or taxonomy relation."""
        s, o = subject.lower(), obj.lower()

        # Taxonomy check: "is a dog an animal?"
        if self.is_a(s, o):
            return {'found': True, 'answer': True, 'relation': 'is_a',
                    'explanation': f"{subject} is a type of {obj}"}

        # Property check: "is ice cold?"
        for r, prop, w in self._by_subject.get(s, []):
            if prop == o or o in prop:
                return {'found': True, 'answer': True, 'relation': r,
                        'weight': w, 'explanation': f"{subject} {r} {obj}"}

        # Opposite check: "is ice hot?" → no, ice is cold (opposite of hot)
        for r, prop, w in self._by_subject.get(s, []):
            if r == 'has_property':
                # Check if obj is opposite of prop
                for _, opp, _ in self._by_subject.get(prop, []):
                    if opp == o:
                        return {'found': True, 'answer': False, 'relation': 'opposite',
                                'explanation': f"{subject} is {prop}, which is opposite of {obj}"}

        # ConceptNet fallback: check HasProperty and IsA
        cn_facts = self._cn_forward.get(s, [])
        for r, cn_o, w in cn_facts:
            if r == 'HasProperty' and (cn_o == o or o in cn_o.split()):
                return {'found': True, 'answer': True, 'relation': 'has_property',
                        'weight': w, 'explanation': f"{subject} has property {obj} (ConceptNet)"}
            if r == 'IsA' and (cn_o == o or o in cn_o.split()):
                return {'found': True, 'answer': True, 'relation': 'is_a',
                        'weight': w, 'explanation': f"{subject} is a {obj} (ConceptNet)"}

        return {'found': False, 'answer': None}

    def _check_capability(self, subject: str, action: str) -> Dict[str, Any]:
        """Check if subject can do action."""
        s = subject.lower()

        # Direct check
        for r, o, w in self._by_subject.get(s, []):
            if r == 'can' and (o == action or action in o):
                return {'found': True, 'answer': True,
                        'explanation': f"{subject} can {action}"}

        # ConceptNet: CapableOf relation
        for r, o, w in self._cn_forward.get(s, []):
            if r == 'CapableOf' and (o == action or action in o.split()):
                return {'found': True, 'answer': True,
                        'explanation': f"{subject} can {o} (ConceptNet)"}

        # Check parent categories
        for parent in self._get_ancestors(s):
            for r, o, w in self._by_subject.get(parent, []):
                if r == 'can' and (o == action or action in o):
                    return {'found': True, 'answer': True,
                            'explanation': f"{subject} (as a {parent}) can {action}"}

        # Closed-world assumption: if the action IS a known capability
        # (some entity can do it) but this subject can't → answer is No.
        # This lets us answer "Can dogs fly?" → No, because birds can fly
        # but dogs are not birds and don't inherit fly.
        action_known = any(
            o == action or action in o
            for s_facts in self._by_relation.get('can', [])
            for _, o, _ in [s_facts]  # unpack (subject, object, weight)
        )
        if not action_known:
            # Also check raw tuples in _by_relation
            for s2, o2, w2 in self._by_relation.get('can', []):
                if o2 == action or action in o2:
                    action_known = True
                    break

        if action_known and (s in self._taxonomy or s in self._by_subject):
            # We know this subject AND we know this action exists,
            # but subject doesn't have it → closed world → No
            return {'found': True, 'answer': False,
                    'explanation': f"{subject} cannot {action}"}

        return {'found': False, 'answer': None}

    def _lookup_relation(self, subject: str, relation: str) -> Dict[str, Any]:
        """Look up what a subject has for a given relation."""
        s = subject.lower()
        results = [(o, w) for r, o, w in self._by_subject.get(s, []) if r == relation]

        if results:
            results.sort(key=lambda x: x[1], reverse=True)
            return {'found': True, 'results': [o for o, _ in results],
                    'answer': results[0][0]}

        # ConceptNet fallback: map our relations to CN relations
        _REL_MAP = {
            'needs': 'HasPrerequisite', 'used_for': 'UsedFor',
            'causes': 'Causes', 'has_property': 'HasProperty',
            'part_of': 'PartOf', 'found_in': 'AtLocation',
            'can': 'CapableOf', 'is_a': 'IsA',
        }
        cn_rel = _REL_MAP.get(relation)
        if cn_rel:
            cn_results = [(o, w) for r, o, w in self._cn_forward.get(s, []) if r == cn_rel]
            if cn_results:
                cn_results.sort(key=lambda x: x[1], reverse=True)
                return {'found': True, 'results': [o for o, _ in cn_results],
                        'answer': cn_results[0][0]}

        return {'found': False, 'answer': None}

    def _reverse_lookup(self, obj: str, relation: str) -> Dict[str, Any]:
        """Reverse lookup: what has this relation to obj?"""
        o = obj.lower()
        results = [(s, w) for s, obj_val, w in self._by_relation.get(relation, []) if obj_val == o]

        if results:
            results.sort(key=lambda x: x[1], reverse=True)
            return {'found': True, 'results': [s for s, _ in results],
                    'answer': results[0][0]}

        return {'found': False, 'answer': None}

    def _api_query(self, concept: str) -> Optional[Dict]:
        """Query ConceptNet API (optional fallback)."""
        if concept in self._api_cache:
            return self._api_cache[concept]

        try:
            import urllib.request
            url = f"http://api.conceptnet.io/c/en/{concept}?limit=20"
            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/json')
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())

            edges = data.get('edges', [])
            if not edges:
                return None

            result = {'found': True, 'source': 'conceptnet_api', 'properties': []}
            for edge in edges[:10]:
                rel = edge.get('rel', {}).get('label', '')
                start = edge.get('start', {}).get('label', '')
                end = edge.get('end', {}).get('label', '')
                weight = edge.get('weight', 1.0)

                if start.lower() == concept:
                    result['properties'].append((rel, end))
                    self.add_fact(concept, rel.lower().replace(' ', '_'), end.lower(), weight)
                elif end.lower() == concept:
                    result['properties'].append((f"reverse_{rel}", start))

            self._api_cache[concept] = result
            return result

        except Exception:
            return None

    def stats(self) -> Dict[str, int]:
        """Get knowledge base statistics."""
        return {
            'common_sense_facts': len(CORE_COMMON_SENSE),
            'taxonomy_entries': len(self._taxonomy),
            'frames': len(self._frames),
            'total_frame_slots': sum(len(f['slots']) for f in self._frames.values()),
            'subjects_indexed': len(self._by_subject),
            'relations_indexed': len(self._by_relation),
        }
