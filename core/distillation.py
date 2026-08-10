"""
Distillation Engine — Complete Capability Distillation from Local LLMs
======================================================================
FOSS-KI identifies its own knowledge gaps via Dict-Index,
generates targeted questions, sends them to a local LLM
(Ollama/llama.cpp), extracts triplets from answers, verifies
them, and stores in the brain file. On-the-fly, no retraining.

Architecture (from pipeline_v4 Sovereign Discovery):
- Gap Detection: 6 gap types, entity-type-aware, Dict-Index O(1)
- Query Dedup: attempted_queries set prevents infinite loops
- Consistency Check: ask twice, discard if answers differ
- Conflict Detection: O(1) via forward index
- Reasoning Logger: every decision logged with WHY
- Feedback Loop: verify stored fact is queryable after store

This is Knowledge Distillation WITHOUT Gradient Descent.
Fact in → immediately queryable. No weights, no epochs.
"""

import json
import os
import time
import logging
import urllib.request
import urllib.error
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================================
# ENTITY-TYPE-AWARE GAP DETECTION
# ============================================================================

# Only ask questions that make sense for the entity type
ENTITY_TYPE_RELATIONS = {
    # === Programming & CS ===
    'programming_language': ['type', 'creator', 'founded', 'paradigm', 'used_for',
                             'syntax_example', 'similar_to', 'advantage', 'disadvantage'],
    'data_structure': ['type', 'complexity', 'used_for', 'operations', 'example',
                       'advantage', 'disadvantage', 'similar_to'],
    'algorithm': ['type', 'complexity', 'used_for', 'input', 'output',
                  'invented_by', 'example', 'similar_to'],
    'design_pattern': ['type', 'used_for', 'example', 'similar_to', 'components',
                       'advantage', 'disadvantage'],
    'protocol': ['type', 'used_for', 'layer', 'port', 'creator', 'similar_to'],
    'file_format': ['type', 'used_for', 'creator', 'extension', 'similar_to'],
    'library': ['type', 'language', 'used_for', 'creator', 'similar_to'],
    # === Language & Grammar ===
    'grammar_concept': ['type', 'definition', 'example', 'used_in', 'similar_to',
                        'opposite'],
    'word_class': ['type', 'definition', 'example', 'function', 'similar_to'],
    'linguistic_concept': ['type', 'definition', 'example', 'used_in', 'similar_to'],
    # === Math & Science ===
    'math_concept': ['type', 'definition', 'formula', 'used_for', 'example',
                     'invented_by', 'similar_to'],
    'science_concept': ['type', 'definition', 'used_for', 'example', 'discovered_by'],
    # === General ===
    'country': ['capital', 'language', 'currency', 'population', 'location',
                'borders', 'founded', 'type'],
    'city': ['country', 'population', 'location', 'founded', 'type'],
    'person': ['born', 'died', 'nationality', 'occupation', 'known_as', 'type'],
    'company': ['creator', 'founded', 'location', 'industry', 'type'],
    'language': ['type', 'creator', 'founded'],
    'invention': ['creator', 'founded', 'type'],
    'default': ['type', 'used_for', 'example', 'similar_to', 'creator'],
}

# Question templates per relation
REL_TO_QUESTION = {
    'type': "What is {entity}? Answer in one sentence.",
    # Programming & CS
    'paradigm': "What programming paradigm does {entity} use? Answer briefly.",
    'used_for': "What is {entity} used for? Answer in one sentence.",
    'syntax_example': "Give a simple code example of {entity}. One line only.",
    'advantage': "What is the main advantage of {entity}? One sentence.",
    'disadvantage': "What is the main disadvantage of {entity}? One sentence.",
    'similar_to': "What is {entity} similar to or an alternative for? Name one.",
    'complexity': "What is the time complexity of {entity}? Answer briefly.",
    'operations': "What operations does {entity} support? List 3-5, comma-separated.",
    'input': "What input does {entity} take? Answer briefly.",
    'output': "What output does {entity} produce? Answer briefly.",
    'invented_by': "Who invented or discovered {entity}?",
    'components': "What are the main components of {entity}? List them briefly.",
    'layer': "What OSI layer does {entity} operate on?",
    'port': "What port number does {entity} typically use?",
    'extension': "What file extension does {entity} use?",
    # Language & Grammar
    'definition': "Define {entity} in one sentence.",
    'example': "Give one example of {entity}.",
    'function': "What is the function of {entity} in a sentence? Answer briefly.",
    'opposite': "What is the opposite of {entity}?",
    'used_in': "Where or when is {entity} used? Answer briefly.",
    # Math
    'formula': "What is the formula for {entity}? Answer briefly.",
    'discovered_by': "Who discovered {entity}?",
    # General (kept from before)
    'capital': "What is the capital of {entity}?",
    'language': "What language is spoken in {entity}?",
    'currency': "What currency is used in {entity}?",
    'population': "What is the population of {entity}?",
    'location': "Where is {entity} located?",
    'creator': "Who created or founded {entity}?",
    'founded': "When was {entity} founded or created?",
    'borders': "What countries border {entity}?",
    'country': "What country is {entity} in?",
    'born': "When was {entity} born?",
    'died': "When did {entity} die?",
    'nationality': "What nationality is {entity}?",
    'occupation': "What is {entity}'s occupation?",
    'known_as': "What is {entity} also known as?",
    'industry': "What industry does {entity} operate in?",
}

# ============================================================================
# TRIPLET EXTRACTION PATTERNS
# ============================================================================

# ORDER MATTERS: specific before generic
# (regex, relation, subject_group, object_group)
EXTRACTION_PATTERNS = [
    # "The capital of X is Y" → (X, capital, Y)
    (r'(?:the\s+)?capital\s+(?:of|city\s+of)\s+(.+?)\s+is\s+(.+?)\.?$',
     'capital', 1, 2),
    # "X is the capital of Y" → (Y, capital, X)
    (r'^(.+?)\s+is\s+the\s+capital\s+(?:of|city\s+of)\s+(.+?)\.?$',
     'capital', 2, 1),
    # "The official language of X is Y"
    (r'(?:the\s+)?(?:official\s+)?language\s+(?:of|spoken\s+in)\s+(.+?)\s+is\s+(.+?)\.?$',
     'language', 1, 2),
    # "X speaks Y" / "They speak Y in X" — common LLM phrasing
    (r'^(?:in\s+)?(.+?),?\s+(?:the\s+)?(?:people|they)\s+speak\s+(.+?)\.?$',
     'language', 1, 2),
    # "X is located/situated in Y"
    (r'^(.+?)\s+is\s+(?:located|situated|found)\s+in\s+(.+?)\.?$',
     'location', 1, 2),
    # "X is in Y" (short form)
    (r'^(.+?)\s+is\s+in\s+(.+?)\.?$',
     'location', 1, 2),
    # "X was founded/created/established in YEAR"
    (r'^(.+?)\s+was\s+(?:founded|created|established|invented|born)\s+(?:in|on)\s+(\d{3,4})',
     'founded', 1, 2),
    # "X was founded/created by Y"
    (r'^(.+?)\s+was\s+(?:founded|created|invented|developed|built)\s+by\s+(.+?)\.?$',
     'creator', 1, 2),
    # "X has a population of Y"
    (r'^(.+?)\s+has\s+(?:a\s+)?population\s+of\s+(?:approximately\s+|about\s+|around\s+|over\s+)?(.+?)\.?$',
     'population', 1, 2),
    # "The population of X is Y"
    (r'(?:the\s+)?population\s+of\s+(.+?)\s+is\s+(?:approximately\s+|about\s+|around\s+|over\s+)?(.+?)\.?$',
     'population', 1, 2),
    # "X uses Y as its currency" / "The currency of X is Y"
    (r'^(.+?)\s+uses?\s+(?:the\s+)?(.+?)\s+as\s+(?:its\s+)?currency\.?$',
     'currency', 1, 2),
    (r'(?:the\s+)?(?:official\s+)?currency\s+of\s+(.+?)\s+is\s+(?:the\s+)?(.+?)\.?$',
     'currency', 1, 2),
    # "X borders Y" / "X is bordered by Y" / "X shares a border with Y"
    (r'^(.+?)\s+(?:borders?|is\s+bordered\s+by|shares?\s+a?\s*borders?\s+with)\s+(.+?)\.?$',
     'borders', 1, 2),
    # "X was born in Y"
    (r'^(.+?)\s+was\s+born\s+(?:in|on)\s+(.+?)\.?$',
     'born', 1, 2),
    # "X died in Y"
    (r'^(.+?)\s+died\s+(?:in|on)\s+(.+?)\.?$',
     'died', 1, 2),
    # "X is known as Y" / "X is also known as Y"
    (r'^(.+?)\s+is\s+(?:also\s+)?known\s+as\s+(.+?)\.?$',
     'known_as', 1, 2),
    # "X operates in the Y industry"
    (r'^(.+?)\s+operates?\s+in\s+(?:the\s+)?(.+?)\s+(?:industry|sector)\.?$',
     'industry', 1, 2),
    # "X is a Y" (type — LAST, most generic)
    (r'^(.+?)\s+is\s+(?:a|an|the)\s+(.+?)\.?$',
     'type', 1, 2),
]


# ============================================================================
# REASONING LOGGER (adapted from pipeline_v4)
# ============================================================================

class ReasoningLogger:
    """Log every decision the engine makes, with WHY."""

    def __init__(self, log_dir=None):
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), 'data')
        self.log_path = os.path.join(log_dir, 'distillation_reasoning.log')
        self.stats = {
            'gaps_processed': 0, 'questions_asked': 0,
            'triplets_extracted': 0, 'triplets_stored': 0,
            'conflicts': 0, 'duplicates': 0, 'unverified': 0,
            'feedback_failures': 0,
        }
        self._init_log()

    def _init_log(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'a') as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"DISTILLATION SESSION — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n\n")

    def _ts(self):
        return datetime.now().strftime('%H:%M:%S')

    def log(self, event, **kwargs):
        """Log an event with context."""
        with open(self.log_path, 'a') as f:
            f.write(f"[{self._ts()}] {event}")
            if kwargs:
                details = ', '.join(f'{k}={v}' for k, v in kwargs.items())
                f.write(f" | {details}")
            f.write('\n')

    def log_gap(self, entity, entity_type, missing):
        self.stats['gaps_processed'] += 1
        self.log('GAP', entity=entity, type=entity_type,
                 missing=', '.join(missing))

    def log_question(self, entity, relation, question):
        self.stats['questions_asked'] += 1
        self.log('ASK', entity=entity, rel=relation, q=question[:80])

    def log_extract(self, triplets, answer_preview):
        self.stats['triplets_extracted'] += len(triplets)
        self.log('EXTRACT', n=len(triplets), answer=answer_preview[:60])

    def log_store(self, subj, rel, obj):
        self.stats['triplets_stored'] += 1
        self.log('STORE', fact=f'({subj}, {rel}, {obj})')

    def log_skip(self, reason, subj, rel, obj, detail=''):
        self.stats[reason] = self.stats.get(reason, 0) + 1
        self.log(f'SKIP:{reason.upper()}', fact=f'({subj}, {rel}, {obj})',
                 detail=detail)

    def log_feedback(self, subj, rel, success):
        if not success:
            self.stats['feedback_failures'] += 1
        self.log('FEEDBACK', fact=f'({subj}, {rel})',
                 result='OK' if success else 'FAIL')

    def summary(self):
        lines = [f"\n{'='*50}", "DISTILLATION SUMMARY"]
        for k, v in self.stats.items():
            lines.append(f"  {k}: {v}")
        return '\n'.join(lines)


# ============================================================================
# OLLAMA BACKEND
# ============================================================================

class OllamaBackend:
    """Talk to Ollama's HTTP API."""

    def __init__(self, model='qwen3:4b', base_url='http://localhost:11434',
                 timeout=60):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def ask(self, question, system_prompt=None):
        """Send a question to Ollama, return the text response."""
        url = f"{self.base_url}/api/generate"
        payload = {
            'model': self.model,
            'prompt': question,
            'stream': False,
            'options': {
                'temperature': 0.1,
                'num_predict': 512,
            },
        }
        if system_prompt:
            payload['system'] = system_prompt

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url, data=data,
            headers={'Content-Type': 'application/json'}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result.get('response', '').strip()
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, json.JSONDecodeError) as e:
            logger.warning(f"Ollama request failed: {e}")
            return None

    def is_available(self):
        """Check if Ollama is running and model is available."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                models = [m.get('name', '') for m in data.get('models', [])]
                return any(self.model in m for m in models)
        except Exception:
            return False


# ============================================================================
# TRIPLET EXTRACTOR
# ============================================================================

class TripletExtractor:
    """Extract (Subject, Relation, Object) triplets from LLM responses."""

    SYSTEM_PROMPT = (
        "Answer with short, factual statements. One fact per sentence. "
        "Use simple structure: 'The capital of France is Paris.' "
        "'Python was created by Guido van Rossum in 1991.' "
        "'Germany has a population of 83 million.' "
        "No opinions, no speculation. If unsure, say 'I don't know.'"
    )

    def __init__(self):
        import re
        self._re = re
        self._patterns = [
            (self._re.compile(p, self._re.IGNORECASE), rel, sg, og)
            for p, rel, sg, og in EXTRACTION_PATTERNS
        ]

    def _normalize_entity(self, entity, hint=None):
        """Normalize entity: strip articles, match hint casing."""
        # Strip leading articles
        entity = self._re.sub(r'^(?:the|a|an)\s+', '', entity,
                              flags=self._re.IGNORECASE).strip()
        # If hint provided and matches case-insensitively, use hint's casing
        if hint and entity.lower() == hint.lower():
            return hint
        return entity

    def extract(self, question, answer, subject_hint=None):
        """Extract triplets from an LLM answer."""
        if not answer:
            return []

        # Strip Qwen3 thinking tags
        clean = self._re.sub(r'<think>.*?</think>', '', answer,
                             flags=self._re.DOTALL).strip()
        if not clean:
            return []
        if 'don\'t know' in clean.lower() or 'not sure' in clean.lower():
            return []

        triplets = []

        for line in clean.split('.'):
            line = line.strip()
            if not line or len(line) < 5:
                continue

            for pattern, relation, s_group, o_group in self._patterns:
                m = pattern.match(line)
                if m:
                    subj = self._normalize_entity(
                        m.group(s_group).strip().rstrip('.,;'),
                        subject_hint)
                    obj = m.group(o_group).strip().rstrip('.,;')
                    if len(subj) > 1 and len(obj) > 1:
                        triplets.append((subj, relation, obj))
                    break

        # Fallback: subject_hint anchor parsing
        if not triplets and subject_hint and clean:
            first = clean.split('.')[0].strip()
            if first and len(first) > 3:
                # "X is the capital city of Y" → (Y, capital, X)
                m = self._re.search(
                    r'(.+?)\s+is\s+the\s+capital\s+(?:city\s+)?of\s+(.+)',
                    first, self._re.IGNORECASE)
                if m:
                    triplets.append((m.group(2).strip(), 'capital',
                                     m.group(1).strip()))
                    return triplets

                # Generic: "HintEntity is a/an X"
                hint_lower = subject_hint.lower()
                if first.lower().startswith(hint_lower):
                    rest = first[len(subject_hint):].strip()
                    if rest.lower().startswith('is '):
                        obj = rest[3:].strip()
                        obj = self._re.sub(r'^(?:a|an|the)\s+', '', obj)
                        if len(obj) > 1:
                            triplets.append((subject_hint, 'type', obj))

        return triplets


# ============================================================================
# DISTILLATION ENGINE
# ============================================================================

class DistillationEngine:
    """
    Curiosity-Driven Knowledge Distillation from local LLMs.

    Pipeline (from pipeline_v4 Sovereign Discovery, adapted):
    1. Gap Detection — find sparse entities via Dict-Index O(1)
    2. Question Generation — entity-type-aware, targeted questions
    3. LLM Query — Ollama backend, system prompt for factual answers
    4. Triplet Extraction — regex patterns, subject-hint fallback
    5. Dedup Check — O(1) via _fact_keys
    6. Conflict Check — O(1) via forward index
    7. Consistency Check — ask twice, discard on mismatch
    8. Store — immediate, no retraining
    9. Feedback Loop — verify the stored fact is queryable
    10. Reasoning Log — every decision logged with WHY
    """

    def __init__(self, knowledge_store, model='qwen3:4b',
                 verify=True, verbose=True):
        self.knowledge = knowledge_store
        self.backend = OllamaBackend(model=model)
        self.extractor = TripletExtractor()
        self.verify = verify
        self.verbose = verbose
        self.reasoning = ReasoningLogger()
        # Query dedup: never ask the same question twice (from pipeline_v4)
        self._attempted_queries = set()

    def check_available(self):
        return self.backend.is_available()

    # ==================================================================
    # GAP DETECTION (6 types, adapted from pipeline_v4 GapDetector)
    # ==================================================================

    # Skip entities that are NOT worth asking an LLM about
    _SKIP_ENTITY_PREFIXES = (
        'os.', 'sys.', 'io.', 're.', 'json.', 'http.', 'email.', 'html.',
        'xml.', 'csv.', 'sqlite3.', 'logging.', 'unittest.', 'collections.',
        'itertools.', 'functools.', 'pathlib.', 'asyncio.', 'threading.',
        'multiprocessing.', 'subprocess.', 'socket.', 'ssl.', 'urllib.',
        'ctypes.', 'struct.', 'array.', 'math.', 'decimal.', 'fractions.',
        'statistics.', 'random.', 'string.', 'textwrap.', 'difflib.',
        'pprint.', 'copy.', 'pdb.', 'profile.', 'timeit.', 'traceback.',
        'gc.', 'inspect.', 'dis.', 'abc.', 'contextlib.', 'typing.',
        'dataclasses.', 'enum.', 'numbers.', 'cmath.', 'operator.',
        'pickle.', 'shelve.', 'marshal.', 'dbm.', 'gzip.', 'bz2.',
        'zipfile.', 'tarfile.', 'tempfile.', 'glob.', 'shutil.', 'fileinput.',
        'hashlib.', 'hmac.', 'secrets.', 'base64.', 'binascii.',
        'signal.', 'mmap.', 'platform.', 'sysconfig.', 'builtins.',
    )

    # Entities that are harvested function/method names — skip for LLM distillation
    _HARVESTED_TYPE_SKIP = (
        'method', 'function', 'class', 'built-in function', 'module',
        'builtin_function_or_method', 'type', 'wrapper_descriptor',
    )

    # Relations that indicate a harvested stdlib entity (from harvester.py)
    _HARVESTED_RELATION_SKIP = frozenset({
        'computes', 'performs', 'operates_on', 'has_function',
        'provides_function', 'has_method', 'has_class',
    })

    # All seed entity names (flattened, lowercased) for priority detection
    _SEED_ENTITIES = None

    @classmethod
    def _get_seed_entities(cls):
        """Lazily build seed entity set from run_overnight.SEED_TOPICS."""
        if cls._SEED_ENTITIES is None:
            try:
                from run_overnight import SEED_TOPICS
                cls._SEED_ENTITIES = set()
                for seeds in SEED_TOPICS.values():
                    for s in seeds:
                        cls._SEED_ENTITIES.add(s.lower())
            except ImportError:
                cls._SEED_ENTITIES = set()
        return cls._SEED_ENTITIES

    def find_gaps(self, min_relations=5, max_entities=100):
        """
        Find entities with sparse knowledge. Entity-type-aware.
        Prioritizes seeded entities over random KB entities.
        Returns list of (entity, entity_type, missing_relations) tuples.
        """
        sparse = self.knowledge.get_sparse_entities(
            min_relations=min_relations)
        seed_set = self._get_seed_entities()

        # Also collect ALL seeded entities in KB (even non-sparse)
        seed_entities_in_kb = set()
        for entity in self.knowledge.all_entities():
            if entity.lower() in seed_set:
                seed_entities_in_kb.add(entity)

        # Partition: seeded entities first, then rest
        seeded = []
        rest = []
        seen = set()
        # First pass: sparse entities
        for entity, n_rels in sparse:
            # Skip stdlib module paths (not useful to ask LLM about)
            if any(entity.startswith(p) for p in self._SKIP_ENTITY_PREFIXES):
                continue
            # Skip very long entity names (usually garbage)
            if len(entity) > 60:
                continue

            # Check entity type — skip harvested stdlib function/method names
            is_seed = entity.lower() in seed_set
            fwd_key = (entity, 'type')
            indices = self.knowledge._forward_index.get(fwd_key)
            if indices and not is_seed:
                type_val = self.knowledge.facts[indices[0]][2]
                if type_val.lower() in self._HARVESTED_TYPE_SKIP:
                    continue  # stdlib garbage like popitem, fdopen, decode

            # Skip harvested stdlib function names (no type fact, only harvester relations)
            if not is_seed:
                rels = set(self.knowledge.get_entity_relations(entity))
                if rels and rels.issubset(self._HARVESTED_RELATION_SKIP):
                    continue

            seen.add(entity)
            if is_seed:
                seeded.append((entity, n_rels))
            else:
                rest.append((entity, n_rels))

        # Add seeded entities that weren't in sparse (they may have enough
        # total relations but still miss entity-type-specific ones)
        for entity in seed_entities_in_kb:
            if entity not in seen:
                n_rels = len(self.knowledge.get_entity_relations(entity))
                seeded.append((entity, n_rels))

        # Seeded first, then rest
        ordered = seeded + rest

        gaps = []
        for entity, _ in ordered[:max_entities * 2]:
            existing = set(self.knowledge.get_entity_relations(entity))
            entity_type = self._infer_entity_type(entity, existing)
            expected = ENTITY_TYPE_RELATIONS.get(
                entity_type, ENTITY_TYPE_RELATIONS['default'])
            missing = [r for r in expected if r not in existing]
            if missing:
                gaps.append((entity, entity_type, missing))
                if len(gaps) >= max_entities:
                    break
        return gaps

    def find_contradictions(self):
        """
        Find facts where same (S,R) has multiple different O values.
        Adapted from pipeline_v4 detect_contradiction_gaps_fast().
        """
        contradictions = []
        for key, indices in self.knowledge._forward_index.items():
            if len(indices) > 1:
                values = set()
                for idx in indices:
                    values.add(self.knowledge.facts[idx][2].lower())
                if len(values) > 1:
                    s, r = key
                    contradictions.append({
                        'subject': s, 'relation': r,
                        'values': list(values),
                        'count': len(values),
                    })
        return contradictions

    # Keywords for entity type detection from 'type' fact values
    _TYPE_KEYWORDS = {
        'programming_language': ['programming language', 'scripting language', 'compiled language',
                                 'interpreted language', 'markup language', 'query language'],
        'data_structure': ['data structure', 'container', 'collection', 'tree', 'graph',
                           'queue', 'stack', 'heap', 'array', 'list', 'hash', 'linked list',
                           'binary tree', 'trie'],
        'algorithm': ['algorithm', 'sorting', 'search algorithm', 'optimization',
                      'pathfinding', 'traversal', 'heuristic'],
        'design_pattern': ['design pattern', 'pattern', 'architectural pattern',
                           'creational', 'structural', 'behavioral'],
        'protocol': ['protocol', 'network protocol', 'communication protocol',
                     'transport protocol'],
        'file_format': ['file format', 'format', 'encoding', 'serialization format'],
        'library': ['library', 'framework', 'package', 'module', 'toolkit', 'sdk'],
        'grammar_concept': ['grammar', 'syntax', 'clause', 'phrase', 'tense',
                            'conjugation', 'declension', 'morphology'],
        'word_class': ['part of speech', 'word class', 'noun', 'verb', 'adjective',
                       'adverb', 'preposition', 'conjunction', 'pronoun'],
        'linguistic_concept': ['linguistic', 'semantics', 'pragmatics', 'phonology',
                               'morpheme', 'phoneme', 'lexicon'],
        'math_concept': ['mathematical', 'theorem', 'equation', 'formula', 'function',
                         'calculus', 'algebra', 'geometry', 'topology', 'statistics',
                         'probability', 'set theory', 'number theory'],
        'science_concept': ['physics', 'chemistry', 'biology', 'law of', 'principle',
                            'theory', 'phenomenon', 'reaction', 'force'],
        'country': ['country', 'nation', 'sovereign', 'republic', 'kingdom', 'state'],
        'city': ['city', 'town', 'municipality', 'capital city'],
        'company': ['company', 'corporation', 'enterprise', 'startup', 'firm'],
        'person': ['person', 'scientist', 'mathematician', 'programmer', 'inventor',
                   'philosopher', 'physicist', 'engineer'],
        'invention': ['invention', 'device', 'tool', 'instrument'],
    }

    # Entity name patterns (no type fact needed)
    _NAME_PATTERNS = {
        'programming_language': ['python', 'javascript', 'typescript', 'java', 'c++',
                                 'rust', 'go', 'ruby', 'php', 'swift', 'kotlin',
                                 'scala', 'haskell', 'erlang', 'elixir', 'lua',
                                 'perl', 'r', 'matlab', 'sql', 'html', 'css'],
        'data_structure': ['linked list', 'binary tree', 'hash table', 'hash map',
                           'red-black tree', 'b-tree', 'avl tree', 'trie',
                           'priority queue', 'bloom filter', 'skip list',
                           'adjacency list', 'adjacency matrix'],
        'algorithm': ['quicksort', 'mergesort', 'heapsort', 'binary search',
                      'breadth-first', 'depth-first', 'dijkstra', 'a*',
                      'dynamic programming', 'backtracking', 'gradient descent'],
        'design_pattern': ['singleton', 'factory', 'observer', 'strategy',
                           'decorator', 'adapter', 'facade', 'proxy',
                           'iterator', 'command pattern', 'mvc', 'mvvm'],
        'protocol': ['http', 'https', 'tcp', 'udp', 'ftp', 'ssh', 'smtp',
                     'dns', 'dhcp', 'websocket', 'grpc', 'mqtt'],
        'grammar_concept': ['subject', 'predicate', 'clause', 'phrase',
                            'dependent clause', 'relative clause',
                            'gerund', 'infinitive', 'participle',
                            'subjunctive', 'subjunctive mood',
                            'passive voice', 'active voice', 'conditional',
                            'past tense', 'present tense', 'future tense',
                            'modal verb', 'morpheme', 'phoneme',
                            'syntax', 'semantics'],
        'word_class': ['noun', 'verb', 'adjective', 'adverb', 'preposition',
                       'conjunction', 'pronoun', 'article', 'interjection'],
        'linguistic_concept': ['idiom', 'metaphor', 'analogy', 'rhetoric'],
        'math_concept': ['derivative', 'integral', 'matrix', 'vector',
                         'eigenvalue', 'fourier transform', 'probability',
                         'bayes theorem', 'markov chain', 'graph theory',
                         'set theory', 'boolean algebra', 'lambda calculus',
                         'information theory', 'entropy',
                         'kolmogorov complexity'],
        'algorithm': ['turing machine', 'regular expression',
                      'context-free grammar', 'halting problem',
                      'k-means clustering', 'neural network'],
    }

    def _infer_entity_type(self, entity, existing_relations):
        """Infer entity type from explicit type fact, name patterns, or relation pattern."""
        entity_lower = entity.lower()

        # 1. Check name patterns first (fast, no DB lookup)
        for etype, patterns in self._NAME_PATTERNS.items():
            if entity_lower in patterns:
                return etype

        # 2. Check explicit type fact
        fwd_key = (entity_lower, 'type')
        indices = self.knowledge._forward_index.get(fwd_key)
        if indices:
            type_val = self.knowledge.facts[indices[0]][2].lower()
            for etype, keywords in self._TYPE_KEYWORDS.items():
                if any(kw in type_val for kw in keywords):
                    return etype

        # 3. Check harvested relations (e.g. provides_function → it's a library/module)
        if 'provides_function' in existing_relations or 'provides_class' in existing_relations:
            return 'library'
        if 'paradigm' in existing_relations or 'syntax_example' in existing_relations:
            return 'programming_language'
        if 'complexity' in existing_relations or 'operations' in existing_relations:
            return 'data_structure'
        if 'formula' in existing_relations:
            return 'math_concept'

        # 4. Legacy geo/person detection
        if 'capital' in existing_relations or 'borders' in existing_relations:
            return 'country'
        if 'country' in existing_relations and 'capital' not in existing_relations:
            return 'city'
        if 'born' in existing_relations or 'died' in existing_relations:
            return 'person'
        if 'industry' in existing_relations:
            return 'company'
        return 'default'

    # ==================================================================
    # QUESTION GENERATION
    # ==================================================================

    def generate_questions(self, entity, entity_type, missing_relations):
        """
        Generate targeted questions. Deduplicates against attempted_queries.
        Returns list of (question, relation) tuples.
        """
        questions = []
        for rel in missing_relations:
            tmpl = REL_TO_QUESTION.get(rel)
            if not tmpl:
                continue
            q = tmpl.format(entity=entity)
            q_key = q.lower().strip()
            if q_key in self._attempted_queries:
                continue
            self._attempted_queries.add(q_key)
            questions.append((q, rel))
        return questions

    # ==================================================================
    # CORE DISTILLATION LOOP
    # ==================================================================

    def distill_entity(self, entity, entity_type, missing_relations):
        """
        Full pipeline for one entity:
        Ask → Extract → Dedup → Conflict → Verify → Store → Feedback
        """
        questions = self.generate_questions(entity, entity_type,
                                            missing_relations)
        stored = 0

        for question, expected_rel in questions:
            self.reasoning.log_question(entity, expected_rel, question)

            # Ask LLM
            answer = self.backend.ask(question,
                                      system_prompt=TripletExtractor.SYSTEM_PROMPT)
            if not answer:
                continue

            # Extract triplets
            triplets = self.extractor.extract(question, answer,
                                              subject_hint=entity)
            self.reasoning.log_extract(triplets, answer)

            if not triplets:
                continue

            for subj, rel, obj in triplets:
                # 1. O(1) Dedup check
                if self.knowledge.has_fact(subj, rel, obj):
                    self.reasoning.log_skip('duplicates', subj, rel, obj)
                    continue

                # 2. O(1) Conflict check
                conflict = self.knowledge.get_conflict(subj, rel, obj)
                if conflict:
                    self.reasoning.log_skip('conflicts', subj, rel, obj,
                                            detail=f'existing={conflict}')
                    continue

                # 3. Consistency check (ask again, compare)
                if self.verify:
                    answer2 = self.backend.ask(
                        question,
                        system_prompt=TripletExtractor.SYSTEM_PROMPT)
                    triplets2 = self.extractor.extract(
                        question, answer2, subject_hint=entity) if answer2 else []

                    # Strict: exact S+R+O match
                    verified = any(
                        t[0].lower() == subj.lower() and
                        t[1].lower() == rel.lower() and
                        t[2].lower() == obj.lower()
                        for t in triplets2)

                    # Relaxed: same S+R (object may be phrased differently)
                    if not verified:
                        verified = any(
                            t[0].lower() == subj.lower() and
                            t[1].lower() == rel.lower()
                            for t in triplets2)

                    if not verified:
                        self.reasoning.log_skip('unverified', subj, rel, obj)
                        continue

                # 4. Store
                self.knowledge.store_fact(subj, rel, obj)
                stored += 1
                self.reasoning.log_store(subj, rel, obj)

                if self.verbose:
                    print(f"  + ({subj}, {rel}, {obj})")

                # 5. Feedback loop: verify it's queryable
                result = self.knowledge.query(subj, rel)
                queryable = (result.get('fact') is not None and
                             result['confidence_level'] in ('HIGH', 'MEDIUM'))
                self.reasoning.log_feedback(subj, rel, queryable)

                if not queryable and self.verbose:
                    print(f"  ! WARNING: stored but not queryable: ({subj}, {rel})")

        return stored

    # ==================================================================
    # RUN MODES
    # ==================================================================

    def run(self, max_entities=50, min_relations=5, save_brain=True,
            brain_path=None):
        """Single distillation run. Find gaps → fill → save."""
        if not self.check_available():
            print(f"ERROR: Ollama model '{self.backend.model}' not available.")
            print(f"Run: ollama pull {self.backend.model}")
            return self.reasoning.stats

        gaps = self.find_gaps(min_relations=min_relations,
                              max_entities=max_entities)

        if not gaps:
            print("No knowledge gaps found. KB is well-covered.")
            return self.reasoning.stats

        if self.verbose:
            print(f"Found {len(gaps)} entities with knowledge gaps.")
            print(f"Model: {self.backend.model}")
            print(f"Verify: {'ON (ask twice)' if self.verify else 'OFF'}")
            # Show contradictions
            contradictions = self.find_contradictions()
            if contradictions:
                print(f"Contradictions in KB: {len(contradictions)}")
            print()

        t0 = time.time()
        total_stored = 0

        for i, (entity, entity_type, missing) in enumerate(gaps):
            if self.verbose:
                print(f"[{i+1}/{len(gaps)}] {entity} ({entity_type}) "
                      f"missing: {', '.join(missing)}")

            self.reasoning.log_gap(entity, entity_type, missing)
            n = self.distill_entity(entity, entity_type, missing)
            total_stored += n
            time.sleep(0.1)

        elapsed = time.time() - t0

        if self.verbose:
            print(self.reasoning.summary())
            print(f"\nTime: {elapsed:.1f}s")
            print(f"KB total: {self.knowledge.n_facts} facts")

        if save_brain and brain_path:
            self._save_brain(brain_path)

        return self.reasoning.stats

    def run_overnight(self, hours=8, min_relations=3, batch_size=20,
                      brain_path=None):
        """Long-running distillation. Batch-based, periodic brain saves."""
        if not self.check_available():
            print(f"ERROR: Ollama model '{self.backend.model}' not available.")
            return self.reasoning.stats

        end_time = time.time() + hours * 3600
        batch_num = 0
        start_facts = self.knowledge.n_facts

        print(f"=== Overnight Distillation ===")
        print(f"Duration: {hours}h | Model: {self.backend.model}")
        print(f"Verify: {'ON' if self.verify else 'OFF'}")
        print(f"Starting KB: {start_facts} facts")
        print()

        while time.time() < end_time:
            batch_num += 1
            gaps = self.find_gaps(min_relations=min_relations,
                                  max_entities=batch_size)

            if not gaps:
                min_relations += 2
                if min_relations > 20:
                    print("KB fully saturated. Stopping.")
                    break
                print(f"  Raising threshold to min_rels={min_relations}")
                continue

            print(f"--- Batch {batch_num} ({len(gaps)} entities) ---")

            for entity, entity_type, missing in gaps:
                if time.time() >= end_time:
                    break
                self.reasoning.log_gap(entity, entity_type, missing)
                self.distill_entity(entity, entity_type, missing)
                time.sleep(0.1)

            # Self-optimize after each batch
            self.self_optimize()

            # Save brain after each batch
            if brain_path:
                self._save_brain(brain_path)

            new_facts = self.knowledge.n_facts - start_facts
            remaining = (end_time - time.time()) / 3600
            rate = new_facts / max(batch_num, 1)
            print(f"  KB: {self.knowledge.n_facts} | "
                  f"+{new_facts} new | "
                  f"~{rate:.0f}/batch | "
                  f"{remaining:.1f}h left")

        print(f"\n=== Overnight Complete ===")
        print(f"Final KB: {self.knowledge.n_facts} facts")
        print(f"New facts: {self.knowledge.n_facts - start_facts}")
        print(self.reasoning.summary())
        return self.reasoning.stats

    # ==================================================================
    # SELF-OPTIMIZATION (Task #41)
    # ==================================================================

    def self_optimize(self):
        """
        Analyze extraction failures and adapt strategy.
        Called after each batch in overnight mode.

        Optimizations:
        1. Track which relations fail extraction → add better patterns
        2. Track which entity types get nonsense → improve type detection
        3. Find contradictions → resolve via LLM
        """
        stats = self.reasoning.stats
        total_asked = stats['questions_asked']
        total_extracted = stats['triplets_extracted']
        total_stored = stats['triplets_stored']

        if total_asked == 0:
            return

        extraction_rate = total_extracted / total_asked
        store_rate = total_stored / max(total_extracted, 1)

        # If extraction rate < 30%, switch to open-ended questions
        if extraction_rate < 0.3 and total_asked > 10:
            self._use_open_questions = True
            self.reasoning.log('OPTIMIZE',
                               action='switch_to_open_questions',
                               extraction_rate=f'{extraction_rate:.0%}')

        # If conflict rate > 50%, lower threshold for storing
        if stats['conflicts'] > total_extracted * 0.5 and total_extracted > 5:
            self.reasoning.log('OPTIMIZE',
                               action='high_conflict_rate',
                               conflicts=stats['conflicts'])

        # Find contradiction clusters
        contradictions = self.find_contradictions()
        if contradictions and self.backend.is_available():
            self._resolve_contradictions(contradictions[:5])

    def _resolve_contradictions(self, contradictions):
        """Ask LLM to resolve contradictions."""
        for c in contradictions:
            values = c['values']
            if len(values) != 2:
                continue

            q = (f"Which is correct: '{c['subject']}' has {c['relation']} "
                 f"'{values[0]}' or '{values[1]}'? "
                 f"Answer with just the correct value.")

            answer = self.backend.ask(q,
                                      system_prompt=TripletExtractor.SYSTEM_PROMPT)
            if answer:
                import re
                clean = re.sub(r'<think>.*?</think>', '', answer,
                               flags=re.DOTALL).strip().lower()
                # Check which value the LLM confirms
                for v in values:
                    if v in clean:
                        self.reasoning.log('RESOLVE',
                                           subject=c['subject'],
                                           relation=c['relation'],
                                           resolved_to=v)
                        break

    # ==================================================================
    # COMMON SENSE DISTILLATION (Layer 5)
    # ==================================================================

    COMMON_SENSE_TEMPLATES = [
        # Physical properties
        ("Can {entity} fly?", "can_fly"),
        ("Is {entity} alive?", "is_alive"),
        ("Is {entity} edible?", "is_edible"),
        ("Is {entity} bigger than a car?", "bigger_than_car"),
        ("Is {entity} hot?", "is_hot"),
        ("Is {entity} dangerous?", "is_dangerous"),
        ("Can you hold {entity} in your hand?", "is_handheld"),
        ("Is {entity} man-made?", "is_man_made"),
        ("Is {entity} visible to the naked eye?", "is_visible"),
        ("Does {entity} exist in nature?", "is_natural"),
        # Categorical
        ("Is {entity} a place?", "is_place"),
        ("Is {entity} a person?", "is_person"),
        ("Is {entity} an animal?", "is_animal"),
        ("Is {entity} a food?", "is_food"),
        ("Is {entity} a tool?", "is_tool"),
        ("Is {entity} abstract?", "is_abstract"),
    ]

    def distill_common_sense(self, entities, max_questions=500):
        """
        Ask yes/no questions to build common sense rules.
        Returns count of new facts stored.
        """
        stored = 0
        asked = 0

        for entity in entities:
            if asked >= max_questions:
                break

            for template, relation in self.COMMON_SENSE_TEMPLATES:
                if asked >= max_questions:
                    break

                q = template.format(entity=entity)
                q_key = q.lower()
                if q_key in self._attempted_queries:
                    continue
                self._attempted_queries.add(q_key)
                asked += 1

                answer = self.backend.ask(
                    q + " Answer only 'yes' or 'no'.",
                    system_prompt="Answer only 'yes' or 'no'. Nothing else.")
                if not answer:
                    continue

                import re
                clean = re.sub(r'<think>.*?</think>', '', answer,
                               flags=re.DOTALL).strip().lower()

                if 'yes' in clean[:10]:
                    value = 'yes'
                elif 'no' in clean[:10]:
                    value = 'no'
                else:
                    continue

                if not self.knowledge.has_fact(entity, relation, value):
                    self.knowledge.store_fact(entity, relation, value)
                    stored += 1
                    self.reasoning.log_store(entity, relation, value)

        return stored

    # ==================================================================
    # WORD RELATIONS DISTILLATION (Layer 6)
    # ==================================================================

    WORD_RELATION_TEMPLATES = [
        ("What is the opposite of {entity}?", "antonym"),
        ("What is a synonym for {entity}?", "synonym"),
        ("What is {entity} a type of?", "hypernym"),
        ("Give an example of {entity}.", "hyponym"),
        ("What is {entity} part of?", "part_of"),
        ("What does {entity} contain?", "contains"),
    ]

    def distill_word_relations(self, entities, max_questions=500):
        """
        Extract word relations (antonyms, synonyms, hypernyms, etc.).
        Returns count of new facts stored.
        """
        stored = 0
        asked = 0

        for entity in entities:
            if asked >= max_questions:
                break

            for template, relation in self.WORD_RELATION_TEMPLATES:
                if asked >= max_questions:
                    break

                q = template.format(entity=entity)
                q_key = q.lower()
                if q_key in self._attempted_queries:
                    continue
                self._attempted_queries.add(q_key)
                asked += 1

                answer = self.backend.ask(
                    q + " Answer with one word or short phrase only.",
                    system_prompt="Answer with one word or very short phrase. No explanation.")
                if not answer:
                    continue

                import re
                clean = re.sub(r'<think>.*?</think>', '', answer,
                               flags=re.DOTALL).strip()

                # Clean up: take first word/phrase, strip punctuation
                clean = clean.split('\n')[0].strip().rstrip('.,:;')
                if not clean or len(clean) < 2 or len(clean) > 50:
                    continue
                if "don't know" in clean.lower() or "not sure" in clean.lower():
                    continue

                if not self.knowledge.has_fact(entity, relation, clean):
                    self.knowledge.store_fact(entity, relation, clean)
                    stored += 1
                    self.reasoning.log_store(entity, relation, clean)

        return stored

    # ==================================================================
    # TAXONOMY DISTILLATION (Layer 10)
    # ==================================================================

    TAXONOMY_TEMPLATES = [
        ("What are the main types of {entity}?", "has_types"),
        ("What category does {entity} belong to?", "category"),
        ("What are examples of {entity}?", "examples"),
        ("What is the parent category of {entity}?", "parent_category"),
    ]

    def distill_taxonomies(self, entities, max_questions=200):
        """Extract taxonomy relations (is_a, has_types, category)."""
        stored = 0
        asked = 0

        for entity in entities:
            if asked >= max_questions:
                break

            for template, relation in self.TAXONOMY_TEMPLATES:
                if asked >= max_questions:
                    break

                q = template.format(entity=entity)
                q_key = q.lower()
                if q_key in self._attempted_queries:
                    continue
                self._attempted_queries.add(q_key)
                asked += 1

                answer = self.backend.ask(
                    q + " List items separated by commas. Be brief.",
                    system_prompt="Answer briefly. List items separated by commas.")
                if not answer:
                    continue

                import re
                clean = re.sub(r'<think>.*?</think>', '', answer,
                               flags=re.DOTALL).strip()
                if not clean or "don't know" in clean.lower():
                    continue

                # Split comma-separated items
                items = [x.strip().rstrip('.') for x in clean.split(',')]
                for item in items[:5]:  # max 5 per question
                    if len(item) < 2 or len(item) > 80:
                        continue
                    if not self.knowledge.has_fact(entity, relation, item):
                        self.knowledge.store_fact(entity, relation, item)
                        stored += 1

        return stored

    # ==================================================================
    # CAUSAL CHAIN DISTILLATION (Layer 11)
    # ==================================================================

    CAUSAL_TEMPLATES = [
        ("What does {entity} cause?", "causes"),
        ("What causes {entity}?", "caused_by"),
        ("What does {entity} require?", "requires"),
        ("What does {entity} enable?", "enables"),
        ("What is {entity} used for?", "used_for"),
    ]

    def distill_causal_chains(self, entities, max_questions=200):
        """Extract causal relationships (causes, requires, enables)."""
        stored = 0
        asked = 0

        for entity in entities:
            if asked >= max_questions:
                break

            for template, relation in self.CAUSAL_TEMPLATES:
                if asked >= max_questions:
                    break

                q = template.format(entity=entity)
                q_key = q.lower()
                if q_key in self._attempted_queries:
                    continue
                self._attempted_queries.add(q_key)
                asked += 1

                answer = self.backend.ask(
                    q + " Answer with a short list, comma-separated.",
                    system_prompt="Answer briefly. Comma-separated list.")
                if not answer:
                    continue

                import re
                clean = re.sub(r'<think>.*?</think>', '', answer,
                               flags=re.DOTALL).strip()
                if not clean or "don't know" in clean.lower():
                    continue

                items = [x.strip().rstrip('.') for x in clean.split(',')]
                for item in items[:5]:
                    if len(item) < 2 or len(item) > 80:
                        continue
                    if not self.knowledge.has_fact(entity, relation, item):
                        self.knowledge.store_fact(entity, relation, item)
                        stored += 1

        return stored

    # ==================================================================
    # FLM TEXT CORPUS GENERATION (Layer 2)
    # ==================================================================

    FLM_TOPIC_PROMPTS = [
        "Write a paragraph about {entity} in simple English.",
        "Explain {entity} as if to a teenager.",
        "Describe {entity} in 3 sentences.",
        "What are interesting facts about {entity}?",
        "How does {entity} work?",
    ]

    def generate_flm_corpus(self, entities, output_path, max_texts=200):
        """
        Generate text corpus for FLM training by asking LLM about known entities.
        Saves raw text to output_path for offline FLM training.
        """
        import re

        texts = []
        count = 0

        for entity in entities:
            if count >= max_texts:
                break

            template = self.FLM_TOPIC_PROMPTS[count % len(self.FLM_TOPIC_PROMPTS)]
            q = template.format(entity=entity)

            answer = self.backend.ask(q)
            if not answer:
                continue

            clean = re.sub(r'<think>.*?</think>', '', answer,
                           flags=re.DOTALL).strip()
            if clean and len(clean) > 20:
                texts.append(clean)
                count += 1

        # Save corpus
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(texts))

        return count

    # ==================================================================
    # REFORMULATION TEMPLATES (Layer 3)
    # ==================================================================

    REFORMULATION_PROMPTS = [
        "How would you say '{subject} has {relation} {object}' in natural English?",
        "Rephrase: '{subject} {relation} {object}'",
        "Express this fact differently: {subject}'s {relation} is {object}",
    ]

    def generate_reformulations(self, max_templates=100):
        """
        Generate natural language reformulation templates from existing facts.
        Returns list of (pattern, template) tuples.
        """
        import re
        templates = []
        count = 0

        # Sample facts
        sample_size = min(max_templates, len(self.knowledge.facts))
        import random
        sample_indices = random.sample(range(len(self.knowledge.facts)),
                                       sample_size)

        for idx in sample_indices:
            if count >= max_templates:
                break

            s, r, o = self.knowledge.facts[idx]
            prompt_idx = count % len(self.REFORMULATION_PROMPTS)
            q = self.REFORMULATION_PROMPTS[prompt_idx].format(
                subject=s, relation=r, object=o)

            answer = self.backend.ask(
                q, system_prompt="Rephrase the fact in natural English. One sentence.")
            if not answer:
                continue

            clean = re.sub(r'<think>.*?</think>', '', answer,
                           flags=re.DOTALL).strip()
            if clean and len(clean) > 10:
                # Extract template by replacing entity names with placeholders
                tmpl = clean.replace(s, '{subject}').replace(o, '{object}')
                if '{subject}' in tmpl or '{object}' in tmpl:
                    templates.append((r, tmpl))
                    count += 1

        return templates

    # ==================================================================
    # REASONING PATTERNS (Layer 4) — CoT extraction
    # ==================================================================

    COT_PROBLEMS = [
        "A train leaves at 3pm going 60mph. Another leaves at 4pm going 80mph. When do they meet?",
        "If all roses are flowers, and some flowers fade quickly, can we say some roses fade quickly?",
        "You have 3 boxes. One has apples, one oranges, one both. Labels are ALL wrong. You pick one fruit from one box. How do you label all boxes?",
        "A farmer has a fox, a chicken, and a sack of grain. He must cross a river in a boat that carries only him and one item. How?",
        "What comes next: 2, 6, 12, 20, 30, ?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
        "You flip a fair coin 3 times. What is the probability of getting exactly 2 heads?",
        "Sort this list using the algorithm you think is best: [38, 27, 43, 3, 9, 82, 10]",
        "What is the shortest path from A to D in: A-B:3, A-C:5, B-D:4, C-D:2, B-C:1?",
        "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much is the ball?",
        "Explain step by step how to convert the number 42 from decimal to binary.",
        "If you rearrange the letters 'CIFAIPC', you get the name of a: ocean, country, or city?",
        "Three friends split a $30 bill. Each pays $10. The waiter returns $5. Each gets $1 back, $2 is missing. Where is it?",
        "What day is it 100 days from a Monday?",
        "Simplify: (x^2 - 4) / (x - 2)",
    ]

    def distill_reasoning_patterns(self, max_problems=15, output_path=None):
        """
        Layer 4: Extract reasoning STEPS from CoT problems.
        Stores the solution PATTERN, not just the answer.
        Returns count of patterns stored.
        """
        import re
        stored = 0

        for i, problem in enumerate(self.COT_PROBLEMS[:max_problems]):
            q_key = f"cot:{problem[:40]}".lower()
            if q_key in self._attempted_queries:
                continue
            self._attempted_queries.add(q_key)

            answer = self.backend.ask(
                problem + " Think step by step.",
                system_prompt="Solve step by step. Number each step. Be concise.")
            if not answer:
                continue

            clean = re.sub(r'<think>.*?</think>', '', answer,
                           flags=re.DOTALL).strip()
            if not clean or len(clean) < 20:
                continue

            # Extract numbered steps
            steps = re.findall(r'(?:step\s*)?(\d+)[.):]\s*(.+?)(?=\n|$)',
                               clean, re.IGNORECASE)

            if steps:
                # Store as reasoning pattern
                step_text = ' → '.join(s[1].strip() for s in steps[:8])
                # Classify problem type
                ptype = 'general'
                if any(w in problem.lower() for w in ('train', 'speed', 'distance')):
                    ptype = 'rate_distance'
                elif any(w in problem.lower() for w in ('probability', 'coin', 'dice')):
                    ptype = 'probability'
                elif any(w in problem.lower() for w in ('sort', 'path', 'algorithm')):
                    ptype = 'algorithm'
                elif any(w in problem.lower() for w in ('logic', 'roses', 'all', 'some')):
                    ptype = 'logic'
                elif any(w in problem.lower() for w in ('cost', 'price', 'bill', 'pay')):
                    ptype = 'arithmetic'

                self.knowledge.store_fact(
                    f'reasoning:{ptype}', 'steps', step_text)
                self.knowledge.store_fact(
                    f'reasoning:{ptype}', 'example_problem', problem[:80])
                stored += 2
                self.reasoning.log_store(f'reasoning:{ptype}', 'steps', step_text[:60])

            # Also save full solution text for FLM corpus
            if output_path:
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write(f"\nProblem: {problem}\nSolution: {clean}\n\n")

        return stored

    # ==================================================================
    # CODE TEMPLATES (Layer 7) — Extract code patterns
    # ==================================================================

    CODE_TASKS = [
        ("reverse a string", "string_manipulation"),
        ("find the maximum in a list", "list_operations"),
        ("check if a number is prime", "number_theory"),
        ("implement binary search", "search"),
        ("implement a stack using a list", "data_structure"),
        ("read a CSV file", "file_io"),
        ("make an HTTP GET request", "networking"),
        ("sort a dictionary by value", "sorting"),
        ("flatten a nested list", "list_operations"),
        ("find duplicates in a list", "list_operations"),
        ("implement fibonacci with memoization", "dynamic_programming"),
        ("parse JSON from a string", "parsing"),
        ("create a simple decorator", "metaprogramming"),
        ("implement a basic linked list", "data_structure"),
        ("calculate factorial recursively", "recursion"),
        ("merge two sorted lists", "sorting"),
        ("count word frequency in text", "text_processing"),
        ("validate an email address", "validation"),
        ("implement a simple cache with TTL", "caching"),
        ("create a context manager", "metaprogramming"),
        ("implement breadth-first search", "graph"),
        ("convert between bases", "number_theory"),
        ("implement retry with exponential backoff", "error_handling"),
        ("simple producer-consumer with queue", "concurrency"),
        ("implement a trie for prefix search", "data_structure"),
    ]

    def distill_code_templates(self, max_tasks=25):
        """
        Layer 7: Extract code patterns from LLM.
        Stores task→method→template triplets.
        Returns count of patterns stored.
        """
        import re
        stored = 0

        for task, category in self.CODE_TASKS[:max_tasks]:
            q_key = f"code:{task}".lower()
            if q_key in self._attempted_queries:
                continue
            self._attempted_queries.add(q_key)

            answer = self.backend.ask(
                f"Write a Python function to {task}. Code only, no explanation.",
                system_prompt="Write clean Python code. Function definition only. No explanation.")
            if not answer:
                continue

            clean = re.sub(r'<think>.*?</think>', '', answer,
                           flags=re.DOTALL).strip()
            if not clean:
                continue

            # Extract function definition
            code_match = re.search(r'(def \w+\(.+)', clean, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                # Limit to reasonable size
                lines = code.split('\n')[:15]
                code = '\n'.join(lines)

                self.knowledge.store_fact(
                    f'code:{category}', 'task', task)
                self.knowledge.store_fact(
                    f'code:{category}', 'template', code[:200])
                stored += 2
                self.reasoning.log_store(f'code:{category}', 'template', task)

        return stored

    # ==================================================================
    # DIALOG PATTERNS (Layer 8) — Conversation transitions
    # ==================================================================

    DIALOG_SCENARIOS = [
        "Simulate a 4-turn conversation where someone asks about the weather, then about restaurants.",
        "Simulate a conversation where someone asks a question, gets a partial answer, and asks for clarification.",
        "Simulate a conversation where someone disagrees politely and offers an alternative.",
        "Simulate a conversation where the topic changes from sports to technology naturally.",
        "Simulate a conversation with a greeting, a question, a follow-up, and a goodbye.",
        "Simulate a conversation where someone asks for help with a task step by step.",
        "Simulate a conversation where someone corrects a misunderstanding.",
        "Simulate a conversation that transitions from casual to formal.",
    ]

    def distill_dialog_patterns(self, max_scenarios=8):
        """
        Layer 8: Extract dialog transition patterns.
        Returns count of patterns stored.
        """
        import re
        stored = 0

        for scenario in self.DIALOG_SCENARIOS[:max_scenarios]:
            q_key = f"dialog:{scenario[:40]}".lower()
            if q_key in self._attempted_queries:
                continue
            self._attempted_queries.add(q_key)

            answer = self.backend.ask(
                scenario,
                system_prompt="Write a realistic dialog. Label each turn A: or B:. Keep it short.")
            if not answer:
                continue

            clean = re.sub(r'<think>.*?</think>', '', answer,
                           flags=re.DOTALL).strip()
            if not clean or len(clean) < 30:
                continue

            # Extract transition patterns between turns
            turns = re.findall(r'[AB]:\s*(.+?)(?=\n[AB]:|$)', clean, re.DOTALL)
            if len(turns) >= 2:
                # Store the transition connectors
                for i in range(len(turns) - 1):
                    t1 = turns[i].strip()[:80]
                    t2 = turns[i + 1].strip()[:80]
                    # Extract transition words/phrases
                    transition_words = re.findall(
                        r'^(speaking of|by the way|that reminds me|anyway|'
                        r'actually|oh|well|so|sure|right|hmm|'
                        r'i see|good point|interesting)',
                        t2.lower())
                    if transition_words:
                        self.knowledge.store_fact(
                            'dialog_transition', transition_words[0],
                            f'{t1[:40]} → {t2[:40]}')
                        stored += 1

                # Store overall pattern
                pattern = ' → '.join(
                    t.strip().split('.')[0][:30] for t in turns[:4])
                self.knowledge.store_fact(
                    'dialog_pattern', 'flow', pattern[:150])
                stored += 1
                self.reasoning.log_store('dialog_pattern', 'flow', pattern[:60])

        return stored

    # ==================================================================
    # GRAMMAR CHECKS (Layer 9) — Error detection rules
    # ==================================================================

    GRAMMAR_PAIRS = [
        ("The dogs runs fast.", "The dogs run fast.", "subject_verb_agreement"),
        ("Me and him went to store.", "He and I went to the store.", "pronoun_case"),
        ("I could of done that.", "I could have done that.", "modal_verb"),
        ("Their going to the park.", "They're going to the park.", "homophone"),
        ("The data shows that.", "The data show that.", "collective_noun"),
        ("Less people came.", "Fewer people came.", "count_mass"),
        ("Between you and I.", "Between you and me.", "preposition_case"),
        ("Who did you give it to?", "To whom did you give it?", "formal_whom"),
        ("I feel badly about it.", "I feel bad about it.", "linking_verb"),
        ("Irregardless of the facts.", "Regardless of the facts.", "nonstandard"),
        ("She did good on the test.", "She did well on the test.", "adjective_adverb"),
        ("The reason is because.", "The reason is that.", "redundancy"),
    ]

    def distill_grammar_rules(self, max_checks=12):
        """
        Layer 9: Extract grammar correctness rules.
        Returns count of rules stored.
        """
        stored = 0

        for wrong, correct, rule_type in self.GRAMMAR_PAIRS[:max_checks]:
            q_key = f"grammar:{wrong[:30]}".lower()
            if q_key in self._attempted_queries:
                continue
            self._attempted_queries.add(q_key)

            # Store the rule directly (no LLM needed — these are known rules)
            if not self.knowledge.has_fact(f'grammar:{rule_type}', 'wrong', wrong):
                self.knowledge.store_fact(f'grammar:{rule_type}', 'wrong', wrong)
                self.knowledge.store_fact(f'grammar:{rule_type}', 'correct', correct)
                stored += 2

            # Ask LLM to explain WHY (deeper understanding)
            answer = self.backend.ask(
                f"Why is '{wrong}' grammatically incorrect? One sentence.",
                system_prompt="Explain the grammar rule briefly. One sentence.")
            if answer:
                import re
                clean = re.sub(r'<think>.*?</think>', '', answer,
                               flags=re.DOTALL).strip()
                if clean and len(clean) > 10 and len(clean) < 200:
                    if not self.knowledge.has_fact(f'grammar:{rule_type}', 'rule', clean):
                        self.knowledge.store_fact(
                            f'grammar:{rule_type}', 'rule', clean)
                        stored += 1

        return stored

    def _save_brain(self, brain_path):
        """Save brain file (atomic write)."""
        try:
            from .brain import BrainSnapshot
            snap = BrainSnapshot.capture_from_parts(
                knowledge_store=self.knowledge)
            snap.save(brain_path, compressed=True)
            if self.verbose:
                size_kb = os.path.getsize(brain_path) / 1024
                print(f"  Brain saved: {size_kb:.0f} KB")
        except Exception as e:
            logger.warning(f"Brain save failed: {e}")
