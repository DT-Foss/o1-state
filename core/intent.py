"""
Intent Classifier — Pre-Transformer NLU
==========================================
Classifies user input into intent categories using:
  1. Keyword pattern matching (fast path)
  2. TF-IDF similarity to trained exemplars
  3. Entity extraction (names, numbers, dates)

No neural networks. No external dependencies.
"""

import re
import math
from typing import Dict, Any, List, Tuple
from collections import defaultdict


# ============================================================
# Intent Definitions
# ============================================================

INTENT_PATTERNS = {
    'greeting': {
        'keywords': {'hello', 'hi', 'hey', 'greetings', 'good morning',
                     'good afternoon', 'good evening', 'howdy', 'hallo',
                     'guten tag', 'moin'},
        'patterns': [r'^(hi|hey|hello|hallo|moin)\b'],
        'exemplars': [
            "hello there", "hi how are you", "hey", "good morning",
            "greetings", "hallo", "moin moin",
        ],
    },
    'farewell': {
        'keywords': {'bye', 'goodbye', 'see you', 'quit', 'exit',
                     'tschüss', 'auf wiedersehen'},
        'patterns': [r'\b(bye|goodbye|see you|tsch[üu]ss)\b'],
        'exemplars': [
            "goodbye", "bye bye", "see you later", "gotta go",
        ],
    },
    'question_factual': {
        'keywords': set(),
        'patterns': [
            r'^(what|who|where|when|which)\s+(is|are|was|were|does)\b',
            r'^(what|who|where|when)\b.*\?$',
            r'\b(capital|population|president|inventor)\b',
        ],
        'exemplars': [
            "what is the capital of France",
            "who invented the telephone",
            "where is Tokyo",
            "when was Python created",
            "which planet is largest",
            "what is the population of Germany",
        ],
    },
    'question_how': {
        'keywords': set(),
        'patterns': [r'^how\s+(do|does|can|to|is|are|many|much|long|far)\b'],
        'exemplars': [
            "how does photosynthesis work",
            "how to sort a list in Python",
            "how many planets are there",
            "how far is the moon",
        ],
    },
    'question_why': {
        'keywords': set(),
        'patterns': [r'^why\s+(do|does|is|are|did|can|would)\b'],
        'exemplars': [
            "why is the sky blue",
            "why do birds migrate",
            "why does ice float",
            "why is education important",
        ],
    },
    'math': {
        'keywords': {'calculate', 'compute', 'solve', 'sum', 'product',
                     'factorial', 'gcd', 'prime', 'derivative', 'integral'},
        'patterns': [
            r'\b\d+\s*[\+\-\*\/\^]\s*\d+',
            r'\bsolve\b.*=',
            r'\b(calculate|compute)\b',
            r'\b(derivative|integral|integrate)\b',
            r'\b(prime|factorial|gcd|lcm)\b',
            r'\b\d+\s*(squared|cubed)\b',
        ],
        'exemplars': [
            "calculate 17 * 23",
            "what is 5 + 3",
            "solve x^2 - 5x + 6 = 0",
            "is 97 prime",
            "derivative of 3x^2",
            "GCD of 48 and 18",
            "5 squared",
            "factorial of 10",
        ],
    },
    'definition': {
        'keywords': set(),
        'patterns': [
            r'^(what|define|explain)\s+(is|are)\s+(a|an|the)?\s*\w+',
            r'^define\b',
            r'\bwhat\s+does?\s+\w+\s+mean\b',
        ],
        'exemplars': [
            "what is machine learning",
            "define recursion",
            "explain neural networks",
            "what does API mean",
        ],
    },
    'comparison': {
        'keywords': {'compare', 'difference', 'versus', 'vs', 'better',
                     'worse', 'faster', 'slower'},
        'patterns': [
            r'\b(compare|vs|versus)\b',
            r'\bdifference\s+between\b',
            r'\bwhat.*(better|worse|faster|slower)\b',
        ],
        'exemplars': [
            "compare Python and Java",
            "difference between TCP and UDP",
            "which is faster, quicksort or mergesort",
            "Python vs JavaScript",
        ],
    },
    'command_file': {
        'keywords': {'read', 'write', 'create', 'delete', 'find', 'search',
                     'list', 'open', 'save'},
        'patterns': [
            r'\b(read|write|create|open|save|delete)\s+(file|directory)\b',
            r'\bfind\s+(all\s+)?\*\.',
            r'\blist\s+(files|directories)\b',
        ],
        'exemplars': [
            "read file config.py",
            "find all *.py files",
            "list files in current directory",
            "create a new file called test.py",
        ],
    },
    'opinion': {
        'keywords': {'think', 'opinion', 'believe', 'feel', 'should'},
        'patterns': [
            r'\bwhat\s+do\s+you\s+think\b',
            r'\bshould\s+I\b',
            r'\bin\s+your\s+opinion\b',
        ],
        'exemplars': [
            "what do you think about AI",
            "should I learn Python or Java",
            "in your opinion, what is the best language",
        ],
    },
    'code': {
        'keywords': {'code', 'function', 'class', 'program', 'implement',
                     'write', 'debug', 'fix', 'refactor'},
        'patterns': [
            r'\b(write|implement|create)\s+(a\s+)?(function|class|program|script)\b',
            r'\b(debug|fix|refactor)\s+(the|this|my)\b',
            r'\bhow\s+to\s+(code|program|implement)\b',
        ],
        'exemplars': [
            "write a function that sorts a list",
            "implement a binary search",
            "debug this Python code",
            "refactor the router module",
        ],
    },
    'commonsense': {
        'keywords': set(),
        'patterns': [
            r'^(is|are|can|does)\s+(a\s+|an\s+)?\w+\s+(a\s+|an\s+)?\w+\??\s*$',
            r'\b(is\s+\w+\s+(cold|hot|wet|dry|big|small|fast|slow))\b',
        ],
        'exemplars': [
            "is ice cold",
            "can birds fly",
            "is a dog an animal",
            "does water freeze",
        ],
    },
}


class IntentClassifier:
    """Classify user input into intent categories."""

    def __init__(self):
        # Build TF-IDF index from exemplars
        self._idf: Dict[str, float] = {}
        self._exemplar_vecs: List[Tuple[str, Dict[str, float]]] = []
        self._build_index()

    def _build_index(self):
        """Build TF-IDF vectors for all exemplars."""
        # Compute document frequencies
        df: Dict[str, int] = defaultdict(int)
        all_exemplars = []

        for intent, config in INTENT_PATTERNS.items():
            for exemplar in config.get('exemplars', []):
                tokens = self._tokenize(exemplar)
                all_exemplars.append((intent, tokens))
                for token in set(tokens):
                    df[token] += 1

        n_docs = len(all_exemplars)
        self._idf = {
            word: math.log((n_docs + 1) / (count + 1))
            for word, count in df.items()
        }

        # Build TF-IDF vectors
        for intent, tokens in all_exemplars:
            vec = self._tfidf_vector(tokens)
            self._exemplar_vecs.append((intent, vec))

    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        return re.findall(r'\b\w+\b', text.lower())

    def _tfidf_vector(self, tokens: List[str]) -> Dict[str, float]:
        """Compute TF-IDF vector for token list."""
        tf: Dict[str, float] = defaultdict(float)
        for token in tokens:
            tf[token] += 1
        n = len(tokens) if tokens else 1

        vec: Dict[str, float] = {}
        for token, count in tf.items():
            idf = self._idf.get(token, 1.0)
            vec[token] = (count / n) * idf
        return vec

    def _cosine_sim(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        """Cosine similarity between two sparse vectors."""
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in set(a) | set(b))
        norm_a = math.sqrt(sum(v * v for v in a.values())) if a else 0
        norm_b = math.sqrt(sum(v * v for v in b.values())) if b else 0
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify user input.

        Returns:
            {
                'intent': str,
                'confidence': float,
                'method': str,  # 'keyword', 'pattern', 'tfidf'
                'entities': dict,
                'all_scores': dict,
            }
        """
        text_lower = text.lower().strip()
        scores: Dict[str, float] = defaultdict(float)

        # Phase 1: Keyword matching (fast, high confidence)
        words = set(self._tokenize(text))
        for intent, config in INTENT_PATTERNS.items():
            overlap = words & config.get('keywords', set())
            if overlap:
                scores[intent] += 0.6 * len(overlap)

        # Phase 2: Regex pattern matching
        for intent, config in INTENT_PATTERNS.items():
            for pattern in config.get('patterns', []):
                if re.search(pattern, text_lower):
                    scores[intent] += 0.8
                    break

        # Phase 3: TF-IDF similarity
        query_tokens = self._tokenize(text)
        query_vec = self._tfidf_vector(query_tokens)

        intent_sims: Dict[str, List[float]] = defaultdict(list)
        for intent, vec in self._exemplar_vecs:
            sim = self._cosine_sim(query_vec, vec)
            intent_sims[intent].append(sim)

        for intent, sims in intent_sims.items():
            best_sim = max(sims) if sims else 0
            scores[intent] += best_sim * 0.5

        # Determine winner
        if not scores:
            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'method': 'none',
                'entities': self._extract_entities(text),
                'all_scores': {},
            }

        best_intent = max(scores, key=lambda k: scores[k])
        best_score = scores[best_intent]

        # Determine method
        method = 'tfidf'
        for intent, config in INTENT_PATTERNS.items():
            if intent == best_intent:
                if words & config.get('keywords', set()):
                    method = 'keyword'
                    break
                for pattern in config.get('patterns', []):
                    if re.search(pattern, text_lower):
                        method = 'pattern'
                        break

        # Normalize confidence to 0-1
        confidence = min(1.0, best_score / 1.5)

        return {
            'intent': best_intent,
            'confidence': round(confidence, 3),
            'method': method,
            'entities': self._extract_entities(text),
            'all_scores': {k: round(v, 3) for k, v in sorted(
                scores.items(), key=lambda x: x[1], reverse=True
            )[:5]},
        }

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text."""
        entities: Dict[str, List[str]] = {}

        # Numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        if numbers:
            entities['numbers'] = numbers

        # Filenames
        files = re.findall(r'\b[\w\-]+\.\w{1,5}\b', text)
        if files:
            entities['files'] = files

        # Quoted strings
        quoted = re.findall(r'"([^"]+)"', text) + re.findall(r"'([^']+)'", text)
        if quoted:
            entities['quoted'] = quoted

        # Capitalized words (potential proper nouns)
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # Filter out sentence starts
        words = text.split()
        if words and caps:
            # Remove first word if it's just sentence case
            if caps and caps[0] == words[0]:
                caps = caps[1:]
        if caps:
            entities['proper_nouns'] = list(set(caps))

        return entities

    def batch_classify(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple texts."""
        return [self.classify(text) for text in texts]

    def get_intent_names(self) -> List[str]:
        """List all known intent categories."""
        return sorted(INTENT_PATTERNS.keys())

    def add_exemplar(self, intent: str, text: str):
        """Add a new exemplar for an intent (runtime only)."""
        tokens = self._tokenize(text)
        vec = self._tfidf_vector(tokens)
        self._exemplar_vecs.append((intent, vec))

    def stats(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        return {
            'intents': len(INTENT_PATTERNS),
            'exemplars': len(self._exemplar_vecs),
            'vocab_size': len(self._idf),
            'patterns': sum(
                len(c.get('patterns', [])) for c in INTENT_PATTERNS.values()
            ),
        }
