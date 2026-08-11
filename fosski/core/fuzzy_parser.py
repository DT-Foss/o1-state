"""
Fuzzy Intent Parser — Ported from FORGE's intent_parser.py
=============================================================
Natural language → Structured intent via:
1. Tokenization (split + stopword removal + stemming)
2. Entity extraction (Jaro-Winkler fuzzy matching)
3. Intent classification (keyword scoring)
4. Parameter extraction (paths, formats, numbers)
5. Composition detection (multi-step tasks)

Jaro-Winkler is built-in (no jellyfish dependency).
"""

import re
from typing import Dict, List, Any, Optional


def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    """
    Jaro-Winkler similarity. Returns 0.0-1.0.
    Built-in implementation (no external dependency).
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 +
            (matches - transpositions / 2) / matches) / 3

    # Winkler modification: boost for common prefix
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * p * (1 - jaro)


class FuzzyParser:
    """
    Full intent parsing pipeline.
    Ported from FORGE core/intent_parser.py.
    """

    STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my',
        'your', 'his', 'her', 'its', 'our', 'their', 'me', 'him', 'us',
        'them', 'all', 'some', 'any', 'no', 'not', 'so', 'please',
    }

    BUILD_KEYWORDS = {
        'build', 'create', 'make', 'generate', 'write', 'code', 'script',
        'program', 'develop', 'implement', 'construct', 'run', 'execute',
        'add', 'append', 'refactor', 'update', 'modify',
    }

    QUESTION_KEYWORDS = {
        'what', 'why', 'how', 'when', 'where', 'who', 'which', 'explain',
        'tell', 'describe', 'define', 'mean',
    }

    ACTION_VERBS = {
        'list', 'read', 'parse', 'convert', 'download', 'scrape', 'fetch',
        'extract', 'sort', 'filter', 'merge', 'split', 'count', 'rename',
        'copy', 'move', 'delete', 'search', 'find', 'replace', 'process',
        'transform', 'analyze', 'send', 'upload', 'compress',
        'encrypt', 'hash', 'validate', 'scan', 'monitor', 'serve',
        'save', 'archive', 'backup', 'export', 'import',
        'detect', 'check', 'test', 'compile', 'deploy',
        'optimize', 'verify', 'report', 'summarize',
    }

    COMPOSITION_CONJUNCTIONS = {'and', 'then', 'also', 'plus', 'into'}

    def __init__(self, known_entities=None):
        """
        Args:
            known_entities: list of known entity strings for fuzzy matching.
                           Can be fragment keys, knowledge entities, etc.
        """
        self.known_entities = set(known_entities or [])

    def set_entities(self, entities):
        """Update the known entity set."""
        self.known_entities = set(entities)

    def tokenize(self, text: str) -> List[str]:
        """Split, lowercase, remove stopwords."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [t for t in tokens if t not in self.STOPWORDS]

    def stem(self, word: str) -> str:
        """Simple suffix-stripping stemmer."""
        for suffix in ['ing', 'ed', 'es', 's', 'er', 'ly', 'tion', 'ment']:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word

    def extract_entities(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """Extract entities via exact + stemmed + fuzzy matching."""
        entities = []
        known_lower = {e.lower(): e for e in self.known_entities}

        for token in tokens:
            stemmed = self.stem(token)

            # Exact match
            if token in known_lower:
                entities.append({
                    'original': token,
                    'matched': known_lower[token],
                    'confidence': 1.0,
                    'method': 'exact',
                })
                continue

            # Stemmed exact match
            if stemmed in known_lower:
                entities.append({
                    'original': token,
                    'matched': known_lower[stemmed],
                    'confidence': 0.95,
                    'method': 'stem',
                })
                continue

            # Skip short tokens for fuzzy (too noisy)
            if len(stemmed) <= 4:
                continue

            # Jaro-Winkler fuzzy match
            best_match = None
            best_score = 0.0

            for entity in self.known_entities:
                e_lower = entity.lower()
                score = jaro_winkler(stemmed, e_lower)

                # Bonus for shared prefix
                if e_lower.startswith(stemmed[:3]):
                    score += 0.1

                # Penalty for large length difference
                if abs(len(stemmed) - len(entity)) > 8:
                    score -= 0.15

                if score > best_score and score > 0.88:
                    best_score = score
                    best_match = entity

            if best_match:
                entities.append({
                    'original': token,
                    'matched': best_match,
                    'confidence': min(1.0, best_score),
                    'method': 'fuzzy',
                })

        return entities

    def classify(self, tokens: List[str],
                 entities: Optional[List[Dict]] = None) -> str:
        """Classify intent: BUILD, QUESTION, TOOL, CONVERSATION."""
        token_set = set(tokens)

        build_score = len(token_set & self.BUILD_KEYWORDS)
        question_score = len(token_set & self.QUESTION_KEYWORDS)
        action_score = len(token_set & self.ACTION_VERBS)

        # Explicit build keywords
        if build_score > 0:
            return 'BUILD'

        # Action verbs without question words = implicit BUILD
        if action_score > 0 and question_score == 0:
            return 'BUILD'

        # Multiple entities without question words = BUILD
        if entities and len(entities) >= 2 and question_score == 0:
            return 'BUILD'

        if question_score > 0:
            return 'QUESTION'

        # Default: if entities found, assume BUILD
        if entities and len(entities) > 0:
            return 'BUILD'

        return 'QUESTION'

    def extract_parameters(self, text: str) -> Dict[str, Any]:
        """Extract paths, URLs, IPs, ports, numbers from text."""
        params = {}

        # File paths
        paths = re.findall(r'(?:~?/[\w/.-]+|\.{1,2}/[\w/.-]+|\w+/[\w/.-]+)', text)
        if paths:
            params['paths'] = paths
            params['input_path'] = paths[0]

        # URLs
        urls = re.findall(r'https?://\S+', text)
        if urls:
            params['urls'] = urls
            params['url'] = urls[0]

        # IPs
        ips = re.findall(r'\d+\.\d+\.\d+\.\d+', text)
        if ips:
            params['ips'] = ips
            params['target'] = ips[0]

        # Ports
        ports = re.findall(r'port\s*(\d+)', text, re.I)
        if ports:
            params['port'] = int(ports[0])

        # Numbers
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            params['numbers'] = [int(n) for n in numbers]

        # File extensions
        exts = re.findall(r'\b(\w+)\s+file', text, re.I)
        if exts:
            params['format'] = exts[0].upper()

        return params

    def detect_composition(self, text: str,
                            tokens: List[str]) -> Dict[str, Any]:
        """Detect multi-step tasks: 'download X and parse Y'."""
        words = set(text.lower().split())
        has_conjunction = bool(words & self.COMPOSITION_CONJUNCTIONS)
        actions = [t for t in tokens if t in self.ACTION_VERBS]

        is_composition = has_conjunction and len(actions) >= 2

        if not is_composition and len(actions) >= 1:
            if ' to ' in text.lower() or ' into ' in text.lower():
                conversion_verbs = {'convert', 'transform', 'export', 'read'}
                if set(actions) & conversion_verbs:
                    is_composition = True

        return {
            'is_composition': is_composition,
            'sub_tasks': actions if is_composition else [],
        }

    def parse(self, user_input: str) -> Dict[str, Any]:
        """
        Full parsing pipeline. Returns structured intent.

        Returns:
            mode: BUILD | QUESTION
            raw: original input
            tokens: cleaned tokens
            entities: fuzzy-matched entities
            params: extracted parameters
            confidence: overall confidence
            is_composition: whether this is a multi-step task
            sub_tasks: action verbs for composition
        """
        tokens = self.tokenize(user_input)
        entities = self.extract_entities(tokens)
        mode = self.classify(tokens, entities)
        params = self.extract_parameters(user_input)
        composition = self.detect_composition(user_input, tokens)

        confidence = (sum(e['confidence'] for e in entities) / len(entities)
                      if entities else 0.5)

        return {
            'mode': mode,
            'raw': user_input,
            'tokens': tokens,
            'entities': entities,
            'params': params,
            'confidence': confidence,
            'is_composition': composition['is_composition'],
            'sub_tasks': composition['sub_tasks'],
        }
