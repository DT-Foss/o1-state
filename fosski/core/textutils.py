"""
Text Utilities — Core NLP Without Dependencies
================================================
Everything a text processing system needs, in pure Python.

Functions:
  - Tokenization (word, sentence, paragraph)
  - Stemming (Porter-like, simplified)
  - Edit distance (Levenshtein, Damerau-Levenshtein)
  - N-gram generation
  - Text normalization
  - Fuzzy string matching
  - Word frequency analysis
  - Readability scoring (Flesch-Kincaid)
  - String similarity metrics
"""

import re
import math
from typing import List, Dict, Tuple, Optional
from collections import Counter


# === Tokenization ===

def word_tokenize(text: str) -> List[str]:
    """Split text into words, handling contractions and punctuation."""
    # Handle contractions
    text = re.sub(r"(\w)'(t|s|re|ve|ll|d|m)\b", r"\1 '\2", text)
    return re.findall(r"\b\w+(?:'\w+)?\b", text)


def sentence_tokenize(text: str) -> List[str]:
    """Split text into sentences."""
    # Protect abbreviations
    abbrevs = ['Mr.', 'Mrs.', 'Dr.', 'Prof.', 'St.', 'Jr.', 'Sr.',
               'Inc.', 'Ltd.', 'vs.', 'etc.', 'i.e.', 'e.g.',
               'U.S.', 'U.K.']
    protected = text
    for abbr in abbrevs:
        protected = protected.replace(abbr, abbr.replace('.', '<DOT>'))
    protected = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', protected)

    parts = re.split(r'(?<=[.!?])\s+', protected)
    return [p.replace('<DOT>', '.').strip() for p in parts if p.strip()]


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Generate n-grams from a token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def char_ngrams(text: str, n: int) -> List[str]:
    """Generate character n-grams."""
    return [text[i:i+n] for i in range(len(text) - n + 1)]


# === Stemming (Simplified Porter) ===

def stem(word: str) -> str:
    """
    Simplified Porter stemmer.
    Not as good as NLTK's, but works without dependencies.
    """
    word = word.lower()
    if len(word) <= 3:
        return word

    # Step 1: plurals and past tenses
    if word.endswith('sses'):
        word = word[:-2]
    elif word.endswith('ies'):
        word = word[:-2]
    elif word.endswith('ss'):
        pass
    elif word.endswith('s') and not word.endswith('us') and not word.endswith('ss'):
        word = word[:-1]

    if word.endswith('eed'):
        if len(word) > 4:
            word = word[:-1]
    elif word.endswith('ed') and any(c in 'aeiou' for c in word[:-2]):
        word = word[:-2]
        if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
            word += 'e'
    elif word.endswith('ing') and any(c in 'aeiou' for c in word[:-3]):
        word = word[:-3]
        if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
            word += 'e'

    # Step 2: suffixes
    replacements = [
        ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'),
        ('anci', 'ance'), ('izer', 'ize'), ('iser', 'ise'),
        ('alli', 'al'), ('entli', 'ent'), ('eli', 'e'),
        ('ousli', 'ous'), ('ization', 'ize'), ('isation', 'ise'),
        ('ation', 'ate'), ('ator', 'ate'), ('alism', 'al'),
        ('iveness', 'ive'), ('fulness', 'ful'), ('ousness', 'ous'),
        ('aliti', 'al'), ('iviti', 'ive'), ('biliti', 'ble'),
    ]
    for suffix, replacement in replacements:
        if word.endswith(suffix) and len(word) - len(suffix) > 1:
            word = word[:-len(suffix)] + replacement
            break

    # Step 3: remove common suffixes
    suffixes = ['ful', 'ness', 'ment', 'able', 'ible', 'ity', 'ive',
                'ous', 'al', 'ly', 'er', 'or', 'ist', 'ism']
    for s in sorted(suffixes, key=len, reverse=True):
        if word.endswith(s) and len(word) - len(s) > 2:
            word = word[:-len(s)]
            break

    return word


# === Edit Distance ===

def levenshtein(s1: str, s2: str) -> int:
    """Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insert = prev_row[j + 1] + 1
            delete = curr_row[j] + 1
            replace = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row.append(min(insert, delete, replace))
        prev_row = curr_row

    return prev_row[-1]


def damerau_levenshtein(s1: str, s2: str) -> int:
    """Damerau-Levenshtein distance (allows transpositions)."""
    len1 = len(s1)
    len2 = len(s2)
    d = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        d[i][0] = i
    for j in range(len2 + 1):
        d[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,      # deletion
                d[i][j-1] + 1,      # insertion
                d[i-1][j-1] + cost,  # substitution
            )
            # Transposition
            if i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
                d[i][j] = min(d[i][j], d[i-2][j-2] + cost)

    return d[len1][len2]


# === String Similarity ===

def jaccard_similarity(s1: str, s2: str, n: int = 2) -> float:
    """Jaccard similarity based on character n-grams."""
    ng1 = set(char_ngrams(s1.lower(), n))
    ng2 = set(char_ngrams(s2.lower(), n))
    if not ng1 and not ng2:
        return 1.0
    intersection = ng1 & ng2
    union = ng1 | ng2
    return len(intersection) / len(union)


def cosine_similarity(s1: str, s2: str) -> float:
    """Cosine similarity based on word overlap."""
    words1 = Counter(word_tokenize(s1.lower()))
    words2 = Counter(word_tokenize(s2.lower()))

    intersection = set(words1.keys()) & set(words2.keys())
    dot = sum(words1[w] * words2[w] for w in intersection)
    mag1 = math.sqrt(sum(v**2 for v in words1.values()))
    mag2 = math.sqrt(sum(v**2 for v in words2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def fuzzy_match(query: str, candidates: List[str],
                threshold: float = 0.6) -> List[Tuple[str, float]]:
    """Find best fuzzy matches for a query in a list of candidates."""
    results = []
    for candidate in candidates:
        sim = jaccard_similarity(query, candidate, n=2)
        if sim >= threshold:
            results.append((candidate, sim))
    return sorted(results, key=lambda x: x[1], reverse=True)


# === Text Normalization ===

def normalize(text: str) -> str:
    """Normalize text: lowercase, collapse whitespace, strip."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def remove_punctuation(text: str) -> str:
    """Remove all punctuation."""
    return re.sub(r'[^\w\s]', '', text)


def remove_urls(text: str) -> str:
    """Remove URLs."""
    return re.sub(r'https?://\S+', '', text)


def remove_numbers(text: str) -> str:
    """Remove standalone numbers."""
    return re.sub(r'\b\d+\b', '', text)


# === Word Frequency ===

def word_frequency(text: str, top_n: int = 20,
                   stop_words: Optional[set] = None) -> List[Tuple[str, int]]:
    """Get word frequency distribution."""
    tokens = word_tokenize(text.lower())
    if stop_words:
        tokens = [t for t in tokens if t not in stop_words]
    return Counter(tokens).most_common(top_n)


def hapax_legomena(text: str) -> List[str]:
    """Words that appear exactly once."""
    tokens = word_tokenize(text.lower())
    freq = Counter(tokens)
    return [word for word, count in freq.items() if count == 1]


def type_token_ratio(text: str) -> float:
    """Vocabulary richness: unique words / total words."""
    tokens = word_tokenize(text.lower())
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


# === Readability ===

def syllable_count(word: str) -> int:
    """Estimate syllable count for a word."""
    word = word.lower().strip()
    if not word:
        return 0
    count = 0
    vowels = 'aeiouy'
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Adjust for silent e
    if word.endswith('e') and count > 1:
        count -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        count += 1
    return max(count, 1)


def flesch_kincaid(text: str) -> Dict[str, float]:
    """
    Flesch-Kincaid readability metrics.

    Returns:
        reading_ease: 0-100 (higher = easier)
        grade_level: US grade level
    """
    sentences = sentence_tokenize(text)
    words = word_tokenize(text)

    if not sentences or not words:
        return {'reading_ease': 0.0, 'grade_level': 0.0}

    total_syllables = sum(syllable_count(w) for w in words)
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = total_syllables / len(words)

    reading_ease = (206.835
                   - 1.015 * avg_sentence_length
                   - 84.6 * avg_syllables_per_word)

    grade_level = (0.39 * avg_sentence_length
                  + 11.8 * avg_syllables_per_word
                  - 15.59)

    return {
        'reading_ease': round(max(0, min(100, reading_ease)), 1),
        'grade_level': round(max(0, grade_level), 1),
        'sentences': len(sentences),
        'words': len(words),
        'syllables': total_syllables,
        'avg_sentence_length': round(avg_sentence_length, 1),
        'avg_syllables_per_word': round(avg_syllables_per_word, 2),
    }
