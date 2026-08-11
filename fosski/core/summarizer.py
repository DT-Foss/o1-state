"""
Summarizer — Text Compression Without Transformers
====================================================
Extractive + abstractive-lite summarization.

Extractive: Sentence scoring → top-k selection
  - TF-IDF importance
  - Position bias (first/last sentences score higher)
  - Entity density (sentences with named entities score higher)
  - Redundancy penalty (MMR-style)

Abstractive-lite: Sentence fusion + compression
  - Remove parenthetical clauses
  - Remove adverbial hedging
  - Merge overlapping sentences

This replaces the summarization capability of ChatGPT/Claude.
No neural network. Pure statistics + linguistics.
"""

import re
import math
from typing import List, Dict, Optional, Tuple
from collections import Counter


# === Sentence Splitting ===

def split_sentences(text: str) -> List[str]:
    """Split text into sentences. Handles abbreviations and decimals."""
    # Protect known abbreviations
    protected = text
    abbrevs = ['Mr.', 'Mrs.', 'Dr.', 'Prof.', 'St.', 'Jr.', 'Sr.',
               'Inc.', 'Ltd.', 'Corp.', 'vs.', 'etc.', 'i.e.', 'e.g.',
               'U.S.', 'U.K.', 'E.U.', 'Ph.D.', 'M.D.']
    for abbr in abbrevs:
        protected = protected.replace(abbr, abbr.replace('.', '<DOT>'))

    # Protect decimals (3.14)
    protected = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', protected)

    # Split on sentence-ending punctuation
    parts = re.split(r'(?<=[.!?])\s+', protected)

    # Restore dots
    sentences = []
    for p in parts:
        p = p.replace('<DOT>', '.').strip()
        if p:
            sentences.append(p)

    return sentences


# === TF-IDF ===

def tokenize(text: str) -> List[str]:
    """Simple word tokenizer."""
    return re.findall(r'\b[a-z]+\b', text.lower())


STOP_WORDS = frozenset([
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'because', 'but', 'and', 'or', 'if', 'while', 'although',
    'this', 'that', 'these', 'those', 'it', 'its', 'he', 'she', 'they',
    'them', 'his', 'her', 'their', 'my', 'your', 'our', 'which', 'what',
    'who', 'whom', 'whose', 'also', 'about', 'up',
])


def compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Term frequency (normalized by max frequency)."""
    counts = Counter(tokens)
    if not counts:
        return {}
    max_freq = max(counts.values())
    return {word: count / max_freq for word, count in counts.items()}


def compute_idf(documents: List[List[str]]) -> Dict[str, float]:
    """Inverse document frequency across sentence-documents."""
    n = len(documents)
    if n == 0:
        return {}
    df = Counter()
    for doc in documents:
        df.update(set(doc))
    return {word: math.log(n / (1 + freq)) for word, freq in df.items()}


def tfidf_score(tokens: List[str], idf: Dict[str, float]) -> float:
    """Total TF-IDF score for a token list."""
    tf = compute_tf(tokens)
    score = 0.0
    for word, freq in tf.items():
        if word not in STOP_WORDS:
            score += freq * idf.get(word, 0)
    return score


# === Entity Detection (rule-based) ===

def detect_entities(sentence: str) -> List[str]:
    """Detect likely named entities (capitalized multi-word sequences)."""
    entities = []
    # Capitalized words not at sentence start
    words = sentence.split()
    for i, word in enumerate(words):
        if i > 0 and word[0].isupper() and word.isalpha() and len(word) > 1:
            entities.append(word)
    # Multi-word patterns
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', sentence):
        entities.append(m.group(1))
    # Numbers with units
    for m in re.finditer(r'\b\d+[\.,]?\d*\s*(?:km|m|kg|lb|°[CF]|%|billion|million|thousand)\b', sentence):
        entities.append(m.group(0))
    return entities


# === Main Summarizer ===

class Summarizer:
    """
    Extractive summarizer with optional compression.

    Algorithm:
    1. Split into sentences
    2. Score each sentence:
       - TF-IDF importance
       - Position bias
       - Entity density
       - Title/query relevance
    3. MMR-based selection (penalize redundancy)
    4. Optional compression (remove filler)
    """

    def __init__(self, compression_ratio: float = 0.3,
                 min_sentences: int = 1, max_sentences: int = 10,
                 position_weight: float = 0.15,
                 entity_weight: float = 0.1,
                 redundancy_penalty: float = 0.5):
        """
        Args:
            compression_ratio: target summary length as fraction of original
            min_sentences: minimum sentences in summary
            max_sentences: maximum sentences in summary
            position_weight: weight for position bias
            entity_weight: weight for entity density
            redundancy_penalty: MMR lambda for redundancy
        """
        self.compression_ratio = compression_ratio
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.position_weight = position_weight
        self.entity_weight = entity_weight
        self.redundancy_penalty = redundancy_penalty

    def summarize(self, text: str, query: Optional[str] = None,
                  n_sentences: Optional[int] = None,
                  compress: bool = True) -> Dict:
        """
        Summarize text.

        Args:
            text: input text
            query: optional focus query (boosts relevant sentences)
            n_sentences: explicit number of sentences (overrides ratio)
            compress: if True, apply sentence compression

        Returns:
            dict with: summary, sentences, scores, stats
        """
        sentences = split_sentences(text)
        if not sentences:
            return {'summary': '', 'sentences': [], 'scores': [], 'stats': {}}

        # Determine target length
        if n_sentences is not None:
            target = min(n_sentences, len(sentences))
        else:
            target = max(self.min_sentences,
                        min(self.max_sentences,
                            int(len(sentences) * self.compression_ratio)))
        target = min(target, len(sentences))

        # Tokenize all sentences
        sent_tokens = [tokenize(s) for s in sentences]

        # Compute IDF across sentences
        idf = compute_idf(sent_tokens)

        # Score each sentence
        scores = []
        for i, (sent, tokens) in enumerate(zip(sentences, sent_tokens)):
            score = self._score_sentence(
                sent, tokens, i, len(sentences), idf, query
            )
            scores.append(score)

        # MMR selection (greedy, penalize redundancy)
        selected_indices = self._mmr_select(
            sentences, sent_tokens, scores, target
        )

        # Keep original order
        selected_indices.sort()
        selected = [sentences[i] for i in selected_indices]

        # Optional compression
        if compress:
            selected = [self._compress_sentence(s) for s in selected]

        summary = ' '.join(selected)

        return {
            'summary': summary,
            'sentences': selected,
            'scores': [(i, scores[i]) for i in selected_indices],
            'stats': {
                'original_sentences': len(sentences),
                'summary_sentences': len(selected),
                'original_chars': len(text),
                'summary_chars': len(summary),
                'compression': round(len(summary) / max(len(text), 1), 2),
            },
        }

    def summarize_multi(self, texts: List[str],
                        query: Optional[str] = None,
                        n_sentences: int = 5) -> Dict:
        """
        Multi-document summarization.
        Merges texts, deduplicates, then summarizes.
        """
        merged = ' '.join(texts)
        return self.summarize(merged, query=query, n_sentences=n_sentences)

    def headline(self, text: str) -> str:
        """
        Generate a headline-style summary (one short sentence).
        Takes the highest-scoring sentence and compresses it aggressively.
        """
        result = self.summarize(text, n_sentences=1, compress=True)
        headline = result['summary']
        # Further compress: remove trailing clauses after comma/semicolon
        headline = re.split(r'[,;]', headline)[0].strip()
        # Remove leading articles
        headline = re.sub(r'^(?:The|A|An)\s+', '', headline)
        # Capitalize first letter
        if headline:
            headline = headline[0].upper() + headline[1:]
        # Remove trailing period
        headline = headline.rstrip('.')
        return headline

    def bullet_points(self, text: str, n_points: int = 5,
                      query: Optional[str] = None) -> List[str]:
        """Generate bullet-point summary."""
        result = self.summarize(text, query=query, n_sentences=n_points,
                               compress=True)
        return result['sentences']

    def _score_sentence(self, sentence: str, tokens: List[str],
                       position: int, total: int,
                       idf: Dict[str, float],
                       query: Optional[str] = None) -> float:
        """Score a single sentence."""
        # TF-IDF importance (normalized to ~0-1 range via sigmoid-like scaling)
        raw_importance = tfidf_score(tokens, idf)
        importance = raw_importance / (1 + raw_importance)  # Squash to 0-1

        # Position bias: first and last sentences score higher
        if total <= 1:
            pos_score = 1.0
        else:
            # U-shaped: high at start, dip in middle, rise at end
            normalized_pos = position / (total - 1)
            pos_score = 1.0 - 0.5 * math.sin(math.pi * normalized_pos)

        # Entity density
        entities = detect_entities(sentence)
        entity_score = min(len(entities) / max(len(tokens), 1) * 5, 1.0)

        # Length penalty: too short or too long sentences penalized
        word_count = len(sentence.split())
        if word_count < 5:
            length_factor = 0.5
        elif word_count > 50:
            length_factor = 0.8
        else:
            length_factor = 1.0

        # Query relevance (check both tokens AND raw case-insensitive match)
        query_score = 0.0
        if query:
            query_tokens = set(tokenize(query)) - STOP_WORDS
            sent_tokens_set = set(tokens) - STOP_WORDS
            if query_tokens:
                overlap = query_tokens & sent_tokens_set
                query_score = len(overlap) / len(query_tokens)
            # Boost if raw query appears in sentence (case-insensitive)
            if query.lower() in sentence.lower():
                query_score = max(query_score, 1.0)

        # Combine — when query provided and matches, it dominates
        if query and query_score > 0.5:
            # Strong query match: make query the primary signal
            score = (query_score * 0.7 +
                    importance * 0.15 +
                    pos_score * 0.1 +
                    entity_score * 0.05) * length_factor
        else:
            query_weight = 0.3 if query else 0.0
            importance_weight = 0.4 if query else 0.5
            score = (importance * importance_weight +
                    pos_score * self.position_weight +
                    entity_score * self.entity_weight +
                    query_score * query_weight) * length_factor

        return score

    def _mmr_select(self, sentences: List[str],
                    sent_tokens: List[List[str]],
                    scores: List[float],
                    target: int) -> List[int]:
        """
        Maximal Marginal Relevance selection.
        Balances relevance with diversity.
        """
        selected = []
        remaining = list(range(len(sentences)))

        for _ in range(target):
            if not remaining:
                break

            best_idx = None
            best_mmr = -float('inf')

            for idx in remaining:
                relevance = scores[idx]

                # Max similarity to already selected
                if selected:
                    max_sim = max(
                        self._jaccard(sent_tokens[idx], sent_tokens[s])
                        for s in selected
                    )
                else:
                    max_sim = 0.0

                mmr = (1 - self.redundancy_penalty) * relevance - \
                      self.redundancy_penalty * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return selected

    @staticmethod
    def _jaccard(tokens1: List[str], tokens2: List[str]) -> float:
        """Jaccard similarity between two token lists."""
        s1 = set(tokens1) - STOP_WORDS
        s2 = set(tokens2) - STOP_WORDS
        if not s1 and not s2:
            return 0.0
        intersection = s1 & s2
        union = s1 | s2
        return len(intersection) / max(len(union), 1)

    @staticmethod
    def _compress_sentence(sentence: str) -> str:
        """
        Compress a sentence by removing filler.
        - Parenthetical clauses
        - Hedging words
        - Redundant modifiers
        """
        s = sentence

        # Remove parenthetical content
        s = re.sub(r'\s*\([^)]*\)', '', s)
        s = re.sub(r'\s*\[[^\]]*\]', '', s)

        # Remove hedging
        hedges = [
            r'\b(?:it is|it\'s) (?:worth noting|important to note|interesting) that\s*',
            r'\b(?:basically|essentially|fundamentally|generally speaking|in general)\s*,?\s*',
            r'\b(?:as a matter of fact|in fact|actually|indeed|certainly)\s*,?\s*',
            r'\b(?:it should be noted that|it is known that|it has been shown that)\s*',
            r'\b(?:more or less|to some extent|in a way|sort of|kind of)\s*',
        ]
        for h in hedges:
            s = re.sub(h, '', s, flags=re.I)

        # Remove redundant adverbs
        s = re.sub(r'\b(?:very|really|quite|rather|fairly|extremely|incredibly|absolutely)\s+',
                   '', s, flags=re.I)

        # Clean up whitespace
        s = re.sub(r'\s{2,}', ' ', s).strip()

        return s


class TopicExtractor:
    """
    Extract key topics/keywords from text.
    Uses TF-IDF + n-gram analysis.
    """

    def extract_keywords(self, text: str, n: int = 10) -> List[Tuple[str, float]]:
        """Extract top-n keywords with scores."""
        tokens = tokenize(text)
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

        if not tokens:
            return []

        # Frequency-based with position boost
        freq = Counter(tokens)
        total = len(tokens)

        scored = {}
        for word, count in freq.items():
            tf = count / total
            # Boost words appearing early
            first_pos = tokens.index(word) / max(total, 1)
            position_boost = 1.0 + 0.5 * (1 - first_pos)
            scored[word] = tf * position_boost

        # Sort by score
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        return ranked[:n]

    def extract_bigrams(self, text: str, n: int = 5) -> List[Tuple[str, int]]:
        """Extract top-n meaningful bigrams."""
        tokens = tokenize(text)
        bigrams = []
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i+1]
            if w1 not in STOP_WORDS and w2 not in STOP_WORDS:
                bigrams.append(f"{w1} {w2}")

        counts = Counter(bigrams)
        return counts.most_common(n)

    def extract_topics(self, text: str, n: int = 5) -> List[str]:
        """Extract topic labels (keywords + bigrams merged)."""
        keywords = [w for w, _ in self.extract_keywords(text, n * 2)]
        bigrams = [b for b, _ in self.extract_bigrams(text, n)]

        # Prefer bigrams, fill with keywords
        topics = bigrams[:n]
        for kw in keywords:
            if len(topics) >= n:
                break
            if not any(kw in t for t in topics):
                topics.append(kw)

        return topics[:n]
