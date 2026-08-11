"""
Stacked Hopfield Layers — Hierarchical Pattern Retrieval
=========================================================
Three Hopfield stages with p-adic hierarchy (Z/3Z → Z/9Z → Z/27Z):
  - Layer 1: Word-level associations (GloVe dim, single words)
  - Layer 2: Phrase-level (takes L1 output, retrieves phrase patterns)
  - Layer 3: Sentence-level (takes L2 output, retrieves sentence patterns)

Each layer builds on the previous — like MLP layers in Transformers
(Geva et al. 2021: FFN = Key-Value Memory).

Uses Modern Hopfield energy (exponential) for high capacity.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import re


class ModernHopfieldLayer:
    """Single Modern Hopfield layer with exponential energy.

    Capacity: exponential in dimension (vs polynomial for classical).
    Retrieval: softmax-weighted sum of stored patterns.
    """

    def __init__(self, dim: int, beta: float = 1.0):
        """
        Args:
            dim: pattern dimension
            beta: inverse temperature (higher = sharper retrieval)
        """
        self.dim = dim
        self.beta = beta
        self.patterns: List[np.ndarray] = []
        self.metadata: List[dict] = []

    @property
    def n_stored(self) -> int:
        return len(self.patterns)

    def store(self, pattern: np.ndarray, meta: Optional[dict] = None):
        """Store a pattern."""
        p = np.array(pattern, dtype=np.float32)
        assert len(p) == self.dim, f"Expected dim {self.dim}, got {len(p)}"
        # Normalize
        norm = np.linalg.norm(p)
        if norm > 0:
            p = p / norm
        self.patterns.append(p)
        self.metadata.append(meta or {})

    def retrieve(self, query: np.ndarray, top_k: int = 1) -> List[Tuple[np.ndarray, dict, float]]:
        """Retrieve patterns via Modern Hopfield attention.

        Returns list of (pattern, metadata, attention_weight) tuples.
        """
        if not self.patterns:
            return []

        q = np.array(query, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        # Compute similarities
        P = np.array(self.patterns)  # (N, dim)
        sims = P @ q  # (N,)

        # Softmax attention with temperature
        sims_scaled = self.beta * sims
        sims_scaled -= sims_scaled.max()  # numerical stability
        weights = np.exp(sims_scaled)
        weights /= weights.sum() + 1e-10

        # Return top-k
        top_indices = np.argsort(-weights)[:top_k]
        results = []
        for idx in top_indices:
            results.append((
                self.patterns[idx],
                self.metadata[idx],
                float(weights[idx])
            ))
        return results

    def retrieve_weighted(self, query: np.ndarray) -> np.ndarray:
        """Retrieve as attention-weighted sum of all patterns.

        This is the "Modern Hopfield" update rule:
        x_new = softmax(beta * X^T * query) * X
        """
        if not self.patterns:
            return np.zeros(self.dim)

        q = np.array(query, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        P = np.array(self.patterns)  # (N, dim)
        sims = P @ q  # (N,)

        sims_scaled = self.beta * sims
        sims_scaled -= sims_scaled.max()
        weights = np.exp(sims_scaled)
        weights /= weights.sum() + 1e-10

        # Weighted sum of patterns
        return weights @ P  # (dim,)


class StackedHopfield:
    """Three-layer hierarchical Hopfield with p-adic weighting.

    Layer 1 (Word): Retrieves word-level associations
      p-adic weight: 1 (strongest, most local)

    Layer 2 (Phrase): Retrieves phrase-level patterns
      p-adic weight: 1/3 (medium, composition level)

    Layer 3 (Sentence): Retrieves sentence-level completions
      p-adic weight: 1/9 (weakest, most global)

    Output = w1*L1 + w2*L2 + w3*L3 (additive, like residual stream)
    """

    def __init__(self, dim: int = 100, beta: float = 8.0):
        """
        Args:
            dim: embedding dimension (must match GloVe/embedding dim)
            beta: inverse temperature for all layers
        """
        self.dim = dim
        # p-adic weights: 1, 1/3, 1/9
        self.layer1 = ModernHopfieldLayer(dim, beta=beta)
        self.layer2 = ModernHopfieldLayer(dim, beta=beta * 0.5)  # Lower beta for broader phrase matching
        self.layer3 = ModernHopfieldLayer(dim, beta=beta * 0.25)  # Even broader for sentences
        self.w1 = 1.0      # Z/3Z
        self.w2 = 1.0/3    # Z/9Z
        self.w3 = 1.0/9    # Z/27Z

    def store_word(self, word_vec: np.ndarray, word: str):
        """Store a word-level pattern in Layer 1."""
        self.layer1.store(word_vec, {'word': word, 'level': 'word'})

    def store_phrase(self, phrase_vec: np.ndarray, phrase: str):
        """Store a phrase-level pattern in Layer 2."""
        self.layer2.store(phrase_vec, {'phrase': phrase, 'level': 'phrase'})

    def store_sentence(self, sent_vec: np.ndarray, sentence: str):
        """Store a sentence-level pattern in Layer 3."""
        self.layer3.store(sent_vec, {'sentence': sentence, 'level': 'sentence'})

    def retrieve(self, query: np.ndarray) -> np.ndarray:
        """Hierarchical retrieval: query through all 3 layers.

        Flow: query → L1 → L1_out → L2 → L2_out → L3 → L3_out
        Final = w1*L1_out + w2*L2_out + w3*L3_out
        """
        # Layer 1: Word-level retrieval
        l1_out = self.layer1.retrieve_weighted(query)

        # Layer 2: Phrase-level retrieval (uses L1 output as refined query)
        # Mix original query with L1 output for context
        l2_query = 0.5 * query + 0.5 * l1_out if np.linalg.norm(l1_out) > 0 else query
        l2_out = self.layer2.retrieve_weighted(l2_query)

        # Layer 3: Sentence-level retrieval
        l3_query = 0.5 * l2_query + 0.5 * l2_out if np.linalg.norm(l2_out) > 0 else l2_query
        l3_out = self.layer3.retrieve_weighted(l3_query)

        # Additive fusion with p-adic weights
        result = self.w1 * l1_out + self.w2 * l2_out + self.w3 * l3_out

        # Normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm

        return result

    def score_text(self, text_vec: np.ndarray, query: np.ndarray) -> float:
        """Score a candidate text against a query using hierarchical retrieval.

        Returns cosine similarity between retrieved pattern and text_vec.
        """
        retrieved = self.retrieve(query)
        norm_r = np.linalg.norm(retrieved)
        norm_t = np.linalg.norm(text_vec)
        if norm_r == 0 or norm_t == 0:
            return 0.0
        return float(np.dot(retrieved, text_vec) / (norm_r * norm_t))

    @property
    def stats(self) -> dict:
        return {
            'layer1_patterns': self.layer1.n_stored,
            'layer2_patterns': self.layer2.n_stored,
            'layer3_patterns': self.layer3.n_stored,
            'total': self.layer1.n_stored + self.layer2.n_stored + self.layer3.n_stored,
        }


def build_stacked_hopfield_from_glove(
    glove_words: np.ndarray,
    glove_vecs: np.ndarray,
    word2idx: Dict[str, int],
    training_text: str = '',
    max_words: int = 10000,
    dim: int = 100,
) -> StackedHopfield:
    """Build a StackedHopfield populated from GloVe and training text.

    Layer 1: Top-K GloVe word vectors
    Layer 2: Bigram phrase vectors (average of word pairs)
    Layer 3: Sentence vectors (average of words)
    """
    sh = StackedHopfield(dim=dim)

    # Layer 1: Store top-K word vectors
    n_store = min(max_words, len(glove_words))
    for i in range(n_store):
        word = str(glove_words[i])
        vec = glove_vecs[i]
        sh.store_word(vec, word)

    if not training_text:
        return sh

    # Layer 2: Bigram phrase vectors
    stop = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'and', 'but', 'or', 'it', 'its', 'he', 'she', 'they'}
    sentences = re.split(r'[.!?]+', training_text)
    phrase_set = set()
    for sent in sentences:
        words = [w.lower() for w in re.findall(r'\b[a-z]+\b', sent.lower())
                 if w not in stop and len(w) > 2 and w in word2idx]
        for j in range(len(words) - 1):
            phrase = words[j] + ' ' + words[j+1]
            if phrase not in phrase_set:
                phrase_set.add(phrase)
                v1 = glove_vecs[word2idx[words[j]]]
                v2 = glove_vecs[word2idx[words[j+1]]]
                phrase_vec = (v1 + v2) / 2.0
                sh.store_phrase(phrase_vec, phrase)
                if len(phrase_set) >= 20000:
                    break
        if len(phrase_set) >= 20000:
            break

    # Layer 3: Sentence vectors
    sent_set = set()
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 10:
            continue
        words = [w.lower() for w in re.findall(r'\b[a-z]+\b', sent.lower())
                 if w in word2idx]
        if len(words) < 3:
            continue
        indices = [word2idx[w] for w in words]
        sent_vec = np.mean(glove_vecs[indices], axis=0)
        short = ' '.join(words[:10])
        if short not in sent_set:
            sent_set.add(short)
            sh.store_sentence(sent_vec, short)
            if len(sent_set) >= 10000:
                break

    return sh
