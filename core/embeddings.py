"""
Embeddings — Markov Transition Matrix Semantic Vectors
========================================================
Replaces TF-IDF (1972) with Markov chain transition embeddings.

Key insight: The row of a word's transition matrix IS its embedding.
Words with similar successors are semantically similar.
This is conceptually Word2Vec (2013) but deterministic,
order-preserving, and built on Foss Markov mathematics.

"dog bites man" vs "man bites dog" → different transition rows.
TF-IDF treats them identically. Markov does not.

Capabilities:
  - Document embedding (text → vector)
  - Semantic similarity (cosine distance)
  - Nearest neighbor search
  - Clustering (k-means)
"""

import math
import re
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict


# === Tokenization ===

STOP_WORDS = frozenset([
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
    'under', 'again', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
    'if', 'while', 'this', 'that', 'these', 'those', 'it', 'its',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who',
    'whom', 'about', 'up',
])


def tokenize(text: str, remove_stops: bool = True) -> List[str]:
    """Tokenize and optionally remove stop words."""
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    if remove_stops:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens


# === Markov Transition Embeddings ===

class MarkovEmbedder:
    """Build word embeddings from Markov chain transition matrices.

    Each word's embedding = its transition probability row.
    Words with similar successors are semantically similar.
    Order-preserving: "dog bites" ≠ "bites dog".
    """

    def __init__(self, dim: int = 64, context_window: int = 2):
        self.dim = dim
        self.context_window = context_window
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.transitions: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.totals: Dict[int, int] = defaultdict(int)
        self._embeddings: Optional[List[List[float]]] = None
        self._fitted = False

    def fit(self, documents: List[str]) -> 'MarkovEmbedder':
        """Build transition matrix from corpus."""
        # Build vocabulary from all documents
        word_freq = Counter()
        for doc in documents:
            word_freq.update(tokenize(doc))

        # Take top dim words as vocabulary (transition matrix dimension)
        top_words = word_freq.most_common(self.dim * 4)  # oversample, then hash
        for i, (word, _) in enumerate(top_words):
            self.word2idx[word] = i
            self.idx2word[i] = word

        # Build transition matrix
        for doc in documents:
            tokens = tokenize(doc)
            for i, token in enumerate(tokens):
                if token not in self.word2idx:
                    continue
                src = self.word2idx[token]
                # Successors within context window
                for j in range(1, self.context_window + 1):
                    if i + j < len(tokens) and tokens[i + j] in self.word2idx:
                        tgt = self.word2idx[tokens[i + j]]
                        self.transitions[src][tgt] += 1
                        self.totals[src] += 1

        # Convert to probability vectors (embeddings)
        n_words = len(self.word2idx)
        self._embeddings = []
        for i in range(n_words):
            row = [0.0] * self.dim
            total = self.totals.get(i, 0)
            if total > 0:
                for tgt, count in self.transitions.get(i, {}).items():
                    idx = tgt % self.dim  # hash to fixed dim
                    row[idx] += count / total
            # Normalize
            norm = math.sqrt(sum(x * x for x in row))
            if norm > 0:
                row = [x / norm for x in row]
            self._embeddings.append(row)

        self._fitted = True
        return self

    def word_embedding(self, word: str) -> List[float]:
        """Get embedding for a single word."""
        if word in self.word2idx:
            return self._embeddings[self.word2idx[word]]
        return [0.0] * self.dim

    def text_embedding(self, text: str) -> List[float]:
        """Get embedding for a text (weighted average of word embeddings)."""
        tokens = tokenize(text)
        if not tokens or not self._fitted:
            return [0.0] * self.dim

        # Position-weighted average (earlier words matter more for context)
        vec = [0.0] * self.dim
        total_weight = 0.0
        for i, token in enumerate(tokens):
            if token in self.word2idx:
                w = 1.0 / (1 + i * 0.05)  # position decay
                emb = self._embeddings[self.word2idx[token]]
                for j in range(self.dim):
                    vec[j] += w * emb[j]
                total_weight += w

        if total_weight > 0:
            vec = [x / total_weight for x in vec]
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


# === TF-IDF (kept for backward compatibility) ===

class TFIDFVectorizer:
    """Build TF-IDF vectors from a corpus."""

    def __init__(self, max_features: int = 5000, min_df: int = 1, max_df_ratio: float = 0.95):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._fitted = False

    def fit(self, documents: List[str]) -> 'TFIDFVectorizer':
        """Build vocabulary and IDF from documents."""
        n = len(documents)
        doc_tokens = [tokenize(doc) for doc in documents]
        df = Counter()
        for tokens in doc_tokens:
            df.update(set(tokens))
        max_df = int(n * self.max_df_ratio)
        valid_terms = {
            term: freq for term, freq in df.items()
            if self.min_df <= freq <= max_df
        }
        sorted_terms = sorted(valid_terms.items(), key=lambda x: x[1], reverse=True)
        top_terms = sorted_terms[:self.max_features]
        self.vocabulary = {term: i for i, (term, _) in enumerate(top_terms)}
        self.idf = {
            term: math.log(n / (1 + df[term])) + 1
            for term in self.vocabulary
        }
        self._fitted = True
        return self

    def transform(self, documents: List[str]) -> List[List[float]]:
        """Transform documents to TF-IDF vectors."""
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        vectors = []
        for doc in documents:
            tokens = tokenize(doc)
            tf = Counter(tokens)
            max_tf = max(tf.values()) if tf else 1
            vec = [0.0] * len(self.vocabulary)
            for term, idx in self.vocabulary.items():
                if term in tf:
                    tf_val = 0.5 + 0.5 * (tf[term] / max_tf)
                    vec[idx] = tf_val * self.idf.get(term, 0)
            vectors.append(vec)
        return vectors

    def fit_transform(self, documents: List[str]) -> List[List[float]]:
        self.fit(documents)
        return self.transform(documents)


# === Embedding Engine (now uses Markov embeddings) ===

class EmbeddingEngine:
    """
    Complete embedding pipeline: text → dense vector.
    Uses Markov transition embeddings instead of TF-IDF + SVD.
    """

    def __init__(self, dim: int = 64, max_vocab: int = 5000):
        self.dim = dim
        self.max_vocab = max_vocab
        self.markov = MarkovEmbedder(dim=dim, context_window=3)
        # Keep TF-IDF as fallback
        self.vectorizer = TFIDFVectorizer(max_features=max_vocab)
        self.embeddings: List[List[float]] = []
        self.documents: List[str] = []
        self.labels: List[str] = []
        self._fitted = False

    def fit(self, documents: List[str], labels: Optional[List[str]] = None):
        """Fit the embedding space from a corpus."""
        self.documents = documents
        self.labels = labels or [f"doc_{i}" for i in range(len(documents))]

        # Markov transition embeddings
        self.markov.fit(documents)

        # Compute document embeddings
        self.embeddings = []
        for doc in documents:
            emb = self.markov.text_embedding(doc)
            self.embeddings.append(emb)

        self._fitted = True

    def embed(self, text: str) -> List[float]:
        """Embed a single text."""
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        return self.markov.text_embedding(text)

    def similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity between two texts."""
        v1 = self.embed(text1)
        v2 = self.embed(text2)
        return cosine_similarity(v1, v2)

    def most_similar(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar documents to a query."""
        if not self._fitted:
            return []
        query_vec = self.embed(query)
        similarities = []
        for i, emb in enumerate(self.embeddings):
            sim = cosine_similarity(query_vec, emb)
            label = self.labels[i] if i < len(self.labels) else f"doc_{i}"
            similarities.append((label, sim, i))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [(label, sim) for label, sim, _ in similarities[:top_k]]

    def cluster(self, k: int = 3, max_iter: int = 50) -> Dict:
        """K-means clustering on embeddings."""
        if not self._fitted or not self.embeddings:
            return {'clusters': [], 'centroids': []}
        import random
        n = len(self.embeddings)
        dim = len(self.embeddings[0])

        # Initialize centroids (k-means++)
        centroids = [self.embeddings[random.randint(0, n-1)][:]]
        for _ in range(1, min(k, n)):
            dists = []
            for emb in self.embeddings:
                min_d = min(
                    sum((a-b)**2 for a, b in zip(emb, c))
                    for c in centroids
                )
                dists.append(min_d)
            total = sum(dists)
            if total == 0:
                centroids.append(self.embeddings[random.randint(0, n-1)][:])
                continue
            r = random.random() * total
            cumsum = 0
            for i, d in enumerate(dists):
                cumsum += d
                if cumsum >= r:
                    centroids.append(self.embeddings[i][:])
                    break

        assignments = [0] * n
        for _ in range(max_iter):
            changed = False
            for i, emb in enumerate(self.embeddings):
                dists = [sum((a-b)**2 for a, b in zip(emb, c)) for c in centroids]
                new_cluster = min(range(len(dists)), key=lambda x: dists[x])
                if new_cluster != assignments[i]:
                    assignments[i] = new_cluster
                    changed = True
            if not changed:
                break
            for c_idx in range(len(centroids)):
                members = [self.embeddings[i] for i in range(n) if assignments[i] == c_idx]
                if members:
                    centroids[c_idx] = [
                        sum(m[d] for m in members) / len(members)
                        for d in range(dim)
                    ]

        clusters = {}
        for i, c in enumerate(assignments):
            if c not in clusters:
                clusters[c] = []
            label = self.labels[i] if i < len(self.labels) else f"doc_{i}"
            clusters[c].append(label)

        return {
            'assignments': assignments,
            'clusters': clusters,
            'centroids': centroids,
            'k': len(centroids),
        }


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
