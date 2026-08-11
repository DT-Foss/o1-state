"""
Hopfield Language Memory — Transformer Mechanisms Without Transformers
=====================================================================
Three mechanisms inspired by Mechanistic Interpretability findings,
implemented on Hopfield + GloVe infrastructure:

1. Hopfield-Sequence-Memory (≈ Induction Heads)
   Store word sequences (3-5 tokens) as Hopfield patterns.
   When context matches beginning of a sequence, retrieve the continuation.
   Classical Induction Head: "A B ... A → predict B" over arbitrary distance.
   Our version: Hopfield content-addressable memory over stored sequences.

2. Per-Token Hopfield-Retrieval (≈ QKV Attention over Knowledge)
   Every content word in the input triggers a Hopfield lookup.
   Retrieved relations inform which continuation is more plausible.
   "kitchen" → retrieves (kitchen, contains, knife), (kitchen, used_for, cooking).

3. Additive Fiber-Fusion (≈ Residual Stream)
   All scoring strategies run in parallel. Results are ADDED, not selected.
   KB-Score + FLM-Score + Sequence-Memory-Score + Per-Token-Retrieval-Score.
   Each contributes to the final belief, like Transformer residual stream layers.

Uses GloVe embeddings for encoding. No gradients, no training beyond
Hopfield Hebbian storage.
"""

import re
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional


class HopfieldSequenceMemory:
    """
    Induction Head on Hopfield basis.

    Stores sequences of word vectors. Given a partial sequence (query),
    retrieves the most similar stored sequence and returns the continuation.

    Storage: sequences of 3-5 content words, encoded as concatenated GloVe vectors.
    Retrieval: Modern Hopfield (softmax attention) over stored prefix vectors.
    Output: continuation words and their GloVe vectors.
    """

    def __init__(self, word2vec: dict, vecs: np.ndarray, window: int = 4):
        """
        Args:
            word2vec: word → index mapping (GloVe vocabulary)
            vecs: numpy array of GloVe vectors (vocab_size, dim)
            window: sequence window size (3-5)
        """
        self._w2v = word2vec
        self._vecs = vecs
        self._dim = vecs.shape[1] if vecs is not None else 100
        self._window = window

        # Stored sequences: prefix_vec → continuation info
        self._prefix_vecs = []  # list of prefix vectors (window-1 words averaged)
        self._continuations = []  # list of continuation word sets
        self._full_seqs = []  # original word sequences for debugging

    def train_from_text(self, text: str, min_word_len: int = 3):
        """
        Extract word sequences from training text and store them.

        Extracts sliding windows of content words. For each window:
        - prefix = first (window-1) words → averaged GloVe vector
        - continuation = last word(s) → stored as word set with vectors
        """
        # Tokenize to content words
        words = [w for w in re.findall(r'\b[a-z]+\b', text.lower())
                 if len(w) >= min_word_len and w in self._w2v]

        if len(words) < self._window:
            return

        # Sliding window
        for i in range(len(words) - self._window + 1):
            seq = words[i:i + self._window]

            # Prefix = first (window-1) words
            prefix_words = seq[:self._window - 1]
            prefix_indices = [self._w2v[w] for w in prefix_words]
            prefix_vec = np.mean(self._vecs[prefix_indices], axis=0)

            # Continuation = remaining words
            cont_words = set(seq[self._window - 1:])

            self._prefix_vecs.append(prefix_vec)
            self._continuations.append(cont_words)
            self._full_seqs.append(seq)

        # Build matrix for fast retrieval
        if self._prefix_vecs:
            self._prefix_matrix = np.array(self._prefix_vecs)
            # Normalize rows
            norms = np.linalg.norm(self._prefix_matrix, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            self._prefix_matrix_normed = self._prefix_matrix / norms

    def train_from_sequences(self, word_sequences: List[List[str]]):
        """Store pre-extracted word sequences."""
        for seq in word_sequences:
            valid = [w for w in seq if w in self._w2v]
            if len(valid) < 2:
                continue

            prefix_words = valid[:-1]
            prefix_indices = [self._w2v[w] for w in prefix_words]
            prefix_vec = np.mean(self._vecs[prefix_indices], axis=0)

            cont_words = set(valid[-1:])

            self._prefix_vecs.append(prefix_vec)
            self._continuations.append(cont_words)
            self._full_seqs.append(valid)

        if self._prefix_vecs:
            self._prefix_matrix = np.array(self._prefix_vecs)
            norms = np.linalg.norm(self._prefix_matrix, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            self._prefix_matrix_normed = self._prefix_matrix / norms

    def retrieve(self, context_words: List[str], top_k: int = 20,
                 beta: float = 8.0) -> Dict[str, float]:
        """
        Given context words, retrieve likely continuation words.

        Uses Modern Hopfield dynamics (softmax attention):
        scores = softmax(β * prefix_matrix @ query)
        result = weighted sum of continuations

        Args:
            context_words: list of content words from context
            top_k: return top-k continuation words
            beta: inverse temperature (higher = sharper)

        Returns:
            dict of word → score (higher = more likely continuation)
        """
        if not hasattr(self, '_prefix_matrix_normed') or len(self._prefix_vecs) == 0:
            return {}

        # Build query from last (window-1) content words
        valid = [w for w in context_words if w in self._w2v]
        if not valid:
            return {}

        # Use last few words as query
        query_words = valid[-(self._window - 1):]
        query_indices = [self._w2v[w] for w in query_words]
        query_vec = np.mean(self._vecs[query_indices], axis=0)
        query_norm = np.linalg.norm(query_vec)
        if query_norm < 1e-10:
            return {}
        query_normed = query_vec / query_norm

        # Softmax attention over stored prefixes
        scores = beta * self._prefix_matrix_normed @ query_normed
        scores -= scores.max()
        weights = np.exp(scores)
        weights /= weights.sum()

        # Aggregate continuation words weighted by attention
        word_scores = defaultdict(float)
        for i, w in enumerate(weights):
            if w < 1e-6:
                continue
            for word in self._continuations[i]:
                word_scores[word] += w

        # Return top-k
        sorted_words = sorted(word_scores.items(), key=lambda x: -x[1])
        return dict(sorted_words[:top_k])

    @property
    def n_sequences(self):
        return len(self._prefix_vecs)


class PerTokenRetriever:
    """
    QKV-Attention over Knowledge Base — on every token.

    For each content word in the input, perform a knowledge lookup.
    Returns associated words/concepts that can boost continuation scoring.

    Query = word vector
    Keys = stored relation subjects/objects
    Values = related words

    Example:
        "kitchen" → retrieves: knife, cooking, stove, food
        "hospital" → retrieves: doctor, patient, medicine, sick
    """

    def __init__(self, word2vec: dict, vecs: np.ndarray):
        self._w2v = word2vec
        self._vecs = vecs
        self._dim = vecs.shape[1] if vecs is not None else 100

        # Knowledge entries: (key_word, value_words, key_vec)
        self._keys = []  # key word strings
        self._key_vecs = []  # key vectors
        self._values = []  # lists of associated words

    def add_relation(self, subject: str, relation: str, obj: str):
        """Store a (subject, relation, object) triple.

        Creates bidirectional lookups:
        subject → {object words}  (content words only, no relation labels)
        object → {subject words}
        """
        subj_lower = subject.lower()
        obj_lower = obj.lower()
        # Extract content words from multi-word entries
        subj_words = set(subj_lower.split()) - {'a', 'an', 'the', 'is', 'of', 'for', 'to'}
        obj_words = set(obj_lower.split()) - {'a', 'an', 'the', 'is', 'of', 'for', 'to'}

        if subj_lower in self._w2v:
            self._keys.append(subj_lower)
            self._key_vecs.append(self._vecs[self._w2v[subj_lower]])
            self._values.append(obj_words)

        if obj_lower in self._w2v and obj_lower != subj_lower:
            self._keys.append(obj_lower)
            self._key_vecs.append(self._vecs[self._w2v[obj_lower]])
            self._values.append(subj_words)

    def add_relations(self, relations: List[Tuple[str, str, str]]):
        """Store multiple relations."""
        for s, r, o in relations:
            self.add_relation(s, r, o)
        self._build_matrix()

    def _build_matrix(self):
        """Build key matrix for fast retrieval."""
        if self._key_vecs:
            self._key_matrix = np.array(self._key_vecs)
            norms = np.linalg.norm(self._key_matrix, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            self._key_matrix_normed = self._key_matrix / norms

    def retrieve_for_tokens(self, words: List[str], beta: float = 4.0,
                            sim_threshold: float = 0.7) -> Dict[str, float]:
        """
        For each input word, retrieve associated knowledge words.

        Returns aggregated word → score dict across all input tokens.
        """
        if not hasattr(self, '_key_matrix_normed') or len(self._key_vecs) == 0:
            return {}

        result = defaultdict(float)

        for word in words:
            if word not in self._w2v:
                continue

            query_vec = self._vecs[self._w2v[word]]
            query_norm = np.linalg.norm(query_vec)
            if query_norm < 1e-10:
                continue
            query_normed = query_vec / query_norm

            # Cosine similarity with all keys
            sims = self._key_matrix_normed @ query_normed

            # Only use entries above threshold
            for i, sim in enumerate(sims):
                if sim >= sim_threshold:
                    for value_word in self._values[i]:
                        result[value_word] += sim

        return dict(result)


class AdditiveScorer:
    """
    Residual Stream — additive fusion of all scoring fibers.

    Instead of selecting ONE strategy or doing weighted consensus,
    all strategies run in parallel and their scores are ADDED.

    Each strategy returns a probability distribution over choices.
    The sum (un-normalized) is the final belief.

    This is simpler than PS-Lifted Consensus and matches what
    Transformers actually do with their residual stream.
    """

    def __init__(self):
        self._strategies = []
        self._weights = []

    def add_strategy(self, name: str, weight: float = 1.0):
        """Register a scoring strategy with a weight."""
        self._strategies.append(name)
        self._weights.append(weight)

    def fuse(self, score_arrays: List[np.ndarray]) -> np.ndarray:
        """
        Additive fusion: sum all score arrays with weights.

        Args:
            score_arrays: list of np.ndarray, each shape (n_choices,)
                         Each should be a probability distribution or raw scores.

        Returns:
            Combined scores (un-normalized sum).
        """
        if not score_arrays:
            return np.array([])

        n_choices = len(score_arrays[0])
        combined = np.zeros(n_choices)

        for i, scores in enumerate(score_arrays):
            weight = self._weights[i] if i < len(self._weights) else 1.0
            combined += weight * np.array(scores)

        return combined


# ── Commonsense Relations for Per-Token Retrieval ──
# Format: (subject, relation, object)
# These are the "Values" that get retrieved when a matching "Key" word appears.
COMMONSENSE_RELATIONS = [
    # Kitchen/cooking
    ("kitchen", "contains", "knife"),
    ("kitchen", "contains", "stove"),
    ("kitchen", "used_for", "cooking"),
    ("kitchen", "contains", "pan"),
    ("kitchen", "contains", "oven"),
    ("stove", "used_for", "heating"),
    ("oven", "used_for", "baking"),
    ("knife", "used_for", "cutting"),
    ("pan", "used_for", "frying"),
    ("pot", "used_for", "boiling"),

    # Bathroom
    ("bathroom", "contains", "shower"),
    ("bathroom", "contains", "sink"),
    ("bathroom", "contains", "mirror"),
    ("shower", "used_for", "washing"),
    ("soap", "used_for", "cleaning"),
    ("towel", "used_for", "drying"),
    ("toothbrush", "used_for", "brushing"),

    # Living room / house
    ("house", "contains", "room"),
    ("room", "contains", "furniture"),
    ("chair", "used_for", "sitting"),
    ("table", "used_for", "eating"),
    ("bed", "used_for", "sleeping"),
    ("lamp", "used_for", "lighting"),
    ("window", "allows", "light"),
    ("door", "allows", "entry"),

    # Outdoor / nature
    ("garden", "contains", "plants"),
    ("garden", "contains", "flowers"),
    ("tree", "has_part", "branches"),
    ("tree", "has_part", "leaves"),
    ("rain", "causes", "wet"),
    ("sun", "causes", "warm"),
    ("wind", "causes", "cold"),
    ("snow", "has_property", "cold"),
    ("ice", "has_property", "slippery"),

    # Body
    ("hand", "used_for", "grabbing"),
    ("hand", "used_for", "holding"),
    ("foot", "used_for", "walking"),
    ("eye", "used_for", "seeing"),
    ("ear", "used_for", "hearing"),
    ("mouth", "used_for", "eating"),
    ("nose", "used_for", "smelling"),

    # Sports / activities
    ("ball", "used_for", "playing"),
    ("bicycle", "used_for", "riding"),
    ("swimming", "requires", "water"),
    ("running", "requires", "legs"),
    ("exercise", "causes", "sweat"),

    # Animals
    ("dog", "can", "bark"),
    ("dog", "can", "fetch"),
    ("cat", "can", "climb"),
    ("bird", "can", "fly"),
    ("fish", "lives_in", "water"),
    ("horse", "can", "gallop"),

    # Vehicles
    ("car", "used_for", "driving"),
    ("car", "needs", "fuel"),
    ("car", "has_part", "wheels"),
    ("car", "has_part", "engine"),
    ("bicycle", "has_part", "pedals"),
    ("boat", "used_for", "sailing"),

    # Tools
    ("hammer", "used_for", "hitting"),
    ("screwdriver", "used_for", "turning"),
    ("saw", "used_for", "cutting"),
    ("drill", "used_for", "boring"),
    ("wrench", "used_for", "tightening"),
    ("pliers", "used_for", "gripping"),

    # Food
    ("bread", "made_from", "flour"),
    ("cake", "made_from", "flour"),
    ("salad", "contains", "vegetables"),
    ("soup", "contains", "broth"),
    ("pizza", "contains", "cheese"),
    ("sandwich", "contains", "bread"),

    # Clothing
    ("shirt", "worn_on", "torso"),
    ("shoes", "worn_on", "feet"),
    ("hat", "worn_on", "head"),
    ("gloves", "worn_on", "hands"),
    ("jacket", "provides", "warmth"),

    # Social
    ("school", "used_for", "learning"),
    ("hospital", "used_for", "healing"),
    ("doctor", "treats", "patients"),
    ("teacher", "teaches", "students"),
    ("store", "used_for", "shopping"),
    ("library", "contains", "books"),

    # Materials / properties
    ("glass", "has_property", "fragile"),
    ("glass", "has_property", "transparent"),
    ("metal", "has_property", "strong"),
    ("wood", "has_property", "burnable"),
    ("plastic", "has_property", "flexible"),
    ("rubber", "has_property", "stretchy"),
    ("paper", "has_property", "thin"),
    ("water", "has_property", "liquid"),
    ("oil", "has_property", "slippery"),
    ("ice", "has_property", "cold"),

    # Cleaning
    ("vacuum", "used_for", "cleaning"),
    ("broom", "used_for", "sweeping"),
    ("mop", "used_for", "mopping"),
    ("sponge", "used_for", "scrubbing"),
    ("detergent", "used_for", "washing"),
    ("bleach", "used_for", "whitening"),

    # Electronics
    ("computer", "used_for", "computing"),
    ("phone", "used_for", "calling"),
    ("television", "used_for", "watching"),
    ("camera", "used_for", "photographing"),
    ("microphone", "used_for", "recording"),

    # Music
    ("guitar", "used_for", "playing"),
    ("piano", "used_for", "playing"),
    ("drum", "used_for", "beating"),
    ("violin", "used_for", "playing"),

    # Time concepts
    ("morning", "associated_with", "breakfast"),
    ("evening", "associated_with", "dinner"),
    ("night", "associated_with", "sleeping"),
    ("winter", "has_property", "cold"),
    ("summer", "has_property", "hot"),

    # Causation chains
    ("fire", "causes", "heat"),
    ("fire", "causes", "smoke"),
    ("heat", "causes", "melting"),
    ("cold", "causes", "freezing"),
    ("gravity", "causes", "falling"),
    ("force", "causes", "motion"),
    ("friction", "causes", "heat"),
    ("pressure", "causes", "compression"),
]


def build_sequence_memory(word2vec: dict, vecs: np.ndarray,
                          text_path: str = None,
                          text: str = None,
                          window: int = 4) -> HopfieldSequenceMemory:
    """Build a Hopfield Sequence Memory from text data."""
    mem = HopfieldSequenceMemory(word2vec, vecs, window=window)

    if text:
        mem.train_from_text(text)
    elif text_path:
        with open(text_path, 'r', encoding='utf-8') as f:
            content = f.read()
        mem.train_from_text(content)

    return mem


def build_token_retriever(word2vec: dict, vecs: np.ndarray,
                          extra_relations: List[Tuple[str, str, str]] = None
                          ) -> PerTokenRetriever:
    """Build a Per-Token Retriever with commonsense + optional extra relations."""
    retriever = PerTokenRetriever(word2vec, vecs)

    all_relations = list(COMMONSENSE_RELATIONS)
    if extra_relations:
        all_relations.extend(extra_relations)

    retriever.add_relations(all_relations)
    return retriever
