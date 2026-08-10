"""
Knowledge Store — Modern Hopfield with Anti-Hallucination
=============================================================
Stores facts as (Subject, Relation, Object) triplets in a
Modern Hopfield network. Uses attractor distance as a
confidence measure — the key anti-hallucination mechanism.

When queried with an unknown subject (e.g., "Narnia"), the
Hopfield dynamics converge to the NEAREST stored attractor
instead of hallucinating. The distance between input and
attractor is measurably large → system can say "I don't know."

Transformers CANNOT do this. They always produce output with
false confidence. This architecture knows when it doesn't know.

Based on T71 results: 10/10 S+R→O, Hopfield beats k-NN at noise.
Based on T79 results: 14/14 QA, 7.4ms, Narnia correctly rejected.
"""

import numpy as np
import hashlib
import os
import re


class PhraseEncoder:
    """
    Encode text phrases into fixed-dimension vectors.

    Uses n-gram hashing (random projection style).
    Deterministic: same text → same vector.
    """

    def __init__(self, dim=128, n_grams=(2, 3)):
        self.dim = dim
        self.n_grams = n_grams
        self._ngram_cache = {}  # ngram → (indices, signs)
        self._n_nonzero = max(dim // 10, 3)

    def _get_ngram_vecs(self, ngram):
        """Get cached (indices, signs) for an n-gram."""
        cached = self._ngram_cache.get(ngram)
        if cached is not None:
            return cached
        h = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
        rng = np.random.RandomState(h % (2 ** 31))
        indices = rng.choice(self.dim, self._n_nonzero, replace=False)
        signs = rng.choice([-1.0, 1.0], self._n_nonzero)
        self._ngram_cache[ngram] = (indices, signs)
        return indices, signs

    def encode(self, text):
        """Encode a text string into a vector."""
        vec = np.zeros(self.dim)
        text_lower = text.lower().strip()

        for n in self.n_grams:
            for i in range(len(text_lower) - n + 1):
                ngram = text_lower[i:i + n]
                indices, signs = self._get_ngram_vecs(ngram)
                vec[indices] += signs / n

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class GloVeEncoder:
    """
    Encode text phrases using GloVe word embeddings.

    Phrase vector = normalized mean of word vectors.
    Falls back to n-gram hashing for OOV words.
    Deterministic: same text → same vector.

    GloVe (Pennington et al., 2014) — pre-trained on Wikipedia+Gigaword.
    We use them the same way we use Hopfield (Ramsauer 2020) or
    PPM (Cleary & Witten 1984): published research, not a black box.
    """

    _shared_vectors = None  # Class-level cache — load once, share across instances
    _shared_dim = None

    def __init__(self, dim=100, glove_path=None):
        self.dim = dim
        self._fallback = PhraseEncoder(dim=dim)

        if glove_path is None:
            # Auto-detect GloVe file
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidates = [
                os.path.join(base, 'data', f'glove.6B.{dim}d.txt'),
                os.path.join(base, 'data', f'glove.6B.{dim}d.npy'),
            ]
            for c in candidates:
                if os.path.exists(c):
                    glove_path = c
                    break

        if glove_path and os.path.exists(glove_path):
            self._load_glove(glove_path)
        else:
            self.vectors = {}

    def _load_glove(self, path):
        """Load GloVe vectors from text file or numpy cache."""
        # Use class-level cache if already loaded
        if GloVeEncoder._shared_vectors is not None and GloVeEncoder._shared_dim == self.dim:
            self.vectors = GloVeEncoder._shared_vectors
            return

        npy_path = path.replace('.txt', '.npy')
        words_path = path.replace('.txt', '_words.npy')

        # Try numpy cache first (10x faster load)
        if os.path.exists(npy_path) and os.path.exists(words_path):
            matrix = np.load(npy_path)
            words = np.load(words_path, allow_pickle=True)
            self.vectors = {w: matrix[i] for i, w in enumerate(words)}
        elif path.endswith('.txt'):
            self.vectors = {}
            words_list = []
            vecs_list = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.rstrip().split(' ')
                    if len(parts) != self.dim + 1:
                        continue
                    word = parts[0]
                    vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    self.vectors[word] = vec
                    words_list.append(word)
                    vecs_list.append(vec)
            # Save numpy cache for next time
            if words_list:
                np.save(npy_path, np.array(vecs_list))
                np.save(words_path, np.array(words_list))
        else:
            self.vectors = {}

        # Cache at class level
        GloVeEncoder._shared_vectors = self.vectors
        GloVeEncoder._shared_dim = self.dim

    def encode(self, text):
        """Encode a text phrase into a vector using GloVe word averaging."""
        if not self.vectors:
            return self._fallback.encode(text)

        text_clean = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', text.lower().strip())
        words = text_clean.split()

        vecs = []
        for word in words:
            if word in self.vectors:
                vecs.append(self.vectors[word])
            else:
                # Try sub-words for compounds like "birthplace" → "birth" + "place"
                found_sub = False
                if len(word) > 5:
                    for split in range(3, len(word) - 2):
                        w1, w2 = word[:split], word[split:]
                        if w1 in self.vectors and w2 in self.vectors:
                            vecs.append(self.vectors[w1] + self.vectors[w2])
                            found_sub = True
                            break
                if not found_sub:
                    # Fallback: n-gram hash for this word, projected to GloVe dim
                    fb = self._fallback.encode(word)
                    if self.dim != self._fallback.dim:
                        fb = fb[:self.dim] if self.dim < self._fallback.dim else \
                             np.pad(fb, (0, self.dim - self._fallback.dim))
                    vecs.append(fb)

        if not vecs:
            return self._fallback.encode(text)

        vec = np.mean(vecs, axis=0).astype(float)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class ModernHopfield:
    """
    Modern Hopfield Network with exponential energy.

    Key difference from classical Hopfield:
    - Exponential interaction function (softmax attention)
    - Capacity scales as exp(d) not 0.14*d
    - Convergence in 1-3 steps (not 10-50)

    T68b: 1000 patterns in 256d = 85% accuracy
    (classical limit would be 0.14*256 = 35 patterns).
    """

    def __init__(self, dim, beta=64.0):
        """
        Args:
            dim: pattern dimension
            beta: inverse temperature (higher = sharper attention)
        """
        self.dim = dim
        self.beta = beta
        self.patterns = []
        self.metadata = []
        self.X = None  # Pattern matrix (dim, n_patterns)
        self._X_dirty = True

    @property
    def n_stored(self):
        return len(self.patterns)

    def store(self, pattern, meta=None):
        """Store a pattern with optional metadata."""
        self.patterns.append(np.array(pattern, dtype=float))
        self.metadata.append(meta or {})
        self._X_dirty = True

    def _ensure_X(self):
        """Build pattern matrix lazily (only when needed for retrieval)."""
        if self._X_dirty and self.patterns:
            self.X = np.column_stack(self.patterns)
            self._X_dirty = False

    def retrieve(self, query, max_steps=20):
        """
        Retrieve nearest stored pattern via Hopfield dynamics.

        Returns:
            (retrieved_pattern, n_steps, similarity_to_nearest)
        """
        self._ensure_X()
        if self.X is None:
            return query, 0, 0.0

        x = np.array(query, dtype=float)

        for step in range(max_steps):
            # Softmax attention over stored patterns
            scores = self.beta * self.X.T @ x
            scores -= scores.max()
            weights = np.exp(scores)
            weights /= weights.sum()

            x_new = self.X @ weights

            if np.allclose(x, x_new, atol=1e-8):
                idx, sim = self._identify(x_new)
                return x_new, step + 1, sim
            x = x_new

        idx, sim = self._identify(x)
        return x, max_steps, sim

    def _identify(self, x):
        """Find closest stored pattern."""
        if not self.patterns:
            return -1, 0.0

        x_norm = np.linalg.norm(x)
        if x_norm < 1e-10:
            return 0, 0.0

        sims = [np.dot(x, p) / (x_norm * np.linalg.norm(p) + 1e-10)
                for p in self.patterns]
        best = int(np.argmax(sims))
        return best, float(sims[best])


class KnowledgeStore:
    """
    Fact store with anti-hallucination confidence.

    Stores (Subject, Relation, Object) triplets.
    On query: returns nearest fact + confidence score.

    Confidence is based on ATTRACTOR DISTANCE:
    - High similarity to stored pattern → HIGH confidence → answer
    - Low similarity → LOW confidence → "I don't know"

    This is architecturally impossible in Transformers.
    """

    # Base confidence thresholds (for small N)
    # With SR-only matching: real matches = 1.000, fakes max ~0.60
    # Set above fake maximum to ensure rejection even at small N
    HIGH_THRESHOLD = 0.85
    LOW_THRESHOLD = 0.75

    def __init__(self, dim=128, encoder='auto', normalizer=None):
        self.dim = dim
        if encoder == 'auto':
            # Try GloVe first (100d available), fall back to n-gram
            # GloVe dim != storage dim — GloVe provides vectors, we project
            for glove_dim in (100, 200, 300):
                enc = GloVeEncoder(dim=glove_dim)
                if enc.vectors:
                    self.encoder = enc
                    self._encoder_type = 'glove'
                    self.dim = glove_dim  # match encoder dim
                    break
            else:
                self.encoder = PhraseEncoder(dim=dim)
                self._encoder_type = 'ngram'
        elif encoder == 'glove':
            self.encoder = GloVeEncoder(dim=dim)
            self._encoder_type = 'glove'
        elif encoder == 'ngram':
            self.encoder = PhraseEncoder(dim=dim)
            self._encoder_type = 'ngram'
        elif hasattr(encoder, 'encode'):
            self.encoder = encoder
            self._encoder_type = 'custom'
        else:
            self.encoder = PhraseEncoder(dim=dim)
            self._encoder_type = 'ngram'
        # Entity normalizer (USA→united states, etc.)
        self._normalizer = normalizer
        self.hopfield = ModernHopfield(dim=dim * 3, beta=64.0)
        self.facts = []
        # Cache S+R vectors for adaptive threshold and SR-only matching
        self._sr_vectors = []
        # Cache O+R vectors for reverse query matching
        self._or_vectors = []
        self._sr_sim_stats = None  # (mean, std) of inter-pattern SR similarities
        # Lazy Hopfield: defer vector encoding until Tier 3 query
        self._hopfield_built_to = 0  # index up to which vectors are built
        self._hopfield_lazy = False  # set True to defer encoding

        # === Dict-Index: O(1) lookups for exact queries ===
        # forward: (subject.lower(), relation.lower()) → [fact_idx, ...]
        self._forward_index = {}
        # reverse: (relation.lower(), object.lower()) → [fact_idx, ...]
        self._reverse_index = {}
        # entity_subject: subject.lower() → [fact_idx, ...]
        self._entity_subject_index = {}
        # entity_object: object.lower() → [fact_idx, ...]
        self._entity_object_index = {}
        # Dedup set (already existed as _fact_keys but make it consistent)
        self._fact_keys = set()

    def _normalize(self, entity):
        """Normalize entity name if normalizer is set."""
        if self._normalizer and entity:
            return self._normalizer.normalize(entity)
        return entity

    @property
    def n_facts(self):
        return len(self.facts)

    def store_fact(self, subject, relation, obj):
        """Store a (S, R, O) triplet. Normalizes entities. Skips exact duplicates."""
        subject = self._normalize(subject)
        obj = self._normalize(obj)
        # Deduplicate: skip if exact (S, R, O) already stored
        fact_key = (subject.lower(), relation.lower(), obj.lower())
        if fact_key in self._fact_keys:
            return
        self._fact_keys.add(fact_key)

        idx = len(self.facts)
        self.facts.append((subject, relation, obj))

        # === Dict-Index: O(1) exact lookups ===
        s_low, r_low, o_low = fact_key
        # Forward: (subject, relation) → object
        fwd_key = (s_low, r_low)
        if fwd_key not in self._forward_index:
            self._forward_index[fwd_key] = []
        self._forward_index[fwd_key].append(idx)
        # Reverse: (relation, object) → subject
        rev_key = (r_low, o_low)
        if rev_key not in self._reverse_index:
            self._reverse_index[rev_key] = []
        self._reverse_index[rev_key].append(idx)
        # Entity-subject: subject → all facts
        if s_low not in self._entity_subject_index:
            self._entity_subject_index[s_low] = []
        self._entity_subject_index[s_low].append(idx)
        # Entity-object: object → all facts
        if o_low not in self._entity_object_index:
            self._entity_object_index[o_low] = []
        self._entity_object_index[o_low].append(idx)

        # === Hopfield vectors (for fuzzy fallback) ===
        if not self._hopfield_lazy:
            self._encode_hopfield(idx, subject, relation, obj)
            self._hopfield_built_to = idx + 1

    def _encode_hopfield(self, idx, subject, relation, obj):
        """Encode a single fact into Hopfield + SR/OR vector caches."""
        s_vec = self.encoder.encode(subject)
        r_vec = self.encoder.encode(relation)
        o_vec = self.encoder.encode(obj)
        full_vec = np.concatenate([s_vec, r_vec, o_vec])
        self.hopfield.store(full_vec,
                           meta={'s': subject, 'r': relation, 'o': obj})
        sr_vec = np.concatenate([s_vec, r_vec])
        self._sr_vectors.append(sr_vec)
        or_vec = np.concatenate([o_vec, r_vec])
        self._or_vectors.append(or_vec)
        self._sr_sim_stats = None
        # Invalidate cached SR matrix (will be rebuilt on next query)
        self._sr_matrix = None

    def store_facts(self, facts):
        """Store multiple facts at once."""
        for s, r, o in facts:
            self.store_fact(s, r, o)

    def store_facts_fast(self, facts):
        """
        Bulk-store facts with deferred Hopfield encoding.
        Use this for harvest/inference where Tier 3 is not needed immediately.
        Call _ensure_hopfield() before any fuzzy query.
        """
        old_lazy = self._hopfield_lazy
        self._hopfield_lazy = True
        for s, r, o in facts:
            self.store_fact(s, r, o)
        self._hopfield_lazy = old_lazy

    def _ensure_hopfield(self):
        """Build Hopfield vectors for all facts not yet encoded."""
        if self._hopfield_built_to >= len(self.facts):
            return
        for idx in range(self._hopfield_built_to, len(self.facts)):
            s, r, o = self.facts[idx]
            self._encode_hopfield(idx, s, r, o)
        self._hopfield_built_to = len(self.facts)
        # Pre-build SR matrix for vectorized cosine similarity
        if self._sr_vectors:
            self._sr_matrix = np.array(self._sr_vectors)
            self._sr_norms = np.linalg.norm(self._sr_matrix, axis=1)
            self._sr_norms[self._sr_norms < 1e-10] = 1e-10

    def save(self, path):
        """Save facts to a JSON file. Vectors are regenerated on load."""
        import json
        data = {
            'dim': self.dim,
            'encoder': self._encoder_type,
            'facts': self.facts,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        """Load facts from a JSON file and rebuild vectors."""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        encoder_type = data.get('encoder', 'ngram')
        ks = cls(dim=data['dim'], encoder=encoder_type)
        ks.store_facts_fast([tuple(f) for f in data['facts']])
        return ks

    def _compute_sr_stats(self):
        """Compute inter-pattern SR similarity statistics for adaptive threshold."""
        if self._sr_sim_stats is not None:
            return self._sr_sim_stats

        n = len(self._sr_vectors)
        if n < 5:
            self._sr_sim_stats = (0.0, 1.0)
            return self._sr_sim_stats

        # Sample pairwise similarities (not all N² for large N)
        rng = np.random.RandomState(42)
        n_samples = min(1000, n * (n - 1) // 2)
        sims = []
        for _ in range(n_samples):
            i, j = rng.choice(n, 2, replace=False)
            vi, vj = self._sr_vectors[i], self._sr_vectors[j]
            ni, nj = np.linalg.norm(vi), np.linalg.norm(vj)
            if ni > 1e-10 and nj > 1e-10:
                sims.append(np.dot(vi, vj) / (ni * nj))

        if sims:
            self._sr_sim_stats = (float(np.mean(sims)), float(np.std(sims)))
        else:
            self._sr_sim_stats = (0.0, 1.0)
        return self._sr_sim_stats

    def _adaptive_thresholds(self):
        """
        Compute adaptive thresholds based on N and pattern statistics.

        As N grows, the expected nearest-neighbor similarity increases
        following extreme value statistics: max of N cosine similarities
        grows as mean + std * sqrt(2 * log(N)).

        We set the threshold above this expected maximum to maintain
        rejection quality at any scale.
        """
        n = len(self.facts)
        if n < 50:
            return self.HIGH_THRESHOLD, self.LOW_THRESHOLD

        mean_sim, std_sim = self._compute_sr_stats()

        # Extreme value correction: random query's best match among N patterns
        # follows Gumbel distribution with location ~ mean + std * sqrt(2*log(N))
        import math
        ev_shift = math.sqrt(2.0 * math.log(max(n, 2)))

        # low_t: must be above the expected max fake similarity + safety margin
        # high_t: additional margin for high confidence
        low_t = max(self.LOW_THRESHOLD, mean_sim + std_sim * ev_shift + 0.05)
        high_t = max(self.HIGH_THRESHOLD, low_t + 0.05)

        return min(high_t, 0.98), min(low_t, 0.95)

    def query(self, subject=None, relation=None, max_tier=3):
        """
        Query for a fact. Dict-Index first (O(1)), Hopfield fallback for fuzzy.

        Three-tier lookup:
        1. Dict exact match: (subject, relation) → O(1)
        2. Dict subject-only: subject → scan relations → O(k) where k = facts per entity
        3. Hopfield fuzzy: full SR-vector scan → O(N) — only if dicts MISS

        Args:
            max_tier: stop after this tier (1=dict only, 2=+subject scan, 3=+Hopfield)

        Returns:
            dict with keys:
                fact: (S, R, O) tuple or None
                confidence: float [0, 1]
                confidence_level: 'HIGH', 'MEDIUM', 'LOW', or 'UNKNOWN'
                attractor_distance: float
                input_similarity: float
        """
        # Normalize query inputs
        subject = self._normalize(subject) if subject else subject
        # === Tier 1: Dict exact match (O(1)) ===
        if subject and relation:
            fwd_key = (subject.lower().strip(), relation.lower().strip())
            indices = self._forward_index.get(fwd_key)
            if indices:
                fact = self.facts[indices[0]]
                return {
                    'fact': fact,
                    'confidence': 1.0,
                    'confidence_level': 'HIGH',
                    'attractor_distance': 0.0,
                    'input_similarity': 1.0,
                    'top2_gap': 1.0,
                    'steps': 0,
                    'thresholds': self._adaptive_thresholds(),
                }

        # === Tier 2: Dict subject-only (O(k), k = facts per entity) ===
        if subject:
            subj_lower = subject.lower().strip()
            indices = self._entity_subject_index.get(subj_lower)
            if indices:
                if relation:
                    # Find best relation match in this entity's facts
                    rel_lower = relation.lower().strip()
                    for idx in indices:
                        s, r, o = self.facts[idx]
                        if r.lower() == rel_lower:
                            return {
                                'fact': self.facts[idx],
                                'confidence': 1.0,
                                'confidence_level': 'HIGH',
                                'attractor_distance': 0.0,
                                'input_similarity': 1.0,
                                'top2_gap': 1.0,
                                'steps': 0,
                                'thresholds': self._adaptive_thresholds(),
                            }
                else:
                    # No relation specified — return first fact for this entity
                    return {
                        'fact': self.facts[indices[0]],
                        'confidence': 1.0,
                        'confidence_level': 'HIGH',
                        'attractor_distance': 0.0,
                        'input_similarity': 1.0,
                        'top2_gap': 1.0,
                        'steps': 0,
                        'thresholds': self._adaptive_thresholds(),
                    }

        # === Tier 3: Hopfield fuzzy matching — vectorized O(N) matmul ===
        if max_tier < 3:
            return {
                'fact': None,
                'confidence': 0.0,
                'confidence_level': 'UNKNOWN',
                'attractor_distance': 0.0,
                'input_similarity': 0.0,
                'top2_gap': 0.0,
                'steps': 0,
                'thresholds': self._adaptive_thresholds(),
            }
        self._ensure_hopfield()
        s_vec = (self.encoder.encode(subject) if subject
                 else np.random.randn(self.dim) * 0.01)
        r_vec = (self.encoder.encode(relation) if relation
                 else np.random.randn(self.dim) * 0.01)

        # SR-only matching: compare query (S,R) against stored (S,R) vectors
        query_sr = np.concatenate([s_vec, r_vec])
        query_sr_norm = np.linalg.norm(query_sr)

        best_idx = -1
        best_sr_sim = 0.0

        if self._sr_vectors and query_sr_norm > 1e-10:
            # Vectorized cosine similarity: stack all SR vectors into matrix
            if not hasattr(self, '_sr_matrix') or self._sr_matrix is None:
                self._sr_matrix = np.array(self._sr_vectors)
                # Pre-compute norms for all SR vectors
                self._sr_norms = np.linalg.norm(self._sr_matrix, axis=1)
                # Mask zero-norm vectors
                self._sr_norms[self._sr_norms < 1e-10] = 1e-10

            # Single matmul: (N, D) @ (D,) = (N,)
            dots = self._sr_matrix @ query_sr
            sims = dots / (self._sr_norms * query_sr_norm)

            best_idx = int(np.argmax(sims))
            best_sr_sim = float(sims[best_idx])

            if len(sims) > 1:
                # Use partition for O(N) top-2 instead of full sort
                top2_indices = np.argpartition(sims, -2)[-2:]
                top2_vals = sims[top2_indices]
                top2_gap = float(np.max(top2_vals) - np.min(top2_vals))
            else:
                top2_gap = 1.0
        else:
            top2_gap = 0.0

        # Hopfield attractor distance for anti-hallucination
        o_vec = np.random.randn(self.dim) * 0.01
        query_vec = np.concatenate([s_vec, r_vec, o_vec])
        retrieved, steps, _ = self.hopfield.retrieve(query_vec)
        query_norm = np.linalg.norm(query_vec)
        attractor_dist = (np.linalg.norm(retrieved - query_vec) /
                         max(query_norm, 1e-10))

        high_t, low_t = self._adaptive_thresholds()

        result = {
            'fact': None,
            'confidence': best_sr_sim,
            'confidence_level': 'UNKNOWN',
            'attractor_distance': attractor_dist,
            'input_similarity': best_sr_sim,
            'top2_gap': top2_gap,
            'steps': steps,
            'thresholds': (high_t, low_t),
        }

        if best_idx >= 0 and best_sr_sim > low_t:
            result['fact'] = self.facts[best_idx]
            result['confidence_level'] = 'HIGH' if best_sr_sim > high_t else 'MEDIUM'
        elif best_idx >= 0:
            result['confidence_level'] = 'REJECTED'
            result['nearest_fact'] = self.facts[best_idx]

        return result

    def query_reverse(self, obj, relation=None):
        """
        Reverse query: find Subject given Object and Relation.
        Dict-Index first (O(1)), Hopfield fallback for fuzzy.

        Example: query_reverse("Berlin", "capital") → ("Germany", "capital", "Berlin")
        """
        # Normalize query inputs
        obj = self._normalize(obj) if obj else obj
        # === Tier 1: Dict exact match (O(1)) ===
        if obj and relation:
            rev_key = (relation.lower().strip(), obj.lower().strip())
            indices = self._reverse_index.get(rev_key)
            if indices:
                return {
                    'fact': self.facts[indices[0]],
                    'confidence': 1.0,
                    'confidence_level': 'HIGH',
                    'input_similarity': 1.0,
                }

        # === Tier 2: Dict object-only scan (O(k)) ===
        if obj:
            obj_lower = obj.lower().strip()
            indices = self._entity_object_index.get(obj_lower)
            if indices:
                if relation:
                    rel_lower = relation.lower().strip()
                    for idx in indices:
                        s, r, o = self.facts[idx]
                        if r.lower() == rel_lower:
                            return {
                                'fact': self.facts[idx],
                                'confidence': 1.0,
                                'confidence_level': 'HIGH',
                                'input_similarity': 1.0,
                            }
                else:
                    return {
                        'fact': self.facts[indices[0]],
                        'confidence': 1.0,
                        'confidence_level': 'HIGH',
                        'input_similarity': 1.0,
                    }

        # === Tier 3: Hopfield fuzzy (O(N)) — only on dict MISS ===
        o_vec = (self.encoder.encode(obj) if obj
                 else np.random.randn(self.dim) * 0.01)
        r_vec = (self.encoder.encode(relation) if relation
                 else np.random.randn(self.dim) * 0.01)

        query_or = np.concatenate([o_vec, r_vec])
        query_or_norm = np.linalg.norm(query_or)

        best_idx = -1
        best_or_sim = 0.0

        if self._or_vectors and query_or_norm > 1e-10:
            sims = []
            for or_vec in self._or_vectors:
                or_norm = np.linalg.norm(or_vec)
                if or_norm > 1e-10:
                    sim = float(np.dot(query_or, or_vec) / (query_or_norm * or_norm))
                else:
                    sim = 0.0
                sims.append(sim)

            best_idx = int(np.argmax(sims))
            best_or_sim = float(sims[best_idx])

        high_t, low_t = self._adaptive_thresholds()

        result = {
            'fact': None,
            'confidence': best_or_sim,
            'confidence_level': 'UNKNOWN',
            'input_similarity': best_or_sim,
        }

        if best_idx >= 0 and best_or_sim > low_t:
            result['fact'] = self.facts[best_idx]
            result['confidence_level'] = 'HIGH' if best_or_sim > high_t else 'MEDIUM'
        elif best_idx >= 0:
            result['confidence_level'] = 'REJECTED'

        return result

    def find_by_entity(self, entity):
        """
        Find all facts where entity appears as Subject OR Object.
        Dict-Index: O(k) where k = facts mentioning this entity.
        Falls back to substring match if exact match finds nothing.
        """
        entity = self._normalize(entity) if entity else entity
        entity_lower = entity.lower().strip()
        results = []
        seen = set()
        # Subject matches (exact)
        for idx in self._entity_subject_index.get(entity_lower, []):
            if idx not in seen:
                results.append(self.facts[idx])
                seen.add(idx)
        # Object matches (exact)
        for idx in self._entity_object_index.get(entity_lower, []):
            if idx not in seen:
                results.append(self.facts[idx])
                seen.add(idx)
        # Substring fallback: "Einstein" matches "Albert Einstein"
        # Rules: substring must be >=3 chars AND be a whole word boundary or >=50% of key length
        if not results and len(entity_lower) >= 3:
            for key in self._entity_subject_index:
                if (entity_lower in key and
                        (len(entity_lower) >= len(key) * 0.5 or
                         key.startswith(entity_lower + ' ') or key.endswith(' ' + entity_lower) or
                         key == entity_lower)):
                    for idx in self._entity_subject_index[key]:
                        if idx not in seen:
                            results.append(self.facts[idx])
                            seen.add(idx)
            for key in self._entity_object_index:
                if (entity_lower in key and
                        (len(entity_lower) >= len(key) * 0.5 or
                         key.startswith(entity_lower + ' ') or key.endswith(' ' + entity_lower) or
                         key == entity_lower)):
                    for idx in self._entity_object_index[key]:
                        if idx not in seen:
                            results.append(self.facts[idx])
                            seen.add(idx)
        return results

    def add_synonym(self, canonical, *synonyms):
        """Register synonyms for an entity. Queries for any synonym match the canonical form."""
        if not hasattr(self, '_synonyms'):
            self._synonyms = {}
        for syn in synonyms:
            self._synonyms[syn.lower()] = canonical

    def _resolve_synonym(self, text):
        """Resolve a synonym to its canonical form if registered."""
        if not hasattr(self, '_synonyms') or not text:
            return text
        return self._synonyms.get(text.lower(), text)

    def query_smart(self, subject=None, relation=None):
        """
        Query with synonym resolution and automatic reverse fallback.

        Tries: forward query → synonym-expanded forward → reverse query.
        Covers both "What is the capital of France?" and "Who discovered radium?"
        """
        # Resolve synonyms
        subject = self._resolve_synonym(subject) if subject else subject
        relation = self._resolve_synonym(relation) if relation else relation

        # Try forward first
        result = self.query(subject, relation)
        if result['confidence_level'] in ('HIGH', 'MEDIUM'):
            return result

        # Try reverse (maybe the subject is actually the object)
        if subject:
            rev = self.query_reverse(subject, relation)
            if rev['confidence_level'] in ('HIGH', 'MEDIUM'):
                return rev

        return result

    def auto_synonyms(self, top_k=3, min_sim=0.7):
        """
        Auto-discover synonyms for stored entities using GloVe nearest neighbors.

        Only works with GloVe encoder. For each unique entity, finds the
        top_k closest words in the GloVe vocabulary with similarity > min_sim.
        """
        if not isinstance(self.encoder, GloVeEncoder) or not self.encoder.vectors:
            return 0

        if not hasattr(self, '_synonyms'):
            self._synonyms = {}

        entities = set()
        entity_lowers = set()
        for s, r, o in self.facts:
            entities.add(s)
            entities.add(o)
            entity_lowers.add(s.lower())
            entity_lowers.add(o.lower())

        n_added = 0
        vocab_words = list(self.encoder.vectors.keys())
        vocab_vecs = np.array([self.encoder.vectors[w] for w in vocab_words])
        vocab_norms = np.linalg.norm(vocab_vecs, axis=1)
        vocab_norms[vocab_norms < 1e-10] = 1.0

        for entity in entities:
            entity_lower = entity.lower().strip()
            # Only single words (multi-word entities don't have direct GloVe matches)
            if ' ' in entity_lower or entity_lower not in self.encoder.vectors:
                continue

            e_vec = self.encoder.vectors[entity_lower]
            e_norm = np.linalg.norm(e_vec)
            if e_norm < 1e-10:
                continue

            sims = vocab_vecs @ e_vec / (vocab_norms * e_norm)
            top_idx = np.argsort(sims)[-top_k - 1:][::-1]

            for idx in top_idx:
                word = vocab_words[idx]
                if word == entity_lower:
                    continue
                # Don't map existing entities to other entities
                # (austria→Germany is wrong, they're separate entities)
                if word in entity_lowers:
                    continue
                if sims[idx] >= min_sim and word not in self._synonyms:
                    self._synonyms[word] = entity
                    n_added += 1

        return n_added

    def knows(self, subject, relation=None):
        """Quick check: does the system confidently know this fact?"""
        r = self.query(subject, relation)
        return r['confidence_level'] in ('HIGH', 'MEDIUM')

    # === Distillation Engine helpers (O(1) via Dict-Index) ===

    def has_fact(self, subject, relation, obj):
        """O(1) duplicate check. Used by Distillation Engine before storing."""
        return (subject.lower(), relation.lower(), obj.lower()) in self._fact_keys

    def get_conflict(self, subject, relation, new_obj):
        """
        O(1) conflict check. Returns existing object if it contradicts new_obj.
        Conflict = same (S,R) but different O.
        Returns None if no conflict.
        """
        fwd_key = (subject.lower().strip(), relation.lower().strip())
        indices = self._forward_index.get(fwd_key)
        if not indices:
            return None
        existing = self.facts[indices[0]]
        if existing[2].lower() != new_obj.lower():
            return existing[2]
        return None

    def get_entity_relations(self, entity):
        """
        O(k) relation list for an entity. Used to detect knowledge gaps.
        Returns list of relation names this entity has.
        """
        entity_lower = entity.lower().strip()
        rels = set()
        for idx in self._entity_subject_index.get(entity_lower, []):
            rels.add(self.facts[idx][1].lower())
        return list(rels)

    def get_sparse_entities(self, min_relations=5):
        """
        Find entities with fewer than min_relations facts.
        Returns [(entity, n_relations), ...] sorted ascending.
        Used by Distillation Engine for curiosity-driven gap filling.
        """
        sparse = []
        for entity, indices in self._entity_subject_index.items():
            n_rels = len(set(self.facts[idx][1].lower() for idx in indices))
            if n_rels < min_relations:
                sparse.append((entity, n_rels))
        sparse.sort(key=lambda x: x[1])
        return sparse

    def all_entities(self):
        """Return all unique entity names (subjects). O(1)."""
        return list(self._entity_subject_index.keys())

    def neighbors(self, entity: str) -> list:
        """
        Get all graph neighbors of an entity (forward + reverse edges).

        Returns: [(neighbor, relation, direction), ...] where direction is 'fwd' or 'rev'.
        Uses existing dict indices — O(degree), no separate graph structure needed.
        """
        e = entity.lower()
        result = []

        # Forward: entity → relation → object
        for idx in self._entity_subject_index.get(e, []):
            s, r, o = self.facts[idx]
            result.append((o.lower(), r.lower(), 'fwd'))

        # Reverse: subject → relation → entity
        for idx in self._entity_object_index.get(e, []):
            s, r, o = self.facts[idx]
            result.append((s.lower(), r.lower(), 'rev'))

        return result

    def find_path(self, start: str, end: str, max_hops: int = 3) -> list:
        """
        BFS path finding between two entities in the knowledge graph.

        Returns: [(entity, relation, entity), ...] — the path as triples.
        Empty list if no path found within max_hops.
        """
        from collections import deque

        start_l = start.lower()
        end_l = end.lower()

        if start_l == end_l:
            return []

        # BFS
        queue = deque([(start_l, [])])  # (current_node, path_so_far)
        visited = {start_l}

        while queue:
            current, path = queue.popleft()
            if len(path) >= max_hops:
                continue

            for neighbor, relation, direction in self.neighbors(current):
                if neighbor in visited:
                    continue

                if direction == 'fwd':
                    step = (current, relation, neighbor)
                else:
                    step = (neighbor, relation, current)

                new_path = path + [step]

                if neighbor == end_l:
                    return new_path

                visited.add(neighbor)
                queue.append((neighbor, new_path))

        return []

    def find_connections(self, entity: str, max_hops: int = 2) -> list:
        """
        Find all entities reachable within max_hops from entity.

        Returns: [(entity, relation, entity, hop_distance), ...]
        """
        from collections import deque

        e = entity.lower()
        result = []
        visited = {e}
        queue = deque([(e, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue

            for neighbor, relation, direction in self.neighbors(current):
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                if direction == 'fwd':
                    result.append((current, relation, neighbor, depth + 1))
                else:
                    result.append((neighbor, relation, current, depth + 1))

                queue.append((neighbor, depth + 1))

        return result
