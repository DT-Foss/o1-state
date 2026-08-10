"""
HellaSwag Solver — Additive Fiber-Fusion for Commonsense Completion
====================================================================
Given a context (activity description + partial sentence),
select the most plausible continuation from 4 choices.

Architecture (post-MI redesign):
  Multiple scoring fibers run in parallel, results ADDED (≈ Residual Stream).
  Hopfield-Sequence-Memory provides Induction Head functionality.
  Per-Token Hopfield-Retrieval provides QKV-Attention over knowledge.
  FLM/PPM provides character-level language statistics.
  All fiber outputs are summed — no selection, no gating.

Uses GloVe embeddings (100d) for semantic matching.
Uses Hopfield Sequence Memory for continuation prediction.
Uses Per-Token Retriever for knowledge-informed scoring.
"""

import re
import numpy as np
from typing import List
from collections import Counter

from .consensus import FossConsensus


# Stop words
_STOP = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'and', 'but', 'or', 'if',
    'he', 'she', 'it', 'they', 'we', 'you', 'me', 'him', 'her', 'us',
    'them', 'his', 'its', 'my', 'your', 'our', 'their', 'this', 'that',
    'these', 'those', 'what', 'which', 'who', 'whom',
    'just', 'also', 'even', 'back', 'still', 'around', 'about',
    'gets', 'goes', 'makes', 'takes', 'comes', 'puts', 'says',
    'starts', 'begins', 'continues', 'keeps',
}


def _content_words(text: str) -> set:
    return set(w for w in re.findall(r'\b[a-z]+\b', text.lower())
               if w not in _STOP and len(w) > 2)


def _bigrams(text: str) -> set:
    words = [w for w in re.findall(r'\b[a-z]+\b', text.lower()) if len(w) > 1]
    return {words[i] + ' ' + words[i+1] for i in range(len(words) - 1)}


def _trigrams(text: str) -> set:
    words = [w for w in re.findall(r'\b[a-z]+\b', text.lower()) if len(w) > 1]
    return {words[i] + ' ' + words[i+1] + ' ' + words[i+2]
            for i in range(len(words) - 2)}


def _softmax(scores, temperature=1.0):
    s = np.array(scores, dtype=float)
    s = s / max(temperature, 0.01)
    s = s - s.max()
    e = np.exp(s)
    total = e.sum()
    if total == 0:
        return np.ones(len(scores)) / len(scores)
    return e / total


class HellaSwagSolver:
    """Solve HellaSwag with Additive Fiber-Fusion."""

    N_STRATEGIES = 5  # 4 original + PPM

    # ConceptNet class-level cache (lazy load)
    _cn_fwd = None
    _cn_rev = None

    _pmi_index = None  # Class-level PMI cache

    def __init__(self, glove_path: str = None, lm_model=None,
                 sequence_memory=None, token_retriever=None,
                 stacked_hopfield=None, conceptnet_path: str = None,
                 pmi_path: str = None, word_ppm=None, apple_nlu=None,
                 rwkv=None):
        self._word2vec = None
        self._vecs = None
        self._sent_encoder = None  # SentenceEncoder (SIF-weighted, position-aware)
        self._lm = lm_model  # PPMModel instance (optional)
        self._word_ppm = word_ppm  # WordPPM instance (optional)
        self._seq_mem = sequence_memory  # HopfieldSequenceMemory (optional)
        self._tok_ret = token_retriever  # PerTokenRetriever (optional)
        self._stacked = stacked_hopfield  # StackedHopfield (optional)
        self._nlu = apple_nlu  # AppleNLU instance (optional)
        self._rwkv = rwkv  # RWKVLanguageModel instance (optional)
        if self._nlu is None:
            try:
                from .nlu import get_nlu
                self._nlu = get_nlu()
            except Exception:
                pass
        if self._rwkv is None:
            try:
                from .rwkv_lm import load_rwkv
                self._rwkv = load_rwkv()
            except Exception:
                pass
        if glove_path:
            self._load_glove(glove_path)
        if conceptnet_path:
            self._load_conceptnet(conceptnet_path)
        if pmi_path:
            self._load_pmi(pmi_path)

        # Bottleneck graph: 2+3 barbell with bridge (PPM as 5th node)
        A = np.zeros((self.N_STRATEGIES, self.N_STRATEGIES))
        A[0, 1] = A[1, 0] = 1
        A[2, 3] = A[3, 2] = 1
        A[3, 4] = A[4, 3] = 1  # PPM connected to discrimination
        A[1, 2] = A[2, 1] = 1
        self._consensus = FossConsensus(A)

    @classmethod
    def _load_conceptnet(cls, path: str):
        """Load ConceptNet pickle index for commonsense reasoning."""
        if cls._cn_fwd is not None:
            return
        import pickle, os
        if os.path.exists(path):
            with open(path, 'rb') as f:
                cn = pickle.load(f)
            cls._cn_fwd = cn['forward']
            cls._cn_rev = cn['reverse']

    # Relation types useful for activity completion / temporal reasoning
    _HS_RELS = {
        'UsedFor': 2.0, 'CapableOf': 1.5, 'HasPrerequisite': 1.0,
        'Causes': 1.5, 'HasSubevent': 1.5, 'AtLocation': 0.8,
        'HasProperty': 1.0, 'PartOf': 0.5,
    }

    def _cn_related_words(self, word: str) -> dict:
        """Get ConceptNet-related words with weights, filtered by useful relations."""
        related = {}
        if self._cn_fwd and word in self._cn_fwd:
            for rel, target, weight in self._cn_fwd[word]:
                rel_mult = self._HS_RELS.get(rel, 0)
                if rel_mult == 0:
                    continue
                for tw in target.lower().split():
                    if tw and tw not in _STOP and len(tw) > 2:
                        related[tw] = related.get(tw, 0) + weight * rel_mult
        if self._cn_rev and word in self._cn_rev:
            for rel, source, weight in self._cn_rev[word]:
                rel_mult = self._HS_RELS.get(rel, 0)
                if rel_mult == 0:
                    continue
                for sw in source.lower().split():
                    if sw and sw not in _STOP and len(sw) > 2:
                        related[sw] = related.get(sw, 0) + weight * rel_mult * 0.3
        return related

    def _load_glove(self, base_path: str):
        import os
        words_path = base_path.replace('.txt', '_words.npy')
        vecs_path = base_path.replace('.txt', '.npy')
        if os.path.exists(words_path) and os.path.exists(vecs_path):
            words = np.load(words_path, allow_pickle=True)
            self._vecs = np.load(vecs_path)
            self._word2vec = {w: i for i, w in enumerate(words)}
            # Build SIF sentence encoder
            try:
                from .sentence_encoder import SentenceEncoder
                self._sent_encoder = SentenceEncoder(
                    self._word2vec, self._vecs, mode='sif'
                )
            except Exception:
                self._sent_encoder = None

    def _sent_vec(self, text: str) -> np.ndarray:
        # Use SIF encoder if available (position/frequency aware)
        if self._sent_encoder is not None:
            return self._sent_encoder.encode(text)
        # Fallback to bag-of-words
        if self._word2vec is None:
            return np.zeros(100)
        tokens = re.findall(r'\b[a-z]+\b', text.lower())
        indices = [self._word2vec[w] for w in tokens if w in self._word2vec]
        if indices:
            return np.mean(self._vecs[indices], axis=0)
        return np.zeros(100)

    def _nlu_vec(self, text: str) -> np.ndarray:
        """512d contextual sentence embedding via Apple NaturalLanguage."""
        if self._nlu is None:
            return None
        v = self._nlu.embed(text)
        return v

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # === Strategy 1: Full Scoring (best single strategy) ===
    def _strategy_full(self, ctx, endings, activity_label,
                       ctx_vec, act_vec, end_vecs, all_end_words):
        """Combined scoring: IDF overlap + GloVe + bigrams + anti-adversarial."""
        ctx_words = _content_words(ctx)
        activity_words = _content_words(activity_label)
        all_ctx = ctx_words | activity_words
        ctx_bg = _bigrams(ctx)
        ctx_tg = _trigrams(ctx)

        # Get last sentence for continuity scoring
        ctx_sents = [s.strip() for s in re.split(r'[.!?]+', ctx) if s.strip()]
        last_ctx_words = _content_words(ctx_sents[-1]) if ctx_sents else set()

        scores = []
        for i, ending in enumerate(endings):
            score = 0.0
            end_words = _content_words(ending)

            # 1. IDF-weighted context overlap
            for w in end_words & all_ctx:
                freq = all_end_words.get(w, 1)
                score += 4.0 / freq

            # 2. Activity label overlap (strongest signal)
            for w in end_words & activity_words:
                freq = all_end_words.get(w, 1)
                score += 10.0 / freq

            # 3. GloVe similarity
            if self._word2vec is not None:
                act_sim = self._cosine(act_vec, end_vecs[i])
                score += act_sim * 6.0
                ctx_sim = self._cosine(ctx_vec, end_vecs[i])
                score += ctx_sim * 2.0

            # 4. Bigram + trigram overlap
            end_bg = _bigrams(ending)
            end_tg = _trigrams(ending)
            score += len(end_bg & ctx_bg) * 5.0
            score += len(end_tg & ctx_tg) * 8.0

            # 5. Last-sentence → first-sentence continuity
            end_sents = [s.strip() for s in re.split(r'[.!?]+', ending) if s.strip()]
            first_end_words = _content_words(end_sents[0]) if end_sents else set()
            if last_ctx_words and first_end_words:
                overlap = len(last_ctx_words & first_end_words)
                score += overlap * 3.0

            # 6. Topic drift penalty (higher = more aggressive)
            if len(end_words) > 0:
                new_words = end_words - all_ctx
                drift = len(new_words) / len(end_words)
                score -= drift * 5.0

            # 7. Adversarial artifact detection
            periods = ending.count('.')
            if periods > 3:
                score -= (periods - 3) * 0.3
            word_list = re.findall(r'\b[a-z]+\b', ending.lower())
            if len(word_list) > 5:
                unique_ratio = len(set(word_list)) / len(word_list)
                if unique_ratio < 0.5:
                    score -= 2.0

            scores.append(score)
        return _softmax(scores, temperature=2.0)

    # === Strategy 2: GloVe Continuity ===
    def _strategy_continuity(self, ctx, endings):
        """GloVe similarity between tail of context and head of each ending."""
        if self._word2vec is None:
            return np.ones(len(endings)) / len(endings)
        ctx_tokens = [w for w in re.findall(r'\b[a-z]+\b', ctx.lower())
                      if w not in _STOP and len(w) > 2 and w in self._word2vec]
        tail = ctx_tokens[-8:] if len(ctx_tokens) >= 8 else ctx_tokens
        if not tail:
            return np.ones(len(endings)) / len(endings)
        tail_vec = np.mean(self._vecs[[self._word2vec[w] for w in tail]], axis=0)

        scores = []
        for ending in endings:
            end_tokens = [w for w in re.findall(r'\b[a-z]+\b', ending.lower())
                          if w not in _STOP and len(w) > 2 and w in self._word2vec]
            head = end_tokens[:8] if len(end_tokens) >= 8 else end_tokens
            if not head:
                scores.append(0.0)
                continue
            head_vec = np.mean(self._vecs[[self._word2vec[w] for w in head]], axis=0)
            scores.append(self._cosine(tail_vec, head_vec))
        return _softmax(scores, temperature=0.3)

    # === Strategy 3: Activity Focus ===
    def _strategy_activity(self, activity_label, endings, end_vecs, act_vec):
        """Pure activity-label matching: GloVe + word overlap."""
        activity_words = _content_words(activity_label)
        scores = []
        for i, ending in enumerate(endings):
            score = 0.0
            end_words = _content_words(ending)
            # Direct word overlap with activity
            score += len(end_words & activity_words) * 3.0
            # GloVe similarity with activity
            if self._word2vec is not None:
                score += self._cosine(act_vec, end_vecs[i]) * 5.0
            scores.append(score)
        if max(scores) == 0:
            return np.ones(len(endings)) / len(endings)
        return _softmax(scores, temperature=0.5)

    # === Strategy 4: Cross-Ending Discrimination ===
    def _strategy_discrimination(self, ctx, endings, all_end_words):
        """Reward endings with unique context-matching words."""
        ctx_words = _content_words(ctx)
        end_word_sets = [_content_words(e) for e in endings]
        scores = []
        for i, ew in enumerate(end_word_sets):
            score = 0.0
            for w in ew:
                in_others = sum(1 for j, other in enumerate(end_word_sets)
                                if j != i and w in other)
                if in_others == 0 and w in ctx_words:
                    # Unique to this ending AND matches context
                    score += 4.0
                elif in_others == 3:
                    # In all endings = no discriminative power
                    pass
                elif w in ctx_words:
                    score += 1.0 / (1 + in_others)
            scores.append(score)
        if max(scores) == 0:
            return np.ones(len(endings)) / len(endings)
        return _softmax(scores, temperature=2.0)

    # === Strategy 5b: Word-level PPM ===
    def _strategy_word_ppm(self, ctx, endings):
        """Score by word-trigram continuation probability."""
        if self._word_ppm is None:
            return np.ones(len(endings)) / len(endings)
        scores = []
        for ending in endings:
            sc = self._word_ppm.score_continuation(ctx, ending)
            scores.append(sc)  # higher = more natural
        return _softmax(scores, temperature=1.0)

    # === Strategy 5: PPM Perplexity ===
    def _strategy_ppm(self, ctx, endings):
        """Score by PPM continuation probability: only the ending, conditioned on ctx."""
        if self._lm is None:
            return np.ones(len(endings)) / len(endings)
        scores = []
        for ending in endings:
            # score_continuation scores ONLY the ending chars, conditioned on ctx
            # Lower = more natural text
            if hasattr(self._lm, 'score_continuation'):
                sc = self._lm.score_continuation(ctx, ending)
                scores.append(-sc)  # negate: lower score_continuation = better
            else:
                # Fallback to full perplexity
                ppl = self._lm.perplexity(list(ctx + ' ' + ending))
                scores.append(-ppl)
        return _softmax(scores, temperature=2.0)

    # === Strategy 6: Hopfield Sequence Memory (Induction Head) ===
    def _strategy_sequence_memory(self, ctx, endings):
        """Score endings by overlap with Hopfield-retrieved continuation words."""
        if self._seq_mem is None or self._seq_mem.n_sequences == 0:
            return np.ones(len(endings)) / len(endings)

        # Extract context content words
        ctx_words = [w for w in re.findall(r'\b[a-z]+\b', ctx.lower())
                     if w not in _STOP and len(w) > 2]

        # Retrieve predicted continuation words from sequence memory
        retrieved = self._seq_mem.retrieve(ctx_words, top_k=30)
        if not retrieved:
            return np.ones(len(endings)) / len(endings)

        scores = []
        for ending in endings:
            end_words = _content_words(ending)
            score = 0.0
            for w in end_words:
                if w in retrieved:
                    score += retrieved[w]
            scores.append(score)

        if max(scores) == 0:
            return np.ones(len(endings)) / len(endings)
        return _softmax(scores, temperature=1.0)

    # === Strategy 7: ConceptNet Relation Bridging ===
    def _strategy_conceptnet(self, ctx, activity_label, endings):
        """Score endings by ConceptNet relation bridging.

        For each context content word, look up ConceptNet relations.
        Score each ending by how many of its words are connected to context words.
        Uses weighted relations: UsedFor, CapableOf, HasSubevent, Causes, etc.
        """
        if not self._cn_fwd:
            return np.ones(len(endings)) / len(endings)

        # Build ConceptNet neighborhood for context words
        ctx_all = ctx + ' ' + activity_label
        ctx_words = _content_words(ctx_all)

        # Collect all related words from context
        ctx_cn = {}
        for w in ctx_words:
            for rw, weight in self._cn_related_words(w).items():
                ctx_cn[rw] = ctx_cn.get(rw, 0) + weight

        scores = []
        for ending in endings:
            end_words = _content_words(ending)
            score = 0.0

            # Score ending words by ConceptNet bridge to context
            for w in end_words:
                if w in ctx_cn:
                    # Direct match to context's CN neighborhood
                    score += min(ctx_cn[w], 10.0) * 0.3
                # Also check ending word's CN neighbors against context
                for rw, weight in self._cn_related_words(w).items():
                    if rw in ctx_words:
                        score += min(weight, 5.0) * 0.2

            scores.append(score)

        if max(scores) == 0:
            return np.ones(len(endings)) / len(endings)
        return _softmax(scores, temperature=1.0)

    # === Strategy 8: PMI Co-occurrence ===
    @classmethod
    def _load_pmi(cls, path: str):
        """Load PMI index (lazy, cached)."""
        if cls._pmi_index is not None:
            return
        import pickle as _pkl
        import os as _os
        if _os.path.exists(path):
            with open(path, 'rb') as f:
                cls._pmi_index = _pkl.load(f)

    def _strategy_pmi(self, ctx, endings):
        """Score endings by PMI co-occurrence with context words.

        PMI (Pointwise Mutual Information) measures how much more often
        two words co-occur than expected by chance. This is an INDEPENDENT
        signal from GloVe (distributional) — it's corpus-based co-occurrence.
        """
        if self._pmi_index is None:
            return np.ones(len(endings)) / len(endings)

        ctx_words = _content_words(ctx)
        scores = []
        for ending in endings:
            end_words = _content_words(ending)
            score = 0.0
            for ew in end_words:
                for cw in ctx_words:
                    pair = tuple(sorted([ew, cw]))
                    pmi_val = self._pmi_index.get(pair, 0)
                    if pmi_val > 0:
                        score += min(pmi_val, 15.0)  # cap extreme values
            scores.append(score)

        if max(scores) == 0:
            return np.ones(len(endings)) / len(endings)
        return _softmax(scores, temperature=2.0)

    # === Strategy 9: Per-Token GloVe Attention ===
    def _strategy_token_attention(self, ctx, activity_label, endings):
        """Score endings by per-word GloVe similarity with context words.

        Each context content word acts as a Query. Each ending word is a Key.
        Similarity = dot product in GloVe space. This IS QKV attention
        over the vocabulary, using GloVe as the embedding.
        """
        if self._word2vec is None:
            return np.ones(len(endings)) / len(endings)

        # Get context content words (last 15 for recency bias)
        ctx_all = ctx + ' ' + activity_label
        ctx_tokens = [w for w in re.findall(r'\b[a-z]+\b', ctx_all.lower())
                      if w not in _STOP and len(w) > 2 and w in self._word2vec]
        if not ctx_tokens:
            return np.ones(len(endings)) / len(endings)

        # Use last 15 context words (recency matters for completion)
        ctx_tokens = ctx_tokens[-15:]
        ctx_indices = [self._word2vec[w] for w in ctx_tokens]
        ctx_vecs = self._vecs[ctx_indices]  # (n_ctx, dim)

        scores = []
        for ending in endings:
            end_tokens = [w for w in re.findall(r'\b[a-z]+\b', ending.lower())
                          if w not in _STOP and len(w) > 2 and w in self._word2vec]
            if not end_tokens:
                scores.append(0.0)
                continue

            end_indices = [self._word2vec[w] for w in end_tokens]
            end_vecs = self._vecs[end_indices]  # (n_end, dim)

            # Per-token attention: each ctx word × each ending word
            # similarity matrix: (n_ctx, n_end)
            sim_matrix = ctx_vecs @ end_vecs.T
            # Normalize by vector norms
            ctx_norms = np.linalg.norm(ctx_vecs, axis=1, keepdims=True)
            end_norms = np.linalg.norm(end_vecs, axis=1, keepdims=True)
            denom = ctx_norms @ end_norms.T
            denom = np.maximum(denom, 1e-10)
            sim_matrix = sim_matrix / denom

            # Score = mean of max similarity per context word
            # (each ctx word "attends to" its best-matching ending word)
            max_per_ctx = sim_matrix.max(axis=1)  # best match for each ctx word
            scores.append(float(max_per_ctx.mean()))

        if max(scores) == 0:
            return np.ones(len(endings)) / len(endings)
        return _softmax(scores, temperature=0.2)

    def _strategy_rwkv(self, ctx, endings):
        """Score endings by RWKV-7 continuation probability.

        RWKV is a pure RNN (not a transformer). It gives real next-token
        probabilities conditioned on context — exactly what HellaSwag needs.
        Lower score_continuation = more natural text.
        """
        scores = []
        for ending in endings:
            nll = self._rwkv.score_continuation(ctx, ending)
            scores.append(-nll)  # negate: lower NLL = better
        return _softmax(scores, temperature=1.0)

    def _strategy_apple_nlu(self, ctx, activity_label, endings):
        """Score endings using Apple NLU 512d contextual embeddings.

        Contextual embeddings capture sentence structure, not just bag-of-words.
        Three signals: context-continuation similarity, activity match,
        and tail-head continuity (last sentence → first sentence of ending).
        """
        ctx_full = activity_label + ' ' + ctx if activity_label else ctx
        ctx_vec = self._nlu_vec(ctx_full)
        if ctx_vec is None:
            return np.ones(len(endings)) / len(endings)

        # Also get tail-of-context vector for continuity
        ctx_sents = [s.strip() for s in re.split(r'[.!?]+', ctx) if s.strip()]
        tail_text = ctx_sents[-1] if ctx_sents else ctx
        tail_vec = self._nlu_vec(tail_text)

        scores = []
        for ending in endings:
            # Full context → ending similarity
            end_vec = self._nlu_vec(ending)
            if end_vec is None:
                scores.append(0.0)
                continue

            sim_full = float(np.dot(ctx_vec, end_vec))

            # Tail → head continuity
            sim_tail = 0.0
            if tail_vec is not None:
                end_sents = [s.strip() for s in re.split(r'[.!?]+', ending) if s.strip()]
                head_text = end_sents[0] if end_sents else ending
                head_vec = self._nlu_vec(head_text)
                if head_vec is not None:
                    sim_tail = float(np.dot(tail_vec, head_vec))

            scores.append(sim_full * 0.6 + sim_tail * 0.4)

        return _softmax(scores, temperature=0.15)

    def solve(self, ctx: str, endings: List[str],
              activity_label: str = '') -> int:
        """Select the best ending using Additive Fiber-Fusion.

        All scoring fibers run in parallel. Results are ADDED (≈ Residual Stream).
        No gating, no selection — every fiber contributes to the final belief.

        If FLM/PPM is available, its score is ADDED to the ensemble
        rather than used standalone (additive fusion, not replacement).
        """
        n_choices = len(endings)

        # Pre-compute shared representations
        ctx_vec = self._sent_vec(ctx)
        act_vec = self._sent_vec(activity_label)
        end_vecs = [self._sent_vec(e) for e in endings]

        all_end_words = Counter()
        for e in endings:
            for w in _content_words(e):
                all_end_words[w] += 1

        # === Additive Fiber-Fusion with entropy confidence ===
        # Each fiber produces a belief distribution. Confident fibers
        # (low entropy) contribute more — same math as Transformer
        # residual stream where strong activations dominate.
        max_entropy = np.log(n_choices)
        combined = np.zeros(n_choices)

        def _add_fiber(belief, base_weight=1.0):
            """Add a fiber's belief weighted by its confidence (1 - normalized entropy)."""
            ent = -np.sum(belief * np.log(belief + 1e-10))
            confidence = max(0.01, 1.0 - ent / max_entropy)
            combined[:] += confidence * base_weight * belief

        # Fiber 1: Full scoring (word overlap + GloVe + bigrams + anti-adversarial)
        f1 = self._strategy_full(ctx, endings, activity_label,
                                 ctx_vec, act_vec, end_vecs, all_end_words)
        _add_fiber(f1, 1.0)

        # Fiber 2: GloVe Continuity (tail-of-context → head-of-ending)
        f2 = self._strategy_continuity(ctx, endings)
        _add_fiber(f2, 1.0)

        # Fiber 3: Activity Focus
        f3 = self._strategy_activity(activity_label, endings, end_vecs, act_vec)
        _add_fiber(f3, 1.0)

        # Fiber 4: Cross-Ending Discrimination
        f4 = self._strategy_discrimination(ctx, endings, all_end_words)
        _add_fiber(f4, 1.0)

        # Fiber 5: RWKV-7 continuation scoring (the real language model)
        # Pure RNN, not a transformer. Gives actual next-token probabilities.
        if self._rwkv is not None:
            f_rwkv = self._strategy_rwkv(ctx, endings)
            _add_fiber(f_rwkv, 5.0)

        # Fiber 5b: FLM/PPM continuation scoring — DISABLED
        # Character-level PPM sees too little context for HellaSwag.
        if self._lm is not None:
            f5 = self._strategy_ppm(ctx, endings)
            _add_fiber(f5, 0.0)

        # Fiber 5b: Word-level PPM continuation scoring — DISABLED
        # TESTED: 32.0% on full dataset (= baseline). Word trigrams from
        # tech+wiki corpus don't help because HellaSwag's adversarial
        # filtering selects for endings that are EQUALLY fluent.
        # First 2000 show 37.5% (easier examples), but full 10K = 32%.
        # if self._word_ppm is not None:
        #     f5b = self._strategy_word_ppm(ctx, endings)
        #     _add_fiber(f5b, 2.0)

        # Fiber 6: Hopfield Sequence Memory (Induction Head)
        if self._seq_mem is not None:
            f6 = self._strategy_sequence_memory(ctx, endings)
            _add_fiber(f6, 1.5)

        # Fiber 7: ConceptNet Relation Bridging — DISABLED
        # Tested: 30.3% with CN vs 32.1% without. Adds noise, not signal.
        # ConceptNet relations are too broad for activity completion.
        # if self._cn_fwd is not None:
        #     f7cn = self._strategy_conceptnet(ctx, activity_label, endings)
        #     _add_fiber(f7cn, 1.0)

        # Fiber 7a: Stacked Hopfield hierarchical retrieval
        if self._stacked is not None and self._word2vec is not None:
            ctx_vec_sh = self._sent_vec(ctx + ' ' + activity_label)
            sh_scores = []
            for ending in endings:
                end_vec = self._sent_vec(ending)
                s = self._stacked.score_text(end_vec, ctx_vec_sh)
                sh_scores.append(s)
            # Normalize to probability
            sh_arr = np.array(sh_scores)
            sh_arr = sh_arr - sh_arr.min()
            sh_sum = sh_arr.sum()
            if sh_sum > 0:
                sh_prob = sh_arr / sh_sum
            else:
                sh_prob = np.ones(n_choices) / n_choices
            _add_fiber(sh_prob, 1.0)

        # Fiber 7b: Apple NLU — 512d contextual sentence embeddings
        # Non-GloVe signal source: contextual, multilingual, ANE-native.
        if self._nlu is not None:
            f_nlu = self._strategy_apple_nlu(ctx, activity_label, endings)
            _add_fiber(f_nlu, 3.0)

        # Fiber 8: PMI Co-occurrence — DISABLED
        # Tested: 30.3% with PMI+CN vs 32.1% without. Tech/Wiki PMI
        # doesn't match HellaSwag's everyday activity domain.
        # if self._pmi_index is not None:
        #     f_pmi = self._strategy_pmi(ctx, endings)
        #     _add_fiber(f_pmi, 1.0)

        # Fiber 9: Expert Scorer Ensemble (Mixture of Experts)
        try:
            from .expert_scorers import score_all_experts
            expert_weights = {
                'subject_verb': 0.3,
                'temporal': 0.5,     # HellaSwag = temporal completion
                'causal': 0.4,
                'negation': 0.6,
                'entity_continuity': 0.6,  # Continuation = entity tracking
                'specificity': 0.2,
                'lexical_overlap': 0.5,
                'ngram_overlap': 0.5,
                'sentiment': 0.3,
                'topic_coherence': 0.6,  # Stay on topic
                'anti_adversarial': 0.8,  # HellaSwag has adversarial distractors
                'cross_discrimination': 0.5,
            }
            full_ctx = activity_label + ' ' + ctx
            expert_scores = score_all_experts(full_ctx, endings, expert_weights)
            # Normalize to probability distribution
            e_min = min(expert_scores)
            e_range = max(expert_scores) - e_min
            if e_range > 0:
                e_norm = [(s - e_min) / e_range for s in expert_scores]
                e_sum = sum(e_norm) or 1.0
                e_prob = np.array([s / e_sum for s in e_norm])
            else:
                e_prob = np.ones(n_choices) / n_choices
            _add_fiber(e_prob, 1.5)
        except ImportError:
            pass

        return int(np.argmax(combined))


def evaluate_hellaswag(solver: HellaSwagSolver, data_path: str):
    """Evaluate solver on HellaSwag JSONL data."""
    import json

    correct = 0
    total = 0
    errors = []

    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            ctx = item.get('ctx', '')
            endings = item.get('endings', [])
            label = item.get('label', 0)
            activity = item.get('activity_label', '')

            prediction = solver.solve(ctx, endings, activity)
            total += 1

            if prediction == label:
                correct += 1
            elif len(errors) < 20:
                errors.append({
                    'ctx': ctx[:80],
                    'pred': f"{prediction}: {endings[prediction][:50]}",
                    'correct': f"{label}: {endings[label][:50]}",
                })

    return correct, total, errors
