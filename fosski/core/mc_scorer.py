"""
Multiple-Choice Scorer — Qwen3 Perplexity + Reservoir Ensemble
===============================================================
Primary: Qwen3-1.7B perplexity scoring (actual LM capability)
Secondary: Reservoir state coherence + embedding similarity

Uses MLX for fast on-device inference.
"""

import numpy as np
from typing import List, Optional


class MultipleChoiceScorer:

    def __init__(self, pipeline, use_qwen=True):
        self.p = pipeline
        self._qwen_model = None
        self._qwen_tokenizer = None
        self._use_qwen = use_qwen

    def _load_qwen(self):
        if self._qwen_model is not None:
            return True
        if not self._use_qwen:
            return False
        try:
            import mlx.core as mx
            from mlx_lm import load
            self._qwen_model, self._qwen_tokenizer = load("Qwen/Qwen3-1.7B")
            self._mx = mx
            return True
        except Exception:
            return False

    def predict(self, context: str, options: List[str]) -> int:
        scores = self.score_options(context, options)
        return int(np.argmax(scores))

    def score_options(self, context: str, options: List[str]) -> np.ndarray:
        n = len(options)
        if n == 0:
            return np.array([])

        scores = np.zeros(n, dtype=np.float64)

        # Primary: Qwen3 perplexity (dominant)
        qwen = self._qwen_ppl(context, options)
        if qwen is not None:
            scores += 10.0 * qwen

        # Reservoir: slight signal above random
        res = self._reservoir_score(context, options)
        if res is not None:
            scores += 1.0 * res

        return scores

    def _qwen_ppl(self, context: str, options: List[str]) -> Optional[np.ndarray]:
        """Score options by Qwen3-1.7B perplexity.

        For each option: tokenize context+option, run forward pass,
        compute average log-prob of option tokens only.
        """
        if not self._load_qwen():
            return None

        mx = self._mx
        model = self._qwen_model
        tokenizer = self._qwen_tokenizer

        # Tokenize context once
        ctx_ids = tokenizer.encode(context)
        ctx_len = len(ctx_ids)

        scores = np.zeros(len(options), dtype=np.float64)

        for i, opt in enumerate(options):
            # Full sequence: context + option
            full_text = context + " " + opt
            full_ids = tokenizer.encode(full_text)

            if len(full_ids) <= ctx_len:
                continue

            # Forward pass
            x = mx.array([full_ids])
            logits = model(x)  # (1, seq_len, vocab_size)

            # Score only option tokens (after context)
            # logits[t] predicts token[t+1]
            total_logp = 0.0
            n_scored = 0
            for t in range(max(ctx_len - 1, 0), len(full_ids) - 1):
                token_logits = logits[0, t, :]  # (vocab_size,)
                target_id = full_ids[t + 1]

                # Log-softmax
                max_logit = mx.max(token_logits)
                shifted = token_logits - max_logit
                log_sum_exp = mx.log(mx.sum(mx.exp(shifted)))
                logp = float(shifted[target_id] - log_sum_exp)

                total_logp += logp
                n_scored += 1

            scores[i] = total_logp / max(n_scored, 1)

        # Normalize to [0, 1]
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-10:
            scores = (scores - s_min) / (s_max - s_min)

        return scores

    def _reservoir_score(self, context: str, options: List[str]) -> Optional[np.ndarray]:
        """Reservoir sequential prediction cosine."""
        if (self.p.reservoir is None or self.p.emb_store is None
                or self.p.reservoir.W_out is None):
            return None
        if self.p.raw_emb is None or not hasattr(self.p, '_token2id'):
            return None

        from .tokenutils import find_token_id

        zero = np.zeros(self.p.emb_store.dim, dtype=np.float32)
        ctx_words = context.lower().split()
        scores = np.zeros(len(options), dtype=np.float64)

        for i, opt in enumerate(options):
            self.p.reservoir.reset_state()
            for w in ctx_words:
                w_c = w.strip('.,!?;:\'"()-')
                if not w_c:
                    continue
                v = self.p.emb_store.encode(w_c)
                self.p.reservoir.step(v if v is not None else zero)

            opt_words = opt.lower().split()
            total_sim = 0.0
            n_scored = 0
            for w in opt_words:
                w_c = w.strip('.,!?;:\'"()-')
                if not w_c:
                    continue
                tid = find_token_id(w_c, self.p._token2id)
                state = (self.p.reservoir.state_pos + self.p.reservoir.state_neg) / 2.0
                pred = self.p.reservoir.predict(state)
                if pred is not None and tid is not None:
                    actual = self.p.raw_emb[tid].astype(np.float32)
                    p_n = pred / (np.linalg.norm(pred) + 1e-8)
                    a_n = actual / (np.linalg.norm(actual) + 1e-8)
                    total_sim += float(np.dot(p_n, a_n))
                    n_scored += 1
                v = self.p.emb_store.encode(w_c)
                self.p.reservoir.step(v if v is not None else zero)

            scores[i] = total_sim / max(n_scored, 1)

        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-10:
            scores = (scores - s_min) / (s_max - s_min)
        return scores

    def _embedding_score(self, context: str, options: List[str]) -> Optional[np.ndarray]:
        """Cosine similarity context ↔ option."""
        if self.p.emb_store is None:
            return None

        STOP = self.p.STOP_WORDS
        def tok(text):
            clean = text.lower()
            for ch in '?.,!;:"\'-()[]{}':
                clean = clean.replace(ch, ' ')
            return [w for w in clean.split() if w not in STOP and len(w) > 1]

        ctx_emb = self.p._query_embedding(tok(context))
        if ctx_emb is None:
            return None
        ctx_norm = ctx_emb / (np.linalg.norm(ctx_emb) + 1e-8)

        scores = np.zeros(len(options), dtype=np.float64)
        for i, opt in enumerate(options):
            opt_emb = self.p._query_embedding(tok(opt))
            if opt_emb is not None:
                opt_norm = opt_emb / (np.linalg.norm(opt_emb) + 1e-8)
                scores[i] = float(np.dot(ctx_norm, opt_norm))

        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-10:
            scores = (scores - s_min) / (s_max - s_min)
        return scores
