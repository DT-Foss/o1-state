"""
Extracted LM Head — Token Prediction from Qwen3-1.7B
=====================================================
The LM Head is the final layer of a Transformer LM:
  hidden_state (2048d) → RMSNorm → lm_head (151936 logits) → softmax → probabilities

We extracted this from Qwen3-1.7B. With it we can:
  1. Given ANY 2048d vector, predict which tokens are most likely
  2. Score how "natural" a word sequence is (perplexity)
  3. Generate text by iteratively picking the most probable next token

This is NOT running the full Transformer — we skip the 28 attention layers.
We feed in:
  - Raw embeddings (baseline)
  - Reservoir output states (projected back to 2048d)
  - Attention-weighted pooled embeddings

The LM head was trained on the same multi-TB corpus as the embeddings.
It knows: embedding → word probability.
"""

import numpy as np
import os
import json


def _rms_norm(x, weight, eps=1e-6):
    """RMSNorm: x * weight / sqrt(mean(x²) + eps)."""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight


class ExtractedLMHead:
    """Frozen Qwen3-1.7B language model head for token prediction."""

    def __init__(self, data_dir=None):
        if data_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base, 'data')

        self._data_dir = data_dir
        self._loaded = False
        self.lm_head = None       # (151936, 2048)
        self.final_norm = None    # (2048,)
        self._id2token = None

    def _load(self):
        if self._loaded:
            return
        head_path = os.path.join(self._data_dir, 'qwen3_lm_head.npy')
        norm_path = os.path.join(self._data_dir, 'qwen3_final_norm.npy')
        vocab_path = os.path.join(self._data_dir, 'qwen3_1.7b_vocab.json')

        if not os.path.exists(head_path):
            return

        self.lm_head = np.load(head_path).astype(np.float32)    # (151936, 2048)
        self.final_norm = np.load(norm_path).astype(np.float32)  # (2048,)

        with open(vocab_path) as f:
            t2id = json.load(f)
        self._id2token = {v: k for k, v in t2id.items()}
        self._token2id = t2id

        self._loaded = True

    @property
    def available(self):
        self._load()
        return self.lm_head is not None

    def predict_tokens(self, hidden_state, top_k=10, temperature=1.0):
        """Predict most likely tokens from a 2048d hidden state.

        Args:
            hidden_state: (2048,) vector
            top_k: number of top predictions
            temperature: >1 = more diverse, <1 = more focused

        Returns:
            list of (token_string, probability) tuples
        """
        self._load()
        if self.lm_head is None:
            return []

        # Final RMSNorm
        normed = _rms_norm(hidden_state.astype(np.float32), self.final_norm)

        # Logits: (151936,)
        logits = self.lm_head @ normed

        # Temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k selection (avoid softmax over 151K)
        top_idx = np.argpartition(logits, -top_k)[-top_k:]
        top_logits = logits[top_idx]

        # Softmax over top-k only
        top_logits = top_logits - top_logits.max()
        probs = np.exp(top_logits)
        probs /= probs.sum()

        # Sort by probability
        order = np.argsort(probs)[::-1]

        results = []
        for idx in order:
            token_id = int(top_idx[idx])
            token_str = self._id2token.get(token_id, f'<{token_id}>')
            # Clean token string
            clean = token_str.replace('Ġ', ' ').replace('Ċ', '\n')
            results.append((clean.strip(), float(probs[idx])))

        return results

    def score_sequence(self, embeddings, raw_emb_matrix):
        """Score how natural a token sequence is.

        Uses the LM head to check if each token is predicted by
        the previous tokens' average embedding.

        Lower score = more natural/predictable.

        Args:
            embeddings: list of token IDs
            raw_emb_matrix: (vocab_size, 2048) raw embeddings

        Returns:
            float: average negative log probability (like perplexity)
        """
        self._load()
        if self.lm_head is None or len(embeddings) < 2:
            return float('inf')

        total_nll = 0.0
        for i in range(1, len(embeddings)):
            # Context: average of previous embeddings
            context = np.mean(raw_emb_matrix[embeddings[:i]], axis=0)
            normed = _rms_norm(context.astype(np.float32), self.final_norm)

            # Logits for the actual next token
            logit_next = float(self.lm_head[embeddings[i]] @ normed)

            # Approximate log-softmax (log_softmax ≈ logit - logsumexp)
            # Use top-1000 for approximate normalization
            all_logits = self.lm_head @ normed
            top1k = np.partition(all_logits, -1000)[-1000:]
            lse = np.log(np.exp(top1k - top1k.max()).sum()) + top1k.max()

            nll = -(logit_next - lse)
            total_nll += nll

        return total_nll / (len(embeddings) - 1)

    def generate_next(self, context_embedding, temperature=0.7):
        """Generate the single most likely next token.

        Args:
            context_embedding: (2048,) context vector
            temperature: sampling temperature

        Returns:
            (token_string, probability) or (None, 0)
        """
        preds = self.predict_tokens(context_embedding, top_k=5, temperature=temperature)
        if not preds:
            return None, 0.0

        # Filter out special tokens and punctuation-only
        for token, prob in preds:
            if token and len(token) > 0 and any(c.isalpha() for c in token):
                return token, prob

        return preds[0] if preds else (None, 0.0)
