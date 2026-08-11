"""
RWKV Language Model Backend for FOSS-KI
========================================
RWKV-7 "Goose" — an RNN with transformer-level performance.
NOT a transformer. Linear time, constant space, no KV-cache.

Provides the same interface as FossLanguageModel so it's a drop-in
replacement in the REPL pipeline.
"""

import os
import math
import torch
import torch.nn.functional as F
import numpy as np

# RWKV env must be set BEFORE import
os.environ.setdefault('RWKV_JIT_ON', '1')
os.environ.setdefault('RWKV_V7_ON', '1')
os.environ.setdefault('RWKV_CUDA_ON', '0')

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS


class RWKVLanguageModel:
    """RWKV-7 language model wrapper matching FossLanguageModel interface.

    Architecture: Pure RNN (Receptance Weighted Key Value).
    - Linear time complexity (vs quadratic for Transformers)
    - Constant memory during inference (no KV-cache)
    - Trained like a Transformer (parallelizable)
    - 100% RNN at inference time
    """

    def __init__(self, model_path, strategy='cpu fp32'):
        self.model = RWKV(model=model_path, strategy=strategy)
        self.pipeline = PIPELINE(self.model, 'rwkv_vocab_v20230424')
        self.vocab_size = 65536
        self._state_cache = {}  # context hash → state

    def encode(self, text):
        """Encode text to token IDs."""
        return self.pipeline.encode(text)

    def decode(self, tokens):
        """Decode token IDs to text."""
        return self.pipeline.decode(tokens)

    def forward(self, tokens, state=None):
        """Run forward pass, return (logits, new_state)."""
        for t in tokens:
            logits, state = self.model.forward([t], state)
        return logits, state

    def score_continuation(self, context, continuation):
        """Score how natural a continuation is. Lower = more natural.

        Returns average negative log-probability per token.
        Compatible with FossLanguageModel.score_continuation().
        """
        ctx_tokens = self.encode(context)
        # Add space prefix if continuation doesn't start with space/punct
        if continuation and continuation[0].isalnum():
            continuation = ' ' + continuation
        cont_tokens = self.encode(continuation)

        if not cont_tokens:
            return float('inf')

        # Process context to get state
        state = None
        for t in ctx_tokens:
            logits, state = self.model.forward([t], state)

        # Score each continuation token
        total_nll = 0.0
        for t in cont_tokens:
            log_probs = F.log_softmax(logits, dim=-1)
            total_nll -= log_probs[t].item()
            logits, state = self.model.forward([t], state)

        return total_nll / len(cont_tokens)

    def log_prob(self, context, token_text):
        """Log2 probability of token given context."""
        ctx_tokens = self.encode(context)
        tok_tokens = self.encode(token_text)

        if not tok_tokens:
            return -100.0

        state = None
        for t in ctx_tokens:
            logits, state = self.model.forward([t], state)

        log_probs = F.log_softmax(logits, dim=-1)
        # Sum log probs for multi-token text
        total = 0.0
        for t in tok_tokens:
            total += log_probs[t].item()
            logits, state = self.model.forward([t], state)
            log_probs = F.log_softmax(logits, dim=-1)

        return total / math.log(2)  # convert ln → log2

    def predict_next(self, context, top_k=10):
        """Predict top-k next tokens with probabilities."""
        tokens = self.encode(context)
        state = None
        for t in tokens:
            logits, state = self.model.forward([t], state)

        probs = F.softmax(logits, dim=-1)
        topk = torch.topk(probs, top_k)

        results = []
        for i in range(top_k):
            idx = topk.indices[i].item()
            p = topk.values[i].item()
            text = self.decode([idx])
            results.append((text, p))
        return results

    def rank_continuations(self, context, candidates):
        """Rank candidate continuations by naturalness.

        Args:
            context: The preceding text
            candidates: List of candidate continuation strings

        Returns:
            List of (candidate, score) sorted best-first (lowest score)
        """
        scored = []
        for c in candidates:
            s = self.score_continuation(context, c)
            scored.append((c, s))
        scored.sort(key=lambda x: x[1])
        return scored

    def generate(self, context, max_tokens=100, temperature=1.0, top_p=0.7):
        """Generate text continuation."""
        args = PIPELINE_ARGS(
            temperature=temperature,
            top_p=top_p,
            top_k=40,
        )
        return self.pipeline.generate(
            context,
            token_count=max_tokens,
            args=args,
        )

    def perplexity(self, text):
        """Compute perplexity on text."""
        tokens = self.encode(text)
        if len(tokens) < 2:
            return float('inf')

        state = None
        total_nll = 0.0
        n = 0

        logits, state = self.model.forward([tokens[0]], state)
        for i in range(1, len(tokens)):
            log_probs = F.log_softmax(logits, dim=-1)
            total_nll -= log_probs[tokens[i]].item()
            n += 1
            logits, state = self.model.forward([tokens[i]], state)

        return math.exp(total_nll / n)


def load_rwkv(model_dir=None, size='0.4B'):
    """Load RWKV model from standard location.

    Args:
        model_dir: Directory containing .pth file. Default: data/rwkv/
        size: Model size string for filename matching

    Returns:
        RWKVLanguageModel instance or None if not found
    """
    if model_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base, 'data', 'rwkv')

    if not os.path.isdir(model_dir):
        return None

    # Find matching .pth file
    for f in os.listdir(model_dir):
        if f.endswith('.pth') and size in f:
            path = os.path.join(model_dir, f)
            # Strip .pth extension for RWKV loader
            model_path = path[:-4]
            return RWKVLanguageModel(model_path)

    return None
