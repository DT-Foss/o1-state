"""
Token Utilities — Qwen3 Vocabulary Handling
============================================
Qwen3 uses Ġ (U+0120) as space prefix marker in its BPE vocabulary.
"Einstein" exists only as "ĠEinstein", "France" as both "France" and "ĠFrance".

This module provides:
  - clean_token(): strip Ġ prefix for display
  - find_token_id(): robust lookup trying all prefix/case variants
  - find_token_embedding(): get embedding for any word form
"""

import numpy as np
from typing import Optional, Tuple, Dict


def clean_token(token: str) -> str:
    """Strip Qwen3 BPE markers for display.

    Ġ (U+0120) = space prefix in GPT/Qwen BPE tokenizers.
    ĉ = tab marker.
    Ċ = newline marker.
    """
    return token.replace('Ġ', '').replace('ĉ', '').replace('Ċ', '').strip()


def find_token_id(word: str, token2id: Dict[str, int]) -> Optional[int]:
    """Find token ID trying all Qwen3 prefix/case variants.

    Search order (first match wins):
      1. Exact match: "France"
      2. With Ġ prefix: "ĠFrance"
      3. Lowercase: "france"
      4. Ġ + lowercase: "Ġfrance"
      5. Capitalized: "France"
      6. Ġ + capitalized: "ĠFrance"
      7. Uppercase: "FRANCE"
      8. Ġ + uppercase: "ĠFRANCE"

    Returns token ID or None.
    """
    candidates = [
        word,
        f'Ġ{word}',
        word.lower(),
        f'Ġ{word.lower()}',
        word.capitalize(),
        f'Ġ{word.capitalize()}',
        word.upper(),
        f'Ġ{word.upper()}',
    ]
    for c in candidates:
        if c in token2id:
            return token2id[c]
    return None


def find_token_embedding(word: str, token2id: Dict[str, int],
                         embeddings: np.ndarray) -> Optional[np.ndarray]:
    """Get embedding vector for a word, trying all Qwen3 variants.

    Returns (dim,) float32 array or None.
    """
    tid = find_token_id(word, token2id)
    if tid is not None:
        return embeddings[tid].astype(np.float32)
    return None


def clean_token_list(tokens):
    """Clean a list of (token, score) tuples for display.

    Input: [("ĠParis", 0.87), ("ĠFrance", 0.65), ...]
    Output: [("Paris", 0.87), ("France", 0.65), ...]
    """
    return [(clean_token(t), s) for t, s in tokens]
