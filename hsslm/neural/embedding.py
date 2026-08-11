"""Embedding module with positional encoding."""

import math
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Used instead of LayerNorm for better training stability with SSMs.
    LN subtracts mean then divides by std; RMSNorm only divides by RMS.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: (*, dim) -> (*, dim)"""
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class HierarchicalEmbedding(nn.Module):
    """Token embedding layer with learned positional encoding.

    Simple but effective: token embeddings + position embeddings + norm + dropout.
    The BPE tokenizer naturally provides subword units that map to morpheme-like
    structure, so we don't need separate character-level decomposition.
    """

    def __init__(
        self,
        vocab_size: int = 16384,
        d_model: int = 256,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding: (16384, 256) = 4,194,304 params
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx
        )

        # Positional embedding: (2048, 256) = 524,288 params
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Normalization and dropout
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embeddings with small random values."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        # Ensure padding token has zero embedding
        with torch.no_grad():
            self.token_embedding.weight[0].fill_(0)

    def forward(self, input_ids: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) token IDs.
            position_offset: Absolute position of input_ids[:, 0] in the full
                sequence. Defaults to 0 (one-shot / first chunk). A streaming
                caller processing chunk k of chunk_size C must pass
                position_offset=k*C so position embeddings continue correctly
                across chunk boundaries instead of restarting at 0 every call
                (restarting at 0 is a real bug for chunked/streaming
                inference: the SSM recurrent state carries real history, but
                the position embedding would otherwise silently claim every
                chunk starts the sequence).

        Returns:
            (B, L, 256) contextualized embeddings.
        """
        B, L = input_ids.shape

        # Token embeddings
        tok_emb = self.token_embedding(input_ids)  # (B, L, 256)

        # Position embeddings, offset for chunked/streaming continuation
        positions = torch.arange(
            position_offset, position_offset + L, device=input_ids.device
        ).unsqueeze(0)  # (1, L)
        positions = positions.clamp(max=self.max_seq_len - 1)
        pos_emb = self.position_embedding(positions)  # (1, L, 256)

        # Combine and normalize
        x = tok_emb + pos_emb
        x = self.norm(x)
        x = self.dropout(x)

        return x

    def get_embedding_table(self) -> torch.Tensor:
        """Return the token embedding weight matrix for weight tying."""
        return self.token_embedding.weight
