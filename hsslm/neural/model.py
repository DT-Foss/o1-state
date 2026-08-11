"""
HSSLM - Hierarchical State-Space Language Module.

Full model combining:
    - HierarchicalEmbedding: token + position embeddings
    - StateSpaceCore: stack of 6 MambaBlocks (selective SSM)
    - HierarchicalComposer: multi-level linguistic composition
    - LMHead: next-token prediction (weight-tied)
    - AuxiliaryHeads: training supervision at multiple levels

Total parameters: ~7.3M
Complexity: O(n) in sequence length
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any

from .config import HSSLMConfig
from .embedding import HierarchicalEmbedding
from .core_engine import StateSpaceCore
from .hierarchy import HierarchicalComposer
from .lm_head import LMHead, AuxiliaryHeads


class HSSLM(nn.Module):
    """Hierarchical State-Space Language Module.

    A minimal non-transformer language model (~7.3M params) with explicit
    hierarchical linguistic processing across 8 linguistic levels:

    Phoneme/Grapheme -> Syllable -> Morpheme -> Word -> Phrase ->
        Sentence -> Utterance -> Discourse

    Architecture:
        Input tokens -> Embedding -> [SSM x6] -> Hierarchical Composer -> LM Head
                                                           |
                                                    Auxiliary Heads (training)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Args:
            config: Configuration dict. See HSSLMConfig for defaults.
        """
        super().__init__()

        # Handle both dict and HSSLMConfig
        if config is None:
            config = HSSLMConfig()
        if isinstance(config, HSSLMConfig):
            config = config.to_dict()

        self.config = config

        # Extract key dimensions
        self.vocab_size = config.get("vocab_size", 16384)
        self.d_model = config.get("d_model", 256)
        self.n_layers = config.get("n_layers", 6)
        self.d_state = config.get("d_state", 16)
        self.d_conv = config.get("d_conv", 4)
        self.expand = config.get("expand", 2)
        self.dt_rank = config.get("dt_rank", 8)
        self.max_seq_len = config.get("max_seq_len", 2048)
        self.dropout = config.get("dropout", 0.1)
        self.hierarchical = config.get("hierarchical", True)
        self.aux_loss_weight = config.get("aux_loss_weight", 0.1)

        # 1. Embedding module
        self.embedding = HierarchicalEmbedding(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            padding_idx=0,
        )

        # 2. Core SSM engine
        self.core = StateSpaceCore(
            n_layers=self.n_layers,
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            dt_rank=self.dt_rank,
            dropout=self.dropout,
        )

        # 3. Hierarchical composer
        self.composer = HierarchicalComposer(
            d_model=self.d_model,
            enabled=self.hierarchical,
            dropout=self.dropout,
        )

        # 4. LM head (weight-tied to embeddings to save ~4.2M params)
        self.lm_head = LMHead(
            d_model=self.d_model,
            vocab_size=self.vocab_size,
            embedding_weight=self.embedding.token_embedding.weight,
        )

        # 5. Auxiliary heads (training only)
        self.aux_heads = AuxiliaryHeads(d_model=self.d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize remaining weights."""
        # Xavier init for all Linear layers not already initialized
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if not hasattr(module, '_custom_init'):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        boundaries: Optional[Dict[str, List[torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        return_hierarchy: bool = False,
        states: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: (B, L) token IDs.
            boundaries: Optional word/sentence boundaries from tokenizer.
            labels: Optional (B, L) target IDs for loss.
            return_hierarchy: Return hierarchical representations.
            states: Optional per-layer states for recurrent inference.

        Returns:
            Dict with:
                - "logits": (B, L, vocab_size) next-token logits
                - "loss": scalar (if labels)
                - "hierarchy": hierarchical representations (if requested)
                - "states": final layer states (for recurrent generation)
                - "aux": auxiliary predictions (if hierarchical)
        """
        # 1. Embed
        x = self.embedding(input_ids)  # (B, L, D)

        # 2. SSM core
        hidden, new_states = self.core(x, states)  # (B, L, D)

        # 3. Hierarchical composition
        result = {"states": new_states}

        if self.hierarchical and boundaries is not None:
            hierarchy = self.composer(hidden, boundaries)
            result["hierarchy"] = hierarchy if return_hierarchy else None

            # 4a. LM head on token-level states
            logits = self.lm_head(hidden)  # (B, L, V)
            result["logits"] = logits

            # 4b. Auxiliary predictions
            aux_preds = self.aux_heads(hierarchy)
            result["aux"] = aux_preds

            # 5. Loss computation
            if labels is not None:
                result["loss"] = self._compute_loss(logits, labels, aux_preds)
        else:
            # Flat mode: just LM head
            logits = self.lm_head(hidden)
            result["logits"] = logits

            if labels is not None:
                result["loss"] = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    labels.view(-1),
                    ignore_index=0,  # pad
                )

        return result

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        aux_preds: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute total loss: LM + auxiliary hierarchical losses.

        L_total = L_lm + 0.1 * (L_pos + L_phrase + L_sentence + L_discourse)
        """
        # Primary: next-token cross-entropy
        lm_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            ignore_index=0,  # pad
        )

        # Auxiliary losses (only if we have predictions)
        aux_loss_total = torch.tensor(0.0, device=logits.device)
        weights = {"pos": 0.1, "phrase": 0.05, "sentence": 0.05, "coherence": 0.02}

        # POS loss (sample from logits shape)
        if "pos_logits" in aux_preds:
            B, W, _ = aux_preds["pos_logits"].shape
            # Create dummy targets for now (in practice these come from labeled data)
            pos_targets = torch.randint(0, 17, (B * W,), device=logits.device)
            pos_loss = F.cross_entropy(
                aux_preds["pos_logits"].view(-1, 17), pos_targets
            )
            aux_loss_total = aux_loss_total + weights["pos"] * pos_loss

        # Phrase boundary loss
        if "phrase_boundary_logits" in aux_preds:
            B, P, _ = aux_preds["phrase_boundary_logits"].shape
            phrase_targets = torch.randint(0, 2, (B * P,), device=logits.device)
            phrase_loss = F.cross_entropy(
                aux_preds["phrase_boundary_logits"].view(-1, 2),
                phrase_targets,
            )
            aux_loss_total = aux_loss_total + weights["phrase"] * phrase_loss

        # Sentence relation loss
        if "sentence_relation_logits" in aux_preds:
            B, S, _ = aux_preds["sentence_relation_logits"].shape
            sent_targets = torch.randint(0, 8, (B * S,), device=logits.device)
            sent_loss = F.cross_entropy(
                aux_preds["sentence_relation_logits"].view(-1, 8), sent_targets
            )
            aux_loss_total = aux_loss_total + weights["sentence"] * sent_loss

        # Coherence loss
        if "coherence_scores" in aux_preds:
            coh_labels = torch.randn_like(aux_preds["coherence_scores"])
            coh_loss = F.mse_loss(aux_preds["coherence_scores"], coh_labels)
            aux_loss_total = aux_loss_total + weights["coherence"] * coh_loss

        return lm_loss + self.aux_loss_weight * aux_loss_total

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Auto-regressive text generation with recurrent state.

        Uses the SSM's recurrent mode for O(1) per-step memory.

        Args:
            input_ids: (B, L) prompt token IDs.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Sample from top-k tokens only.
            top_p: Nucleus sampling threshold.
            eos_token_id: Stop generation on this token.

        Returns:
            (B, L + generated) generated token IDs.
        """
        self.eval()
        device = input_ids.device
        B = input_ids.shape[0]

        # Initialize states
        states = self.core.init_states(B, device)

        # Process prompt through model to get initial states
        # Feed tokens one at a time to build state
        for t in range(input_ids.shape[1]):
            tok = input_ids[:, t:t + 1]  # (B, 1)
            outputs = self.forward(tok, states=states)
            states = outputs["states"]

        # Generate new tokens
        generated = []
        current_id = input_ids[:, -1:]

        for _ in range(max_new_tokens):
            # Forward pass with state
            outputs = self.forward(current_id, states=states)
            logits = outputs["logits"][:, -1, :]  # (B, vocab_size)
            states = outputs["states"]

            # Apply temperature
            logits = logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None and top_p > 0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                # Remove tokens with cumulative prob above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                for b in range(B):
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    logits[b, indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated.append(next_token)
            current_id = next_token

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        if generated:
            generated_ids = torch.cat(generated, dim=1)
            return torch.cat([input_ids, generated_ids], dim=1)
        return input_ids

    @torch.no_grad()
    def analyze(
        self,
        text: str,
        tokenizer: "HierarchicalTokenizer",
    ) -> Dict[str, Any]:
        """Run hierarchical linguistic analysis on input text.

        Returns representations at all linguistic levels.

        Args:
            text: Input text string.
            tokenizer: Tokenizer instance.

        Returns:
            Dict with:
                - "tokens": List of token strings.
                - "words": Word-level analysis.
                - "phrases": Phrase-level analysis.
                - "sentences": Sentence-level analysis.
                - "discourse": Discourse-level vector.
                - "representations": Raw tensors at each level.
        """
        self.eval()

        # Encode
        encoded = tokenizer.encode(text)
        input_ids = encoded["input_ids"].unsqueeze(0)  # (1, L)
        boundaries = {
            "word_boundaries": [encoded["word_boundaries"]],
            "sentence_boundaries": [encoded["sentence_boundaries"]],
        }

        # Forward
        outputs = self.forward(
            input_ids,
            boundaries=boundaries,
            return_hierarchy=True,
        )

        hierarchy = outputs.get("hierarchy", {})

        # Decode tokens
        tokens = tokenizer.decode(input_ids[0], skip_special=True)

        # Build analysis result
        analysis = {
            "tokens": tokens,
            "representations": {
                "token": hierarchy.get("token", torch.zeros(1, 1)).squeeze(0),
            },
        }

        # Word level
        if "word" in hierarchy:
            word_reps = hierarchy["word"][0]  # (W, D)
            analysis["representations"]["word"] = word_reps
            analysis["words"] = {
                "count": word_reps.shape[0],
                "vector_dim": word_reps.shape[-1],
            }

        # Phrase level
        if "phrase" in hierarchy:
            phrase_reps = hierarchy["phrase"][0]
            analysis["representations"]["phrase"] = phrase_reps
            analysis["phrases"] = {
                "count": phrase_reps.shape[0],
            }

        # Sentence level
        if "sentence" in hierarchy:
            sent_reps = hierarchy["sentence"][0]
            analysis["representations"]["sentence"] = sent_reps
            analysis["sentences"] = {
                "count": sent_reps.shape[0],
            }

        # Discourse level
        if "discourse" in hierarchy:
            disc_reps = hierarchy["discourse"][0]
            analysis["representations"]["discourse"] = disc_reps
            analysis["discourse"] = {
                "vector": disc_reps[-1] if disc_reps.dim() > 1 else disc_reps,
            }

        return analysis

    def get_parameter_count(self) -> Dict[str, int]:
        """Return parameter count breakdown by component."""
        return {
            "embedding": sum(p.numel() for p in self.embedding.parameters()),
            "core": sum(p.numel() for p in self.core.parameters()),
            "composer": sum(p.numel() for p in self.composer.parameters()),
            "lm_head": sum(p.numel() for p in self.lm_head.parameters()),
            "aux_heads": sum(p.numel() for p in self.aux_heads.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }

    def print_parameter_summary(self) -> None:
        """Print a formatted parameter count summary."""
        counts = self.get_parameter_count()
        total = counts["total"]

        print("=" * 50)
        print(f"HSSLM Parameter Summary (~{total / 1e6:.1f}M total)")
        print("=" * 50)
        for name, count in counts.items():
            if name != "total":
                pct = count / total * 100
                print(f"  {name:20s}: {count:>10,} ({pct:5.1f}%)")
        print("-" * 50)
        print(f"  {'TOTAL':20s}: {total:>10,}")
        print("=" * 50)
