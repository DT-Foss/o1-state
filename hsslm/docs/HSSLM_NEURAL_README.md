# HSSLM - Hierarchical State-Space Language Module

A minimal non-transformer language model with explicit hierarchical linguistic processing, based on selective state space models (Mamba-style S6).

**~8.6M parameters | O(n) complexity | Pure PyTorch | No CUDA kernels required**

---

## Architecture

```
Input Tokens -> Embedding -> [MambaBlock x6] -> Hierarchical Composer -> LM Head
                                                   |                        (tied)
                                            Auxiliary Heads
```

### Key Features

- **Selective SSM Core**: 6 Mamba-style blocks with input-dependent state transitions (B, C, Delta are functions of input)
- **Hierarchical Composition**: 4 learned composers building representations at word, phrase, sentence, and discourse levels
- **Weight Tying**: Input/output embeddings shared, saving 4.2M parameters
- **Pure PyTorch**: No custom CUDA, no mamba_ssm dependency - every line is inspectable
- **Modular Design**: Hierarchical composers can be disabled; model works as flat LM

### Linguistic Hierarchy Mapping

The 8 linguistic layers from the source report map to architecture components:

| Linguistic Layer | Component | Type |
|---|---|---|
| Phoneme/Grapheme | Subword token embedding (BPE) | Learned |
| Syllable | Conv1D local feature extractor | Learned |
| Morpheme | Subword boundary detection | Learned auxiliary |
| Word | **WordComposer** - mean-pool + MLP | Learned |
| Phrase | **PhraseComposer** - local attention + MLP | Learned |
| Sentence | **SentenceComposer** - max-pool + MLP | Learned |
| Utterance/Turn | Running state per turn | Learned |
| Discourse | **DiscourseComposer** - gated recurrence | Learned |

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Demo

```bash
# Run full demo (model creation, forward pass, generation, analysis, benchmark)
python scripts/demo.py
```

### Training

```bash
# Train on text files in ./data
python scripts/train.py --data_dir ./data --output_dir ./checkpoints

# Or with synthetic data for testing
python scripts/train.py --max_steps 1000 --batch_size 4 --seq_length 128
```

### Inference

```bash
# Interactive mode
python scripts/inference.py --checkpoint checkpoints/final_model.pt --interactive

# Single prompt
python scripts/inference.py --checkpoint checkpoints/final_model.pt \
    --prompt "The quick brown fox"

# Hierarchical analysis
python scripts/inference.py --checkpoint checkpoints/final_model.pt \
    --analyze "The cat sat on the mat. It was sunny."
```

### Python API

```python
from hsslm import HSSLM, HSSLMConfig
from hsslm.tokenizer import HierarchicalTokenizer

# Create model (~8.6M params)
config = HSSLMConfig()
model = HSSLM(config.to_dict())
model.print_parameter_summary()

# Tokenize
tokenizer = HierarchicalTokenizer()
encoded = tokenizer.encode("Hello world", add_bos=True)

# Forward pass
import torch
outputs = model(
    encoded["input_ids"].unsqueeze(0),
    boundaries={
        "word_boundaries": [encoded["word_boundaries"]],
        "sentence_boundaries": [encoded["sentence_boundaries"]],
    },
    labels=encoded["input_ids"].unsqueeze(0),  # for loss
)
print(f"Loss: {outputs['loss'].item():.4f}")

# Generate
output_ids = model.generate(
    encoded["input_ids"].unsqueeze(0),
    max_new_tokens=50,
    temperature=0.8,
)
print(tokenizer.decode(output_ids[0]))

# Hierarchical analysis
analysis = model.analyze("The cat sat on the mat.", tokenizer)
for level, tensor in analysis["representations"].items():
    print(f"{level}: {tensor.shape}")
```

---

## Parameter Breakdown

| Component | Parameters | Percentage |
|---|---|---|
| Embedding (token + position) | 4,718,848 | 54.7% |
| SSM Core (6 layers) | 2,579,200 | 29.9% |
| Hierarchical Composer (4 levels) | 1,184,257 | 13.7% |
| Auxiliary Heads | 138,524 | 1.6% |
| LM Head (weight-tied) | 0 | 0.0% |
| **TOTAL** | **8,620,829** | **100%** |

**Under 10M parameter ceiling with 14% headroom.**

---

## File Structure

```
output/
  ARCHITECTURE_DECISIONS.md   # Full architecture justification
  SPEC.md                     # Detailed technical specification
  README.md                   # This file
  requirements.txt            # Python dependencies
  hsslm/                      # Main package
    __init__.py               # Package exports
    config.py                 # HSSLMConfig dataclass
    tokenizer.py              # HierarchicalTokenizer
    embedding.py              # HierarchicalEmbedding + RMSNorm
    core_engine.py            # SelectiveSSM, MambaBlock, StateSpaceCore
    hierarchy.py              # HierarchicalComposer + 4 composers
    lm_head.py                # LMHead + AuxiliaryHeads
    model.py                  # Full HSSLM model
  scripts/
    train.py                  # Training script
    inference.py              # Inference CLI + interactive mode
    demo.py                   # Quick demonstration
  tests/
    (test files)
```

---

## Configuration

Key hyperparameters (see `hsslm/config.py` for full list):

| Parameter | Default | Description |
|---|---|---|
| `d_model` | 256 | Model dimension |
| `n_layers` | 6 | Number of SSM layers |
| `d_state` | 16 | SSM state dimension |
| `expand` | 2 | Expansion factor |
| `vocab_size` | 16384 | Vocabulary size |
| `hierarchical` | True | Enable hierarchical composition |
| `learning_rate` | 6e-4 | Learning rate |
| `grad_clip` | 1.0 | Gradient clipping (CRITICAL for SSM) |

---

## Training Notes

### SSM Stability (Important!)

SSMs can be sensitive to training dynamics:

1. **Always use gradient clipping** (`grad_clip=1.0`) - without this, SSMs will NaN
2. **Use bf16 or fp32** - fp16 can cause numerical instability
3. **Initialize Delta bias to small positive values** (done automatically)
4. **Use warmup** (2000 steps) to prevent early instability

### Loss Function

The total loss combines next-token prediction with auxiliary hierarchical losses:

```
L_total = L_lm + 0.1*(L_pos + L_phrase_boundary + L_sentence_relation + L_coherence)
```

Auxiliary losses guide the model to learn meaningful representations at each linguistic level.

### Training Data

- Minimum: 100M tokens (basic coherence)
- Recommended: 2-3B tokens (Chinchilla-optimal for 8.6M params)
- Format: Plain text files (.txt, .md) in a directory

---

## Performance

### Speed (CPU, batch=1)

| Sequence Length | Generation Time (20 tokens) | Tokens/sec |
|---|---|---|
| 64 | ~1.0s | ~20 |
| 256 | ~3.2s | ~6 |
| 512 | ~6.0s | ~3 |

Note: This is a pure PyTorch implementation. With CUDA-optimized kernels
(selective_scan from mamba_ssm), expect 5-10x speedup on GPU.

### Memory

- Model: ~35 MB (float32)
- Inference state: ~100 KB (constant, doesn't grow with sequence length)
- Training (batch=32, seq=2048): ~2-3 GB

---

## Design Decisions

See [ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md) for:
- Why SSM over RNN/CNN/Transformer
- Exact dimension justifications
- Risk assessment and mitigations
- Fallback architectures

See [SPEC.md](SPEC.md) for:
- Complete module interfaces
- Data flow diagrams
- Training/inference protocols
- Performance targets

---

## License

MIT License - Free for research and commercial use.
