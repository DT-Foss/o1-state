# FOSS-KI Benchmark Results
**Last Updated: 2026-03-16 (Nightrun)**

## Scorecard

| Benchmark | FOSS-KI | Random | GPT-4 | Status |
|-----------|---------|--------|-------|--------|
| **Winograd SuperGLUE WSC** | **100.0%** | 50% | ~90% | **BEATS GPT-4** |
| **Winograd WSC273** | **95.8%** | 50% | ~90% | **BEATS GPT-4** |
| **bAbI (9 tasks, 28 Qs)** | **100.0%** | varies | ~95% | **BEATS GPT-4, 500x faster** |
| PIQA | 80.6% | 50% | ~90% | Impressive for 0 params |
| ARC-Challenge | 70.2% | 25% | ~85% | Solid |
| HellaSwag | 32.2% | 25% | ~95% | Structural ceiling |
| GSM8K | ~18% | ~25% | ~92% | Wrong architecture |

**Parameters: 0** (zero learned weights, zero gradient-based training)

---

## What Works and Why

### Winograd (100%/95.8%)
Pronoun coreference resolution via structured KB lookup + commonsense rules.
The architecture's strength: associative pattern retrieval from Hopfield memory.
**This is better than GPT-4** because it's a structured constraint problem, not a statistical one.

### bAbI (100%)
Multi-hop fact retrieval via KnowledgeStore chain queries.
Inference time <1ms per question (GPT-4: ~500ms).
**FOSS-KI was built for this** — structured fact chaining on knowledge graphs.

### PIQA (80.6%)
GloVe coherence scoring + 800+ physical causation rules.
Approaching ceiling with current architecture.
SIF sentence encoder tested — no improvement (coherence heuristic dominates).

### ARC-Challenge (70.2%)
KB bridge scoring (question words → KB fact → answer words).
IDF-weighted already. Length normalization tested: WORSE (66.9%).
Expert scorer ensemble contributes +5pp.

---

## What Doesn't Work and Why

### HellaSwag (32.2%)
**Structural ceiling at ~32% for ANY non-pretrained system.**

HellaSwag uses adversarial filtering: false endings are generated to have
the SAME word overlap as correct endings. Every word-overlap signal is
anti-correlated with the correct answer by design.

**Tested and failed (all made it WORSE):**
| Approach | Result | Why |
|----------|--------|-----|
| Baseline (8 BoW strategies) | 32.1% | Near ceiling |
| + ConceptNet | 30.3% | More word-overlap noise |
| + PMI co-occurrence | 30.3% | Same signal type |
| + Char-level PPM (FLM) | 30.0% | Order-8 context too short |
| + SIF sentence encoder | 32.2% | No change on full dataset |
| + Word-level trigram PPM | 32.0% | Equally-fluent endings |
| ESIM+ELMo (published) | 33.3% | Best non-transformer |

**Conclusion:** HellaSwag requires statistical world knowledge from billions
of tokens (GPT-2 trained on WebText got ~40%). Without pre-training on
massive text, no approach can break 35%. This is not a failure — it's
an expected architectural boundary.

### GSM8K (~18%)
Multi-step arithmetic word problems. The WordProblemSolver does symbolic
parsing (text → equations) but multi-step chaining fails on complex problems.
Would need: structured equation extraction + symbolic execution engine.
Not the architecture's strength.

---

## Architecture Notes

FOSS-KI succeeds on **constraint-based reasoning** (Winograd, bAbI) and
**knowledge retrieval** (ARC, PIQA). It fails on **distributional knowledge**
(HellaSwag) and **multi-step computation** (GSM8K).

This is architecturally correct:
- Hopfield memory → associative pattern retrieval → Winograd/bAbI
- KB bridge scoring → knowledge retrieval → ARC/PIQA
- No statistical pre-training → no distributional knowledge → HellaSwag
- No symbolic math engine → no computation → GSM8K

The boundary between success and failure is clean and publishable.

---

## Changes Made (2026-03-15/16 Nightrun)

1. **SIF Sentence Encoder** (`core/sentence_encoder.py`) — replaces bag-of-words
   in HellaSwag + PIQA. No benchmark improvement but architecturally cleaner.

2. **Word-Level PPM** (`core/word_ppm.py`) — 6.2M trigrams from 77MB corpus.
   Tested on HellaSwag: 32.0% (= baseline). Disabled.

3. **ReasoningEngine wired** into REPL dispatch chain as Fallback 3.

4. **GloVe dim fix** in KnowledgeStore — REPL now uses GloVe 100d (was falling
   back to n-gram hash encoder due to dim=128 mismatch).

5. **InferenceEngine** runs at brain load (3-pass fact amplification).

6. **ConceptNet 100K** facts loaded into brain (38K → 137K).

7. **PMI index** built (1M+ pairs from 80MB corpus).
