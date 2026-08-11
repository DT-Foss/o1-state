# FOSS-KI Changelog

## v3.0 — Research Findings Implementation (2026-03-17)

### Task #191: Sinkhorn auf Hopfield Attention [19:30]
- **Problem**: Attention sink bias — patterns near embedding center match everything (FB2: 1049% compounding over 4 layers)
- **Fix**: Precompute per-pattern popularity bias in `_build_matrix()`, subtract before softmax
- **Theory**: 1 Sinkhorn iteration captures 86% of correction (BC11/F7)
- **Files**: `core/hopfield_bank.py` (_build_matrix, retrieve)

### Task #193: Fix CASI Gate — 1D-CASI with 7 Strategies [19:30]
- **Problem**: `casi_gate.py` used crypto-CASI (byte-level bit_correlation, xor_distribution) on quantized float similarities = measures NOTHING
- **Fix**: Complete rewrite with 7 statistical strategies: runs_test, durbin_watson, crossing_radius, adjacent_correlation, sign_change_rate, radial_trend, quartile_shift
- **Formula**: CASI(x) = Σsᵢ(x) / E[Σsᵢ(π(x))], 50 permutations for null model
- **Result**: AUC=0.988 for hallucination detection (CF248)
- **Files**: `core/casi_gate.py` (complete rewrite, ~170 LOC)

### Task #194: CASI Confidence Penalty — bits_wasted [19:30]
- **Formula**: bits_wasted = max(0, log₂(CASI(source_confidences))), penalty = bits_wasted / max_bits
- **Wired into**: `foss_pipeline.py` Step 3b after consensus, before Z₂ gating
- **Effect**: Detects systematic overconfidence across sources → reduces confidence
- **Files**: `core/casi_gate.py` (confidence_penalty method), `core/foss_pipeline.py`

### Task #195: SVD Hopfield Keys 512d → 32d [19:30]
- **Problem**: 512d keys for 2000 patterns = 22× over-provisioned (F2: ID_needed = 2×log₂(K)+1.2 ≈ 23d)
- **Fix**: SVD compression in `_build_matrix()`, project queries through same SVD in `retrieve()`
- **Result**: 16× noise reduction, faster cosine similarity (32d vs 512d)
- **Files**: `core/hopfield_bank.py` (_build_matrix, retrieve)

### Task #196: GCV Ridge Alpha [19:30]
- **Problem**: `ridge_alpha=1.0` hardcoded in `train_readout()` — arbitrary
- **Fix**: Marchenko-Pastur formula: α_opt = (D/N) × σ²_noise, cross-validation-free
- **Theory**: F3 from Formelbot audit, RMT theory
- **Files**: `core/reservoir_lm.py` (train_readout)

### Task #197: Expand MLP Facts — Script Ready [19:30]
- **Status**: `extract_mlp_facts.py` already exists, needs MLX GPU run
- **Current**: 198 entity vectors, target: 1300+ subjects
- **Pipeline**: Already handles expanded data when available

### Task #198: Situation Memory — Anti-Hallucination [19:30]
- **NEW FILE**: `core/situation_memory.py` (~120 LOC)
- **What**: Circular buffer (500 entries) storing (query_emb, answer, quality)
- **Anti-hallucination**: Novel query (sim < 0.3 to all stored) → confidence halved
- **EMA update**: Duplicate queries blend quality scores (0.7 old + 0.3 new)
- **Wired into**: `foss_pipeline.py` Step 5 after Z₂ gating
- **Source**: doom/foss-v2/modules/situation_memory.py (D1)

### Task #199: Experience Pool — Online Learning [19:30]
- **NEW FILE**: `core/experience_pool.py` (~100 LOC)
- **What**: Circular buffer storing successful query experiences for priming
- **Properties**: Max 500, quality threshold 0.5, dedup via cosine > 0.9
- **Wired into**: `foss_pipeline.py` Step 6 — stores after successful queries
- **Source**: doom/foss-v2/modules/experience_pool.py (D3)

### Task #200: Cerebellum Reasoning Loop [19:30]
- **NEW FILE**: `core/cerebellum.py` (~80 LOC)
- **What**: Predict → CASI → Correct → Re-inject → Repeat (3-5×)
- **Key insight**: Reservoir Echo State = different trajectories on re-injection = "thinking" without transformer layers
- **CASI monitors**: If improvement < 0.1 → stop iterating
- **Source**: doom/foss-v2/modules/cerebellum.py (D2)

### Task #201: Effective Rank + TwoNN Diagnostics [19:30]
- **NEW FILE**: `core/diagnostics.py` (~130 LOC)
- **effective_rank()**: Shannon entropy of singular values (F9)
- **twonn_id()**: Facco et al. (2017) intrinsic dimension estimator (F10)
- **reservoir_health()**: Full health check (readout rank, condition number, MERA prediction)
- **diagnose_states()**: Batch state analysis (rank, variance, ID)

### Task #202: DS Layer Stack — Sinkhorn Depth [19:30]
- **NEW FILE**: `core/ds_layers.py` (~100 LOC)
- **What**: 3-4 Sinkhorn normalization layers for depth without transformers
- **Each layer**: Similarity → Sinkhorn → Matrix-vector product
- **Theory**: FB1 (universal spectral stats), FB2 (anti-sink), F7 (86% per iteration)
- **From gottformel**: DS > Zeno penalty, structural topology control

### Task #203: Multi-Timescale Reservoir [19:30]
- **NEW FILE**: `core/multi_timescale.py` (~130 LOC)
- **3 bands**: Fast (512, leak=0.5), Medium (1024, leak=0.3), Slow (512, leak=0.1)
- **Z₂ mismatch**: Fast vs Slow disagreement = novelty detector
- **Combined state**: 2048d total ([fast; medium; slow])
- **Source**: doom/foss-v2/ARCHITECTURE.md (D4)

### Task #204: Online Ridge — Learning during Inference [19:30]
- **NEW FILE**: `core/online_ridge.py` (~100 LOC)
- **Formula**: XtX *= decay; XtX += outer(state, state); W_out = solve(XtX, XtY)
- **Properties**: Rank-1 updates O(d²), exponential forgetting (0.999), no gradients
- **Recompute W_out**: Every 10 updates (amortized cost)
- **Source**: doom/foss-v2/ARCHITECTURE.md lines 596-618 (D5)

### Regression Tests v3.0: 20/20 (100%), Median 26ms, Max 63ms

## v3.1 — Full Wiring + Bug Fixes (2026-03-17)

### Task #205: FIX SituationMemory empty-memory bug [20:00]
- **Bug**: Empty memory (0 entries) → every query is "novel" → confidence halved on first query
- **Fix**: Skip novelty check when `situation_memory.size < 10`
- **Files**: `core/foss_pipeline.py` (Step 5)

### Task #206: FIX ExperiencePool cross-contamination [20:00]
- **Bug 1**: ExperiencePool stored results but never retrieved them (no priming step)
- **Bug 2**: First fix used sim > 0.5 threshold → "born einstein" got "1564" from "Shakespeare born"
- **Fix**: Added Step 1e priming with sim > 0.85 threshold (near-duplicate only), top_k=1, conf×0.5
- **Files**: `core/foss_pipeline.py` (Step 1e)

### Task #207: WIRE Cerebellum Loop [20:00]
- **Was**: `core/cerebellum.py` existed but never imported
- **Now**: Wired into Step 5b — activated when confidence < 0.4, runs 3 iterations
- **Re-query**: Calls hopfield + KB with augmented content words
- **Files**: `core/foss_pipeline.py` (Step 5b)

### Task #208: WIRE DS Layer Stack [20:00]
- **Was**: `core/ds_layers.py` existed but never imported
- **Now**: Wired into Step 1d — 3 Sinkhorn layers on content word embeddings
- **Effect**: Cross-token information integration (depth without transformers)
- **Files**: `core/foss_pipeline.py` (Step 1d)

### Task #209: WIRE Multi-Timescale Reservoir [20:00]
- **Was**: `core/multi_timescale.py` existed but never imported
- **Now**: Init in configure(), wired into Step 4b as novelty detector
- **Effect**: Fast/slow Z₂ mismatch supplements single-band Z₂ novelty
- **Files**: `core/foss_pipeline.py` (configure + Step 4b)

### Task #210: WIRE Online Ridge [20:00]
- **Was**: `core/online_ridge.py` existed but never imported
- **Now**: Init in configure(), learns in Step 7 after successful queries (conf > 0.6)
- **Files**: `core/foss_pipeline.py` (configure + Step 7)

### Task #211: WIRE Diagnostics Health Check [20:00]
- **Was**: `core/diagnostics.py` existed but never imported
- **Now**: `reservoir_health()` called in configure(), exposed via `pipeline.health()`
- **Health API**: hopfield_patterns, kb_facts, memory sizes, online_ridge_updates, nn_index
- **Files**: `core/foss_pipeline.py` (configure + health method)

### Task #212: FIX Hopfield confidence after SVD [20:00]
- **Bug**: SVD 512d→32d dropped Hopfield confidence from 1.0 to 0.48 (smaller cosine sims in 32d)
- **Fix**: Auto-calibrate beta after SVD based on 95th vs 50th percentile similarity gap
- **Formula**: beta = clamp(3.0 / (p95 - p50), 4.0, 30.0)
- **Files**: `core/hopfield_bank.py` (_build_matrix)

### Task #213: Full Regression Test [20:00]
- **Result**: 20/20 (100%), Median 28ms, Max 65ms
- **17/19 components active** (attention + ricci optional, need pre-extracted weights)
- **All 7 new modules wired**: DS Layers, Cerebellum, Multi-Timescale, Online Ridge, Diagnostics, SituationMemory, ExperiencePool
- **Memory growing**: SituationMemory + ExperiencePool accumulate during query sequence

### Pipeline Data Flow (v3.1)
```
Query → Step 0: Tokenize → content_words
      → Step 1a: Extracted Attention (optional)
      → Step 1b: Ricci Attention (optional)
      → Step 1c: Z₂ Disagreement Attention
      → Step 1d: DS Layers (3× Sinkhorn mixing)
      → Step 1e: Experience Pool priming (sim>0.85)
      → Step 2a: Reservoir ESN
      → Step 2b: Hopfield Bank (SVD 32d + Sinkhorn bias)
      → Step 2c: Knowledge Store (3-tier)
      → Step 2d: Causal DAG
      → Step 2e: Residual Hopfield
      → Step 2f: MLP Facts
      → Step 2g: Multi-Hop (causal questions)
      → Step 3: Foss Consensus (barbell gossip)
      → Step 3b: CASI Confidence Penalty
      → Step 4: Z₂ Novelty gating
      → Step 4b: Multi-Timescale mismatch
      → Step 5: Situation Memory (anti-hallucination)
      → Step 5b: Cerebellum Loop (if conf < 0.4)
      → Step 6: Experience Pool (store)
      → Step 7: Online Ridge (learn)
      → Answer
```

## v3.2 — Phrase Encoding + Attention Wiring (2026-03-17)

### Task #222: Multi-word phrase encoding [20:30]
- **Problem**: 88% of KB subjects are multi-word (e.g., "United States", "solar_system") → zero embedding → invisible to Hopfield/retrieval
- **Fix**: `encode()` in both `EmbeddingStore` and `EmbShim` now detects spaces/underscores, splits, encodes each word, averages
- **Result**: +1381 newly encodable subjects in first 5000 facts (58%→86% coverage)
- **Files**: `core/reservoir_lm.py` (EmbeddingStore.encode), `test_pipeline_regression.py` (EmbShim.encode)

### Task #214: Wire ExtractedAttention [20:30]
- **Was**: `self.attention = None` in pipeline — never initialized despite `extracted_attention.py` existing
- **Now**: Auto-loads in `configure()` from `qwen3_attention_layers.npz`
- **Status**: Loaded and active, but layers 0-2 give near-uniform weights (0.50/0.50) — needs deeper layers (#215)
- **Files**: `core/foss_pipeline.py` (configure)

### Task #217: PHRASE EMBEDDINGS — resolved by #222
- Multi-word encode() IS the phrase embedding fix. No separate module needed.

### Task #218: Audit unused core/ modules [20:45]
- **Found**: 102 of 122 core/ files NOT imported by pipeline
- **Assessment**: Most are from earlier iterations (GloVe-era, pre-Qwen3), standalone agents/tools
- **Worth watching**: `icl.py` (TranslationVectorICL), `stacked_hopfield.py` (needs Qwen3 port)
- **Decision**: No wiring — the 20 active modules cover all critical pipeline paths. Adding more adds complexity without clear wins.

### Task #219: qwen3_transformer_embeddings.npy audit [20:45]
- **Shape**: (151936, 2048), 1.2GB, float32
- **What**: Post-28-layer hidden states per token (NOT static embeddings)
- **Evidence**: 3.5× larger norms (5.47 vs 1.57), cos sim 0.187 to static
- **Decision**: Not loading (1.2GB too large for full vocab). FUTURE: extract subset for top-2000 KB entities.

### Task #215: Extract Layer 10+18 from Qwen3-1.7B [21:00]
- **Extracted**: 23 arrays from layers 10+18 via MLX full model load
- **Contents**: Q/K/V/O projections, QK norms, MLP gate/up/down, layernorms, final_norm
- **File**: `data/qwen3_deep_layers.npz` (181MB)
- **Script**: `scripts/extract_deep_layers.py`

### Task #215b: Wire deep layers into ExtractedAttention [21:00]
- **Was**: ExtractedAttention used shallow layers 0-2 → uniform weights (0.50/0.50)
- **Now**: Prefers deep layers 10+18, falls back to shallow if unavailable
- **Result**: "who wrote hamlet" → hamlet=0.512, wrote=0.244 (spread 0.268 vs ~0 before)
- **Files**: `core/extracted_attention.py` (complete rewrite of _load)

### Task #216: MLP down_proj as standalone fact memory — NEGATIVE [21:15]
- **Test**: Raw MLP down_proj @ token embedding → predicts input word back (no cross-token association)
- **Root cause**: MLP fact recall requires prior attention contextualization (residual stream after attention)
- **Decision**: Use full forward-pass MLP extraction (#220) instead of standalone weight matrices

### Task #220: Expand MLP Facts 198→1814 [21:30]
- **Was**: 198 MLP facts (top 200 subjects only)
- **Now**: 1814 MLP facts (all KB subjects + entity objects), extracted via full Qwen3 forward pass at Layer 18
- **Script**: `scripts/expand_mlp_facts.py`
- **File**: `data/qwen3_mlp_facts.npz` (7.7MB, was 0.8MB)
- **Coverage**: 9× more stolen transformer knowledge vectors

### Task #221: Multiple-Choice Benchmark — HellaSwag + PIQA [22:00]
- **Built**: `core/mc_scorer.py` — MultipleChoiceScorer with Qwen3 PPL + Reservoir + Embedding
- **Built**: `benchmark_mc.py` — HellaSwag (10042 examples) + PIQA (1838 examples)
- **Results (n=500)**:
  - **HellaSwag: 59.4%** (random 25%, +34.4pp)
  - **PIQA: 72.2%** (random 50%, +22.2pp)
- **Per-scorer**: Qwen3 PPL=59.8%, Reservoir=26% (random), Embedding=26.6% (random)
- **Honest assessment**: Qwen3 perplexity does 99% of the work on sentence completion.
  Reservoir readout was trained on KB facts, not language continuation — needs retraining on text data.
- **Latency**: HellaSwag avg 1007ms, PIQA avg 265ms (MLX on M4)

### Regression: 20/20 (100%), avg 40ms, median 36ms, max 102ms

---

## v2.0 — Performance Revolution + Infrastructure (2026-03-17)

### Task #187: Hopfield Bank Integration [17:46]
- **Problem**: `HopfieldTemplateBank` existed in `core/hopfield_bank.py` but was never instantiated
- **Fix**: Auto-build in `configure()` from KB facts (2000 patterns, Modern Hopfield, 512d Qwen3 projected)
- **Result**: Hopfield now provides parallel retrieval alongside KB — cross-source agreement via Consensus
- **Example**: "capital france" → Hopfield: "paris" (conf=1.0) + KB: "Paris" (conf=1.0) → Consensus: "Paris" (1.0)
- **Files**: `core/foss_pipeline.py` (configure + _hopfield_retrieve)

### Task #188: MLP Fact Vectors + Residual Hopfield Integration [17:46]
- **Problem 1**: 198 MLP fact vectors (`qwen3_mlp_facts.npz`) never loaded — stolen transformer knowledge unused
- **Problem 2**: Residual Hopfield (1877 pre-contextualized states) loaded but never queried
- **Problem 3**: `find_token_embedding()` args were SWAPPED → 864ms per MLP query (iterating ndarray as dict)
- **Fix**: Load MLP vectors in configure(), add `_mlp_retrieve()` and `_residual_hopfield_retrieve()` as sources
- **Fix**: Corrected arg order: `find_token_embedding(word, token2id, embeddings)` not `(word, embeddings, token2id)`
- **Result**: 1200ms → 21ms query time (arg swap fix), all 3 new sources wired into Consensus
- **Active sources**: Reservoir Z₂, Hopfield Bank, KB, Residual Hopfield, MLP Facts, Multi-Hop, Causal DAG
- **Files**: `core/foss_pipeline.py` (configure, query, _mlp_retrieve, _residual_hopfield_retrieve)

### Task #182: Benchmark vollständig [17:46]
- **All components loaded**: Reservoir, Hopfield Bank (2000 patterns), KB (137K facts), Residual Hopfield (1877 states), MLP Facts (198 vectors), Annoy NN Index, Causal DAG, Multi-Hop, CommonsenseEngine
- **Result**: 8/8 queries correct, Median 23ms, Max 32ms
- **Cross-source validation**: Hopfield + KB agree → Consensus conf=1.0

### Task #184: Pipeline Regression Test Suite [17:46]
- **20 Gold-Standard Q&A pairs** in `test_pipeline_regression.py`
- **Categories**: capital(3), author(2), property, superlative(2), discoverer, numeric(2), composition, date(2), language, currency, inventor, symbol, relation, type
- **Result**: 20/20 (100%), Median 27ms, Max 53ms, all under 500ms budget
- **Key**: Hopfield Bank now active as parallel retrieval source alongside KB for every query
- **Files**: `test_pipeline_regression.py` (NEW)

### Task #176: NumPy BLAS → Apple Accelerate [14:xx]
- **Problem**: NumPy used OpenBLAS with SANDYBRIDGE (x86) profile on ARM M4 → 28-875× slower than native
- **Fix**: `pip install numpy` from source → builds against Apple Accelerate framework
- **Result**: 197-875× speedup on all matmul operations
- **Side effect**: scipy recompile needed (`pip install --upgrade scipy` → 1.17.1)
- **Files**: System-level (no code changes)

### Task #177: Annoy NN Index [14:xx]
- **Problem**: Brute-force NN search over 151,936 × 2048d = 48ms/query (O(V·d))
- **Fix**: Spotify Annoy (angular distance ≈ cosine), 10 trees, disk-cached
- **Result**: 0.09ms/query → **533× faster**
- **Files**:
  - `core/nn_index.py` (NEW) — NNIndex class with Annoy/FAISS/numpy fallback
  - `data/annoy_index_151936_2048_10.ann` (1.3GB) — pre-built index, instant reload
- **Note**: FAISS crashes with Anaconda (OpenMP conflict, segfault 139) → Annoy is primary

### Task #178: AR Generation Fixes [15:xx]
- **Problem**: generate_sequence() returned 0 tokens — cosine sims in 2048d are ~0.08, threshold was 0.15
- **Fix**: Threshold 0.15 → 0.02, dimension mismatch fixed (W_in expects 512d, was getting 2048d)
- **Files**: `core/foss_pipeline.py` — generate_sequence(), _nearest_raw()

### Task #181: Vocab Case Sensitivity [14:xx]
- **Problem**: "Einstein" not found — exists only as "ĠEinstein" in Qwen3 BPE vocab
- **Fix**: 8-variant lookup (exact, Ġ-prefixed, lower, Ġ+lower, capitalized, Ġ+cap, upper, Ġ+upper)
- **Files**: `core/tokenutils.py` (NEW) — find_token_id(), find_token_embedding(), clean_token()

### Task #183: Temperature Sampling + Repetition Penalty [15:xx]
- **Problem**: AR generation was greedy (argmax) → always same output; hard-block set killed diversity
- **Fix**: Softmax temperature (0.7), top-k=8, frequency-based repetition penalty (1.5^count)
- **Files**: `core/foss_pipeline.py` — generate_sequence() signature changed

### Task #186: Ġ-Token Cleanup [14:xx]
- **Problem**: Display showed raw BPE markers (Ġ, ĉ, Ċ) to user
- **Fix**: clean_token() strips all markers, _nearest_raw() deduplicates via seen_clean set
- **Files**: `core/tokenutils.py`, `core/foss_pipeline.py`

### Speed Results (Apple M4, Pure CPU, No GPU)
| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| NN search (151K×2048d) | 48ms (78ms old BLAS) | 0.09ms | 533× |
| Reservoir step (2048²) | 83ms | 0.42ms | 197× |
| SVD projection (2048→512) | 35ms | 0.04ms | 875× |
| Query "capital france" | 2167ms | 127ms | 17× |
| Query "author hamlet" | 120ms | 7ms | 17× |

### Task #179: Pipeline vollständig konfigurieren + Tier 3 Performance Fix [16:xx]
- **Problem 1**: `KnowledgeStore()` startete leer — Brain-Datei wurde nicht geladen
- **Problem 2**: `kb.query()` lief IMMER bis Tier 3 Hopfield (O(N) über 137K Fakten, 2-7s pro Call)
- **Problem 3**: Subject-only Fallback gab beliebige Fakten zurück ("speed of light" → "heavy")
- **Fix**: `max_tier` Parameter in `kb.query()` — Pipeline nutzt nur Tier 1/2 (Dict-Lookup, <0.1ms)
- **Fix**: Brain auto-load im Setup, vollständige Komponentenkonfiguration
- **Fix**: Relation-Alias-Map erweitert (20+ Mappings: discovered→discoverer, made→composed_of, etc.)
- **Fix**: Case-insensitive Subject-Lookup (france/France/FRANCE alle getestet)
- **Fix**: Superlative-Scanner ("largest planet" → scannt descriptions für Keyword-Match)
- **Fix**: Expanded Query-Parser (12 neue Patterns: "X born", "X made of", "who discovered X", etc.)
- **Result**: 8/8 Benchmark-Queries korrekt, alle <0.1ms (vorher: 3 korrekt, avg 3.7s)
- **Files**: `core/knowledge.py` (max_tier), `core/foss_pipeline.py` (_kb_retrieve komplett neu)

### Git + Documentation
- `git init` + LFS for data files (`.npy`, `.npz`, `.pkl`, `.csv.gz`, `.txt`, `.pth`)
- `README.md` — project overview, architecture diagram, quick start
- `ARCHITECTURE.md` — 546-line handoff doc (all components, data flow, training)
- `benchmark_speed.py` — full speed benchmark (components, queries, AR, bottleneck analysis)
- `.gitattributes` — LFS tracking rules

---

## Session 2026-03-14c — Brain Architecture + Wikidata Live Import

### Architecture: Brain File (.brain)
- **Persistent knowledge**: One file = complete AI state (facts + FLM + Hopfield patterns)
- **Brain-first bootstrap**: If `.brain` exists → load (no network, no regeneration). If not → bootstrap + save.
- **Atomic writes**: temp file + `os.replace()` — never leaves corrupted state
- **Analogous to transformer weights**, but transparent (readable facts), mergeable, and incrementally updatable
- **Format**: zlib-compressed JSON, 37 KB for 4350 facts

### Wikidata Live Import (SPARQL)
- **6 query categories**: countries (1273), borders (815), cities (579), companies (221), inventions (151), languages (265)
- **Total**: 3303 new facts from Wikidata, 4350 total KB (was 1341)
- **7-day cache**: `data/wikidata_cache.json` — network only needed once per week
- **CLI**: `python cli.py wikidata` — pull all categories, save to brain

### Bugs Fixed (Mechanism-Level)
- **Reverse lookup pattern priority**: "What country has Berlin as its capital?" → specific regex now before generic "What country has/uses X?"
- **`country` alias removed**: Was mapping `country→location`, breaking city queries (Tokyo→Japan)
- **Relation fallback chain**: `location→country→part_of`, `founder→creator→inventor` — if primary relation misses, try alternatives
- **Wrong-relation detection**: Fallback triggers when Hopfield returns correct subject but wrong relation (e.g., Tokyo+location→population instead of country)
- **"When was X invented?"**: `invented`→`founded` mapping for date queries (was giving inventor name instead of year)
- **"founded" in pronoun resolver**: `_parse_for_relation('founded')` now returns `founded` (date), not `founder` (person)
- **"symbol for X"**: `of|for` both accepted in "What is the X of/for Y?" pattern
- **Compare patterns expanded**: "What is bigger, X or Y?", "X vs Y", "X versus Y" all route to comparison

### New Response Formatters
- `country`: "X is in Y."
- `industry`: "X operates in the Y industry."
- `nationality`: "X is Y."
- `occupation`: "X is a/an Y."
- `born`/`died`: "X was born/died in Y."
- `symbol`/`formula`: "The symbol/formula for X is Y."
- `description`/`known_as`: "X is Y." / "X is also known as Y."
- `founded` (inventions): "X was invented in Y." (context-aware: checks type=invention)

### CLI Commands Added
| Command | Description |
|---------|-------------|
| `python cli.py wikidata` | Pull structured knowledge from Wikidata SPARQL |
| `python cli.py wikidata --categories countries cities` | Import specific categories |
| `python cli.py brain` | Show brain file info (facts, size, creation date) |
| `python cli.py brain rebuild` | Force rebuild from scratch |
| `python cli.py brain delete` | Delete brain file |
| `python cli.py bootstrap --wikidata` | Bootstrap + Wikidata in one step |

### Regression Tests: 72 → 83
| New Tests | Category |
|-----------|----------|
| Nigeria capital, Norway currency | wikidata_country |
| Germany borders Poland | wikidata_borders |
| London population | wikidata_city |
| Google founder, Google founded | wikidata_company |
| Where is London/Tokyo | location_fallback |
| Tokyo belongs to Japan | city_country |
| Symbol for gold | symbol_query |
| Telephone invented 1876 | invention_date |
| Multi-hop Spain (any neighbor) | multi_hop (expected_contains_any) |

### Test Results (2026-03-14c, final)
```
Regression: 83/83 passed (100%)
KB: 4350 facts (1341 bootstrap + 3303 Wikidata)
Brain: 37 KB (compressed)
Time: 15.3s (185ms avg, budget: 500ms)
```

---

## Session 2026-03-14b — Mechanism-Level Red-Teaming (Haiku Ping-Pong)

### Method
4 Haiku agents as parallel QA testers (knowledge, code gen, conversation flow, response quality), 2 Haiku agents as REFERENCE AI (ideal answer patterns). All findings fixed mechanistically — not per-query, per-mechanism.

### Features Added
- **Yes/No Questions** — "Does France border Germany?" → "Yes, France borders Germany." / "No, I don't have information that Japan borders France."
  - Works for borders (bidirectional check) and type ("Is Python a programming language?" → "Yes")
- **Reverse Lookups** — "What country has Berlin as its capital?" → "Germany has Berlin as its capital."
  - Scans KB for (?, relation, object) matches
- **Informal Currency** — "What money do they use in Japan?" → "Japan uses the Yen as its currency."
- **More Coding Verbs** — check, find, merge, validate, compute, parse, convert, extract, flatten, decode, encode, hash, detect, scan, remove, count all route to CODE_REQUEST
- **"X using Y" Pattern** — "Queue using two stacks" → CODE_REQUEST
- **Anti-Hallucination** — CBR fallback DISABLED. Was overriding "I don't know" with random garbage (Narnia → AI explanation).

### Bugs Fixed (Mechanism-Level)
- **"it" pronoun carried old relation** — Pronoun resolver now extracts NEW relation from current query
- **"its" false-triggered in "has X as its Y"** — Added guard: "its" only triggers if "as" not in query
- **"Tell me about Germany" → "German is a Germanic language"** — About-query type check now requires EXACT subject match
- **Multi-fact generalized** — `founded|created|invented|built` by X in Y → all extract 2 facts
- **"The telephone" subject pattern** — Now accepts `(?:[Tt]he\s+)?` prefix
- **Code gen threshold** — Lowered from >= 7 to >= 5 chars
- **"How do I" / "I need to" patterns** — Added for 30+ coding verbs
- **Case sensitivity** — `do\s+I` → `do\s+i` (input is lowercased)
- **Template aliases** — Added for natural phrasing: "read a csv", "write to a file", "sort a list", "todo list", etc.
- **"type" missing in relation keywords** — "What type is it?" now resolves correctly
- **"money"/"big" relation mapping** — Added to _parse_for_relation

### Code Templates Added (20+ new)
- fibonacci, prime, factorial, read file, write file, web scraper, calculator, todo list, password generator, matrix multiply
- binary tree, min heap, balanced parentheses, merge sorted lists, find duplicates, queue from two stacks, longest common substring
- Sort variants with spaces (bubble sort, merge sort, quick sort)

### Regression Tests: 47 → 69
| New Tests | Category |
|-----------|----------|
| prime, read file, sort, todo, stack, scraper, calculator | code_gen |
| balanced parentheses, duplicates, merge sorted, queue | code_gen |
| "created by X in Y" → 2 facts | learn_compound |
| currency via multi-hop | multi_hop |
| "it" with new relation, "type" resolution | pronoun |
| "I need to read CSV" | code_gen |
| "Tell me about Germany" → overview | about_query |
| Yes/No border/type, reverse lookup, money→currency | new features |
| Anti-hallucination (Narnia ≠ Paris) | anti_hallucination |

### Haiku Reference Patterns (applied)
1. Answer first, context second — no preamble
2. Concrete numbers beat vague language
3. One-line support for yes/no
4. Honest about unknowns — "I don't have reliable information"
5. Parallel structure for comparisons

### Additional Fixes (late session)
- **"Square root of 256" failed** — Case-sensitive regex: `square root` didn't match `Square root` → Added `re.I` flag
- **"its" false trigger in "has X as its Y"** — Possessive "its" inside "as its" ≠ pronoun reference → Guard: skip if "as" in words
- **FORGE domain pollution** — For queries without templates, FORGE fragments could generate security tools (os.remove, subprocess) → Mitigated by expanding template library to cover all common algorithm/data structure requests

### Test Results (2026-03-14b, final)
```
Regression: 72/72 passed (100%)
Time: 7.7s (107ms avg, budget: 500ms)
```

---

## Session 2026-03-14 — Product Iteration

### Features Added
- **"Did you mean?" (5-Strategy PS-Lifted Consensus)** — Google-style entity suggestions, fused from 5 independent strategies via PS-Lifted Consensus on barbell topology:
  1. Lexical (Levenshtein distance, per-word matching for multi-word entities)
  2. Phonetic (Soundex 1918, from scratch)
  3. Semantic (FLM perplexity scoring)
  4. Embedding (char trigram cosine similarity)
  5. KB-Relation Overlap (expanded common_rels incl. person relations)
  - Anti-hallucination threshold (0.15) prevents false suggestions
  - Handles typos: "Frence"→France, "Pyton"→Python, "Einsten"→Einstein, "Shakspeare"→Shakespeare, "Japn"→Japan
  - Negative test: "Xylandia" correctly produces no suggestion

- **LEARN Intent** — Agent auto-detects declarative statements ("FOSS-KI is a Markov chain AI engine") via `_is_declarative()`, extracts triplets via `TripletExtractor`, stores facts, and makes them immediately queryable

- **Pronoun Resolution (multi-turn)** — EntityTracker resolves:
  - "there" → last mentioned entity + relation extraction from context
  - "how many people live there" → population of last entity
  - "And what about X?" → ellipsis resolution, carries forward relation

- **Multi-Hop Reasoning** — Decompose nested queries into chained knowledge lookups:
  - "What is the capital of the country that borders France?" → borders(France)→Germany → capital(Germany)→Berlin
  - "What language do they speak in the country that borders Spain?" → borders(Spain)→France → language(France)→French
  - Handles both "of" and "in" connectors, forward and reverse fact lookup

- **Entity Comparison** — Side-by-side comparison of any two entities:
  - "Compare France and Germany" → table with capital, language, population, currency, borders, location
  - "Compare Python and Linux" → table with creator, type, founded, paradigm, etc.
  - Also triggered by "differences between X and Y", "X versus Y"

- **Conversation Export** — Export chat history as Markdown or JSON:
  - In chat mode: `export path.md` or `export path.json`
  - Markdown: formatted with headers, user/AI turns, metadata
  - JSON: structured with timestamps, intent, source, latency per turn

- **PDF Feed** — `feed --file document.pdf` now reads PDFs via PyMuPDF (fitz)
  - Extracts text from all pages, feeds through metacognition pipeline

- **Graceful Failure** — Double-wrapped error handling: outer try/except in `Agent.process()` + individual try/except per intent handler + `DialogSystem.turn()` wrapping `_turn_inner()`. No query can crash the system.

- **Latency Budget** — 500ms max per query enforced in regression runner. Current benchmark: 32ms avg. All queries under budget.

- **Code Gen Priority Fix** — Specific template library (keyword >= 7 chars) checked BEFORE FORGE fragments

- **Code Templates** — Added `sort by`, `sort tuple`, `second element` templates

### Bugs Fixed (Session 2)
- **Currency query returned all facts** — Parser had no pattern for "What currency does X use?" → added dedicated currency patterns + generic "What {relation} does X have/use?"
- **Math echoed full question** — "What is 15 * 7? = 105" → now "15 * 7 = 105" (uses cleaned expression)
- **"What is Python?" dumped all facts** — About queries went straight to composer → now tries type/identity FIRST
- **Multi-hop entity resolution** — "borders" relation: subject/object correctly resolved based on which side matches the query entity

### Bugs Fixed (Session 1)
- `agent.py` broken syntax (try without except) — orphaned elif/else branches
- Regression runner: CLI didn't persist state between bootstrap and regression → auto-bootstrap via Agent
- Math queries fail via `dialog.turn()` → route all queries through `Agent.process()`
- "Frence" suggestions returning wrong entities → tightened pre-filter (same_first + ed_ratio)
- Single-char entities ("C") appearing in suggestions → `len(s) > 1` filter
- Multi-word entities ("Albert Einstein") not matching → per-word Levenshtein in pre-filter + lexical
- Hamming distance couldn't handle insertions → proper Levenshtein implementation
- "Did you mean?" overridden by CBR fallback → check `has_suggestions` before fallbacks
- "there" not resolved → added to EntityTracker with `_parse_for_relation()`
- "how many" mapped to 'number' (nonexistent relation) → mapped to 'population'
- LEARN intent: regex negative lookahead failed, QUESTION handler overwrote learned response, `store()` → `store_fact()`
- Code gen wrong template: FORGE fragment matched before specific templates

### Response Formatting
Natural language responses for all relation types:
- capital: "Paris is the capital of France."
- language: "The official language of France is French."
- currency: "France uses the Euro as its currency."
- population: "France has a population of 67 million."
- location: "France is located in Europe."
- type: "Python is a programming language."
- creator: "Guido van Rossum is the creator of Python."
- borders: "France borders Germany."
- founded: "Python was founded in 1991."

### Architecture
- 93 core modules, 1341 bootstrapped facts
- FLM (Foss Language Model) — PS-Lifted Consensus over 4 PPM context trees (orders 3,5,8,12) + word-level Markov
- Hopfield memory for pattern storage
- Constraint solver for reasoning
- No transformer, no GPU, no API calls

### CLI Commands
| Command | Description |
|---------|-------------|
| `python cli.py ask "question"` | Single question |
| `python cli.py chat` | Interactive chat (full Agent pipeline) |
| `python cli.py feed "text"` | Extract facts from text |
| `python cli.py feed --file path` | Extract facts from file (TXT, PDF) |
| `python cli.py train --text path` | Train FLM on text |
| `python cli.py bootstrap` | Load world knowledge |
| `python cli.py selftest` | Run self-test benchmark |
| `python cli.py regression` | Run 47 NL regression tests |
| `python cli.py shovel` | Export KB with dummy names |
| `python cli.py gaps` | Knowledge gap report |
| `python cli.py explore` | Frontier exploration |
| `python cli.py patterns` | Discover extraction patterns |
| `python cli.py status` | Metacognition status |
| `python cli.py nexus` | Nexus agent mode |
| `python cli.py save/load path` | Persist/restore knowledge |
| `python cli.py info` | System info |

- **Shovel Mode** — Export KB with semantically equivalent dummy substitutions for external debugging. Structure-preserving, type-preserving, deterministic, reversible. 273 entities mapped across 7 types (country, city, person, language, currency, continent, generic). Zero real name leakage verified.

- **Swarm Mode (tested)** — Multi-agent UDP system with domain-specific specialists (geography, medicine, law, science, code). MetaAgent routes queries to appropriate domain(s) including multi-domain detection ("treat diabetes and what country" → medicine + geography). Keyword expansion for medicine domain (headache, cure, remedy, diabetes, etc.).

- **Text Generation** — Email templates (late, sick, resign, update, follow-up), explanation templates (quicksort, binary search, recursion, hash table), knowledge-based explanations, extractive summarization, joke templates. Intent routing fixed: "Write me a joke" → GENERATION not CODE_REQUEST.

### Regression Tests (47 total)
| Category | Count | Status |
|----------|-------|--------|
| basic_capital | 3 | PASS |
| language_query | 2 | PASS |
| continent_alias | 2 | PASS |
| about_type | 2 | PASS |
| about_query | 2 | PASS |
| creator_query | 2 | PASS |
| currency_query | 2 | PASS |
| population_query | 2 | PASS |
| math | 3 | PASS |
| did_you_mean | 6 | PASS |
| compare | 2 | PASS |
| compare_edge | 1 | PASS |
| multi_hop | 2 | PASS |
| text_gen | 3 | PASS |
| multi-turn sequences | 5 | PASS |
| code_gen | 3 | PASS |
| conversation | 1 | PASS |
| edge_case | 5 | PASS |

### Test Results (2026-03-14, final)
```
Regression: 47/47 passed (100%)
Time: 1.9s (40ms avg, budget: 500ms)
```
