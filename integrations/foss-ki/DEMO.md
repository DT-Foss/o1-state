# DEMO — FOSS-KI forgets and remembers a fact, live, no rebuild

```
FossKIRepl(live_causal_store="<converted store>", knowledge_only=True)
```

1. **Ask:** "what is the capital of France?" → **"Paris"**
2. **Cut** every segment that cites the France↔Paris relationship, in EVERY direction and source it's stored in — `find_segments_citing("france", "paris")` (the forward knowledge_full.json fact) union `find_segments_citing("paris", "france")` (the reverse ConceptNet `AtLocation` fact, a different source entirely) — via `drop_segments()`. No restart, no reload, same running process.
3. **Ask the identical question again:** → **"I don't have information about that topic."** Not a hallucination, not a guess from another layer — an honest, correct refusal. (Cutting only ONE direction is provably NOT enough — the demo shows this as an explicit intermediate step, not a hidden caveat.)
4. **Append** the exact same records back (content-addressed: every re-appended segment's sha256 is byte-identical to the one just dropped).
5. **Ask again:** → **"Paris."** Nothing was rebuilt in between; the same `FossKIRepl` object answered all three times.
6. A control question ("who wrote Hamlet?") is asked at every step and returns the identical answer throughout — the cut removed exactly one relationship, not the knowledge base.

**Why nobody else can show this:** the fact FOSS-KI just forgot and re-learned isn't a row in an opaque database update log — it's a specific, content-addressed file (`<sha256>.seg`) that can be read, hashed, and verified by a stranger with nothing but the store directory. Knowledge here is an editable, addressable, checkable artifact, not a black-box weight update. `experimental/livecausal/demo_cut_append.py` runs this end to end against a store built from BOTH `knowledge_full.json` and a ConceptNet slice, and checks all nine claims above automatically — see its transcript for the exact trace of every step.

**What this demo does NOT claim:** the `knowledge_only=True` flag turns off FOSS-KI's other, independent fact sources (the Foss Pipeline's generative Reservoir/Hopfield, its CBR case library, multi-hop reasoning over CommonSense, web search) for the run — those still exist as separate FOSS-KI subsystems and are untouched by the cut, by design. This demo proves forgetting for the knowledge source(s) actually converted into the LiveCausalAdapter's store — as of Phase 4, that's `knowledge_full.json` (4,855 facts, fully converted) plus a 15,000-source-relation ConceptNet slice (11,217 facts after relation-type filtering; the full 500K-relation file converts fine but does not YET mount safely — see the README's "Known scaling limit" section for the measured OOM and the exact curve). Reasoning and math (formula solving, the ReasoningEngine) stay on throughout, because they compute over the same adapter-backed knowledge rather than bypassing it.

---

# DEMO 2 — the full loop: world text -> live segments -> answer -> forget (Phase 5)

```
1. builder_run.py --text-file <article.txt> --store-dir <store>   # organism + fabel, live
2. FossKIRepl(live_causal_store=<store>, knowledge_only=True)     # same store, same process
```

1. A fixed local text file (a Marie Curie facts article, no web scraping) is streamed by `builder_run.py` — the organism's windowing + fabel's deterministic 14-step `curator_yield_run.extract_validated` (no LLM anywhere) validate causal triplets from it and fold them straight into a `LiveStore`/`LiveGraph`, on top of a pre-seeded `knowledge_full.json` base.
2. **Ask, in the SAME process that just built the store:** "what does high doses of radiation cause?" → **"high doses of radiation causes cancer"** — the article's own fact, answered immediately, no separate conversion step.
3. **Ask an unrelated control question:** "who wrote Hamlet?" → **"Shakespeare"** — from the pre-seeded base, proving the article didn't overwrite anything.
4. **Cut only the article's 8 segments** (not the 98 pre-seeded `knowledge_full.json` segments) via `drop_segments()`.
5. **Ask the article question again:** → **"I don't have information about that topic."** The radiation/cancer fact is gone.
6. **Ask the control question again:** → **"Shakespeare."** Unchanged.

**Why this is the whole stack, not a subset:** every other demo in this directory starts from an already-converted store. This one starts from a raw text file and ends at a spoken-language answer, with the SAME process doing the building and the answering — nothing was pre-computed, cached, or staged between steps 1 and 2.

**Honest scope note, traced not assumed:** the fabel extractor's real output on this kind of text is short causal-mechanism triplets ("X causes Y"), and even getting there took real iteration — the first attempts (a Wikipedia "Penicillin" article, then longer multi-sentence paragraphs) produced multi-sentence, unnaturally-long trigger/outcome keys unreachable by ANY natural question (documented in `demo_e2e_loop.py`'s own module docstring); a one-sentence-per-line article with a small `--window-tokens` value was needed to get clean, short causal facts out of this extractor. Separately, none of `core/router.py`'s pre-existing natural-language query patterns (`_parse_query_for_knowledge`, all attribute-shaped: capital/known_for/location/born/...) covered "what causes X" / "what does X cause" against `self.knowledge` — this demo's Phase 5 change adds exactly that one lookup to `repl.py`'s `_direct_kb_lookup`, returning a full sentence ("{subject} causes {outcome}") rather than the bare outcome specifically because a short bare answer can fail `_answer_quality_gate`'s topic-overlap check when it shares no words with the question and isn't a capitalized proper noun — a real, traced fix, not a workaround around the gate.

---

# DEMO 3 — receipts: the trace shows exactly which segments backed an answer (Task 16)

```
1. "what is the capital of France?" -> "Paris"
   [Trace] ... Receipt: (8478f493a348,11), (b079ae4797d4,40) | evidence_count=2
2. Cut ONE of the two cited segments.
3. Same question again -> STILL "Paris"
   [Trace] ... Receipt: (b079ae4797d4,40) | evidence_count=1
   (one citation survived the cut -- the receipt tracked it correctly)
4. Cut the LAST cited segment -> "I don't have information about that topic."
   (the receipt's prediction held: zero citations left, zero fact left)
```

A second run against a genuine conflict store (`Coffee causes alertness`, 3 independently-sourced segments, vs. `Coffee correlates_with alertness`, 1 segment) shows the trace's `Contested: 3:1 (winner='causes', ...)` line — then cutting 2 of the 3 `causes` segments flips it LIVE to `Contested: 1:1 (winner='correlates_with', ...)` on the very next question, proving the ratio is recomputed from the current store every time, never cached.

**Why this matters:** a confidence number alone ("HIGH", "0.9") is a claim you have to trust. A receipt is a claim you can check — every citation is a real `(segment_sha, record_index)` pointer a stranger can open with nothing but the store directory, and this demo proves the pointer is load-bearing: cut what it cites, and the answer (or the confidence ratio) changes exactly as the receipt said it would. `experimental/livecausal/demo_receipts.py` runs both proofs automatically, 8 checks, all PASS.

**What "contested" does and does not mean here:** `evidence.py`'s own conflict definition is mechanism-based — two records citing the SAME `(trigger_key, outcome_key)` pair with DIFFERENT `mechanism` strings (e.g. "causes" vs. "correlates_with" for the same Coffee→alertness pair). Two segments claiming genuinely DIFFERENT outcomes for the same trigger (e.g. "France's capital is Paris" vs. "France's capital is Lyon") are NOT contested in this model — they're two separate, uncontested edges, and the adapter silently picks the better-evidenced one without flagging a conflict. Building that comparison would mean inventing new conflict logic beyond `evidence.py`'s existing folds, which this task was explicitly told not to do.
