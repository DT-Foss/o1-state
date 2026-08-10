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
