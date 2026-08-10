# DEMO — FOSS-KI forgets and remembers a fact, live, no rebuild

```
FossKIRepl(live_causal_store="<converted store>", knowledge_only=True)
```

1. **Ask:** "what is the capital of France?" → **"Paris"**
2. **Cut** the two segments that cite France→capital→Paris via `drop_segments()` — no restart, no reload, same running process.
3. **Ask the identical question again:** → **"I don't have information about that topic."** Not a hallucination, not a guess from another layer — an honest, correct refusal.
4. **Append** the exact same records back (content-addressed: the new segment's sha256 is byte-identical to the one just dropped).
5. **Ask again:** → **"Paris."** Nothing was rebuilt in between; the same `FossKIRepl` object answered all three times.
6. A control question ("who wrote Hamlet?") is asked at every step and returns the identical answer throughout — the cut removed exactly one fact, not the knowledge base.

**Why nobody else can show this:** the fact FOSS-KI just forgot and re-learned isn't a row in an opaque database update log — it's a specific, content-addressed file (`<sha256>.seg`) that can be read, hashed, and verified by a stranger with nothing but the store directory. Knowledge here is an editable, addressable, checkable artifact, not a black-box weight update. `experimental/livecausal/demo_cut_append.py` (`knowledge_only=True` variant) runs this end to end and checks all seven claims above automatically — see its transcript for the exact trace of every step.

**What this demo does NOT claim:** the `knowledge_only=True` flag turns off FOSS-KI's other, independent fact sources (ConceptNet, its CBR case library, multi-hop reasoning over ConceptNet, web search) for the run — those still exist and are untouched by the cut, by design. This demo isolates and proves forgetting for the one knowledge source the LiveCausalAdapter actually controls. Reasoning and math (formula solving, the ReasoningEngine) stay on throughout, because they compute over the same adapter-backed knowledge rather than bypassing it.
