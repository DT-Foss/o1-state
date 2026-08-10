"""
LIVE-CAUSAL cut/append demo, Phase 3 (hardened) -- the full-REPL
forgetting proof for revival-probe Phase 3.

Phase 2's version of this script proved the cut/append/re-append cycle
at the LiveCausalAdapter's own query() level, but noted honestly that
the FULL repl.process() answer in step (ii) could still say "Paris" via
FOSS-KI's independent ConceptNet/CommonSense layer -- a real, separate
knowledge source never converted into this adapter's store.

This version boots FossKIRepl(knowledge_only=True) (see repl.py's
knowledge_only flag, added for exactly this purpose): it disables the
redundant FACT-answering fallbacks that do not route through
self.knowledge at all (ConceptNet/CommonSense, the CBR case-answer
fallback, MultiHop, Web search) while leaving Reasoning/Math untouched
(_solve_reasoning, self.formulas, self.reasoning -- none of these are a
facts bypass; reasoning/math is computation, not an alternate fact
store). With that flag on, the SAME three-step cut/append cycle now
proves out on repl.process()'s own returned answer, not just an
isolated adapter query.

Asks "what is the capital of France?" three times through ONE
FossKIRepl(live_causal_store=..., knowledge_only=True) instance, no
rebuild between steps:

  (i)   baseline, all segments present -> "Paris."
  (ii)  the segment(s) citing France/capital/Paris are cut via
        drop_segments() -> the SAME question, asked through the FULL
        repl.process() (not just adapter.query()) -> honest "I don't
        have information about that topic."
  (iii) the segments are appended back (content-addressed, same sha256
        as the originals) -> "Paris." is back.

A control question ("who wrote Hamlet?") is asked at every step and
must be byte-identical throughout, proving the cut/knowledge_only mode
does not just break everything -- it forgets exactly the one fact it
was told to forget.
"""
import json
import os
import sys

sys.path.insert(0, '/root/mac_offload/desktop/foss-ki')
sys.path.insert(0, '/root/fosski-venv/adapter')
sys.path.insert(0, '/root')
os.chdir('/root/mac_offload/desktop/foss-ki')

from repl import FossKIRepl  # noqa: E402
from livecausal_bridge.infer import LiveGraph  # noqa: E402

STORE_DIR = '/root/fosski-venv/livecausal_store_demo_v2'
TRANSCRIPT_PATH = '/root/fosski-venv/cut_append_transcript_v2.txt'

lines = []


def log(s=""):
    print(s)
    lines.append(s)


def ask(repl, question, label):
    log(f"\n--- [{label}] Q: {question} ---")
    answer = repl.process(question)
    log(f"A: {answer}")
    return answer


def main():
    log("=" * 72)
    log("LIVE-CAUSAL cut/append demo v2 (Phase 3, hardened) --")
    log("capital of France, three times, FULL repl.process() answers")
    log("=" * 72)

    probe_graph = LiveGraph(STORE_DIR)
    edges = probe_graph.query('france')
    france_paris_shas = sorted({
        sha for e in edges if e['to_key'] == 'paris' for sha, _idx in e['derivation']
    })
    log(f"\nSegments citing france->paris (to be cut): {france_paris_shas}")
    log(f"Total segments in store before cut: {len(probe_graph.store.segments())}")

    saved_records_by_sha = {
        sha: [rec for _s, _i, rec in probe_graph.store.iter_records(sha)]
        for sha in france_paris_shas
    }

    log("\n" + "=" * 72)
    log("Booting ONE FossKIRepl(live_causal_store=..., knowledge_only=True)")
    log("-- reused for all 3 steps, no rebuild")
    log("=" * 72)
    repl = FossKIRepl(live_causal_store=STORE_DIR, knowledge_only=True)
    repl.show_trace = True
    log(f"using_live_causal = {repl.using_live_causal}")
    log(f"knowledge_only    = {repl.knowledge_only}")

    # ------------------------------------------------------------------
    # Step (i): baseline
    # ------------------------------------------------------------------
    log("\n" + "=" * 72)
    log("STEP (i): baseline, all segments present")
    log("=" * 72)
    a1 = ask(repl, "what is the capital of France?", "i: baseline")
    control1 = ask(repl, "who wrote Hamlet?", "i: control (unrelated fact)")

    # ------------------------------------------------------------------
    # Step (ii): cut the France/capital/Paris segment(s)
    # ------------------------------------------------------------------
    log("\n" + "=" * 72)
    log("STEP (ii): drop_segments() on the France/capital citing segments")
    log("           -- via repl.knowledge.drop_segments, NOT a rebuild.")
    log("=" * 72)
    repl.knowledge.drop_segments(france_paris_shas)
    repl.knowledge._facts_cache = None
    log(f"Segments remaining in store: {len(repl.knowledge.segments())}")

    a2 = ask(repl, "what is the capital of France?", "ii: after cut (FULL repl.process)")
    control2 = ask(repl, "who wrote Hamlet?", "ii: control (unrelated fact)")

    # ------------------------------------------------------------------
    # Step (iii): append the segments back
    # ------------------------------------------------------------------
    log("\n" + "=" * 72)
    log("STEP (iii): re-append the same records (content-addressed,")
    log("            same sha256 as the originals) -- NOT a rebuild.")
    log("=" * 72)
    restored_shas = []
    for sha in france_paris_shas:
        records = saved_records_by_sha[sha]
        new_sha = repl.knowledge.append_segment(records)
        restored_shas.append(new_sha)
    repl.knowledge._facts_cache = None
    log(f"Re-appended segment shas: {restored_shas}")
    log(f"Original shas matched (content-addressed, same bytes): "
        f"{sorted(restored_shas) == sorted(france_paris_shas)}")
    log(f"Segments in store after re-append: {len(repl.knowledge.segments())}")

    a3 = ask(repl, "what is the capital of France?", "iii: after re-append")
    control3 = ask(repl, "who wrote Hamlet?", "iii: control (unrelated fact)")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    log("\n" + "=" * 72)
    log("VERDICT")
    log("=" * 72)
    checks = {
        "(i) baseline FULL repl answer contains Paris": "Paris" in a1,
        "(ii) FULL repl answer after cut is honest IDK "
        "(not a hallucination, not another layer's guess)":
            "don't have information" in a2.lower(),
        "(ii) FULL repl answer after cut does NOT contain Paris":
            "Paris" not in a2,
        "(ii) control question (Hamlet) unaffected by the cut":
            "Shakespeare" in control2,
        "(iii) FULL repl answer after re-append contains Paris again":
            "Paris" in a3,
        "(iii) re-appended sha == original sha (content-addressed)":
            sorted(restored_shas) == sorted(france_paris_shas),
        "control question identical across all 3 steps":
            control1 == control2 == control3,
    }
    all_pass = True
    for desc, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        log(f"  [{status}] {desc}")
    log(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")

    log("\nWhat changed vs. the Phase 2 demo: knowledge_only=True disables")
    log("repl.py's OTHER independent fact sources (ConceptNet/CommonSense,")
    log("CBR case library, MultiHop, Web search) for the duration of this")
    log("repl instance, so the FULL repl.process() answer -- not just an")
    log("isolated adapter.query() call -- now honestly reflects the cut.")
    log("Reasoning/Math (_solve_reasoning, formulas, ReasoningEngine) stay")
    log("on throughout: they compute over self.knowledge (the adapter)")
    log("rather than bypassing it, so disabling them would prove nothing")
    log("about forgetting.")

    with open(TRANSCRIPT_PATH, 'w') as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nTranscript written to {TRANSCRIPT_PATH}")

    return all_pass


if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)
