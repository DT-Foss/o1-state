"""
LIVE-CAUSAL cut/append demo, Phase 4 (multi-source forgetting) --
revival-probe Task 14.

Phase 3 proved forgetting for a fact stored in ONE source
(knowledge_full.json, converted 1:1). This version uses a store built
from BOTH knowledge_full.json AND a ConceptNet slice, so the France/
Paris relationship is represented TWICE, by two independently-converted
sources, in two different directions:

  - knowledge_full.json: ("France", "capital", "Paris") -- forward,
    trigger_key=france, outcome_key=paris (2 duplicate records, same
    fact asserted twice in the source file itself)
  - ConceptNet (AtLocation): ("paris", "AtLocation", "france") --
    REVERSE direction, trigger_key=paris, outcome_key=france

A cut that only targets the France->Paris direction (the Phase 2/3
demo's hand-written approach) would leave the ConceptNet-derived
Paris->France citation, and everything inferred from it, fully intact
-- an incomplete forgetting, not a bug in the adapter, but a real
consequence of facts being stored in more than one direction by more
than one source.

This demo uses LiveCausalAdapter.find_segments_citing(subject, obj) in
BOTH directions to build the complete citation set before cutting,
and shows that cutting only one direction is NOT enough -- then that
cutting both directions IS enough, then append restores everything.
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

STORE_DIR = '/root/fosski-venv/merged_store_demo'
TRANSCRIPT_PATH = '/root/fosski-venv/cut_append_transcript_v3.txt'

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
    log("LIVE-CAUSAL cut/append demo v3 (Phase 4, multi-source forgetting)")
    log("Two independent sources both assert the France<->Paris relationship")
    log("=" * 72)

    probe_graph = LiveGraph(STORE_DIR)
    log(f"\nTotal segments in merged store (knowledge_full.json + ConceptNet slice): "
        f"{len(probe_graph.store.segments())}")

    fwd_edges = probe_graph.query('france')
    fwd_shas = sorted({sha for e in fwd_edges if e['to_key'] == 'paris'
                        for sha, _idx in e['derivation']})
    rev_edges = probe_graph.query('paris')
    rev_shas = sorted({sha for e in rev_edges if e['to_key'] == 'france'
                        for sha, _idx in e['derivation']})
    log(f"\nfind_segments_citing('france', 'paris') [forward, knowledge_full.json]: "
        f"{len(fwd_shas)} segments")
    log(f"find_segments_citing('paris', 'france') [reverse, ConceptNet AtLocation]: "
        f"{len(rev_shas)} segments")
    all_shas = sorted(set(fwd_shas) | set(rev_shas))
    log(f"Union (the complete evidence set for this relationship): {len(all_shas)} segments")

    saved_records_by_sha = {
        sha: [rec for _s, _i, rec in probe_graph.store.iter_records(sha)]
        for sha in all_shas
    }

    log("\n" + "=" * 72)
    log("Booting ONE FossKIRepl(live_causal_store=..., knowledge_only=True)")
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
    control1 = ask(repl, "who wrote Hamlet?", "i: control")

    # ------------------------------------------------------------------
    # Step (ii-partial): cut ONLY the forward direction -- shows this is
    # NOT enough by itself, the honest negative result this demo needs.
    # ------------------------------------------------------------------
    log("\n" + "=" * 72)
    log("STEP (ii-partial): cut ONLY the forward (knowledge_full.json)")
    log("  citations -- checking the adapter directly (not the full repl,")
    log("  which still has other fast-path solvers that could mask this)")
    log("=" * 72)
    repl.knowledge.drop_segments(fwd_shas)
    repl.knowledge._facts_cache = None
    partial_query = repl.knowledge.query(subject='France', relation='capital')
    reverse_still_present = repl.knowledge.query(subject='paris')
    log(f"Direct query('France', 'capital') after PARTIAL cut: {partial_query}")
    log(f"Direct query('paris') after PARTIAL cut (reverse ConceptNet "
        f"citation untouched): {reverse_still_present}")

    # ------------------------------------------------------------------
    # Step (ii-full): also cut the reverse (ConceptNet) direction.
    # ------------------------------------------------------------------
    log("\n" + "=" * 72)
    log("STEP (ii-full): ALSO cut the reverse (ConceptNet AtLocation)")
    log("  citations -- now the FULL repl.process() answer must forget.")
    log("=" * 72)
    repl.knowledge.drop_segments(rev_shas)
    repl.knowledge._facts_cache = None
    log(f"Segments remaining in store: {len(repl.knowledge.segments())}")

    a2 = ask(repl, "what is the capital of France?", "ii-full: after BOTH directions cut")
    control2 = ask(repl, "who wrote Hamlet?", "ii-full: control")

    # ------------------------------------------------------------------
    # Step (iii): append everything back
    # ------------------------------------------------------------------
    log("\n" + "=" * 72)
    log("STEP (iii): re-append all segments from both sources")
    log("=" * 72)
    restored_shas = []
    for sha in all_shas:
        records = saved_records_by_sha[sha]
        new_sha = repl.knowledge.append_segment(records)
        restored_shas.append(new_sha)
    repl.knowledge._facts_cache = None
    log(f"Re-appended {len(restored_shas)} segments")
    log(f"All shas content-addressed identical to originals: "
        f"{sorted(restored_shas) == sorted(all_shas)}")

    a3 = ask(repl, "what is the capital of France?", "iii: after re-append")
    control3 = ask(repl, "who wrote Hamlet?", "iii: control")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    log("\n" + "=" * 72)
    log("VERDICT")
    log("=" * 72)
    checks = {
        "(i) baseline answers Paris": "Paris" in a1,
        "(ii-partial) cutting ONLY the forward direction is NOT enough "
        "-- the reverse ConceptNet citation is still queryable "
        "(honest negative result, not a failure of the tool)":
            reverse_still_present.get('confidence_level') != 'REJECTED',
        "(ii-partial) forward direction itself IS rejected after its own cut":
            partial_query['confidence_level'] == 'REJECTED',
        "(ii-full) FULL repl answer after cutting BOTH directions is "
        "honest IDK": "don't have information" in a2.lower(),
        "(ii-full) FULL repl answer does NOT contain Paris": "Paris" not in a2,
        "(ii-full) control question unaffected": "Shakespeare" in control2,
        "(iii) FULL repl answer after full re-append contains Paris again":
            "Paris" in a3,
        "(iii) re-appended shas == original shas (content-addressed)":
            sorted(restored_shas) == sorted(all_shas),
        "control question identical across all 3 main steps":
            control1 == control2 == control3,
    }
    all_pass = True
    for desc, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        log(f"  [{status}] {desc}")
    log(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")

    log("\nThe story: forgetting means cutting ALL evidence segments for a")
    log("relationship, not just the ones from the source you happen to be")
    log("thinking about. find_segments_citing(subject, obj) lets a caller")
    log("find every segment citing a given directed edge; a caller building")
    log("a real 'forget this relationship' feature must query BOTH")
    log("(subject, obj) and (obj, subject) and union the results, because")
    log("different converted sources can and do encode the same real-world")
    log("relationship in different directions with different mechanism labels.")

    with open(TRANSCRIPT_PATH, 'w') as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nTranscript written to {TRANSCRIPT_PATH}")

    return all_pass


if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)
