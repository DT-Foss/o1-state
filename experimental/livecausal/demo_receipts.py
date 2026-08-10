"""
LIVE-CAUSAL receipts demo -- revival-probe Task 16.

Proves the receipt shown in a FOSS-KI trace is a REAL, checkable citation,
not a decorative confidence number: ask a question, see the exact
(segment_sha[:12], idx) citations behind the answer, cut ONE of those
cited segments, ask again -- if the receipt was real, the answer must
either change (fewer surviving citations) or the fact must disappear
entirely once ALL its citations are gone. This demo does both:

  PART A -- evidence_count proof: the France/capital/Paris fact has TWO
  citing segments (both from knowledge_full.json's own duplicate entry).
  Cut ONE -> the receipt's evidence_count drops from 2 to 1, the answer
  still holds (one citation survives) -- proving the receipt tracked a
  REAL, separately-cuttable citation, not a static label. Cut the SECOND
  -> the fact is gone entirely, receipt-based prediction confirmed.

  PART B -- contested proof: the Coffee/causes/alertness vs.
  Coffee/correlates_with/alertness conflict (3 "causes" sources vs 1
  "correlates_with" source). The trace shows "Contested: 3:1". Cut TWO of
  the three "causes" segments -> the ratio must flip to correctly reflect
  the new count (now 1:1, no longer contested-with-a-clear-winner) --
  proving the contested ratio is computed live from the current store,
  not cached or hardcoded.
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

FRANCE_STORE = '/root/fosski-venv/receipts_france_store'
CONTESTED_STORE = '/root/fosski-venv/receipts_contested_store'
TRANSCRIPT_PATH = '/root/fosski-venv/receipts_transcript.txt'

lines = []


def log(s=""):
    print(s)
    lines.append(s)


def ask(repl, question, label):
    log(f"\n--- [{label}] Q: {question} ---")
    answer = repl.process(question)
    log(f"A: {answer}")
    return answer


def extract_receipt(answer_text):
    """Pull the Receipt/Contested lines out of a show_trace answer string."""
    receipt_line = None
    contested_line = None
    for line in answer_text.splitlines():
        if 'Receipt:' in line:
            receipt_line = line.strip()
        if 'Contested:' in line:
            contested_line = line.strip()
    return receipt_line, contested_line


def main():
    all_pass = True
    checks = {}

    # ==================================================================
    # PART A: evidence_count receipt proof (France/capital/Paris)
    # ==================================================================
    log("=" * 72)
    log("PART A -- evidence_count receipt proof (France/capital/Paris)")
    log("=" * 72)

    if os.path.exists(FRANCE_STORE):
        import shutil
        shutil.rmtree(FRANCE_STORE)
    import shutil
    shutil.copytree('/root/fosski-venv/livecausal_store', FRANCE_STORE)

    probe = LiveGraph(FRANCE_STORE)
    edges = probe.query('france')
    france_paris_shas = sorted({
        sha for e in edges if e['to_key'] == 'paris' for sha, _idx in e['derivation']
    })
    log(f"\nSegments citing france->paris: {france_paris_shas}")
    assert len(france_paris_shas) == 2, "expected exactly 2 citing segments"

    repl_a = FossKIRepl(live_causal_store=FRANCE_STORE, knowledge_only=True)
    repl_a.show_trace = True

    a1 = ask(repl_a, "what is the capital of France?", "A1: baseline, 2 citations")
    receipt1, _ = extract_receipt(a1)
    log(f"Receipt: {receipt1}")
    checks["A1: baseline receipt shows evidence_count=2"] = (
        receipt1 is not None and "evidence_count=2" in receipt1)

    # Cut ONE of the two citing segments.
    log(f"\nCutting ONE segment: {france_paris_shas[0]}")
    repl_a.knowledge.drop_segments([france_paris_shas[0]])
    repl_a.knowledge._facts_cache = None

    a2 = ask(repl_a, "what is the capital of France?", "A2: after cutting 1 of 2 citations")
    receipt2, _ = extract_receipt(a2)
    log(f"Receipt: {receipt2}")
    checks["A2: answer still holds (1 citation survives)"] = "Paris" in a2
    checks["A2: receipt now shows evidence_count=1"] = (
        receipt2 is not None and "evidence_count=1" in receipt2)
    checks["A2: receipt no longer cites the cut segment"] = (
        receipt2 is not None and france_paris_shas[0][:12] not in receipt2)

    # Cut the SECOND (last) citing segment -- fact must disappear entirely.
    log(f"\nCutting the SECOND (last) segment: {france_paris_shas[1]}")
    repl_a.knowledge.drop_segments([france_paris_shas[1]])
    repl_a.knowledge._facts_cache = None

    a3 = ask(repl_a, "what is the capital of France?", "A3: after cutting both citations")
    checks["A3: fact fully forgotten once ALL citations are cut"] = (
        "don't have information" in a3.lower())

    # ==================================================================
    # PART B: contested-ratio receipt proof (Coffee/causes vs correlates_with)
    # ==================================================================
    log("\n" + "=" * 72)
    log("PART B -- contested-ratio receipt proof (Coffee: causes vs correlates_with)")
    log("=" * 72)

    if os.path.exists(CONTESTED_STORE):
        import shutil
        shutil.rmtree(CONTESTED_STORE)
    shutil.copytree('/root/fosski-venv/contested_store', CONTESTED_STORE)

    probe_b = LiveGraph(CONTESTED_STORE)
    edges_b = probe_b.query('coffee')
    causes_shas = []
    correlates_shas = []
    for sha in probe_b.store.segments():
        for _s, idx, rec in probe_b.store.iter_records(sha):
            if rec['mechanism'] == 'causes':
                causes_shas.append(sha)
            elif rec['mechanism'] == 'correlates_with':
                correlates_shas.append(sha)
    causes_shas = sorted(set(causes_shas))
    correlates_shas = sorted(set(correlates_shas))
    log(f"\n'causes' segments (3 expected): {causes_shas}")
    log(f"'correlates_with' segments (1 expected): {correlates_shas}")

    repl_b = FossKIRepl(live_causal_store=CONTESTED_STORE, knowledge_only=True)
    repl_b.show_trace = True

    b1 = ask(repl_b, "what does Coffee cause?", "B1: baseline, 3:1 contested")
    _, contested1 = extract_receipt(b1)
    log(f"Contested: {contested1}")
    checks["B1: baseline contested ratio is 3:1"] = (
        contested1 is not None and "3:1" in contested1)

    # Cut TWO of the three "causes" segments -- ratio must flip to 1:1.
    to_cut = causes_shas[:2]
    log(f"\nCutting 2 of the 3 'causes' segments: {to_cut}")
    repl_b.knowledge.drop_segments(to_cut)
    repl_b.knowledge._facts_cache = None

    b2 = ask(repl_b, "what does Coffee cause?", "B2: after cutting 2 of 3 causes sources")
    _, contested2 = extract_receipt(b2)
    log(f"Contested: {contested2}")
    checks["B2: contested ratio recomputed live, now 1:1"] = (
        contested2 is not None and "1:1" in contested2)
    checks["B2: ratio is NOT still 3:1 (would prove it's cached/hardcoded)"] = (
        contested2 is not None and "3:1" not in contested2)

    # ==================================================================
    # Verdict
    # ==================================================================
    log("\n" + "=" * 72)
    log("VERDICT")
    log("=" * 72)
    for desc, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        log(f"  [{status}] {desc}")
    log(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")

    log("\nBoth receipts (evidence_count citations and the contested ratio) are")
    log("computed live from the current store's actual segments on every")
    log("query() call -- cutting a cited segment changes the receipt's own")
    log("numbers in exactly the way the citation predicts, proving the")
    log("receipt is a real, checkable pointer into the store, not a")
    log("decorative confidence label.")

    with open(TRANSCRIPT_PATH, 'w') as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nTranscript written to {TRANSCRIPT_PATH}")

    return all_pass


if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)
