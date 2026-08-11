"""
LIVE-CAUSAL souveraenitaets-probe -- Task 19 (revival-probe).

Boots FossKIRepl TWICE (Qwen embeddings, then A3-sovereign embeddings)
against the SAME default KnowledgeStore path (not the LiveCausalAdapter
-- the mapping found the adapter is structurally incompatible with the
Foss Pipeline regardless of which embedding store backs it: _kb_retrieve
reads self.knowledge._entity_subject_index, an attribute
LiveCausalAdapter deliberately does not implement -- so this is the only
path where the Reservoir/Attention/Hopfield similarity tier can even run
long enough to compare).

Question set:
  (i)   The DirectKB/adapter-shaped standard set (France/Hamlet/ice/
        97-prime/Narnia/radiation-style) -- these are EXPECTED to be
        identical under both embedding backends, because the mapping
        found DirectKB/_solve_reasoning/_solve_compositional/
        _answer_quality_gate never touch self.emb_store at all. Included
        here as a control: if these differ, that is itself a finding
        (something the mapping missed), not an expected result.
  (ii)  Three probes confirmed by direct testing to route through the
        Foss Pipeline's Reservoir+Attention+Hopfield similarity tier
        (nonsense-noun + real-verb phrasing defeats every structured
        solver and CommonSense, forcing the generative fallback) -- this
        is where a Qwen vs. A3 difference is actually measurable.
"""
import json
import os
import sys

sys.path.insert(0, '/root/mac_offload/desktop/foss-ki')
os.chdir('/root/mac_offload/desktop/foss-ki')
os.environ.setdefault('PYTHONPATH', '/root/o1lab/reference:/root/o1lab/src')
sys.path.insert(0, '/root/o1lab/reference')
sys.path.insert(0, '/root/o1lab/src')

from repl import FossKIRepl  # noqa: E402

TRANSCRIPT_PATH = '/root/fosski-venv/sovereignty_transcript.txt'

CONTROL_QUESTIONS = [
    "what is the capital of France?",
    "who wrote Hamlet?",
    "why does ice float?",
    "is 97 prime?",
    "what is the capital of Narnia?",
]

PIPELINE_PROBES = [
    "morvath enables",
    "the kelvithorn approaches",
    "protivex generates",
]

lines = []


def log(s=""):
    print(s)
    lines.append(s)


def ask_with_trace(repl, question):
    repl.show_trace = True
    full = repl.process(question)
    if "\n\n[Trace]\n" in full:
        answer, trace_block = full.split("\n\n[Trace]\n", 1)
    else:
        answer, trace_block = full, ""
    winner_line = None
    for line in trace_block.splitlines():
        if 'Winner:' in line:
            winner_line = line.strip()
    method = '?'
    if winner_line and "method='" in winner_line:
        method = winner_line.split("method='")[1].split("'")[0]
    return answer.strip(), method


def main():
    log("=" * 72)
    log("SOUVERAENITAETS-PROBE -- Qwen vs. A3-sovereign embeddings")
    log("=" * 72)

    log("\n" + "=" * 72)
    log("Booting Qwen-backed FossKIRepl (default, unchanged)")
    log("=" * 72)
    repl_qwen = FossKIRepl()
    log(f"emb_store.dim = {repl_qwen.emb_store.dim}")

    qwen_results = {}
    for q in CONTROL_QUESTIONS + PIPELINE_PROBES:
        answer, method = ask_with_trace(repl_qwen, q)
        qwen_results[q] = (answer, method)
        log(f"\n[QWEN] Q: {q}")
        log(f"  A: {answer}")
        log(f"  method: {method}")

    del repl_qwen

    log("\n" + "=" * 72)
    log("Booting A3-sovereign FossKIRepl (sovereign_embeddings=True)")
    log("=" * 72)
    repl_a3 = FossKIRepl(sovereign_embeddings=True)
    log(f"emb_store.dim = {repl_a3.emb_store.dim}")

    a3_results = {}
    for q in CONTROL_QUESTIONS + PIPELINE_PROBES:
        answer, method = ask_with_trace(repl_a3, q)
        a3_results[q] = (answer, method)
        log(f"\n[A3] Q: {q}")
        log(f"  A: {answer}")
        log(f"  method: {method}")

    log("\n" + "=" * 72)
    log("SIDE-BY-SIDE COMPARISON")
    log("=" * 72)
    log(f"\n{'Question':<40} {'Qwen answer':<25} {'Qwen method':<15} {'A3 answer':<25} {'A3 method':<15} Status")
    log("-" * 145)
    control_identical = True
    for q in CONTROL_QUESTIONS:
        qa, qm = qwen_results[q]
        aa, am = a3_results[q]
        status = "IDENTICAL" if qa == aa else "DIFFERENT"
        if qa != aa:
            control_identical = False
        log(f"{q[:38]:<40} {qa[:23]:<25} {qm:<15} {aa[:23]:<25} {am:<15} {status}")

    log("")
    for q in PIPELINE_PROBES:
        qa, qm = qwen_results[q]
        aa, am = a3_results[q]
        status = "IDENTICAL" if qa == aa else "DIFFERENT"
        log(f"{q[:38]:<40} {qa[:23]:<25} {qm:<15} {aa[:23]:<25} {am:<15} {status}")

    pipeline_ran_both = all(
        qwen_results[q][1] == 'foss_pipeline' and a3_results[q][1] == 'foss_pipeline'
        for q in PIPELINE_PROBES
    )

    log("\n" + "=" * 72)
    log("VERDICT")
    log("=" * 72)
    checks = {
        "Control questions (DirectKB/adapter-shaped) are byte-identical "
        "under both embedding backends -- confirms the mapping's finding "
        "that this path never touches self.emb_store":
            control_identical,
        "All 3 pipeline probes actually routed through method='foss_pipeline' "
        "under BOTH backends (the test measures what it claims to measure)":
            pipeline_ran_both,
    }
    all_pass = True
    for desc, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        log(f"  [{status}] {desc}")
    log(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")

    with open(TRANSCRIPT_PATH, 'w') as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nTranscript written to {TRANSCRIPT_PATH}")

    return all_pass


if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)
