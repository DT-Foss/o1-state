"""
LIVE-CAUSAL end-to-end loop demo -- revival-probe Phase 5 (Task 15).

The full stack, once, live:

  fresh text (a file) -> builder_run.py (organism streams windows, fabel's
  deterministic curator_yield_run.extract_validated validates causal
  triplets, LiveGraph.append_segment folds them in) -> LiveCausalAdapter
  mounts the resulting store -> FossKIRepl answers a question about the
  text IMMEDIATELY (same run, no separate conversion step -- builder_run.py
  writes directly into a LiveStore/LiveGraph) -> the article's segments are
  cut -> FOSS-KI forgets exactly that content, control question unaffected.

Honest scope note (documented in DEMO.md, not hidden): the fabel
extractor's real output is causal-mechanism triplets ("X causes Y", "X
leads to Y"), not attribute-style facts ("X is the capital of Y"). None
of repl.py's existing natural-language query patterns
(core/router.py's _parse_query_for_knowledge) covered "what causes X" /
"what does X cause" against self.knowledge before this demo -- a real,
traced gap between the fabel extractor's output shape and repl.py's
pre-existing query surface, not a workaround. This demo's Phase 5 change
adds that one reverse/forward causal-mechanism lookup to
_direct_kb_lookup (returning a full sentence, "{subject} causes
{outcome}", specifically because a bare short outcome like "cancer" can
fail _answer_quality_gate's topic-overlap check when it happens to share
no content words with the question and isn't capitalized as a proper
noun -- embedding the subject guarantees overlap without touching the
gate itself, a mechanism every OTHER caller of _direct_kb_lookup also
depends on).
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, '/root/mac_offload/desktop/foss-ki')
sys.path.insert(0, '/root/fosski-venv/adapter')
sys.path.insert(0, '/root')
os.chdir('/root/mac_offload/desktop/foss-ki')

from repl import FossKIRepl  # noqa: E402
from livecausal_bridge.infer import LiveGraph  # noqa: E402

ARTICLE_PATH = '/root/fosski-venv/marie_curie_article.txt'
STORE_DIR = '/root/fosski-venv/e2e_loop_store'
TRANSCRIPT_PATH = '/root/fosski-venv/e2e_loop_transcript.txt'

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
    log("LIVE-CAUSAL end-to-end loop demo -- world text -> live segments -> answer -> forget")
    log("=" * 72)

    if os.path.exists(STORE_DIR):
        import shutil
        shutil.rmtree(STORE_DIR)

    log(f"\nSource article: {ARTICLE_PATH}")
    with open(ARTICLE_PATH) as f:
        article_text = f.read()
    log(f"({len(article_text)} chars, {article_text.count(chr(10))} lines -- a fixed local file, no web scraping)")

    # ------------------------------------------------------------------
    # Step 0: pre-seed the SAME store with knowledge_full.json (Phase 2's
    # converter) BEFORE the organism ever touches it -- this is what a
    # real deployment looks like: a base knowledge store already exists,
    # then a fresh article gets folded in live on top of it. Also gives
    # this demo an honest, non-article control fact ("who wrote Hamlet?")
    # that must survive cutting ONLY the article's segments.
    # ------------------------------------------------------------------
    log("\n" + "=" * 72)
    log("STEP 0: pre-seed the store with knowledge_full.json (base knowledge, already there)")
    log("=" * 72)
    sys.path.insert(0, '/root/fosski-venv')
    from convert_knowledge_full import convert as convert_kb  # noqa: E402
    kb_result = convert_kb(
        "/root/mac_offload/desktop/foss-ki/data/knowledge_full.json", STORE_DIR)
    log(f"Pre-seeded {kb_result['n_records_converted']} knowledge_full.json records "
        f"({len(kb_result['segment_shas'])} segments) into {STORE_DIR}")
    pre_existing_segments = set(kb_result["segment_shas"])

    # ------------------------------------------------------------------
    # Step 1: the organism + fabel build the store LIVE from the article,
    # appending onto the pre-seeded store above. Run as a subprocess
    # exactly the way a human operator would invoke it -- no in-process
    # shortcuts, this is the real builder_run.py CLI.
    # ------------------------------------------------------------------
    log("\n" + "=" * 72)
    log("STEP 1: builder_run.py streams the article, fabel extracts, folds into LiveStore")
    log("=" * 72)
    cmd = [
        "nice", "-n", "19", "env",
        "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        "python3", "/root/o1lab/src/livecausal/builder_run.py",
        "--text-file", ARTICLE_PATH,
        "--store-dir", STORE_DIR,
        "--out-prefix", "/root/fosski-venv/e2e_loop_build",
        "--window-tokens", "6",
        "--max-windows", "60",
        "--tag", "fosski_e2e_loop",
    ]
    log(f"$ {' '.join(cmd)}")
    env = dict(os.environ)
    env["PYTHONPATH"] = "/root/o1lab/reference:/root/o1lab/vendor/fabel/extract"
    result = subprocess.run(cmd, cwd="/root/o1lab", env=env,
                             capture_output=True, text=True)
    for line in result.stdout.strip().splitlines():
        log(f"  {line}")
    if result.returncode != 0:
        log(f"BUILDER FAILED (exit {result.returncode}):")
        log(result.stderr[-2000:])
        return False

    probe_graph = LiveGraph(STORE_DIR)
    all_segments_after_build = set(probe_graph.store.segments())
    article_segments = sorted(all_segments_after_build - pre_existing_segments)
    log(f"\nTotal segments in store: {len(all_segments_after_build)} "
        f"({len(pre_existing_segments)} pre-seeded + {len(article_segments)} from the article)")
    log(f"Base edges: {sum(len(v) for v in probe_graph._base_edges.values())}")

    # ------------------------------------------------------------------
    # Step 2: FOSS-KI answers a question about the article IMMEDIATELY --
    # same process, the store the builder just wrote is mounted fresh.
    # ------------------------------------------------------------------
    log("\n" + "=" * 72)
    log("STEP 2: FossKIRepl mounts the just-built store, answers a question about the article")
    log("=" * 72)
    repl = FossKIRepl(live_causal_store=STORE_DIR, knowledge_only=True)
    repl.show_trace = True
    log(f"using_live_causal = {repl.using_live_causal}")
    log(f"knowledge_only    = {repl.knowledge_only}")

    a1 = ask(repl, "what does high doses of radiation cause?", "immediate answer about the article")
    control1 = ask(repl, "who wrote Hamlet?", "control (unrelated, from knowledge_full.json)")

    # ------------------------------------------------------------------
    # Step 3: cut every segment from the article, verify forgetting.
    # ------------------------------------------------------------------
    log("\n" + "=" * 72)
    log("STEP 3: cut ONLY the article's segments -- the whole ingested text, not")
    log("        one fact, and NOT the pre-seeded knowledge_full.json base")
    log("=" * 72)
    log(f"Dropping {len(article_segments)} article-derived segments via drop_segments()")
    log(f"({len(pre_existing_segments)} pre-seeded knowledge_full.json segments untouched)")
    repl.knowledge.drop_segments(article_segments)
    repl.knowledge._facts_cache = None
    log(f"Segments remaining in store: {len(repl.knowledge.segments())} "
        f"(should equal the {len(pre_existing_segments)} pre-seeded segments)")

    a2 = ask(repl, "what does high doses of radiation cause?", "after cutting the article")
    control2 = ask(repl, "who wrote Hamlet?", "control")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    log("\n" + "=" * 72)
    log("VERDICT")
    log("=" * 72)
    checks = {
        "step 1: builder_run.py exited successfully": result.returncode == 0,
        "step 1: at least one article segment was built": len(article_segments) > 0,
        "step 2: immediate answer mentions cancer (the article's fact)":
            "cancer" in a1.lower(),
        "step 2: control question answers Shakespeare (from the pre-seeded base)":
            "Shakespeare" in control1,
        "step 3: after cutting, the article fact is honestly forgotten":
            "don't have information" in a2.lower(),
        "step 3: after cutting, the answer does NOT still say cancer":
            "cancer" not in a2.lower(),
        "step 3: control question unaffected by the cut": "Shakespeare" in control2,
        "control question identical before and after the cut": control1 == control2,
    }
    all_pass = True
    for desc, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        log(f"  [{status}] {desc}")
    log(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")

    log("\nThe full stack, once, live: a text file was streamed by the organism,")
    log("fabel's deterministic 14-step extractor validated causal triplets from")
    log("it with no LLM anywhere, LiveStore sealed them into content-addressed")
    log("segments, LiveCausalAdapter mounted that store and FOSS-KI answered a")
    log("question about the article in the SAME process that just built it --")
    log("then forgot the entire article, on command, leaving everything else")
    log("(knowledge_full.json's facts) untouched.")

    with open(TRANSCRIPT_PATH, 'w') as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nTranscript written to {TRANSCRIPT_PATH}")

    return all_pass


if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)
