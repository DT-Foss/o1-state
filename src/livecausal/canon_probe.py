#!/usr/bin/env python3 -u
"""CANON CROSS-VERSION PROBE — the P70 risk, measured for canon.py.

P70's own lesson (recorded in analysis/PREDICTIONS.md and mirrored in
curator_yield_run.py's İstanbul fix): "same code path, DIFFERENT
behavior depending on what's installed on the host" is not a hypothetical
for anything that touches spaCy or Python's own Unicode case-folding.
canon.py's determinism CLAIM is scoped to "same env_pin" (see its module
docstring) -- this script is what turns that scope into a MEASUREMENT
instead of an assumption, across the two machines P75 actually runs on
(local: spaCy 3.8.11; the x86 runner: spaCy 3.8.15).

Two modes:

  dump --out FILE
      Runs canonical_key over a fixed, deterministic probe list (module-
      level PROBES below -- no RNG, so the SAME list runs on every
      machine) and writes {"env_pin": ..., "probes": [{"raw", "canon"}]}
      as JSON to FILE.

  compare --a FILE --b FILE [--out FILE]
      Loads two dumps (typically one from this machine, one shipped back
      from the x86 runner) and reports: n_total, n_identical, and the
      full list of divergences (raw phrase, canon_a, canon_b, both
      env_pins). ALWAYS exits 0 -- this is a report, not a gate (the
      scorer philosophy this repo already uses elsewhere: a divergence
      is a measured fact that P75's registration reacts to, not a reason
      for this script itself to fail a CI-style check).

Usage:
  python3 src/livecausal/canon_probe.py dump --out probe_local.json
  python3 src/livecausal/canon_probe.py compare --a probe_local.json --b probe_x86.json
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from livecausal.canon import canonicalize_with_default_nlp, env_pin  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
#  Probe list -- FIXED, module-level, no RNG. Every machine that runs
#  `dump` runs the exact same list, so any output divergence is entirely
#  attributable to the (spaCy version, model version, canon_version)
#  triple, never to sampling. Categories mirror the build brief:
# ─────────────────────────────────────────────────────────────────────────

# --- 1. Determiner / adjective / PP patterns from test_canon.py's own
#        worked examples, plus the systematic determiner x noun-phrase
#        cross-product that generates the connectivity mechanic's join
#        cases (a king/war/famine-shaped set, deterministic, no RNG). ---
_DETERMINERS = ["the", "a", "an", "the old", "the resulting", "the ensuing",
                 "that terrible", "this severe", "another", "some"]
_ADJECTIVES = ["", "severe ", "rising ", "global ", "sharp ", "sudden ",
               "growing ", "unexpected "]
_HEAD_NOUNS = ["king", "war", "famine", "economy", "downturn", "inflation",
               "recession", "crisis", "market", "harvest"]
_PP_TAILS = ["", " of france", " in unemployment", " about the weather",
             " between nations", " of the empire"]


def _det_adj_pp_probes():
    """Deterministic cross-product: every (determiner, adjective, head
    noun) combination, plus a subset crossed with PP tails (full 10 x 8 x
    10 x 6 product would be 4,800 phrases -- far more than the ~200
    minimum the build brief asks for; PP tails are applied only to the
    bare-determiner, no-adjective row per head noun to keep the list at a
    deliberately chosen size rather than exploding combinatorially)."""
    probes = []
    for det in _DETERMINERS:
        for adj in _ADJECTIVES:
            for noun in _HEAD_NOUNS:
                probes.append("{} {}{}".format(det, adj, noun).replace("  ", " "))
    for noun in _HEAD_NOUNS:
        for tail in _PP_TAILS:
            probes.append("the {}{}".format(noun, tail))
            probes.append("a {}{}".format(noun, tail))
    return probes


# --- 2. Verb-fragment phrases -- no noun chunk at all in some cases,
#        forcing the fallback path; others have a trailing noun object
#        that DOES have a chunk, testing the boundary directly. ---
_VERB_FRAGMENTS = [
    "was published", "has increased sharply", "quickly disappeared",
    "fell rapidly", "grew without warning", "collapsed overnight",
    "caused widespread damage", "led to further unrest",
    "triggered a chain reaction", "resulted in mass casualties",
    "was widely reported", "continues to worsen",
]

# --- 3. Symbol strings / degenerate inputs -- adversarial-ish, must not
#        crash canonical_key, must resolve via the deterministic
#        fallback path (see canon.py's _surface_fallback). ---
_SYMBOL_STRINGS = [
    "xk7j2q9", "123", "!!!", "", "   ", "\t\n", "---", "a1b2c3",
    "N/A", "TBD", "###header###", "42", "0.001", "$$$",
    "king" * 20,  # pathologically long single "word"
]

# --- 4. Unicode edge cases -- the P70 İstanbul lesson (U+0130 LATIN
#        CAPITAL LETTER I WITH DOT ABOVE, .lower() is NOT length-
#        preserving: 1 codepoint -> 2, 'i' + U+0307 COMBINING DOT ABOVE)
#        applied directly to canon.py's own fallback path (which calls
#        .lower()) and to spaCy's tokenizer/lemmatizer (version-
#        sensitive Turkish-I handling is a documented spaCy edge case).
#        Plus a handful of OTHER classic case-folding/normalization
#        traps in the same family (German ß, combining diacritics
#        entered in both precomposed and decomposed form, full-width
#        Latin, RTL/Arabic presence as a "does it crash" check). ---
_UNICODE_PROBES = [
    "İstanbul",                       # U+0130, the P70 crash case itself
    "the İstanbul uprising",          # same case, embedded in a phrase
    "İ",                              # bare U+0130
    "the resulting İstanbul crisis",
    "Diyarbakır",                     # Turkish dotless-i family (ı, U+0131)
    "the großen Straße",              # German ß (case-folds to "ss" under casefold, not lower)
    "café",                            # precomposed é (U+00E9)
    "café",                     # decomposed e + combining acute (U+0065 U+0301) -- same GRAPHEME, different codepoints
    "ﬁle",                             # U+FB01 LATIN SMALL LIGATURE FI (single codepoint, "file" in one glyph)
    "ＫＩＮＧ",                          # fullwidth Latin (U+FF2B...), a real "does the fallback normalize this" case
    "the war (١٩١٤)",                 # Arabic-Indic digits embedded in an otherwise-Latin phrase
    "MASS STARVATION",           # non-breaking space (U+00A0) instead of a regular space between words
    "the​king",                  # zero-width space (U+200B) INSIDE a word boundary
]

# --- 5. Realistic trigger/outcome phrase shapes, the kind extract_validated
#        actually produces (see curator_yield_run.py/builder_run.py),
#        named directly in the build brief. ---
_REALISTIC_PHRASES = [
    "a sharp rise in unemployment", "the resulting recession",
    "a report about the weather", "severe economic downturn",
    "rising interest rates", "a great famine", "the old king",
    "king of france", "the war between nations", "another bad harvest",
    "a costly war", "reduced life expectancy", "public healthcare budgets",
    "wildfire smoke", "respiratory illness", "smoking", "lung cancer",
    "increased greenhouse gas emissions", "global sea level rise",
    "the collapse of the housing market", "widespread crop failure",
    "a prolonged drought", "the outbreak of civil war",
    "declining birth rates", "an aging population",
    "the introduction of new tariffs", "a slowdown in manufacturing output",
]


def _build_probe_list():
    """Assembles the full deterministic probe list (no RNG anywhere in
    this function or its callees) and de-duplicates while preserving
    first-occurrence order, so the SAME phrase is never probed twice
    (a duplicate would not add information and would just inflate
    n_total without changing n_identical's meaning)."""
    all_probes = (
        _det_adj_pp_probes()
        + _VERB_FRAGMENTS
        + _SYMBOL_STRINGS
        + _UNICODE_PROBES
        + _REALISTIC_PHRASES
    )
    seen = set()
    ordered = []
    for p in all_probes:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


PROBES = _build_probe_list()


# ─────────────────────────────────────────────────────────────────────────
#  dump
# ─────────────────────────────────────────────────────────────────────────
def run_dump():
    """Runs canonical_key (via canon.py's module-cached default pipeline)
    over every entry in PROBES, in list order, and returns the full
    payload dict. Never raises on a single probe's failure -- canon.py's
    own canonical_key already catches parse exceptions internally and
    falls back to the surface normalization (see canon.py's module
    docstring), so a probe reaching this loop either returns a string or
    canon.py itself has a bug; this function does not add a second
    try/except layer on top of canon.py's own documented boundary."""
    pin = env_pin()
    results = []
    for raw in PROBES:
        canon_key = canonicalize_with_default_nlp(raw)
        results.append({"raw": raw, "canon": canon_key})
    return {
        "env_pin": pin,
        "n_probes": len(results),
        "probes": results,
    }


# ─────────────────────────────────────────────────────────────────────────
#  compare
# ─────────────────────────────────────────────────────────────────────────
def run_compare(dump_a, dump_b):
    """Diffs two dump payloads (already-loaded dicts, same shape run_dump
    returns). Report-only -- computes and returns a dict, never raises on
    a mismatch (mismatches are the entire point of this tool: it exists
    to FIND them, not to treat finding one as failure).

    Alignment is by `raw` phrase (a dict keyed on raw -> canon per dump),
    not by list position -- so this tolerates dump_a/dump_b having been
    produced by probe lists that differ in ORDER (should never happen
    since PROBES is one fixed module list, but a dump from an older/
    newer version of this script, or a hand-edited probe subset, should
    still compare correctly on the phrases both sides actually share)."""
    map_a = {p["raw"]: p["canon"] for p in dump_a["probes"]}
    map_b = {p["raw"]: p["canon"] for p in dump_b["probes"]}

    common_raws = sorted(set(map_a.keys()) & set(map_b.keys()))
    only_in_a = sorted(set(map_a.keys()) - set(map_b.keys()))
    only_in_b = sorted(set(map_b.keys()) - set(map_a.keys()))

    divergences = []
    n_identical = 0
    for raw in common_raws:
        ca, cb = map_a[raw], map_b[raw]
        if ca == cb:
            n_identical += 1
        else:
            divergences.append({
                "raw": raw,
                "canon_a": ca,
                "canon_b": cb,
                "env_pin_a": dump_a.get("env_pin"),
                "env_pin_b": dump_b.get("env_pin"),
            })

    return {
        "n_total": len(common_raws),
        "n_identical": n_identical,
        "n_divergent": len(divergences),
        "fraction_identical": round(n_identical / len(common_raws), 4) if common_raws else None,
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "env_pin_a": dump_a.get("env_pin"),
        "env_pin_b": dump_b.get("env_pin"),
        "divergences": divergences,
    }


# ─────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="P75 cross-version canon_key probe (canon.py determinism, measured not assumed)"
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    dump_ap = sub.add_parser("dump", help="run the fixed probe list, write env_pin + results as JSON")
    dump_ap.add_argument("--out", required=True, help="output JSON path")

    cmp_ap = sub.add_parser("compare", help="diff two dumps (e.g. local vs the x86 runner)")
    cmp_ap.add_argument("--a", required=True, help="first dump JSON path")
    cmp_ap.add_argument("--b", required=True, help="second dump JSON path")
    cmp_ap.add_argument("--out", default=None, help="optional: also write the compare report as JSON")

    args = ap.parse_args()

    if args.mode == "dump":
        payload = run_dump()
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        print("=" * 74)
        print("CANON PROBE DUMP")
        print("env_pin: {}".format(payload["env_pin"]))
        print("n_probes: {}".format(payload["n_probes"]))
        print("wrote {}".format(args.out))
        return 0

    if args.mode == "compare":
        with open(args.a, "r", encoding="utf-8") as f:
            dump_a = json.load(f)
        with open(args.b, "r", encoding="utf-8") as f:
            dump_b = json.load(f)
        report = run_compare(dump_a, dump_b)

        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, sort_keys=True)

        print("=" * 74)
        print("CANON PROBE COMPARE")
        print("env_pin a: {}".format(report["env_pin_a"]))
        print("env_pin b: {}".format(report["env_pin_b"]))
        print("n_total={} n_identical={} n_divergent={} fraction_identical={}".format(
            report["n_total"], report["n_identical"], report["n_divergent"], report["fraction_identical"]
        ))
        if report["only_in_a"]:
            print("only in a ({}): {}".format(len(report["only_in_a"]), report["only_in_a"][:10]))
        if report["only_in_b"]:
            print("only in b ({}): {}".format(len(report["only_in_b"]), report["only_in_b"][:10]))
        if report["divergences"]:
            print("-" * 74)
            print("DIVERGENCES:")
            for d in report["divergences"]:
                print("  raw={!r}  a={!r}  b={!r}".format(d["raw"], d["canon_a"], d["canon_b"]))
        if args.out:
            print("wrote {}".format(args.out))
        # Always exit 0 -- report, not gate (this repo's scorer philosophy:
        # a divergence is a measured fact for P75's registration to react
        # to, not a reason for this tool itself to fail a check).
        return 0

    return 0  # unreachable (argparse enforces mode in ("dump", "compare"))


if __name__ == "__main__":
    sys.exit(main())
