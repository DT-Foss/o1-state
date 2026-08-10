"""LIVE-CAUSAL entity canonicalization -- a deterministic READ-TIME layer.

Localizes the one measured bottleneck named twice before this module
existed and then measured: P72 (write side -- 2,047 triplets folded to
2,046 distinct trigger/outcome key PAIRS, so real-prose keys almost never
collide exactly, and only 76/200 inferred edges resulted) and P73 (read
side -- coverage 0.0162: 40 of 2,471 surprise spikes found ANY outgoing
edge in the store's exact-string key space). Both numbers are the same
disease: `trigger_key`/`outcome_key` today is curator_yield_run.py's
normalize_entity, `s.strip().lower()` -- a surface-string match, no
linguistic head extraction, so "the old king" and "king of france" are
different keys even though both name entities a human would call "king".

THIS MODULE NEVER MUTATES ANYTHING SEALED. store.py's segments are
content-addressed and their bytes never change (LIVE_CAUSAL_SPEC.md SS1);
canon.py adds a second, parallel key space computed AT READ TIME from the
same immutable records, joined by canon_key rather than raw_key. Nothing
here writes into a segment, rewrites a manifest, or touches
evidence.py/infer.py's existing base-edge identity. infer.py's
`canon=False` path (the default) is untouched: byte-identical behavior to
every prior LiveGraph caller, checked directly by test_canon.py's
regression test.

Approach (Lead's binding design, this module's concrete realization):
    canonical_key(raw_key, nlp) -> str
  is a PURE function of (raw_key, spaCy model + version). Parse raw_key as
  a standalone noun-phrase-shaped string, take the FIRST noun chunk spaCy
  finds (see _first_noun_chunk_lemma below for why "first", not "last" or
  "syntactic doc-root" -- both alternatives were tried and rejected during
  build, see the module-level NOTE below), lemmatize its root token,
  lowercase, collapse whitespace. Determiners ("the", "a"), adjectival
  modifiers ("old", "severe"), and trailing prepositional-phrase
  attachments ("of france", "in unemployment") fall away because spaCy's
  noun_chunks already excludes them from the chunk's root-token identity
  (the chunk SPAN includes "the old king", but chunk.root is the token
  "king" -- det/amod modify the root, they are not it).

  Fallback (deterministic, no spaCy dependency for identity when spaCy is
  unavailable OR the parse yields no noun chunk at all -- verb-only
  fragments like "was published" or symbol strings like "xk7j2q9" have no
  noun_chunks): normalize_entity-style surface fallback, i.e.
  raw_key.strip().lower() with internal whitespace collapsed to single
  spaces. This is intentionally the SAME normalization curator_yield_run's
  normalize_entity already applies, so a fallback-canonicalized key is
  never WORSE than today's raw_key -- canon=True can only ADD joins,
  never lose ones raw_key already had.

  Every canonical_key call is meant to be wrapped by a caller that also
  records env_pin() alongside its output (spacy_available, spacy_version,
  canon_version) -- the P70 lesson (builder_run.py's own env_pin
  docstring: same code path, DIFFERENT extraction behavior depending on
  what's installed on the host) applies identically here: a canon_key
  computed with spaCy 3.8.11 loaded and one computed via the no-spaCy
  fallback are NOT the same function, and a cache/artifact that doesn't
  record which one ran cannot be re-derived by a stranger.

NOTE on the rejected alternatives (found concretely while building this,
not invented for balance -- see the build report's "known weaknesses"
section for the fuller list):
  - "last noun chunk" (spec brief's literal reading candidate) FAILS the
    Lead's own worked example: on "king of france", spaCy's noun_chunks
    returns TWO separate flat chunks, ["king", "france"] (the PP
    "of france" attaches as its own chunk, spaCy does not fold it into a
    single "king of france" span). The LAST chunk's root is "france" --
    exactly the wrong key, since "the old king" canonicalizes to "king"
    and would then never join "king of france" under a last-chunk rule.
    Measured directly during build (see canon() module tests / build
    report), not assumed.
  - "syntactic doc-root token" (walk to the token with dep_ == 'ROOT',
    take ITS lemma) gives the same answer as "first noun chunk" on every
    noun-phrase-shaped test case tried, but breaks on verb-fragment keys
    ("was published" -> ROOT is the verb "published", lemma "publish" --
    a verb key where a noun key was wanted) and gives no controlled
    fallback path when the ROOT itself isn't inside any noun chunk.
    "First noun chunk" degrades more predictably: no noun chunk found ==>
    fallback path taken, deterministically, every time.
"""

import re
import unicodedata

CANON_VERSION = "1"

_WHITESPACE_RE = re.compile(r"\s+")

# Lazily populated by _get_nlp(); a module-level cache so canonical_key()
# doesn't reload the spaCy pipeline on every call within one process (spaCy
# model load is the expensive part -- parsing individual short key strings
# is cheap once the pipeline exists). None until first attempted load;
# False (not None) once a load has been tried and failed, so repeated
# calls in a spaCy-less environment don't re-attempt an import each time.
_NLP_CACHE = {"nlp": None, "tried": False}


def _surface_fallback(raw_key):
    """Deterministic, spaCy-free normalization -- the floor every
    canonical_key() call can fall back to. Matches curator_yield_run's
    normalize_entity (strip + lower) plus internal-whitespace collapse
    (normalize_entity does not collapse internal whitespace since its
    inputs are already single extracted spans; canon_key's inputs can be
    raw multi-token phrases where "king   of   france" and "king of
    france" should not become different fallback keys)."""
    collapsed = _WHITESPACE_RE.sub(" ", raw_key.strip())
    return collapsed.lower()


def _get_nlp():
    """Returns a loaded spaCy `en_core_web_sm` pipeline, or None if spaCy
    or the model is unavailable. Cached at module scope (see _NLP_CACHE
    docstring above). Never raises -- unavailability is a normal,
    documented condition (env_pin() reports it), not an error."""
    if _NLP_CACHE["tried"]:
        return _NLP_CACHE["nlp"]
    _NLP_CACHE["tried"] = True
    try:
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # spaCy importable, but the model isn't installed. Per the
            # build brief: do NOT install it here -- report the gap via
            # env_pin() and use the deterministic fallback path.
            nlp = None
    except Exception:
        nlp = None
    _NLP_CACHE["nlp"] = nlp
    return nlp


def _first_noun_chunk_lemma(doc):
    """The FIRST noun chunk in the parse (not the last -- see the module
    docstring's NOTE for the concrete failure of "last chunk" on "king of
    france"), root token, lemmatized and lowercased. None if the parse has
    no noun chunk at all (verb-only fragments, bare symbols, empty
    strings)."""
    for chunk in doc.noun_chunks:
        return chunk.root.lemma_.strip().lower()
    return None


def canonical_key(raw_key, nlp=None):
    """Pure function: (raw_key: str, spaCy pipeline or None) -> str.

    `nlp` is an explicit, injectable spaCy pipeline object (or None) --
    NOT re-resolved from global state inside this function -- so that
    a) tests can pin a specific loaded pipeline without touching module
       cache internals, and b) a caller who already paid the model-load
       cost once (e.g. LiveGraph mounting with canon=True over many
       records) can pass the same `nlp` through every call instead of
       this module re-deciding availability per call.

    Pass nlp=None to use the fallback surface normalization unconditionally
    (no spaCy attempted at all) -- this is what a caller uses when it has
    already determined via env_pin() that spaCy/the model is unavailable,
    to avoid a redundant per-call availability check.

    Determinism contract: for a FIXED raw_key and a FIXED (spaCy library
    version, model version) pair, this function returns the same output
    on every call, in every process, on every machine -- the P70 lesson
    (spaCy version differences change tagging behavior) means the
    determinism claim is scoped to "same env_pin", not "always", which is
    exactly why env_pin() must travel with every canon_key output (see
    the module docstring)."""
    if not isinstance(raw_key, str):
        raw_key = str(raw_key)

    if nlp is not None:
        try:
            doc = nlp(raw_key)
            lemma = _first_noun_chunk_lemma(doc)
            if lemma:
                # NFC-normalize the lemma too -- spaCy lemmas can carry
                # combining-character forms depending on tokenizer/model
                # internals; the fallback path already goes through NFC
                # via _surface_fallback below, so both paths agree on
                # Unicode normal form (byte-identical determinism across
                # processes needs this, not just "looks the same").
                return unicodedata.normalize("NFC", lemma)
        except Exception:
            # A malformed/adversarial raw_key that spaCy chokes on must
            # not crash canonicalization -- fall through to the
            # deterministic surface fallback, same as "no noun chunk
            # found." No exception type is assumed here deliberately:
            # this is a boundary against an NLP library's internals, not
            # a spot to enumerate spaCy's exception surface.
            pass

    return unicodedata.normalize("NFC", _surface_fallback(raw_key))


def env_pin():
    """One-shot host/canon fingerprint -- mirrors builder_run.py's
    env_pin() (spacy_available, spacy_version) exactly, plus
    canon_version, so a canon-tagged artifact records BOTH which spaCy
    behavior ran AND which version of this module's canonicalization rule
    produced it (the rule itself, e.g. "first chunk" vs "last chunk", is
    also part of what determinism is scoped to -- CANON_VERSION bumps
    whenever _first_noun_chunk_lemma's algorithm changes, model_available
    covers spaCy's own version drift)."""
    spacy_available = False
    spacy_version = None
    model_available = False
    try:
        import spacy  # noqa: E402

        spacy_available = True
        spacy_version = spacy.__version__
        model_available = _get_nlp() is not None
    except Exception:
        pass
    return {
        "spacy_available": spacy_available,
        "spacy_version": spacy_version,
        "model_available": model_available,
        "canon_version": CANON_VERSION,
    }


def canonicalize_with_default_nlp(raw_key):
    """Convenience wrapper: resolves the module-cached default pipeline
    (loading it on first use) and calls canonical_key. This is what
    infer.py's LiveGraph(canon=True) path calls per raw_key when it does
    not want to manage its own spaCy pipeline lifetime -- the module
    cache in _get_nlp() means the model loads at most once per process
    regardless of how many keys are canonicalized."""
    nlp = _get_nlp()
    return canonical_key(raw_key, nlp=nlp)
