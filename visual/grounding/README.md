# FERTIG Grounding Kernel 0.2

This directory is an isolated research lab. It does not import from or write to
the actively edited live `fertig` package.

The package attacks one precise part of Harnad's symbol-grounding problem:
whether fresh opaque symbols can acquire falsifiable operational content from
raw observation, intervention and consequence, and whether those symbols can
then support a small learned language. It does not equate graph membership,
word co-occurrence, CLIP similarity or a part-of-speech slot with grounding.

The governing impossibility boundary is:

\[
h\sim h' \iff \sup_i D_{TV}
\big(P(Y\mid h,do(i)),P(Y\mid h',do(i))\big)=0.
\]

A predicate can be grounded under the declared sensors/actions only if it is
constant on every such equivalence class. Meaning is therefore relative to an
environment family, intervention set, horizon and target algebra. Absolute
reference beyond world automorphisms and token renaming is not identifiable.

## What is implemented

### v0 transition kernel

`microworld.py`, `binder.py`, `composition.py` and `benchmark.py` provide the
first finite reference: opaque transition binding, token-remapping invariance,
intervention necessity, typed conjunction and honest abstention on a causally
disconnected predicate.

### v1 object/process/language reference

The hardened v1 path adds:

- strict outcome-code-free `PublicTrace` records and support-only generic
  feedback;
- shelter-like affordances, self-sustaining process twins, matched visual
  context twins and an exposed unidentifiable concept;
- hypotheses induced from independent public support worlds;
- real `reset/step` acquisition, exact expected information gain, exhaustive
  random orders, counterbalanced passive orders, evaluator-private target
  commitments, authoritative action cost and an executed
  intervention/consequence derangement;
- a raw-RGB episode binder plus sensorless complete-action-target, action-only,
  target-only and outcome-channel controls;
- a visual target selector that probes every candidate, re-localizes after each
  step, resets, then executes the requested action schema;
- a bidirectional opaque two-slot language learned from three of four factorial
  combinations;
- an independently executed feedback-free trace → expected operational
  referent → description → fresh-world re-execution, not the trace produced by
  the learner's own parse;
- proof-carrying definition → executable referent materialization, with closed
  dictionary cycles and deleted leaves remaining unknown and swapped leaves
  changing behavior;
- finite interventional quotients and constructive anchor-closure witnesses;
- a persistent JSON-only candidate process with pre-codebook artifact
  commitment, evaluator-charged actions, one freeze point and mutation checks.

The integrated v1 CLI is deliberately named a **claim-grade-2 reference
self-test**. It is not an adversarial grounding certificate because its
component stack is still assembled transparently in-process. The persistent
candidate runner exists, but every axis has not yet been routed through one
externally sandboxed candidate. See [`THREAT_MODEL.md`](THREAT_MODEL.md).

## Components

```text
grounding_kernel/
  quotient.py             finite interventional equivalence and closure theorem
  contracts.py            v0 observations/actions/trajectories
  v1_contracts.py         outcome-free session records and candidate protocol
  v1_wire.py              exact size-bounded canonical JSON
  v1_session.py           authoritative phases, costs, freeze and audit ledger
  v1_isolation.py         persistent spawned candidate process
  v1_runner.py            session + candidate + evaluator-cost orchestration
  microworld.py           v0 world with evaluator-only oracle
  processworld.py         object/process/context/negative-control world
  binder.py               v0 sensorimotor binder
  episode_binder.py       v1 raw-trace concept binder
  perceptual_policy.py    visual candidate discovery and active re-binding
  active.py               exact Bayesian version space and EIG policies
  active_process.py       live ProcessWorld acquisition experiment
  language.py             opaque bidirectional language and definition proofs
  closed_loop_programs.py perception-bound role × action execution/recognition
  composition.py          typed Kleene logic and least grounding fixed point
  certificates.py         lower/upper confidence bounds and metric bundles
  v1_controls.py          inversion-aware shortcut statistics
  benchmark.py            v0 reference test
  v1_benchmark.py         claim-grade-2 v1 reference self-test
```

## Run

```bash
python3 -m pytest -q
python3 -m grounding_kernel.benchmark --learner binder --episodes 24
python3 -m grounding_kernel.v1_benchmark --compact
```

The default v1 run uses 32 outer world blocks and a fail-closed registry of 18
mandatory shortcut controls. Positive axes and shortcut controls use exact
one-sided Clopper–Pearson bounds in the appropriate lower/upper direction.
Seed 3 and the built-in evaluator key are explicitly development fixtures, not
a preregistered holdout. For a one-time holdout, a random evaluator masterseed
is generated only after the source commitment, withheld while the ledger runs,
domain-separated into all world seeds and the target-audit key, then revealed.
That chronology is recorded separately in
[`RESULTS_V1.md`](RESULTS_V1.md).

Programmatic post-freeze evaluation uses
`run_v1_benchmark_from_master_seed(master_seed, ...)`; the master must be
generated outside the package only after the printed source commitment. The
development CLI intentionally accepts an explicit public seed and therefore
cannot create that chronology by itself.

Minimal protocol-isolated use:

```python
from grounding_kernel import SealedEvaluationRunner, commit_candidate_artifact

commitment = commit_candidate_artifact("my_candidate:build", ["weights.bin"])
# Construct SessionManifest, then pass a zero-argument codebook factory to one
# SealedEvaluationRunner. It starts and rehashes the staged candidate before
# invoking that factory, then keeps one process for support -> acquisition ->
# freeze -> every query.
```

The spawned JSON boundary removes accidental evaluator-object sharing. It is
not an OS sandbox: adversarial claims additionally require filesystem, network,
clock, process and repository denial by the host/container. The staged files'
read-only mode bits and `checkpoint_commitment()` detect cooperative or
accidental mutation; same-UID code can change those mode bits, and the
checkpoint digest is candidate-declared. Neither is an adversarial full-state
attestation until an evaluator-owned VM/process snapshot exists.

## Claim boundary

The current strongest integrated result is a deterministic finite reference
self-test for outcome-free object/process discrimination and a bounded
compositional motor code. It does not establish open-world natural language,
variable syntax, relations/pragmatics, social convention, cross-simulator or
physical transfer, human concepts in their cultural fullness, consciousness,
or universal meaning. Those are explicit next falsification stages, not facts
smuggled into a high accuracy number.
