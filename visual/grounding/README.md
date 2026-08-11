# FERTIG Grounding Kernel v0

This directory is independent of the live `fertig` package. It is a reference
implementation of the falsifiable path in
[`GROUNDING_THEORY.md`](../primitive_schema_snapshot/GROUNDING_THEORY.md) and
[`GROUNDZERO_OMEGA.md`](../primitive_schema_snapshot/GROUNDZERO_OMEGA.md).

The claim is deliberately scoped:

> Learn fresh opaque symbols from raw observations and controlled actions in a
> sealed finite microworld; retain meaning under token remapping and nuisance
> changes; distinguish interventionally different concepts; abstain on a
> concept the available sensors/actions cannot identify; and transfer grounded
> atoms through a definition never shown as a direct example.

No network, CLIP, pretrained encoder, web source, live graph, English label or
privileged simulator state is available to the learner. The evaluator owns the
latent state and codebook semantics in a separate API boundary.

The included binder is a deterministic finite-world reference, not the final
perceptual architecture. A later learned pixel/world-model stack must retain
these leakage controls while adding active experiment choice and policies.

## Components

```text
grounding_kernel/
  contracts.py       immutable observations, actions and trajectories
  protocol.py        opaque codebooks, split and manifest hashing
  microworld.py      raw-pixel environment plus evaluator-only oracle
  binder.py          operational sensorimotor signatures and abstention
  composition.py     typed definitions and anchor-closure proofs
  certificates.py    noncompensatory confidence-bound certificates
  benchmark.py       remapping, nuisance, intervention and composition gates
  isolation.py       spawn/strict-JSON boundary for untrusted learner code
```

## Run

```bash
python3 -m pytest -q
python3 -m grounding_kernel.benchmark --learner binder --episodes 24
```

The CLI exits non-zero if a mandatory axis fails or a negative control passes
as a grounder. Use `--full-json` for the complete certificate and evidence
ledger. The installed console entry point is `grounding-kernel`. The frozen
seed-3 result, hashes and remaining falsification gates are recorded in
[`RESULTS.md`](RESULTS.md).

Minimal API use:

```python
from grounding_kernel.benchmark import run_benchmark
from grounding_kernel.binder import SensorimotorBinder

result = run_benchmark(
    seed=3,
    episodes=24,
    learner_factory=lambda _seed: SensorimotorBinder(),
)
assert result.passed
```

For adversarial candidate code, use `run_isolated_learner` instead of passing a
`Microworld` object into the same interpreter. This removes the known bound-
method reflection path and accepts only strict serialized RPC records. It is a
process/capability boundary, not an OS sandbox: filesystem and network denial
still belong in a container or operating-system policy.

## What v0 does and does not certify

All mandatory axes are intersection gates: token-remapping equivariance,
nuisance/object transfer, intervention necessity, unseen typed conjunction and
honest abstention on a counterfactually unidentifiable predicate. A static-
pixels/no-action control must fail, and all metrics use conservative confidence
bounds plus a separate coverage floor.

V0 currently certifies ostensive transition classification and a first typed
composition step. It does not yet certify active experiment selection, goal-
conditioned policy learning, long-horizon causal processes, social convention,
sim-to-real transfer or phenomenal meaning. The wider protocol and kill
criteria are specified in
[`GROUNDZERO_OMEGA.md`](../primitive_schema_snapshot/GROUNDZERO_OMEGA.md).

Passing v0 establishes only the declared microworld certificate. It is not a
claim about every possible world, consciousness, values or social meaning.
