# GroundZero-v1 one-time post-freeze holdout

## Status

- Result: **reference self-test passed**
- Claim grade: **2**
- Adversarial certificate: **no**
- `certified`: `false`
- `certificate_eligible`: `false`
- Formal preregistration: **no**
- Retries: **0**

This is a one-time post-freeze record, not a third-party preregistered result.
It establishes a finite reference construction under the scope in
`THREAT_MODEL.md`; it does not establish universal grounding.

## Chronology and commitments

1. The complete package source commitment was computed over every
   `grounding_kernel/*.py` file plus `pyproject.toml`.
2. That commitment was printed before any holdout seed existed.
3. One 256-bit evaluator masterseed was sampled and only its commitment was
   printed.
4. All world/codebook/target/renderer namespaces and the target-audit key were
   domain-separated from that master inside
   `run_v1_benchmark_from_master_seed`.
5. The master remained undisclosed until the 64-block ledger was terminal.
6. The master was then revealed for exact reproduction. No retry was run.

| Item | Value |
|---|---|
| Source commitment | `da7a308aba8df575ff1d31b214399d6a52c4b138c40b149de38eeb6fc044e875` |
| Source files bound | `29` |
| Master commitment | `a6b64482257630bb269aeb8fac0d1f5269bab2abd491aae74de9dde0b62d525a` |
| Master reveal | `1c3588e654ab309c8cfcbfef7972204dcc64d310e220ae031625e8ad82c8a5b5` |
| Derived evaluator seed | `13395644258270279971` |
| Dataset hash | `2f747a34753c69e13a291649592ab1bc6db78083650b3d9e083c3386c616e8ad` |
| Scope hash | `e2879d321d9ee96cf372bd1d3934906fe1486be3dc9034edac8f630af4bc5004` |
| Metric-bundle hash | `4cdf52ea9c14b8d3066523b693cdc5e3aaeb668500cf0fd089875f796bb9fecc` |
| Ledger hash | `13a3cbd74a45ad23a85592364c64a24370c4f1150f990d4ac739d71313bcea9d` |
| Runtime | Python `3.12.7`, NumPy `1.26.4` |

## Predeclared gates

- Outer cluster unit: one independently derived world block
- Blocks: `64`
- Capability threshold: `0.80`
- Answer-coverage threshold: `0.80`
- Shortcut leakage ceiling: `0.20`
- Chance-control ceiling: `0.75`
- Confidence: `0.95`
- Inference: exact one-sided Clopper-Pearson bounds
- Required axes: `10`
- Required shortcut controls: `18`
- Raw control trials: `18 × 64 = 1,152`

Every required control had exactly one hashed raw observation per block. The
ledger rejects missing, duplicate or unregistered control names before any
aggregate can be reported.

## Capability axes

All ten axes achieved `64/64` correct at `64/64` coverage. For each axis:

- estimate: `1.0`
- exact one-sided 95% lower bound: `0.9542702976692375`
- coverage estimate: `1.0`
- exact one-sided 95% coverage lower bound: `0.9542702976692375`

The axes were:

1. active acquisition efficiency;
2. object-affordance invariance;
3. process causality and temporal order;
4. sensor-context necessity;
5. description to action;
6. independent trace to description and fresh-world re-execution;
7. factorial composition;
8. lexicon-permutation equivariance;
9. proof-grounded symbolic theft;
10. open-set honesty.

## Shortcut-control findings

The phrase “insufficient as a full grounder” is an explicit disjunction:
either low leakage was proven, or the shortcut could not meet the `0.80`
coverage floor. Those are reported separately.

### Low leakage and insufficient coverage

The twelve action-only, action/outcome-only, complete-action-target-only and
target-only controls across object/process/context, plus full-utterance lookup,
had:

- informative leakage: `0/64`;
- leakage UCB95: `0.04572970233076241`;
- full-answer coverage: `0/64`;
- coverage UCB95: `0.04572970233076241`.

Thus both low leakage and insufficient coverage were established for these
thirteen controls.

### Low leakage at full execution coverage

Temporal shuffle, definition-leaf deletion and definition-leaf swap each had:

- leakage: `0/64`;
- leakage UCB95: `0.04572970233076241`;
- execution coverage: `64/64`.

### Intervention/consequence shuffle

The actively executed derangement produced no resolved answer in `64/64`
blocks. Coverage UCB95 was `0.04572970233076241`. Because there were zero
answered trials, the result establishes insufficient coverage, **not** a
conditional low-leakage bound.

### Target-rank-only

This shortcut answered in `3/64` blocks:

- coverage estimate: `0.046875`;
- coverage UCB95: `0.11671658790335931`;
- conditional informative outcomes: `3/3`;
- conditional leakage UCB95: `1.0`.

Therefore target rank is insufficient as a full grounder, but it leaks when it
does answer. The result deliberately does not relabel this as “low leakage.”

## Verification before freeze

- `229` tests collected and passed
- `ruff check .`: passed
- `compileall`: passed
- independent benchmark/statistics red-team: no remaining P0
- independent isolation/security red-team: no remaining P0

## Remaining claim boundary

The integrated benchmark still assembles transparent reference components
in-process. It is not one arbitrary frozen learner routed through every axis.
The protocol-isolated runner is stronger, but its read-only staging uses
same-UID mode bits and its checkpoint digest is candidate-declared. A malicious
candidate can only be claimed against after evaluator-owned OS/VM isolation and
state snapshots.

This record also covers only one finite simulator, bounded object/process
concepts and a fixed two-slot language. It does not cover open-world natural
language, variable syntax, pragmatics, social convention, independently
implemented simulators, physical transfer, consciousness or universal
reference. The next research gates are one end-to-end serialized candidate,
an external sandbox, a second environment implementation, multi-agent naming
games/relations, and sim-to-real sensorimotor transfer.

