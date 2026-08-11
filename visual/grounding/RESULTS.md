# GroundZero-v0 result record

Date: 2026-08-11  
Status: scoped operational certificate; not a universal symbol-grounding claim

## Reproducible run

```bash
python3 -m grounding_kernel.benchmark \
  --seed 3 \
  --episodes 24 \
  --learner binder \
  --compact
```

| Mandatory axis | Trials | Estimate | Wilson LCB 95% | Result |
|---|---:|---:|---:|---|
| token-remapping equivariance | 24/24 | 1.000 | 0.862 | pass |
| nuisance/object transfer | 24/24 | 1.000 | 0.862 | pass |
| intervention necessity | 24/24 | 1.000 | 0.862 | pass |
| unseen typed conjunction | 24/24 | 1.000 | 0.862 | pass |
| honest unidentifiable abstention | 24/24 | 1.000 | 0.862 | pass |

The static-pixels/no-action control scored `2/24 = 0.0833` and was rejected as
a grounder. Coverage was 1.0 on every mandatory axis. The noncompensatory
certificate score was `1.034353700928628`.

- Scope hash: `e604387e0e5216b5395950b65478809ed4633b67d607aef1bd6cafc3c0882f6d`
- Certificate hash: `c3f26079b9ba60a2c6b617841ddbe5f4c67dbd0e31795064292a2772b4e4dd75`

## Verification

- `71` tests collected and passed.
- `python3 -m ruff check grounding_kernel tests`: clean.
- `python3 -m compileall -q grounding_kernel tests`: clean.
- The executable module command returned exit code `0` and the hashes above.

## Red-team changes incorporated

1. A bound-method reflection path into the in-process engine was reproduced.
   Adversarial learners now have a spawn/strict-JSON process boundary; this is
   capability isolation, not an operating-system sandbox.
2. Intervention signatures were geometry-confounded. They now run in a
   canonical obstruction-free counterfactual context.
3. Manifest hashing collapsed typed keys. Manifests now require string keys.
4. Static visible identity was a possible shortcut. New objects/renderers are
   held out and a static-pixels/no-action baseline must fail.
5. The negative-control proof checked too little. It now exhaustively checks a
   declared finite action alphabet through depth two and fails closed when its
   transition budget is insufficient.

## Exact claim boundary

This run certifies, for the declared finite microworld family, that the binder
can learn opaque predicate tokens from ostensive sensorimotor transitions,
retain them under a fresh lexicon permutation and nuisance changes, use the
intervention/outcome pairing, compose a previously unseen typed conjunction,
and abstain on a predicate absent from both rendering and transition laws.

It does **not** yet certify active experiment selection, counterfactual policy
planning, deep relational composition, social convention, physical-world
transfer, arbitrary natural-language meaning, consciousness or values.

## Next falsification gates

1. Active intervention choice must beat passive/random probing at equal cost.
2. Description-to-action and action-to-description must both pass on new goals.
3. Typed spatial and causal relation kernels must compose to held-out depth.
4. Passive-identical/intervention-divergent twin worlds must separate.
5. Symbolic theft must transfer a novel definition with zero direct composite
   examples, then survive partner/codebook changes.
6. The frozen system must replicate on a second simulator and two physical
   sites before any broader grounding claim is made.

No live `fertig` file was modified for this experiment.
