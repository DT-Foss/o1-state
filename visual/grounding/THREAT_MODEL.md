# GroundZero threat model and claim ladder

This document is normative. A green unit-test suite is not by itself a symbol-
grounding certificate.

## Identifiability boundary

For public history `h`, allowed intervention `i`, and future learner-visible
consequence `Y`, define

\[
h\sim h' \iff
\sup_i D_{TV}\!\left(P(Y\mid h,\operatorname{do}(i)),
P(Y\mid h',\operatorname{do}(i))\right)=0.
\]

A target predicate is operationally groundable only if it is constant on every
class of this quotient. Token names are identifiable only up to a permutation,
and world meanings only up to automorphisms that preserve all allowed public
interventions and consequences. No benchmark can identify more without adding
new sensors, actions or assumptions.

## Claim grades

1. **Component test.** A module satisfies its local typed contract.
2. **Reference self-test.** Curated reference components solve held-out finite
   worlds. Evaluator construction may still prepare support records. This is
   useful regression evidence, not a candidate certificate.
3. **Protocol-isolated reference run.** One candidate artifact is committed
   before the codebook; one persistent process receives support, performs
   acquisition, freezes once, and answers every query through strict JSON.
   This rules out accidental object/oracle sharing but assumes honest host
   capabilities and an honest checkpoint digest.
4. **Adversarial candidate certificate.** Grade 3 plus an external OS sandbox
   denying repository, evaluator state, undeclared files, clocks, network and
   process escape. Mutation after freeze and every forbidden import/capability
   are active kill tests.
5. **Cross-environment operational evidence.** Grade 4 across independently
   implemented environments and then physical sensorimotor transfer.

The current package may report lower grades while work on higher grades
continues. It must never silently relabel a lower grade as a higher one.

## Forbidden learner channels

- evaluator enums, semantic outcome names, decoded codebooks or latent IDs;
- seeds, renderer/world variant IDs, oracle handles or bound evaluator methods;
- query-time task feedback or support labels;
- exact train/test episode, coordinate, utterance or target lookup tables;
- pretrained text/image models or web corpora in the ground-zero kernel;
- filesystem/network access for an adversarial-candidate claim;
- model revision after the committed freeze point.

Opaque action integers are allowed, but the mandatory v1 path strips legacy
`outcome_code` values. Corrective scalar feedback is support-only. Query traces
are feedback-free.

## Mandatory shortcut kills

- sensorless, action-only, target-coordinate-only and full-utterance lookup;
- temporal shuffle and intervention/consequence shuffle;
- token permutation and same-domain fresh tokens;
- matched causal twins with identical non-sensory transcripts;
- perfectly inverted or selectively abstaining shortcut predictions (both can
  still contain the answer bit);
- a causally disconnected concept exposed with both labels;
- a definition-only symbol whose leaves or definition are deleted/swapped;
- active learning supplied with evaluator-built holdout likelihood tables;
- a random/passive comparator that is incorrect but treated as cheap;
- public seeds substituted for post-freeze secret evaluation seeds.

Negative controls use an exact one-sided Clopper–Pearson upper bound on
shortcut leakage. Taking the negation of a failed positive lower-bound test is
not evidence of absence. Low leakage and insufficient answer coverage are
separate findings even when either makes a shortcut insufficient as a complete
grounder.

## Statistical unit

The outer cluster is an independently generated world × codebook block. Frames,
actions, probes and token combinations inside one block are not independent
samples. Capability axes use conservative lower bounds; shortcut controls use
upper bounds. Active efficiency is a paired block-level cost-to-correct effect,
with failures censored at the committed budget. The evaluation masterseed is
sampled only after source/candidate freeze, withheld until the terminal ledger,
and domain-separated into worlds, codebooks, targets and variants. Any
non-default threshold run is diagnostic unless its thresholds were committed
before sealed data existed.

## Universal-claim guard

Even grade 5 would establish scoped operational grounding, not phenomenal
consciousness, unique metaphysical reference, every human concept, or a proof
over all possible environments. “Complete” can be made exact only relative to
a declared environment family, public intervention set, horizon and target
predicate algebra; outside that scope the result must remain open to
falsification.
