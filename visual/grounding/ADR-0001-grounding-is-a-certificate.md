# ADR-0001: Grounding is a scoped causal-behavioral certificate

Status: accepted for v0

## Context

The surrounding FERTIG code historically used `grounded` for graph membership
and the live prototype currently uses CLIP/Wikipedia or text-extracted
number/unit pairs as terminal anchors. Those signals may be useful evidence,
but none establishes that the agent learned a referent through its own
perception/action loop.

This distinction is difficult to retrofit after evidence types have been
merged. It affects data models, benchmarks, APIs and every future claim.

## Decision

In this project, a symbol is called directly grounded only when it has a
versioned certificate over a declared environment/sensor/action scope and
passes all required perception, action, intervention, invariance,
composition/open-set gates with conservative confidence bounds.

- Graph membership is `graph_resolved`.
- CLIP/image similarity is `cross_modal_proxy`.
- A number/unit claim read from text is `textual_quantity_evidence`.
- Human or agent reports are `social_testimony`.
- Direct sensor/action evidence is `direct_sensorimotor`.
- A derived meaning is `composed` and carries a proof tree to certified
  direct leaves.

Evidence tiers are never combined by a raw maximum confidence. A downstream
adapter may project a certified assertion into a knowledge graph, but the
certificate remains the source of truth.

## Consequences

- The learner path cannot import the evaluator oracle, text sources, live
  graph, CLIP or pretrained semantic encoders.
- Unidentifiable target predicates must remain uncertified.
- Passing v0 means complete only on the v0 scope, not universal human meaning.
- More engineering is required for manifests, evidence genealogy and sealed
  evaluation, but false-positive grounding claims become structurally harder.

## Rejected alternatives

- **Any non-word datum is an anchor:** numbers and pixels can still be selected
  or interpreted through human symbols.
- **Embedding similarity above a threshold:** association is not causal use.
- **One aggregate grounding score:** compensation can hide a dead mandatory
  channel.
- **Static primitive inventory as completeness proof:** names do not establish
  denotation and open-world relations cannot be assumed closed.

