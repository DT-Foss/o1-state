# GroundZero-Ω: Sealed Grounding Benchmark

This benchmark cannot prove metaphysical meaning. It is designed to falsify
the strongest operational symbol-grounding claim and block the most common
shortcuts.

## World protocol

Each world is a hidden structural causal model:

\[
X_{t+1}=f_w(X_t,A_t,U_t),\qquad O_t=g_w(X_t,N_t),
\]

and each codebook is a fresh random bijection from identifiable predicates to
opaque integer tokens. The candidate sees only raw observations, actions,
opaque tokens and sparse task feedback. Privileged state, object IDs,
category names, codebook and random seeds remain in a separate evaluator
process.

## Environment ladder

1. **InvariantShape:** view, light, occlusion and deformation invariance.
2. **AffordanceLab:** visually matched objects differing only in mass,
   friction, elasticity, magnetism or containment.
3. **ProcessWorld:** multi-step causes, prevention, order and counterfactuals;
   train correlations reverse in test.
4. **PerspectiveWorld:** partial observation, object permanence, deixis,
   viewpoint and role-dependent spatial relations.
5. **SocialWorld:** negotiated conventions, clarification, role swap and
   teaching a new agent.
6. **RealTable:** post-freeze physical objects with hidden properties at two
   independent sites.

Use both a Grounding-Core track with reliable primitive motor actions and a
Total track with raw control. This separates semantic failure from robotics
failure.

## Leakage barrier

- Generate codebooks only after hashing/fixing the submitted model.
- Use atomic integer IDs, balanced length/frequency/position and no English
  field names or filenames.
- Never reuse codebooks across train/test worlds.
- Define some referents by fresh post-freeze SCM rules.
- Run the candidate offline, read-only, without network, host clock or seed
  access.
- Keep oracle and generator in a separate process.
- Include secret canary tokens; any appearance invalidates the run.
- Require token-only performance at chance before exposure.
- Include twin worlds that are passively identical but diverge after one
  intervention.

Two tracks keep claims honest:

- **Grounding-origin:** no semantically pretrained text/image model.
- **Foundation:** pretrained perception allowed, but all post-freeze meanings,
  causal rules and codebooks remain novel.

## One evaluation block

1. Create a fresh world and codebook.
2. Permit 48 ostensive/interactive support episodes.
3. Permit 16 actively chosen diagnostic interventions.
4. Freeze graph and long-term memory.
5. Run 24 sealed queries on new objects and compositions.
6. Change one real rule and measure belief revision separately.

Zero-shot is required for new combinations of already grounded atoms, not for
an arbitrary atom that has never had evidence.

## Orthogonal holdouts

- new instances;
- new renderer/camera/light/noise;
- reversed spurious correlations;
- fresh lexicon;
- unseen factorial combinations;
- train depth 1–2, test depth 3–6;
- new interventions and mechanism parameters;
- new partner, dialect, viewpoint and social role;
- cross-modal transfer;
- new embodiment/sim-to-real;
- a relation outside the fixed FERTIG registry;
- an intentionally unidentifiable predicate.

## Required behaviors

The system must discriminate, manipulate, identify, describe, produce
descriptions and act on descriptions. In addition test:

- synonym agreement and lexicon remapping;
- causal prediction under `do`, not just observation;
- creating a described configuration, not only recognizing it;
- negation, quantification, order and counterfactuals;
- first-use symbolic theft from grounded definitions;
- active experiment choice;
- honest unknown and later revision;
- A→B→C social teaching transfer.

## Baselines and ablations

Run identical worlds with no sensors, no actions, shuffled time, shuffled
action/outcome pairs, no symbol channel, no graph, no composition, random
causal edges, no social feedback and no pretrained encoder.

Required baselines include chance/majority, token and texture probes, FERTIG
graph-only, current CLIP/Wikipedia proxies, text LLM/RAG, passive VLM,
world-model/RL without symbols, symbolic Bayesian version-space learner,
oracle perception + FERTIG planner, full oracle and humans under the same
budget.

## Noncompensatory certificate

For token `s`, let `Q(s)` be all required capabilities for its type:

\[
G(s)=\min_{q\in Q(s)}\operatorname{LCB}_{95\%}(M_{s,q}),
\]

\[
GC_\tau=\frac1{|S|}\sum_s\mathbf 1[G(s)\ge\tau].
\]

Report balanced accuracy, open-set false-grounding, Brier/ECE, action success,
intervention regret, ATE error, counterfactual consistency, reversal drop,
exact composition by depth, social coordination, graph precision against the
hidden SCM, revision latency, worst-group performance and bottom-decile CVaR.

Two essential diagnostics:

\[
E_\pi=1-\mathbb E\,d(B(D,q,O),B(\pi D,\pi q,O))
\]

for codebook equivariance, and

\[
\Delta_{do}=U_{full}-U_{shuffled(action,outcome)}
\]

for sensorimotor causal necessity.

## Preregistration

- Intersection-union primary test: every mandatory axis passes.
- Evaluation unit is world × codebook, clustered by world.
- Paired cluster bootstrap confidence intervals.
- Generator and sealed seed hashes registered before model freeze.
- Model/container/dependency hashes and all budgets registered.
- No best-run selection; exclusions only for predefined hardware failures.
- Physical core result independently replicated.

## Kill criteria for a “complete on scope” claim

The claim fails if any one occurs:

- a simulation-gate 95% LCB is below its preregistered threshold;
- `GC_0.90 < 1` on the sealed symbol set;
- fresh-token pre-exposure performance exceeds chance;
- codebook equivariance `<0.95`;
- correlation reversal costs `>10` percentage points;
- depth 4–6 composition is `<85%` or drops `>10` points from depth 1;
- open-set false-grounding is `>1%` or ECE `>0.05`;
- new-partner transfer is `<80%`;
- full system gains `<25` points over symbol-only or shuffled-outcome;
- a passive agent beats chance on intervention-only twin worlds;
- a fixed relation table forces a known label instead of learning/abstaining;
- any semantic metadata, test codebook or hidden seed leaks;
- the physical result does not replicate.

Thresholds are initial targets. A pilot may calibrate them once; the sealed
test remains untouched afterwards.

