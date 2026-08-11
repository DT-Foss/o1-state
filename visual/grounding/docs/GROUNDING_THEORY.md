# FERTIG Grounding Theory v0

Status: research hypothesis and falsifiable specification, not a claim that
human meaning or consciousness has been solved.

## 1. Scope of the strongest defensible claim

No finite experiment can establish grounding for every possible world and
every possible language. The strongest exact target is therefore relative to
a declared domain

\[
\mathfrak D=(\mathcal E,\mathcal I,H,\mathcal L,\mathcal A,\mathcal N),
\]

where `E` is an environment family, `I` the allowed interventions, `H` the
prediction horizon, `L` the language fragment, `A` the action repertoire and
`N` the nuisance transformations.

> A system is operationally completely grounded on `D` when its symbols
> represent every distinction that can be identified through the allowed
> perception/action loop, no distinction that cannot be identified, and its
> derived symbols preserve this semantics under unseen composition.

Uniqueness is possible only up to token renaming and automorphisms of the
world that no allowed experiment can distinguish.

## 2. Text-only non-identifiability

Let a learner observe only strings over alphabet `Σ`. For every bijection
`π:Σ→Σ`, the corpus and a correspondingly permuted model have identical
symbolic structure. Co-occurrence can identify internal roles, but not which
external thing a role denotes. External reference is therefore unidentifiable
without an anchoring channel that breaks this permutation symmetry.

This is the formal core of the Chinese/Chinese dictionary regress. More
formula manipulation inside the dictionary cannot break it.

## 3. The interventionally meaningful world

Model an environment as a controlled partially observed dynamical system:

\[
X_{t+1}=F(X_t,A_t,U_{t+1}),\qquad O_t=G(X_t,N_t).
\]

For sensorimotor history `h`, intervention family `I`, and horizon `H`, define
the interventional signature

\[
\Psi_{\mathcal I,H}(h)=
\left\{
P(Y_{t+1:t+H}\mid h,\operatorname{do}(i)):i\in\mathcal I
\right\},
\]

where `Y` contains future observations, measured events, rewards and action
consequences.

Two histories are operationally equivalent iff no allowed experiment can
distinguish them:

\[
h\sim_{\mathcal I,H}h'
\iff
\sup_{i\in\mathcal I}
D_{\mathrm{TV}}\!\left(
P(Y\mid h,\operatorname{do}(i)),
P(Y\mid h',\operatorname{do}(i))
\right)=0.
\]

The quotient `H/~` is the maximally meaningful world available through those
sensors and actions. A target predicate `c` can be grounded only if

\[
h\sim h'\Longrightarrow c(h)=c(h').
\]

An intentionally indistinguishable negative-control predicate must remain at
chance. Otherwise the evaluator leaks privileged information.

## 4. Meaning as a causal operational kernel

For a token `z`, define its operational meaning as a family of controlled
consequence kernels, not as another string or a single vector:

\[
K_z^E(Y\mid h,\operatorname{do}(a))
=
P_\theta(Y_{t:t+H}\mid h,z,\operatorname{do}(a)).
\]

This supplies three inseparable views of a symbol:

- what it discriminates in perception;
- what it predicts under interventions;
- which policies make, select, prevent or communicate its referent.

Icons and category detectors implement the bottom-up layer described by
Harnad. The explicit FERTIG graph is a later symbolic memory and composition
layer.

## 5. Necessary certificate axes

A weighted average is unsafe because excellent classification could hide a
dead action or causal channel. Let the grounding defect be

\[
\varepsilon_{SG}=
\max\{\varepsilon_{quotient},\varepsilon_{lump},
\varepsilon_{causal},\varepsilon_{use},\varepsilon_{inv},
\varepsilon_{comp},\varepsilon_{closure},\varepsilon_{open},
\varepsilon_{social}\}.
\]

The domain is certified only if every preregistered upper confidence bound is
below its own threshold and every efficacy lower bound exceeds its threshold.
Coverage has a separate floor so universal abstention cannot pass.

### 5.1 Controlled lumpability

If `φ` maps histories to symbols, symbolic transitions are well defined only
when states sharing a symbol induce the same abstract transition law:

\[
\sum_{y:\phi(y)=s'}P(y\mid x,a)
=
\sum_{y:\phi(y)=s'}P(y\mid x',a)
\quad
\forall x,x':\phi(x)=\phi(x').
\]

### 5.2 Causal commutation

World-level interventions and their symbolic counterparts must commute:

\[
\tau_{\#}P_E^{\operatorname{do}(i)}
\approx
P_\Sigma^{\operatorname{do}(\omega(i))}.
\]

### 5.3 Invariance without collapse

For nuisance transforms `n` and semantic transforms `m`:

\[
\phi(nh)=\phi(h),\qquad
\phi(mh)=\rho(m)\phi(h),
\]

with faithful `ρ`. The equivariance condition prevents the constant
representation from passing an invariance-only test.

### 5.4 Internal causal use

Changing the internal symbolic state must alter predictions/actions in the
semantically correct direction:

\[
I_\nu(\operatorname{do}(Z=z);A,Y)\ge\eta.
\]

A probe that can decode a concept which the agent itself never uses is not a
grounded symbol.

### 5.5 Open-set calibration

Unknowns require a selective-risk contract. Report coverage, risk-coverage,
Brier score and ECE separately. Do not call an abstaining system complete
unless it meets the preregistered coverage floor.

## 6. Relations as typed operators

For compatible stochastic kernels, ordered composition is ordinary
marginalisation:

\[
K_{r_1;r_2}(z\mid x)
=
\sum_yK_{r_1}(y\mid x)K_{r_2}(z\mid y).
\]

This is noncommutative and type-checkable. It is the valid compositional idea
in the Hoffman-agent material; scalar Möbius addition is not a general
relation algebra.

If `sup_x TV(K_i(x,·),K̂_i(x,·))≤ε_i`, then for a depth-`d` chain:

\[
\operatorname{TV}(\widehat K_1\cdots\widehat K_d,
K_1\cdots K_d)
\le
\sum_{i=1}^{d}\varepsilon_i
\prod_{j=i+1}^{d}\delta(K_j)
\le\sum_i\varepsilon_i,
\]

where `δ` is the Dobrushin contraction coefficient. This supplies an honest
error budget for composed reasoning paths.

## 7. Grounding closure: the dictionary-cycle test

Let `a_s∈[0,1]` be directly certified grounding for symbol `s`. Each definition
hyperedge `d:(s₁,…,sₖ)→s` has validated operator confidence `c_d`. Initialise

\[
g^{(0)}(s)=a_s
\]

and iterate the monotone rule

\[
g^{(n+1)}(s)=
\max\left(a_s,
\max_{d:\to s}c_d\min_jg^{(n)}(s_j)\right).
\]

Use the least fixed point. A cycle with no directly grounded leaf begins at
zero and remains zero; it cannot manufacture meaning from its own dictionary
loop. Every positive result has a finite proof tree terminating in direct
sensorimotor certificates.

## 8. Relative representational completeness

Let `C` be a finite interventionally identifiable quotient. Suppose grounded
primitive predicates `p₁,…,pₘ` jointly separate all states:

\[
\chi(c)=(p_1(c),\ldots,p_m(c))
\]

is injective. With grounded Boolean composition, every referent `R⊆C` is
expressible:

\[
\varphi_R(c)=
\bigvee_{r\in R}\bigwedge_j
\begin{cases}
p_j(c),&p_j(r)=1,\\
\neg p_j(c),&p_j(r)=0.
\end{cases}
\]

Under these explicit finite-world assumptions, a grounded basis plus faithful
composition is representationally complete. This is a real relative
completeness result; it does not imply efficient learning, concise language,
planning or universality across open worlds.

## 9. Constrained learning objective

A weighted loss can trade away a necessary grounding axis. Prefer constrained
optimisation:

\[
\min_\theta I(Z;H_t)
\]

subject to

\[
H(C_{\mathcal I,H}\mid Z)\le\epsilon,\qquad
\varepsilon_k\le\tau_k\ \forall k,\qquad
I_\nu(\operatorname{do}Z;A,Y)\ge\eta.
\]

This is a causal grounding bottleneck: compress nuisance detail while
retaining every interventionally relevant distinction. A primal-dual training
form is

\[
\min_\theta\max_{\lambda\ge0}
\mathcal L_{task}(\theta)+
\sum_k\lambda_k(\varepsilon_k(\theta)-\tau_k).
\]

The experiment controller should choose actions by expected information gain
per cost:

\[
i^*=\arg\max_i
\frac{I(\Theta;Y\mid\operatorname{do}(i),\mathcal D)}
{\operatorname{Cost}(i)}.
\]

Zeno is useful here only as an explicit staleness rule: stop when the maximum
expected gain or frozen-holdout improvement falls below a preregistered bound.

## 10. Grounding contracts for FERTIG relations

Each future `RelationSpec` needs more than aliases and regexes:

| Family | Operational contract |
|---|---|
| category / property | discriminate across nuisance changes; preserve relevant intervention outcomes |
| located / part / contains | geometry/topology plus movement, occlusion and removal interventions |
| material | predictable response to calibrated physical tests |
| capable / used_for | policy with the object improves goal success over controls |
| causes / prevents / increases | calibrated interventional effect, not text co-occurrence |
| temporal | ordered event observations or calibrated clocks/logs |
| comparative | paired measurements with uncertainty and unit compatibility |
| related_to | observational dependence explicitly labelled noncausal |

A certificate should store domain, codomain, arity, observables,
interventions, invariances, environment scope, raw episode hashes,
calibration, confidence interval and derivation parents.

## 11. Formula trust map

- Directly useful: stochastic kernels, Bayes with dependency assumptions,
  eligibility traces, proper scoring, contraction/error bounds, active
  information gain, frozen holdout and calibrated abstention.
- Conditionally useful: Sinkhorn for genuinely one-to-one lexicon alignment;
  Noether-style tests for real measured sensor symmetries; TwoNN only for
  continuous representation diagnostics.
- Proxy only: CLIP similarity, web agreement, entropy, graph coverage,
  participation ratio and spectral diagnostics.
- Reject as semantic generators: digital roots, universal `D=2`, Möbius as
  relations algebra, or quantum/Lorentz/E8/cosmological analogies.

The intellectual move is not to use every formula. It is to assign each valid
formula one auditable role and refuse category errors.

## 12. Claim tiers

- **Tier A — theorem inside a declared finite formal domain:** quotient,
  identifiability boundary, grounded closure and representational completeness
  under their assumptions.
- **Tier B — empirical certificate:** every necessary test passes on sealed
  unseen worlds, interventions, embodiments and compositions.
- **Tier C — proxy evidence:** cross-modal association, retrieved quantities,
  graph/QA coverage and geometry diagnostics.
- **Unsupported:** universal all-world completeness, subjective meaning,
  consciousness or alignment inferred from any of the above.

## Primary sources

- [Harnad 1990, The Symbol Grounding Problem](https://eprints.soton.ac.uk/250382/)
- [Cangelosi, Greco & Harnad 2000, From Robotic Toil to Symbolic Theft](https://pearl.plymouth.ac.uk/secam-research/1720/)
- [Cangelosi & Harnad, Adaptive Advantage of Symbolic Theft](https://eprints.soton.ac.uk/252616/)
- [Locatello et al. 2019, unsupervised identifiability limit](https://proceedings.mlr.press/v97/locatello19a)
- [Schölkopf et al. 2021, Toward Causal Representation Learning](https://arxiv.org/abs/2102.11107)
- [Taniguchi et al., Symbol Emergence in Cognitive Developmental Systems](https://arxiv.org/abs/1801.08829)

