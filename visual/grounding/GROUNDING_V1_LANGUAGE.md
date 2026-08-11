# GroundZero-v1: from grounded concepts to learned language

Status: executable research contract under implementation. This document does
not widen the already certified v0 claim.

## The question being answered

Knowing that a token occurs where a noun or verb is expected is not knowing
what it means. V1 asks whether an agent can learn an arbitrary word as a stable
sensorimotor distinction, use that word to control behavior, and combine it in
an utterance it has never memorized.

The benchmark does not tell the learner that one word is a noun and another a
verb. It exposes only raw observations, opaque actions/outcomes, opaque token
sequences, task feedback, and a finite interaction budget. Typed internal
operators are allowed as a computational bias; evaluator POS tags, English
names, latent object IDs and codebook maps are forbidden learner inputs.

## Operational meanings

For a history `h`, intervention `i`, and future consequence `Y`, define

\[
\Psi(h)=\{P(Y\mid h,\operatorname{do}(i)):i\in\mathcal I\}.
\]

An entry concept is an equivalence class of histories with the same controlled
consequence signature. A word `w` is directly grounded only when its learned
kernel predicts or selects those consequences:

\[
K_w(Y\mid h,\operatorname{do}(i))
\approx P(Y\mid c_w(h),\operatorname{do}(i)).
\]

The category is therefore neither the spelling of the word nor a POS label.

### Object-like concept

The v1 analogue of *house/shelter* is defined by a conjunction of invariant
structure and affordance: an entity can contain an actor, has a traversable
opening, and prevents a controlled hazard while the actor is inside. Color,
texture, position and instance identity are nuisance variables. A visually
matched object without the protective intervention effect is a negative twin.

### Process-like concept

The v1 analogue of *run/move* is a temporally extended transition schema: an
agent repeatedly applies a self-propelled motor intervention and accumulates
directed displacement over a horizon. Reversing transition order or replacing
the motor cause with an external displacement must change the process
classification even when endpoint frames match.

These examples are intentionally operational. They do not claim to exhaust the
open-ended human concepts *house* or *running*.

## Active sensorimotor toil

Let `Theta` be the version space of operational meanings compatible with an
opaque token. The next experiment maximizes expected posterior entropy
reduction per cost:

\[
i^*=\arg\max_i\frac{H(\Theta\mid D)-
\mathbb E_{y\sim P(y\mid i,D)}H(\Theta\mid D,i,y)}{\operatorname{Cost}(i)}.
\]

The ledger records the prior, every candidate score, the chosen experiment,
the observed public consequence, the posterior, and the stopping reason.
Identical hypotheses remain unresolved; the learner must not invent a label.

Active learning passes only if it reaches the same correct posterior with no
more probes than random and passive baselines over preregistered seeds. All
methods receive the same hypothesis family, observations and action budget.

## Language induction without POS supervision

A demonstration is a pair `(utterance, grounded referent)` where the utterance
is a tuple of fresh opaque tokens and the referent is a public operational
record or a typed program over already grounded anchors. The learner estimates
token-to-meaning support from contrastive co-occurrence across factorially
varied demonstrations. Full utterance memorization is insufficient because
test combinations are absent from training.

The induced language must work in both directions:

\[
\operatorname{parse}(u,h)\to p,
\qquad
\operatorname{execute}(p,h)\to \pi,
\qquad
\operatorname{describe}(p,h)\to u'.
\]

Correctness is semantic rather than textual: `u'` may use a learned synonym,
but parsing it must recover an equivalent grounded program. The round trip must
commute on held-out worlds:

\[
\operatorname{execute}(\operatorname{parse}(
\operatorname{describe}(p,h),h),h)\simeq
\operatorname{execute}(p,h).
\]

Token renaming must commute with parsing and generation. Shuffling the
utterance/referent pairing, deleting sensors or deleting actions must destroy
the corresponding entry grounding.

## Symbolic theft and the dictionary-cycle guard

After direct anchors exist, a new word may be introduced solely by a typed
definition. Grounding confidence is the least fixed point

\[
g^{(n+1)}(s)=\max\left(a_s,
\max_{d:\to s}c_d\min_j g^{(n)}(s_j)\right).
\]

Definitions with grounded leaves can transfer to unseen examples. An
unanchored cycle starts at zero and remains zero. Every positive derived word
must carry a proof tree terminating in sensorimotor episodes.

## Mandatory v1 gates

V1 uses an intersection test; every axis and its coverage floor must pass.

1. **Active acquisition efficiency:** correct posterior under the probe budget;
   active is no worse than random/passive on paired worlds.
2. **Object affordance invariance:** new shelter instances/renderers pass;
   visually matched non-protective twins fail.
3. **Process causality and order:** new trajectories pass; shuffled time and
   externally displaced endpoint twins fail.
4. **Description to action:** a held-out utterance produces the intended
   intervention/policy and world consequence.
5. **World to description:** a held-out operational fact is described with a
   semantically correct learned utterance.
6. **Factorial composition:** withheld token combinations and deeper programs
   are executed without direct composite examples.
7. **Lexicon permutation equivariance:** a fresh bijection of all surface tokens
   changes utterances but not behavior or recovered programs.
8. **Symbolic theft:** a definition-only word transfers with a grounded proof;
   an unanchored definition cycle remains unknown.
9. **Open-set honesty:** causally disconnected and unsupported concepts are
   rejected with calibrated abstention.

## Baselines and kill conditions

Run the same frozen splits with static pixels only, tokens only, actions only,
shuffled action/outcome pairs, shuffled temporal order, shuffled
utterance/referent pairs, full-utterance lookup, majority/chance, passive probe
selection and an evaluator oracle.

The v1 claim is invalid if any shortcut baseline passes a grounding gate, if a
fresh token is predicted before exposure, if the active method receives extra
information, if a learned utterance contains hidden semantic names, or if
train/test share world, object, renderer, lexicon or composite identities.

## Claim boundary

Passing v1 would establish a finite operational instance of learning words and
simple compositional language from interaction. It would not yet establish
open-world natural language, social convention, recursion without depth bound,
pragmatics, autobiographical meaning, human concepts in their full cultural
scope, consciousness or alignment.

## Primary design anchors

- Stevan Harnad, *The Symbol Grounding Problem*:
  https://eprints.soton.ac.uk/250382/
- Angelo Cangelosi and Stevan Harnad, *The Adaptive Advantage of Symbolic
  Theft Over Sensorimotor Toil*:
  https://eprints.soton.ac.uk/252616/
- Deb Roy, *Learning visually grounded words and syntax of natural spoken
  language*: https://doi.org/10.1075/eoc.4.1.04roy
- Luc Steels, *A Self-Organizing Spatial Vocabulary*:
  https://pubmed.ncbi.nlm.nih.gov/8925502/
- David Danks et al., *Actively Learning to Learn Causal Relationships*:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC13292816/
