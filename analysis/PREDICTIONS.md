# PREDICTIONS — committed before the data lands

Registered 2026-07-22 ~22:15 CEST, while the POS long run (started ~21:45, PID
39826) and the WP4 holographic-gap full sweep are mid-flight. Everything below
is falsifiable by files that do not exist yet. Knowledge cutoff for these
predictions: the build-gate smokes (results/pos_g3b_*, pos_mech_*,
holo_stream_recall_smoke*.json) — nothing from the runs in flight.

The point of this file: tomorrow's numbers either confirm or kill these — no
story-fitting after the fact. Verdicts go into POS_THESIS.md / the WP4 note
with explicit references back to each P-number.

## POS long run (results/pos_summary.json, T+40h)

- **P1 — the core ratio.** A3 captures **0.80–0.95** of A2's heldout
  improvement (point estimate **0.85**) at a gradient-token fraction of
  0.18–0.26. Basis: surprise-selected chunks carry above-average learning
  signal per token; smoke ratios ~1.0 were ignition-dominated and don't count.
  Strong-form falsifier: ratio < 0.75. Embarrassment threshold the other way:
  ratio > 1.0 sustained (gating *beats* full gradient per streamed token)
  would be a bigger result than the thesis itself.
- **P2 — gate drift.** Post-ignition gate fraction starts ~0.15 and drifts
  toward the nominal 0.25 (=1−q) as the loss curve flattens; cumulative final
  in **0.18–0.26**.
- **P3 — flat RSS.** Post-warmup RSS span < **0.15 GB** over the whole run;
  zero process restarts (stream_reconnects counts network only).
- **P4 — frozen control.** A1 heldout constant at 8.6656 ± 0.001 for 40h.
- **P5 — the twin signature.** At fork: A3R heldout == A3 (same weights, by
  construction). Then three transients: (i) online surprise excess s3r−s3 in
  the first 2h between **+0.03 and +0.15**; (ii) A3R over-gates ≥ **1.5×**
  A3's post-fork gate rate in those 2h (the restart *pays extra gradient
  tokens* to rebuild what the living state carried); (iii) rolling |s3r−s3|
  converges below 0.05 within 2–8h. End-of-run heldout gap |A3R−A3| < 0.05
  (the warmup is a transient tax, not a permanent scar — that's exactly what
  makes it a *cost of restarting*, not a capability gap).
- **P6 — injection, paired.** mean_d_inj > mean_d_rand with the majority of
  probes d_inj > d_rand (sign test). Magnitudes small: mean_d_inj ∈
  [+0.001, +0.05], mean_d_rand ∈ [−0.005, +0.005]. Second-order prediction:
  the paired difference in the **second half** of the run exceeds the first
  half — the effect grows as the γ-spectrum matures (measured injection
  transport at 60k tokens was ~1e-4; closed_loop's trained-model figure was
  +0.026).
- **P7 — probe volume.** 100–600 probes total (30/h cap, ~5h warmup + 2h
  recurrence latency, C4 4-gram recurrence rates).

### POS long-run scoring (Phase B, day 4 — results/pos_summary.json, verify_pos 16/16 exit 0)

- **P1 CONFIRMED BEYOND ITS OWN CEILING: ratio 1.0091 at grad-token
  fraction 0.2517** — the registered embarrassment threshold ("ratio > 1.0
  would be a bigger result than the thesis itself") fired. A3 8.6588→4.7430
  vs A2 8.6588→4.7782 on 909.7M streamed tokens.
- **P2 CONFIRMED**: cumulative gate fraction 0.2517, post-ignition 0.2516
  (band 0.18–0.26).
- **P3 FALSIFIED as written**: RSS span 0.972 GB vs the <0.15 GB band — the
  prediction ignored the index, the twin's second model, and the windows.
  Absolute ceiling 1.094 GB for the whole organism; zero restarts.
- **P4 PARTIAL**: constancy perfect (Δ0.0000 over 40h); anchor value 8.6588
  vs the smoke-derived 8.6656±0.001.
- **P5 FALSIFIED IN MAGNITUDE, CONFIRMED IN FORM — the restart is free**:
  (i) surprise excess 0.0029 vs [0.03,0.15]; (ii) over-gating 1.0002× vs
  ≥1.5×; (iii) converged at n+1 CHUNK vs 2–8h; end gap −0.0065 (A3R ahead).
  Third independent measurement of the two-timescale law (P23, P38a).
- **P6 FALSIFIED as registered, mechanism intact**: both deltas negative
  (stale 5–11M-era spans on an 80×-older model disturb absolutely) but the
  paired contrast is sharp — inj −0.0866 vs rand −0.2793 (3.2×), helped
  0.35 vs 0.10. Provenance-limited, not mechanism-dead.
- **P7 EXCEEDED**: 1020 probes vs 100–600.

## WP4 full sweep (results/holo_stream_recall.json, tonight)

- **P8 — persistence axis (G).** For every (P, arm=carried) cell whose
  curriculum ignited: accuracy at G=128 within **10 pp** of accuracy at G=8 —
  a plateau, not an exponential decay. This is the theory's sharpest claim
  (the phase does not rotate during a gap; see HOLO_CARRIER_THEORY.md). The
  zeroed-at-gap null sits at chance (~0.0625) in every cell.
- **P9 — capacity axis (P).** Carried recall at G=8 orders as ~1/√P
  interference: P=1 ≈ 1.0, P=2 ≥ 0.5, P=4 ≥ 0.25 — *conditional on ignition*;
  ignition itself is the biggest uncertainty (2000 iters may not clear the
  0.9-curriculum bar at P=4; a stuck curriculum is a budget statement, not a
  capacity statement, and must be labeled as such).
- **P10 — factorization.** recall(P,G) ≈ f(P)·g(G): the G-shape is the same
  across P (correlation of normalized G-profiles > 0.9 across ignited P
  cells). If this holds, gap-persistence and pair-capacity are *independent
  axes* — the disruptive reading in the theory note.
- **P11 — the phase pays rent at P≥2.** holo_on − holo_off ≥ **+10 pp** at
  P=2, G=8 (rank-2 vs rank-1 per channel). At P=1 the two arms tie (both
  100%): one binding needs no key-conditioning — that tie is *predicted*, not
  a failure of the mechanism.

## Day-1 moonshots (registered ~07:20, before the --full runs)

- **P12 — held-out-key binding** (src/holo_heldout_keys.py, full run in
  flight): original form — holo generalizes to unseen keys (≥3× chance,
  ≥ off + 10pp), off drops to ~chance. The smoke already fired the
  pre-written alternative branch (BOTH arms generalize at 600 iters); the
  full-budget question is whether the v3 phase advantage (+25 pp) reappears
  on train keys and extends to test keys. Measured state: the channel-
  allocation lookup story is already wounded; embedding geometry may carry
  key identity for both mechanisms.
- **P13 — γ-knee mobility** (src/holo_gap_knee.py): full-sequence training
  (the equivalence theorem licenses train-unchunked/deploy-chunked) moves
  the recall knee from 32<G*<128 (v3) to ≥256; with a γ-kickstart head to
  ≥1024; knee position correlates with the measured filler-γ across
  variants (the theory's G* ≈ ln-margin/(1−γ) tested directly).

## Day-1 second wave (registered ~08:30, before any of the four runs)

- **P14 — magnitude-normalized read** (M3): the M2 blocker is readout margin,
  not γ (τ≈700 channels exist). Renormalizing |S| before the de-rotation read
  (phase is intact by theory §1) moves the knee from 256 to **≥1024**; zeroed
  null stays at chance.
- **P15 — closed wake/sleep cycles** (M4): iterated collect→consolidate
  cycles beat plain continued training at equal TOTAL gradient budget
  (final heldout lower by ≥0.02), and the per-cycle sleep dividend does not
  collapse to zero after cycle 1 (it is a mechanism, not a one-off).
- **P16 — state+index hybrid on MQAR** (M5): with a surprise-gated external
  index and paired injection at query time, recall at P=16 (far above the
  state's capacity) reaches **≥0.9** (state alone ≤0.15, chance 0.0625);
  random-injection control ≈ state alone; at P=2 the hybrid is not worse
  than state alone (the index must not hurt within-capacity recall).
- **P17 — does the phase advantage return with capacity?** (M6): at P_max=64
  (where M1 found holo==off at d=64), scaling d_model to 256 restores
  holo−off ≥ **+15 pp** at G≤32. If it does NOT return, the v3 phase
  advantage is a small-key-space phenomenon — scored as registered either way.

## Wave 3 — the dynamic portfolio opens (registered day 1 ~15:05, see analysis/MOONSHOTS.md)

- **P18 — learned to be reminded (MS1):** training WITH stochastic
  consultation (p=0.5 correct, p=0.1 wrong injection) lifts hybrid@P=16 from
  M5's 0.51 to **≥0.85**; base (no injection) stays within 5pp of M5's base
  (the skill is arbitration, not degradation); and on wrong injections the
  trained model loses LESS than M5's random arm did (it learns to weigh
  state against reminder).
- **P19 — the dream generator (MS2):** training on the model's own sampled
  continuations ("dreams", same gradient budget, warm opt) beats fresh data
  (self-distillation stabilizes: dream delta > fresh delta by ≥0.03) but
  loses to stored-span replay (real surprises carry information dreams
  cannot invent). If dream ≥ sleep instead: storage-free consolidation —
  MS6/MS7 redesign per MOONSHOTS.md rule 2.
- **P20 — domain shock (MS3):** on C4→code→C4, full-gradient learns code
  fastest but forgets most (WT-2 heldout degrades ≥0.15 during the code
  phase); surprise-gating alone forgets less at comparable code plasticity;
  gating + dosed replay of phase-1 spans forgets the least (≤50% of
  full-gradient's forgetting) — sleep as the anti-forgetting organ.

## Foundations track / wave 4 (P21–P22)

*P21 and P22 were written into their builder scripts' docstrings before the runs
launched; unlike P1–P20 they were committed together with the first harvest, not
in advance — recorded here with that caveat, scored with the same rigor.*

- **P21 — language-stream holographic graft (MS5,
  src/holo_language_graft.py):** the holo graft on the frozen 400M-token POS
  snapshot clears ≥3× chance (≥0.19, chance 0.0625) at G=128 on real text;
  the ctrl graft stays below that; the zeroed null decays to chance.
- **P22 — family transfer (F5, src/pos_family_transfer.py):** S6-POS-ratio ≥
  0.85 × GSSM-POS-ratio (family-generic gating); GSSM-full ≤ S6-full + 0.1
  nats (architecture-competitive, the DD baseline). **SCORED on the 6M full
  run (results/pos_family.json): CONFIRMED, both parts** — ratio-of-ratios
  0.9804 (GSSM 0.9523 at gate 22.6%, S6 0.9337 at gate 23.1%); head-to-head
  GSSM-full 5.177 vs S6-full 5.333 nats — GSSM *leads* by 0.156 nats at scan
  parameter parity 1.0016 and identical pipeline/tokens/seed.
  **HARDENED to n=3 seeds across two CPU architectures (day 4):** seed43
  (Mac/ARM): ratio 0.955, GSSM +0.148; seed44 (core/x86): ratio 0.968,
  GSSM +0.133. Both verdicts replicate on every seed and both machines.

## Wave 5 — the deployment primitives (registered day 3 ~20:00, before any build)

- **P23 — weight hot-swap on the living stream (MS11,
  src/state_weight_swap.py):** carrying an OLD state into NEW weights works.
  With W(359M) fixed and fresh C4 chunks cloned to all arms (forward-only):
  (a) the cold-start arm's online-NLL excess over the native arm in the first
  50 chunks is ≥ 2× the hot-swap-far arm's (state from 128M, 231M tokens of
  training distance); (b) both hot-swap arms converge to within 0.01 nats of
  native inside 300 chunks; (c) the channel-shuffled control is worse than
  hot-swap-far throughout (compatibility is structural, not a bias artifact).
- **P24 — hot-swap growth (MS8, src/hot_swap_growth.py):** function- and
  state-preserving widening (channel duplication, d64→d128, carried Z
  migrated) on a live C4 stream: (a) surgery equivalence |Δlogits| < 1e-4;
  (b) no post-surgery transient (first-20-chunk online NLL within 0.05 of the
  stay-d64 arm); (c) at +1.5M post-surgery tokens the grown arm's held-out
  beats stay-d64 by ≥ 0.03 nats (the new capacity is used); (d) grown beats
  fresh-d128-from-scratch at the same wall-token axis (growth beats restart).
- **P25 — α-shut pollution control (MS12, src/holo_alpha_shut.py):** the F3
  discovery is causal: adding an α(x_filler) regularizer to the T2+MagNorm
  knee recipe (λ sweep incl. λ=0 control) (a) cuts trained filler φ-drift at
  G=512 from ~1.5 rad to < 0.5 rad, (b) moves the recall knee to ≥ 1024, and
  (c) drift reduction and knee position are dose-monotone in λ. If large λ
  strangles the write itself, that trade-off boundary is the measurement.

- **P26 — the stored bit survives the surgery and the swap (MS13,
  src/beacon_swap.py, registered before the build):** on the beacon
  idle-persistence task (write-once-freeze carrier, streaming_train.py §D/E):
  (a) across the MS8 widening surgery (d64→d128, carried-Z migrated,
  post-gate) beacon recall through a 256-token gap stays ≥ 0.99 — the bit
  survives the brain operation exactly; (b) within one training run, writing
  the bit under W(T1)'s encoder and reading it under W(T2)'s decoder (weights
  from 2× further training, state carried across the swap) keeps recall
  ≥ 0.9 — the state CODE (which channel carries the bit, at what scale) is
  stable across training distance once the carrier has locked; (c) the
  channel-shuffled state control collapses to chance at both (a) and (b). If
  (b) fails while (a) passes, the reading is: state code drifts with
  training distance, and a state-alignment map becomes the next disclosed
  primitive — either outcome is the measurement.

### Wave-5 scoring (as results land)

- **P23 SCORED (results/state_weight_swap.json, A3 organism, 600 chunks):**
  (a) **FALSIFIED, cleanly and instructively** — cold (Z=0) shows NEGATIVE
  first-50-token excess (−0.085) vs native; the carried fast-path state holds
  no NLL value beyond the ~5–8-token receptive field. This is F2 measured
  from the other side, not a bug. (b) **CONFIRMED, far beyond the bar** —
  both swap arms converge to the native trajectory in 4 chunks (256 tokens;
  bar was 300 chunks). (c) inconclusive by construction: past chunk ~4 all
  arms are bit-identical (deterministic convergence), so windowed separation
  only exists in the onset (where the 30-trial diagnosis separates shuffled
  at +0.9). Net reading: weight hot-swap on the fast path is a NON-EVENT —
  no lock-in, no compatibility risk, nothing to migrate; the portable value
  lives in the slow channels (→ P26).
- **P24 SCORED (results/hot_swap_growth.json, 1.2M+1.2M):** (a) CONFIRMED
  6.7e-6; (b) CONFIRMED gap 0.046 (and conservatively measured: the grown
  arm restarts Adam, stay64 keeps warm moments — the pure surgery transient
  is smaller); (d) CONFIRMED growth beats restart by 0.127 nats. (c)
  **FALSIFIED at this scale**: grown trails stay64 by 0.036 at 1.2M
  post-surgery tokens — but the deficit HALVED from 0.073 at 300k, monotone;
  next lever registered: migrate Adam moments through the duplication map
  instead of resetting, and/or longer horizon.

## Wave 6 — the pull-system queue (registered day 3 ~20:45, before any build)

- **P27 — Adam-moment migration closes the growth deficit (MS14,
  src/hot_swap_growth.py --migrate-moments):** migrating optimizer moments
  through the duplication map (exp_avg via the gradient transform, exp_avg_sq
  via its square) instead of resetting Adam: (a) the commutation gate holds —
  grow(adam_step(m64)) == adam_step(grow_with_moments(m64)) to < 1e-3 on all
  parameters after one identical-batch step; (b) at 1.2M post-surgery tokens
  the grown arm's deficit vs stay64 shrinks from −0.036 to ≥ −0.01 or turns
  positive; (c) growth still beats restart.
- **P28 — curiosity homeostasis beats fixed q under shock (MS4,
  src/pos_auto_q.py on the MS3 harness):** a homeostat that regulates q to
  hold a target gate rate (~25%): (a) holds gate_frac in [0.18, 0.32] through
  the C4→code→C4 shock while fixed q=0.75 overshoots during phase 2 (R2
  measured 0.58); (b) at MATCHED total gradient tokens, auto-q's WT-2
  forgetting ≤ fixed-q's and its code plasticity ≥ 0.9× fixed-q's — the
  homeostat spends the same budget on better-chosen chunks; (c) the q
  trajectory itself is the measurement: it must RISE during the shock (the
  organism becomes pickier when everything is surprising) and relax after.

- **P25 SCORED (results/holo_alpha_shut.json + _lam0check.json): FALSIFIED
  in the registered regime, with the mechanism found.** The α-filler
  regularizer neither reduces trained drift@512 (1.19 vs 1.20 reference) nor
  moves the knee (λ=0 knees 1024/512 across seeds; λ>0 knees 256–512 — the
  regularizer HURT where it did anything). Root cause, discovered by the
  builder's forensic pass: the M3 recipe's curriculum NEVER ignites —
  final_train_gap=2 in all six cells of the original holo_magread run too;
  the 512 knee was always pure kickstart+magnorm EXTRAPOLATION from a
  gap-2-trained model. So the regularizer only ever acted on 2 filler
  positions; α-behavior on long gaps was never trained in any arm. The
  registered lever cannot reach the mechanism in this regime — clean kill.
- **P29 — inference-time α-clamp, the direct pollution causality test (MS12b,
  registered before the build):** on the M3-recipe model (λ=0, gap-2-trained,
  knee 512), clamping α(x) toward 0 on filler positions AT EVAL ONLY (state
  write suppressed during silence, no training change): (a) trained filler
  φ-drift at G=512 drops below 0.3 rad (untouched: ~1.3); (b) the recall
  knee moves to ≥ 1024; (c) recall at G≤128 is unchanged within 5 pp (the
  clamp must not damage in-range recall). If (b) fails while (a) passes,
  pollution is real but NOT the binding constraint at 512+ — the
  alternative (magnitude floor? phase SNR?) becomes the next measurement.

- **P29 SCORED (results/holo_alpha_clamp.json): (a) CONFIRMED — the eval
  clamp eliminates trained filler drift exactly (0.0 rad @512, both seeds;
  the zero-drive law demonstrated on the trained model); (c) CONFIRMED
  (in-range recall −5pp, at tolerance); (b) NOT MET — the clamped knee (640
  mean) is not higher than unclamped (768). The registered fallback fired, with
  the mechanism visible in the raw data: WITHOUT filler writes the magnitude
  collapses unfed (mag_ratio → 0.0007 @2048, snr-alive 1.0→0.09) while
  unclamped filler writes REFRESH it (30–80×) even as they pollute the
  phase. The filler write is a double agent: phase pollutant, magnitude
  feeder. The knee past 512 is MAGNITUDE-bound, not pollution-bound.**
- **P30 — the 2×2 that disentangles the two axes (MS12c, registered before
  the build):** eval-time arms {clamp, no-clamp} × {in-state magnitude
  refresh at chunk boundaries (renormalize |S|→1, phase untouched — the
  variant disclosed in FOUNDATIONS F3; eps-guarded so a zeroed state stays
  dead), no-refresh} on the M3-recipe model: (a) clamp+refresh knee ≥ 2048;
  (b) ordering clamp+refresh > refresh-only > unclamped > clamp-only (drift
  costs recall once magnitude is guaranteed; feeding beats starving); (c)
  zeroed-at-gap null stays at chance in every arm (the refresh must not
  invent information); (d) in-range recall @G≤128 within 5pp everywhere.

- **P30 SCORED (results/holo_clamp_refresh.json): (a) CONFIRMED — knee
  2176 mean under clamp+refresh (seed0 reaches the 4096 end of range at
  acc 0.33 vs chance 0.06; seed1 stays at 256 — real seed variance, stated);
  (d) CONFIRMED (in-range recall 0.41–0.46 everywhere). (b) NOT MET as a
  strict ordering, and that is the finding: refresh WITHOUT clamp is the
  WORST arm (512 < clamp-only 640 < untouched 768 << both 2176) — the two
  axes INTERACT, they do not add. Refresh rescales the polluted direction
  (it cannot heal what only the clamp prevents); the clamp starves the
  magnitude (only the refresh feeds it). Persistence = clean phase AND fed
  magnitude, jointly. (c) nominally out at 12/48 null cells but symmetric
  around chance with mean deviation +0.005 — sampling noise at the small
  large-G eval batches, not information manufacture (eps-guard verified in
  isolation).**

- **P31 — two organisms, one index: A pre-learns the shock for B (MS7,
  src/pos_shared_index.py, registered before the build):** organism A
  streams a C4+code mix and stores its surprise spans; organism B streams
  pure C4, then hits a code shock (the MS3 protocol). Arms for B, matched
  budgets: (i) B replaying A's shared spans during the shock, (ii) B with
  only its own (pure-C4) spans, (iii) B with no replay, (iv) control: B
  replaying token-shuffled versions of A's spans. Predictions: (a) B's
  code-phase forgetting with A's spans ≤ 0.7× of arm (ii); (b) B's code
  plasticity in arm (i) ≥ arm (ii) (A's spans carry usable code, not just
  regularization); (c) the shuffled control (iv) does NOT beat (ii) —
  the benefit is content, not noise injection; (d) A itself is unharmed
  (its own trajectory is not part of B's budget). This is collective
  memory across O(1) individuals: one organism's surprises become another's
  immunity.

- **P20 SCORED (results/pos_domain_shock.json, full):** PARTIAL — the core
  confirmed, one clause falsified, one surprise. Core CONFIRMED: R3
  (gating+dosed replay) forgets +0.246 = **37% of R1's +0.667** (bar was
  ≤50%) — sleep is the anti-forgetting organ. Clause FALSIFIED: R2's
  "comparable code plasticity" does not hold (0.520 vs 0.854 at 58% of the
  gradient tokens — the forgetting/plasticity ratio is dose-coupled, as the
  smoke already showed). Surprise, inverted from the smoke: in the full run
  R1 recovers almost fully (+0.010) while R3 stays +0.234 above baseline
  after phase 3 — at full scale the replay that protected WT-2 during the
  shock keeps pulling the model toward stored spans during recovery
  (an overdose signature on the recovery phase; the volume-coupled budget
  needs a phase-3 dividend monitor). Method note: the code-eval carries
  unk_rate 0.44 (WT-2 vocab on Python), so absolute code-NLL levels are
  diluted; the WT-2 forgetting signal — the core measurement — is clean,
  and all regimes share the same eval.
- **P27 SCORED (results/hot_swap_growth_mig.json, 1.2M+1.2M on the x86 runner):**
  (a) CONFIRMED (commutation 29/29; full-run gate 5.7e-6); (c) CONFIRMED —
  growth beats restart by +0.166 (third independent replication: +0.127
  fresh-Adam Mac, +0.152 smoke, +0.166 here). (b) FALSIFIED at the 1.2M
  horizon: deficit −0.033 vs −0.037 unmigrated — the smoke's 87% deficit
  reduction was a TRANSIENT effect that decays over Adam's β₂ memory
  horizon (~0.5M tokens). The real, robust gain of moment migration is the
  transient itself: post-surgery gap 0.046 → **0.0035** (13× smaller) —
  migration makes growth seamless, not faster-converging.
- **Seed-hardening correction (results/holo_magread_seeds23.json, the x86 runner):**
  the M3 "knee 512, seed-stable" claim was n=2. At n=4 the M3-recipe knee
  is 512/512/128–256/128–256 (seeds 0–3) and the V1/V2/V3 variants do not
  order consistently within seeds. What replicates: the v1→v3 repair, the
  interventions' direction, and the clamp+refresh interaction jump; the
  absolute knee position is seed-dependent (also seen in P30: 4096 vs 256).
  Documents updated accordingly — the correct unit is the intervention
  effect, not the knee coordinate.
- **P28 smoke note (results/pos_auto_q_smoke.json — not scored, method
  finding):** the q-controller saturates at the 0.95 clip in ALL phases —
  it integrates during the ignition window (forced gating, r̂=1.0) and
  never recovers: an anti-windup bug, not a homeostasis result. Fix: freeze
  the controller during ignition. Efficiency signal already visible: auto
  matches fixed's forgetting (+0.274 vs +0.281) at 18% fewer gradient
  tokens. Full run follows after the fix.

- **P28 SCORED (results/pos_auto_q.json, full, anti-windup fixed):
  FALSIFIED — and the mechanism is instructive.** The controller works as
  designed (q moves freely: medians 0.75/0.73/0.76; phase-2 gate rate lands
  exactly on target, 0.247 vs r*=0.25). That is precisely why it loses:
  under shock it holds the RATE constant instead of getting pickier — q
  even dips during phase 2 (0.733 < 0.75), admitting mid-surprise chunks —
  and forgets MORE than fixed q (+0.401 vs +0.350) at equal code
  plasticity (0.564 vs 0.565). (a) also out: both arms gate ~0.45–0.49 in
  phase 1 (the registered band was too tight for a warm-snapshot resume).
  Verdict: rate-homeostasis is the WRONG controller — the fixed
  quantile's shock response (gate the top surprises, whatever the rate) is
  the better curiosity policy. A surprise-LEVEL target instead of a rate
  target would be a new registered question, not a rescue of this one.

- **P32 — the rent map of the phase (MS10, src/holo_rent_map.py,
  registered before the build):** sweeping P_max ∈ {8,16,32,64} × d_model ∈
  {32,64,128,256} (holo_on vs holo_off, P ∈ {2,4}, G ∈ {8,32}, 2 seeds,
  matched budgets; best-of-2 lr control for the off arm at the corner
  cells — the lr lesson): (a) the phase rent (holo−off) is governed by the
  ratio d/P_max, with a transition in the band d/P_max ≈ 2–4 (anchors
  measured so far: rent at 16/64=×4 [v3, +25–30pp], none at 64/64=×1 [M1],
  small at 64/256=×4-but-large-d [M6, +13–15pp]); (b) cells collapse onto
  one curve in d/P_max (a ratio law, not two independent axes); (c) the
  off arm's recall varies smoothly with d and is flat in P_max beyond
  interference (no phase mechanism to gain rent from). Any cell where the
  lr-control flips the sign is reported as such, not averaged away.

- **P31 SCORED (results/pos_shared_index.json, full): CONFIRMED, all four
  checks.** B replaying A's spans through the code shock forgets +0.156 vs
  +0.233 with only its own spans (ratio 0.67 ≤ 0.7) at BETTER plasticity
  (0.542 vs 0.518); the token-shuffled control (+0.317) is nearly as bad as
  no replay (+0.376) — the benefit is content, not noise regularization.
  A's store: 64 spans, 7.8% code-like. Collective memory across O(1)
  individuals is real: one organism's surprises are another's immunity.
- **P32 SCORED (results/holo_rent_map.json, full 16-cell grid): (a) PARTIAL
  (low-ratio cells < 5pp holds; high-ratio ≥ 10pp fails), (b) FALSIFIED —
  no ratio law (ratio-4 cells span 44pp: +32.1 to −11.9) and the interim
  product law also breaks at d=256. The map itself is the result: the
  phase pays robust rent only in the SCARCE CORNER (P_max ≤ 16 AND
  d ≤ 64: +11 to +32pp), a near-zero valley in between (±3pp), and a
  second positive region at mid-P_max × large d (16–32/256: +12.5/+13.4 —
  the M6/P17 capacity-return, now mapped), with the extreme corners
  lr-sensitive and excluded. No one-parameter law fits; two rent regions
  separated by a valley is the measured shape.
- **P21 SCORED (results/holo_graft.json, full, both seeds): FALSIFIED as
  registered — and explained by the rent map.** The graft on the frozen
  475M-token organism recalls facts over REAL C4 text at 0.90–0.97 through
  G=32 (zeroed null at chance in every cell — the carried state carries the
  binding; real-text filler γ reaches 0.9996), but G=128 lands at 0.12–0.14
  (~2× chance, not the registered 3×; ctrl exactly at chance). And the
  phase rent vs ctrl vanishes at full budget (+37pp at 600 iters → +1–3pp
  at 2500): the graft sits at d=128/P_max=16 — P_max·d=2048, squarely in
  the rent map's valley. Two instruments, one law.

## Wave 7 — the three gaps it would be stupid not to close (registered day 4 ~07:30, before any build)

- **P33 — CHIMERA v0 (MS6, src/chimera.py, spec in analysis/CHIMERA_SPEC.md):**
  on the MS3 shock protocol at matched gradient tokens: (a) CHIMERA
  forgetting ≤ R3's (+0.246) at ≥ R3's plasticity (+0.366); (b) CHIMERA
  recovery beats R3's +0.234 — the dividend monitor must fix P20's
  recovery overdose; (c) each ablation (minus-reminder, minus-monitor) is
  worse than full CHIMERA on ≥1 axis; (d) no single-organ arm dominates
  CHIMERA on all three axes. F1's locking experiment.
  **Run-time note, registered BEFORE the full run (2026-07-24 ~15:00):**
  smoke (results/chimera_smoke.json, 120 chunks/arm): all five arms ran
  end-to-end; the dividend monitor demonstrably intervened (sleep SUSPENDED
  at EMA=−0.040, 7 monitor skips); the reminder organ fired ZERO times in
  every arm. Mechanical verification (unit test, same day): store→harvest→
  lookup fires correctly on an exact 4-gram recurrence — the organ works;
  the smoke's exposure is below the recurrence base rate (POS-measured:
  ≥1020 capped hits / 900M tokens / 20k spans ⇒ ~0.1 expected fires at
  smoke exposure; observed 0, consistent). The full run (150 chunks/phase)
  expects single-digit fires from the C4 base rate; code-phase boilerplate
  may raise it. Scoring consequence, fixed now: if fewer than 10 reminders
  fire in the chimera arm, clause (c)'s minus-reminder ablation is scored
  UNDECIDABLE-AT-EXPOSURE (neither confirmed nor falsified) — the organ's
  in-composition value then needs index-scale exposure or a
  recurrence-seeded protocol as a NEW registered experiment. Instrument
  dimensioning (max_per_chunk=2, spike_min_nll=7.0) deliberately left AS
  SPEC'D — no post-smoke retuning.
  **SCORED 2026-07-24 (full on the second x86 runner, results/chimera_full.json; 150
  chunks/phase, 5 arms):** instrument deviation named first: the
  registered "matched gradient tokens" clause is unenforceable for gated
  arms (the gate chooses); arms ran at matched CHUNKS — chimera used
  100.9k grad tokens vs r3's 141.3k vs r1's 230.4k. (a) FAILED as
  measured: chimera forgetting +0.259 > r3's +0.189 at essentially equal
  plasticity (0.625 vs 0.621) — with the caveat that chimera took 29%
  fewer gradient tokens. (b) CONFIRMED DECISIVELY — the headline: chimera
  residual damage after recovery +0.0028 (fully healed) vs r3's +0.151;
  and the ablation nails the attribution within the run: no_monitor
  (same organs, monitor removed) regresses to +0.146 while the monitor
  actively suspended 9 sleep blocks in the chimera arm. The dividend
  monitor IS the recovery organ — P20's overdose, fixed in composition.
  (c) SPLIT: minus-monitor worse CONFIRMED (recovery collapses);
  minus-reminder UNDECIDABLE-AT-EXPOSURE per the pre-registered rule —
  only 2 reminders fired (<10), the no_reminder deltas cannot be
  attributed to the organ. (d) CONFIRMED: no single-organ arm dominates
  (r3 wins forgetting, chimera wins plasticity+recovery; r1_full is the
  firehose signature — best plasticity +0.866, catastrophic forgetting
  +0.590, echoing the 40h result). Verdict: CHIMERA v0's composition
  trades some shock-forgetting for near-perfect recovery at equal
  plasticity and fewer gradient tokens; the monitor earns its place, the
  reminder organ still awaits index-scale exposure.
- **P34 — the phase lifts binding rank per channel to ≥2 (src/rank_sweep.py):**
  an Eckart–Young-style capacity sweep (K keys vs D channels, the
  rank1_capacity method) on the COMPLEX holographic state: (a) inverting
  the recall bound gives D_eff ≥ 1.8 per channel for the phase arm where
  the scalar arm inverts to ~1.0 (measured anchor: D_eff≈1.02); (b) the
  phase arm's capacity cliff sits at load ≈ 2K/D, the scalar's at ≈ K/D;
  (c) attention validity gate at ~1.0 throughout. If D_eff stays ~1, the
  phase's rent is NOT rank — the alternative (SNR-based) is stated.
  **AMENDED before the sweep ran (day 4 ~08:15), reason on record:** the
  0.1406 anchor's generating script is lost (two documented reconstruction
  attempts land at chance; the artifact paper/evidence_companion/
  hybrid_B.json remains the anchor's source of truth, flagged as
  reproduction debt in RANK1_CAPACITY_THEOREM.md). P34 therefore runs on
  the reproducible mqar.py instrument with the criterion made RELATIONAL:
  (a') D_eff_phase ≥ 2× D_eff_scalar at every K where phase clears 3×
  chance, and D_eff_phase(K=8) consistent with the measured 8.9% ceiling
  (≈0.6 model-wide); (b') the phase cliff sits at ≥2× the scalar cliff-K;
  (c) unchanged. Same law, corrected instrument.
  **SCORED 2026-07-24 (full grid on the reference Mac,
  results/rank_sweep_final.json; K∈{2,4,8,16,32} × 4 arms × 4 seeds):**
  (c) CONFIRMED — attention validity min recall 0.9898, the tasks are
  solvable and the harness correct. Anchor reproduction is EXACT for
  scalar (0.0166 vs anchor 0.0170±0.0022), phase_off, and attn — the
  instrument is calibrated. (a') FALSIFIED at the one eligible cell and
  unscorable elsewhere: at K=2 (3/4 ignited) the ratio is 0.371 — the
  phase's per-channel capacity among ignited seeds is LOWER than the
  scalar's, the opposite sign of the prediction; no higher K clears 3×
  chance on seed-mean. (b') FAIL as computed (cliff ratio 1.0). The
  registered fallback therefore ENGAGES as written: the phase's measured
  rent (P32's map) is NOT per-channel rank — the SNR-based alternative is
  now the standing hypothesis. Dominating phenomenon and method finding:
  phase IGNITION COLLAPSES WITH LOAD (3/4 → 1/4 → 1/4 → 0/4 → 0/4 across
  K) — whatever the phase could pay at high K, training reliability dies
  first; and the single ignited K=8 seed (0.0728) sits ~1σ below the
  5-seed anchor mean, i.e. the ignition coin drifts with machine co-load
  even on the reference machine. Process note: the full was launched
  without --verify-anchor (the harness's own guard); the post-hoc anchor
  comparison above recovers it. The 0.1406 Task-B anchor's reproduction
  debt (P34 amendment) stands unchanged — this scores the mqar.py
  instrument, on which the rank hypothesis is dead.
- **P35 — the gap ladder to a million (src/gap_ladder.py):** eval-only, on
  the M3-recipe model with the P30 clamp+refresh prosthesis, gaps
  {4096, 16384, 65536, 262144, 1048576} chunked-carried: (a) MQAR recall at
  G=65536 ≥ 0.5× its G=4096 level; (b) the beacon bit (write-once-freeze
  carrier, MS13 harness) survives G=1M at recall ≥ 0.9 with the refresh
  prosthesis (γ=0.9995 alone decays at τ≈2000 — the refresh is load-bearing
  and that is the point); (c) zeroed null at chance at every rung (large
  eval batches per P36's protocol). Any rung that breaks is the measured
  wall, reported as such.
  **SCORED 2026-07-24 night (MQAR: results/gap_ladder_full.oom1239.log +
  rerun log, both seeds complete under the O(1) eval fix, bit-identical
  regression; beacon: results/gap_ladder_beacon.json after the
  cold-path fix):** (a) FALSIFIED in both seeds — recall@65536 /
  recall@4096 = 0.125 (seed 0) and 0.31 (seed 1); the MQAR wall stands at
  G=16384, exactly as the interim saw it. (b) FALSIFIED WITH A MEASURED
  BREAKPOINT — the refresh ladder holds 1.0000 through G=65536 (+49pp
  over no-refresh), then breaks: 0.5600 at 262144 (+11.5pp), −1.0pp at
  1M where both arms sit at the cold floor. The beacon knee lives between
  65k and 262k; the γ-compensation is load-bearing to 65k and dead by
  262k. (c) as-written NOT MET at this run's eval power (max null dev
  3.75pp vs the 3pp line at eval_batch=100, where 2σ ≈ 4.9pp —
  underpowered, anticipated; P36's dedicated batch-400 nulls stand and
  are the authority). SUMMARY: silence extrapolation's measured boundary is
  now measured on BOTH instruments — F6 gains two precise walls (MQAR
  16k, beacon 65k→262k) in place of an open-ended promise. Method
  footnote: this measurement cost three OOM kills with two distinct
  mechanisms (MQAR per-chunk logits accumulation; beacon cold-control's
  unchunked (NB,G) forward) — both fixed, both regression-verified
  bit-identical.
- **P36 — P30c null hardening (src/holo_alpha_shut.py --null-hardening):**
  the zeroed-at-gap null re-run with eval batches ≥ 100 at G ∈ {1024, 2048}:
  all null cells within 3pp of chance. Closes the F6-flagged sampling-noise
  gap.

- **P37 — the future-trained organism (MS15, src/horizon_pos.py, registered
  before any build; David's architecture insight, day 4):** add an H-step
  prediction head to the streaming organism (predict a summary functional of
  chunk t+H at chunk t; score on arrival; horizon-surprise = the error).
  On the MS3 shock protocol: (a) horizon-surprise (H ≥ 8 chunks) rises ≥ 5
  chunks EARLIER at the domain boundary than 1-step surprise (the
  early-warning claim); (b) gating on a mix of 1-step and H-step surprise
  at matched gradient tokens forgets ≤ the 1-step gate's forgetting with
  plasticity within 10%; (c) the deposited-prediction mechanism is sound:
  shuffling the deposited predictions destroys (a) and (b). If (a) holds
  but (b) does not, the horizon signal is a detector, not yet a teacher —
  reported as such.
  **SCORED 2026-07-24 (full on the second x86 runner, results/horizon_pos_full.json):**
  (a) FALSIFIED — the H=8 detector fired 65 chunks LATER than 1-step
  surprise at the phase1→2 boundary (fire idx 304 vs 239) and never fired
  at phase2→3. No early warning at this scale — consistent with the
  smoke's structural finding. (b) CONFIRMED with room: horizon-mix gate
  forgetting +0.300 vs base gate +0.392, at plasticity +0.736 vs +0.544 —
  both axes better, not a trade. (c) FALSIFIED, and this kills the
  attribution: the SHUFFLED-deposit control keeps the (b)-shaped benefit
  (forgetting +0.324, also beats base) — whatever improves the mixed
  gate, it is NOT the content of the deposited predictions. Verdict: at
  this scale the future-trained gate is neither detector (a) nor
  attributable teacher (c); the measured (b) win is real but structural —
  a second, differently-tempered surprise stream diversifies the gate.
  Next attack (unregistered design note): a rate-matched noise-gate
  control carrying the same firing statistics but zero predictive
  content — if (b) survives it, gate DIVERSITY is a cheap new F1 organ in
  its own right; if it dies, the horizon content matters and the shuffle
  control was too weak. The multi-horizon architecture idea stays open at
  larger H / richer targets — this scores THIS instrument, not the idea.
- **P40 — gate diversity vs. content (the P37 attribution decider,
  registered 2026-07-24 before any build):** three arms on the MS3 shock
  protocol at matched gradient tokens, same ckpt, seed 42 primary:
  base_gate (1-step only), horizon_gate (P37's mix, unchanged), and
  noise_gate — identical machinery and compute, except at gate-decision
  time the second stream's value is the horizon-surprise from a uniformly
  random EARLIER chunk (large random lag): distribution-identical,
  rate-matched through the same rolling quantile, zero temporal content.
  (a) REGISTERED POINT CALL: the noise arm RETAINS ≥70% of horizon_gate's
  forgetting improvement over base — the P37-(b) win is DIVERSITY, not
  content. If instead horizon beats noise by >0.03 nats forgetting at
  ≥ noise's plasticity, content matters and P37's shuffle control was too
  weak — reported at full strength either way. (b) sanity: noise arm's
  realized second-stream firing rate within 2pp of horizon_gate's.
  Either outcome is a win: diversity ⇒ a near-free new F1 organ (multiple
  gate tempers); content ⇒ horizon-v2 is justified sharper.
  **SCORED 2026-07-24 evening (full on the second x86 runner,
  results/horizon_p40_full.json; head-init seeded, min_lag=50, fallback
  11.1% = the designed cold-start window):** the registered point call is
  FALSIFIED in the most informative direction — retained_fraction is
  **−0.575**: the temporally-scrambled second stream doesn't keep 70% of
  the win, it is WORSE THAN NO second stream at all (noise forgetting
  0.443 vs base 0.392, plasticity 0.559), while horizon_gate replicates
  its P37 win cleanly (0.302 / 0.726). The counter-clause FIRES once the
  implementation's sign slip is corrected at scoring (the code tested
  forg_horizon−forg_noise > 0.03; the registered sense is noise−horizon:
  **0.141 nats, ~5× the threshold, at horizon plasticity ≥ noise's** —
  reviewed-and-missed by the lead, recovered from the logged raw numbers,
  which is what raw-numbers-next-to-every-pass/fail is FOR). (b)
  rate-match fails as anticipated after the smoke: 4.0pp overall, widest
  in the shock phase (0.313 vs 0.220) — content-driven boundary firing,
  measured. SYNTHESIS with P37(c): shuffled DEPOSITS kept the win (a
  scrambled deposit is still scored NOW against NOW — its error still
  carries the current regime); time-scrambled VALUES destroy it. The
  mixed gate's fuel is neither deposit content nor generic diversity but
  the second stream's TEMPORAL COHERENCE with the current regime. This
  is the sharpest mechanism statement the horizon program has produced
  and it directly licenses horizon-v2 (self-surprise ladder), which
  maximizes exactly that component. Design note
  for horizon-v2 (numbered only when its spec freezes): SELF-SURPRISE
  target — the head predicts the organism's own chunk-NLL at t+H (one
  scalar, purer and cheaper than v1's histogram), H-ladder {2, 8, 32}, on
  a shift-denser schedule (early warning needs boundaries to warn about;
  MS3 has only two).
- **P41 — the retrodiction meter (MS18 v0, David's time-mirror, registered
  2026-07-24 before any build):** a BACKWARD head ladder on the streaming
  organism: at chunk t, one linear head per rung reconstructs the
  top-256-bucket histogram of chunk t−H from the current carried
  state/features, H ∈ {2, 8, 32, 128}. Targets are past chunks — available
  immediately from a bounded rolling buffer (no deposit queue); heads
  train online, matched budgets, MS3 shock stream, same ckpt as
  P37/P40. Registered: (a) TWO-REGIME DECAY — the two-timescale law seen
  backward: error rises steeply across the receptive-field scale and
  PLATEAUS beyond it on the weights level. Point call:
  err(H=128) − err(H=32) < 0.25 × (err(H=8) − err(H=2)). (b) Shuffle
  control: temporally shuffled targets erase the H-structure (rungs
  within noise of each other, no monotone ladder). (c) LIVE FORGETTING
  (wording clarified before any run, 2026-07-24 evening — "the phase-2
  boundary" means the boundary INTO phase 2, the phase1→2 shock at B12):
  in the window (B12, B12+32], on decisions whose target lies pre-shock
  (t−H < B12), the qualifying rungs' error rises above their own
  end-of-phase-1 mean + 2σ — the meter sees phase-1 content fade WHILE
  the model adapts to code; H∈{8,32,128} qualify richly at both smoke and
  full phase lengths (direction registered; magnitude exploratory in
  v0). Scope: v0 is the METER only — the consolidation organ
  (decay-triggered targeted replay) is a separate later registration.
  **AMENDED before the full (2026-07-24 evening, from the smoke's own
  diagnosis, results/retro_pos_smoke.json):** the smoke found a real
  instrument confound: rung H only starts training at chunk H, so higher
  rungs carry fewer optimizer steps and a ladder appears from
  under-training alone — the SHUFFLED arm showed nearly the same ladder
  (4.54/4.74/5.41 vs real 4.49/4.77/5.45), and H=128 is structurally
  empty in a 120-chunk smoke. The shuffled arm shares the exact training
  schedule, so the confound cancels in the per-rung contrast. All three
  clauses are therefore re-based on REAL−SHUFFLED per-rung differences:
  (a) two-regime decay on the CONTRAST curve; (b) becomes the contrast
  being significantly > 0 with ladder structure (the old absolute-flat
  test is retired); (c) unchanged in anchor/window but measured on the
  contrast. phase_chunks for the full raised to 200 (top rung trains ≥70
  steps before the boundary). Scoring happens from the logged raw curves
  of both arms (fully recorded), not the in-run pass/fail block.
  **SCORED 2026-07-24 night (full on the second x86 runner, phase_chunks=200,
  results/retro_pos_full.json, contrast-based per the amendment):** the
  instrument reads NULL — and the null teaches. Per-rung REAL−SHUFFLED
  contrasts at end-of-phase-1: +0.030/+0.059/+0.021/+0.021, each within
  ~1–2σ of zero, no ladder; if anything the temporally-locked target is
  slightly HARDER than a random recent chunk. The smoke's ladder is
  confirmed as pure step-count confound (absolutes converge to 4.32–4.41
  under 200-chunk training). (a)'s formal pass on a flat-zero contrast is
  vacuous and NOT claimed; (b) NOT MET; (c) no coherent contrast rise at
  the shock (+0.04/−0.12/−0.04 across eligible rungs). VERDICT:
  chunk-specific retention is INVISIBLE to bulk-histogram reconstruction
  from mean-pooled features on stationary text — the state's marginal
  statistics are interchangeable across recent chunks, so the probe
  measures the marginal, not memory. The time-mirror idea itself stands
  on other evidence: the F3 knee IS keyed retrodiction (specific bindings
  probed with keys survive measured gaps) — the meter's v1 must therefore
  probe with KEYS, not bulk summaries. MS18 v1 design note: keyed
  retrodiction probes (ask the state for a SPECIFIC past binding at lag
  H), numbered when its spec freezes.
- **P42 — the gate law survives width (the transfer axis, registered
  2026-07-24 before launch):** the exact 40h recipe (q=0.75, window 500,
  min 100, ignition 100, B=8, K=64, lr 3e-3, seed 42, from-scratch,
  streamed C4) at d_model=256 — 4× the FLOPs, 2× the width of every POS
  result so far — to 50M streamed tokens, token-scheduled evals, twin and
  index off (the pure three-arm gate law). Anchor, computed from the 40h
  run's own curve at the same token count: at 49.6M tokens d=128 stood at
  ratio_A3_vs_A2 = 0.9729 with gate 0.2472. Registered: (a) ratio(d=256,
  50M) ≥ 0.93 — the gate's token-efficiency advantage does not weaken
  materially with width; point call: it lands ≥ the d=128 anchor 0.9729
  (selection should matter MORE when each gradient is 4× as expensive is
  the intuition, stated but not required); (b) post-ignition gate
  fraction in 0.20–0.30 (the q75 band transfers). Either direction of
  (a) is a law-shaping datapoint: growth ⇒ the law strengthens with
  scale, shrinkage ⇒ width dilutes selection and the GPU-scale question
  sharpens. This is the anti-goalpost move: not another organ, the
  scaling axis of the core result itself.
  **SCORED 2026-07-24 night (results/pos_d256.log, 50,000,384 tokens in
  2.01h on the quiet Mac, RSS ≤ 1.1 GB):** (a) CONFIRMED WITH ROOM AND
  THE POINT CALL HITS — ratio(d=256, 50M) = **0.9953** (A1 8.6834, A2
  5.0363, A3 5.0534) vs the d=128 same-token anchor 0.9729: the gate's
  token-efficiency advantage GROWS with width. At 4× the FLOPs per
  gradient, the surprise-gated arm sits at 99.5% of full-gradient
  progress on 24.7% of the gradient tokens — selection pays MORE where
  gradients cost more, exactly the stated intuition. (b) CONFIRMED: gate
  fraction 0.247 inside the 0.20–0.30 band — the q75 calculus transfers
  untouched. The width axis of the core law bends UPWARD; the GPU-scale
  question sharpens in the favorable direction.

- **P38 — the portable organism (MS16, src/portable_organism.py, registered
  before any build; David's seeding insight, day 4):** three compositions,
  each against a never-moved control at matched token budgets: (a) LIVE
  CROSS-ARCHITECTURE MIGRATION — a running organism checkpointed on the ARM machine
  (ARM) resumes on the x86 runner (x86) mid-stream; its held-out trajectory rejoins
  the control's within 0.02 nats inside 1M tokens (bit-determinism is lost
  across BLAS — behavioral equivalence is the claim); (b) KILL+REJOIN — of
  two replicas sharing an index (P31 harness), one is killed for K chunks
  and rejoins by snapshot; after the measured heal it is within 0.02 nats
  of its uninterrupted twin, and the surviving replica's index writes cover
  the gap (the rejoiner benefits from spans collected while it was dead);
  (c) OFFLINE MODE — a stream outage of K chunks spent SLEEPING (dosed
  replay, dividend-monitored) beats the same outage spent idle by a
  measurable held-out margin at reconnect+N chunks, and both resume
  cleanly. Any part that fails is the measured boundary of portability.
  **P38a SCORED (day 4 ~09:15): CONFIRMED far beyond the registered bar.**
  Local gate: checkpoint→new-process→resume is BIT-identical (line-by-line
  chunk log). Cross-architecture: a live organism checkpointed mid-stream on
  the ARM machine and resumed on the x86 runner (x86) ends at heldout 6.182391 —
  identical to six decimals with the never-migrated control (6.182391),
  with identical gradient tokens (17,664: every gate decision matched).
  The only divergence is the bit-level digest (BLAS rounding differs across
  ISAs) and it does NOT propagate into behavior. The registered bar was
  "rejoin within 0.02 nats inside 1M tokens"; the measurement is immediate
  behavioral identity. Live cross-ISA migration is a solved, free operation.
  (b) and (c) full runs in flight on the second x86 runner; smoke already shows the
  index-cover mechanism (shared ≤ private on rejoin) and a clean
  small-budget boundary on offline-sleep (toy span pools are not
  representative — full budget decides).

- **P39 — stop-free streaming migration (MS17,
  src/portable_organism.py --exp d, registered before the build; David's
  möbius insight, day 4):** organism A streams continuously and NEVER
  pauses; at chunk N a snapshot is taken (sub-second) and transferred while
  A keeps streaming; target B resumes from the snapshot and replays the
  chunk range A consumed during the transfer window, then both process the
  next chunks in lockstep. Checks: (a) zero source downtime (A's chunk
  cadence shows no gap at the snapshot point); (b) B reaches state parity
  with A — identical held-out and identical gate decisions on the first
  shared post-catch-up chunks (locally bit-identical; cross-ISA to six
  decimals per P38a); (c) the catch-up window is bounded and small
  (transfer+replay < the time A needs to stream the same chunks — the
  migration CONVERGES rather than chasing forever); (d) iterated
  delta-sync (repeat snapshot/catch-up at cadence C) keeps a standing
  replica within one sync window of the living source at all times —
  continuous replication as a standing state. **P39 SCORED (day 4): (b)
  STATE PARITY CONFIRMED — the core claim: deterministic catch-up while the
  source keeps living gives bit-identical lockstep (chunk-log tail matches,
  heldout 6.171316 == 6.171316, local; six decimals cross-ISA per P38a). (a)
  the snapshot chunk costs 27.8 ms absolute (Torch serialization) — a
  ratio-10 blip only against the x86 runner's 2.7 ms baseline, NOT a perceptible
  stall; the metric conflates I/O latency with downtime and is being
  re-measured at realistic d_model where per-chunk cost dominates. (c)
  convergence and (d) iterated re-ran at d_model=128 on the quiet x86 runner: same
  picture (snapshot 35 ms absolute vs 4 ms chunks; B/A cpu 2.3 with the
  fixed process-start amortized over toy-sized work). VERDICT on the
  metrics themselves: (a) and (c) as registered are STRUCTURALLY
  UNDECIDABLE at toy scale — both are ratios of fixed constants
  (serialization, process start) to per-chunk costs that are microseconds
  in a d≤128/B=4 organism; they measure overhead arithmetic, not the
  migration property. What IS measured and stands: (b) bit-identical
  catch-up while the source lives (every scale, every machine tried), and
  the absolute snapshot cost (28–35 ms — one chunk slot). Final (a)/(c)
  measurement scheduled on the full POS-scale organism (B=8/K=64, ~100 ms
  chunks) where per-chunk work dominates the constants.
  **P39 FINAL (2026-07-24 evening, exp_d at d_model=128 on the quiet Mac,
  results/portable_organism.json):** (b) replicated once more at POS
  width — bit-identical lockstep, heldout match exact. The absolute
  overheads are now precise: snapshot-adjacent stall **35.5 ms**, catch-up
  replay **14.6 ms CPU/chunk** vs 3.85 ms live. As-measured ratios against
  the instrument's own ~2 ms toy cadence still fail (17.9× / 3.79×) — the
  d=128 run has POS width but toy chunk WEIGHT (32-token chunks), the
  cadence deficiency diagnosed above, unchanged. Scored instead against
  the PRODUCTION cadence the 40h organism actually measured (81 ms/chunk
  = 40.0 h / 1,776,712 chunks): the stall is **0.44 chunk slots** — (a)
  zero-downtime holds with room; replay runs **5.5× faster than the
  source produces** — (c) convergence holds with room. Labeled plainly:
  this is measured-constant arithmetic against a measured cadence, not a
  new in-situ measurement; the fully-live confirmation (and (d) iterated)
  merges into the Möbius cross-machine staging (A on the ARM machine, B on the x86 runner,
  real network in the window), queued behind the x86 runner's beacon finale.
  **P39 IN-SITU AT PRODUCTION CADENCE (2026-07-25,
  results/p39_production_scored.json) — this SUPERSEDES the arithmetic
  above, and overturns its conclusion.** The scheduled measurement was
  never actually runnable: BATCH and CHUNK were module constants with no
  CLI, and `exp_d` forwarded only `--d-model` to its subprocesses, so
  every earlier "d=128" run measured toy chunk WEIGHT (B=4/K=32) no
  matter what the parent was told. Both are now exposed
  (`--batch`/`--chunk-size`, seq_len tracks CHUNK) and forwarded;
  confirmed via the config recorded in A's own snapshot (batch 8,
  chunk 64). Re-measured in situ at d=128/B=8/K=64 — 512 tokens per
  chunk, the exact 40h POS configuration: **(a) 17.935× → 7.085×**
  (38.7 ms post-snapshot gap vs 5.46 ms baseline), **(c) 3.789× → 2.219×**
  (cpu, B/A). The cadence hypothesis is therefore HALF confirmed and HALF
  falsified, and both halves stand: the toy cadence really did inflate
  these ratios — both roughly halve once a chunk carries real work — but
  cadence was not the whole story, and a residual survives. **Neither (a)
  nor (c) clears its threshold in situ.** The projection above (0.44 chunk
  slots; replay 5.5× faster than the source) does not survive contact with
  the measurement it was standing in for: as registered, (a) and (c) do
  NOT pass on a single Mac, and they are not merely toy-scale arithmetic
  artifacts. (b) state parity passes at every cadence tried, unchanged.
  (d) remains gated on (a)-(c) by design and stays SKIPPED.
  **REPLICATED ON BEAST (2026-07-25, results/p39_two_machine_scored.json).**
  Run again at the same production cadence on the x86 runner (x86, 16 cores, B gets
  its own) to test the obvious explanation — that (c) fails on the ARM machine only
  because A and B contend for cores. **That explanation is REFUTED.**
  (a) 5.603× and (c) 2.039× on the x86 runner, against 7.085× / 2.219× on the ARM machine:
  the same picture, both still over threshold. Uncontended cores move (c)
  by 0.18, not by the 2× it would need. The raw cpu seconds locate the
  cause: B spends **45.4 s** replaying the chunks A produced with **22.3 s**
  of cpu — replay costs ~2× live streaming per chunk, on both ISAs,
  contended or not. That is a property of catch-up replay, not of any
  machine, and (c) as registered asks replay to run FASTER than production.
  Two independent machines now agree; (a)/(c) are **FALSIFIED as
  registered**, and the next move is to re-derive what (c) should
  ask for rather than to keep hunting for a machine that clears < 1.0.
  (b) passes on both machines with exact heldout match.

## Scoring rule

Each P-item gets CONFIRMED / PARTIAL / FALSIFIED in the harvest documents,
with the measured number beside the predicted interval. A falsified
prediction is a measurement — the register exists so we can't unknow what we
expected.

## Cadence audit (2026-08-05, results/cadence_audit.json)

Systematic sweep triggered by the P39 lesson (a "d=128" run that silently
measured toy chunk weight): every registered result was classified by
whether per-chunk cost sits in a denominator of its claim, and whether its
cadence was explicit and recorded. Outcome: **P39's cost checks were the
only affected claims, and they were already re-measured and falsified as
registered before this audit.** P38's registered checks are exactness /
heldout claims and are unaffected — but its descriptive timing
side-numbers (transfer_s, catchup_s) are toy-weight and must not be quoted
as production costs. The Möbius parity (delta 0.0 at equal chunk count) is
an exactness claim and stands; the sync-debt magnitudes are toy-weight
dynamics, and the rate-matched rerun must set its cadence explicitly
(moebius_stage.py now forwards --batch/--chunk-size; defaults unchanged).
pos_run, holo_*, scale_to_*, lifetime: clean — cadence always explicit and
recorded, or no cost ratio in any claim. Standing rule now in DECISIONS.md:
a cost-ratio metric without (batch, chunk, d_model) recorded in the same
artifact is a number about the defaults, not about the system.

## Wave 8 — source portability + the N-arm brain + ignition forensics (registered 2026-08-05, before any build)

- **P43 — live source hot-swap (MS-L, David's Umstöpseln insight).** One
  gated organism (POS gate values q=0.75/window 500, d=128, cadence 8/64
  explicit per the audit rule), schedule: C4 (S chunks) → idle pause →
  WT-103 (S) → idle pause → C4 (S), plus a pure-C4 control at equal total
  chunks, same seed. Registered expectations:
  (a) SURVIVABLE: no crash, stream bookkeeping intact, loss on the new
      source falls within its segment.
  (b) THE GATE IS A REGIME DETECTOR, SIGNED: C4→WT103 moves TOWARD the
      vocab/eval domain (WT-2 sibling), so the swap-in transient is a gate
      rate DROP (first-100-chunk rate ≤ 0.6× the pre-swap C4 steady rate);
      the return WT103→C4 is a SPIKE (≥ 1.5× within the first 100 chunks).
      Both transients decay back into the 0.18–0.26 band within one gate
      window (500 chunks) — the quantile re-adapts by construction.
  (c) THE DETOUR IS NOT DESTRUCTIVE: frozen-eval heldout after the full
      cycle within +0.05 of the pure-C4 control at equal total chunks.
      (Flagged, not a confound: heldout may IMPROVE during the WT103
      segment itself — WT103 is the eval's domain sibling. The claim is
      about the post-return state, not the detour minimum.)
  Falsifier for the F1 reading: transients absent or wrong-signed.
- **P44 — five runs, one brain (MS-M, David's multi-arm insight).** N=5
  organisms, IDENTICAL init, different C4 stream offsets, S chunks each
  (the x86 runner, cadence explicit). Fusion = one-shot weight average at end
  ("the brain"), measured against: single arm at S (equal per-arm budget),
  single arm at 5×S (equal total compute), and the P31 shared-store
  collective as the memory-level baseline. Registered:
  (a) NON-COLLAPSE: the merged brain's heldout ≤ single-arm + 0.3 (same
      init ⇒ the five stay linearly connectable; a collapse kills naive
      fusion and the finding is the incompatibility).
  (b) THE BRAIN BEATS ITS PARTS: merged < single-arm-at-S by ≥ 0.02 —
      five disjoint streams' knowledge partially adds under averaging.
  (c) HONEST CEILING: merged does NOT beat the 5×S control — parallel
      fusion is a wall-clock win, not a token-efficiency win.
      Embarrassment threshold: if it DOES beat 5×S, federated streaming
      has a free lunch and that is a bigger result than (b).
- **P45 — where the gate rate is born (MS-K, ignition forensics).** Per-
  chunk traces (s_t, gate_t, q_t, loss) for the first 2000 chunks at
  d ∈ {128, 256, 512} × 2 seeds, exact width-curve recipe. Registered:
  (a) the d256-vs-d512 separation (cum 0.2042 vs 0.1785 at first eval) is
      established within the FIRST 500 chunks: cum gate rate at chunk 500
      differs by ≥ 0.03, same direction, both seeds.
  (b) MECHANISM CANDIDATE, falsifiable: during ignition the quantile
      window is dominated by the fast-falling early loss; wider models
      descend faster, so more fresh chunks sit BELOW the stale quantile →
      fewer fires. Concretely: early per-chunk-NLL slope (chunks 100–500)
      is strictly steeper d512 > d256 > d128, AND the mean margin
      (s_t − q_t) over chunks 100–500 is more negative for wider models.
      Slope ordering broken ⇒ mechanism falsified, trace still localizes
      the divergence.
  (c) determinism: same seed, same width ⇒ identical gate decisions
      chunk-for-chunk. If THIS fails, P34's run-vs-run instability lives
      in the gate itself — important either way.
  **P45 SCORED SAME DAY (results/ignition_forensics.json).** Sequence
  note: the per-chunk traces predate this registration (they were written by
  the July width runs; an earlier claim that only aggregates existed was
  wrong) — the aggregate-level facts were known at registration time, the
  trace contents were not read until after. (a) CONFIRMED at n=1 seed:
  separation established within the first 500 chunks, cum gate 0.304 vs
  0.270, Δ0.034 ≥ 0.03; both-seeds clause running (seed-43 cells + a
  determinism repeat, queued the same hour). (b) HALF CONFIRMED / HALF
  FALSIFIED, mechanism refined: the margin half holds monotonically
  (−0.1422/−0.2057/−0.2504) but the slope ordering is broken (d256 steeper
  than d512 in chunks 100–500). The wide model's fast descent happens
  INSIDE the always-learn ignition phase — s3 at chunk 100 is already
  5.639 (d512) vs 5.801/5.787 — so the quantile window fills from the high
  descent trail and fresh chunks sit persistently below its q75. The gate
  rate is born from the DEPTH of ignition descent, not post-ignition
  slope. (c) pending the repeat cell.
  **P45 FINAL (2026-08-05 evening, confirmation cells + determinism repeat,
  results/ignition_forensics.json):** (a) FALSIFIED — at seed 43 the
  separation REVERSES (cum500 d256 0.274 < d512 0.284); width does not set
  ignition depth. (b) the mechanism survives DECOUPLED from width: across
  six cells, gate rate tracks ignition-exit depth (r=0.77, rho=0.60, n=6,
  n.s.) — at seed 43 it is d256 that exits lowest and gates least. (c)
  FALSIFIED with a measured fork: same seed, same machine — bit-identical
  through chunk 118, then a quantile interpolation flips on sub-print-
  precision drift, one gate decision diverges, and the runs land at
  DIFFERENT 2000-chunk rates (0.191 vs 0.163; gates match 1936/2000).
  P34's run-vs-run instability is now measured inside the gate. The
  uncomfortable consequence, stated in README and paper: at this horizon
  same-seed variation equals the width gap; whether the 50M separation
  (24.7 vs 19.8) is width or frozen lottery is undecided without repeats
  at scale.
  **P43 SCORED (2026-08-05 evening, results/source_swap.json).** (a) PASS.
  (c) PASS WITH ROOM — the WT-103 detour arm ends 0.129 BETTER than the
  pure-C4 control (5.3629 vs 5.4922; bar was "within +0.05"). (b)
  FALSIFIED IN BOTH CLAUSES WITH INVERTED SIGNS, and the inversion is the
  finding: swap-in SPIKES the gate 3.8× (0.630 vs 0.164; surprise
  5.281→5.513) where quiet was registered; the return home QUIETS it to
  0.49× (0.080) where a spike was registered. The registered reasoning
  bet on absolute difficulty (domain proximity); the gate actually fires
  on TRANSITIONS relative to its own window — entering any new regime
  spikes, returning home is quiet, both decaying into steady bands within
  the window length. A cleaner regime-detector law than the one
  registered, and crisp only because the wrong sign was committed first.
  **P44 SCORED (2026-08-05 night, results/five_brain.json).** (a) PASS at
  the bar (5.9791 vs 6.0633). (b) FALSIFIED decisively: the brain is worse
  than EVERY arm (5.9791 vs best 5.7633; required ≥+0.02 over arm0,
  measured −0.216) — after ~768k tokens of disjoint-stream divergence,
  one-shot weight averaging averages away more than it adds. (c) CONFIRMED
  as registered: no free lunch (brain trails 5×S by 0.620). Side finding:
  the five arms gated at 138k–179k grad tokens on identical config — the
  ignition lottery reproduces across STREAM CONTENT. Convergent reading
  with P31: memory-level fusion works (shared store, 0.67× forgetting),
  naive weight-level fusion fails — composability lives in the memory
  layer, not the weight layer (F4 extended to collectivity). Next attack
  (to be registered as P46 before build): iterated merge-redistribute
  every M chunks, and the memory-route hybrid (five arms, one store, one
  reader).

## Wave 8b — the knowledge-file loop (registered 2026-08-05 night, before any build)

- **P46 — five producers, one store, one reader (MS-M v2).** N=5 organisms
  stream disjoint C4 offsets and write ONE shared span store (P31
  machinery); a separate reader organism streams its own budget WITH
  reminder consultation from the union store. Controls: same reader, no
  store; and the P44 one-shot weight merge as the dead baseline.
  Registered: (a) reader-with-union-store beats reader-no-store by
  ≥ 0.02 heldout at equal reader budget — the collective helps through
  the MEMORY interface where weight averaging failed; (b) iterated
  merge-redistribute every 100 chunks (bounded divergence) ≥ the single
  arm at S — may fail exactly like P44b; if it instead BEATS the 5×S
  control, that is the federated free lunch and the bigger result.
- **P47 — the frozen knowledge file (the refinery loop, closed).**
  Organism A streams C4 and harvests surprise spans; the harvest is
  DISTILLED into a frozen, sha256-hashed, provenance-carrying knowledge
  file (each entry: key, span, stream doc-coordinate, surprise) — built
  once, never mutated after. Organism B (fresh, never saw A's stream)
  then lives a domain shock (WT-103) with dosed sleep-replay from that
  file. Arms: intact file / token-shuffled file (keys+coords intact — the
  poisoning control) / no file. Registered:
  (a) FILE-MEDIATED IMMUNITY: B's C4-competence forgetting with the
      intact file ≤ 0.8× the no-file control (P31's live-store 0.67× is
      the reference; the frozen interface may cost some of it).
  (b) PROVENANCE REPLAYS: ≥ 4/5 sampled entries reconstruct exactly by
      re-instantiating the deterministic stream at their recorded
      doc-coordinate — "sources sehbar" as a measurement, not a slogan.
  (c) THE POISONING CONTROL: the corrupted file's benefit < 0.5× the
      intact file's benefit — and, per the MS1 conflict finding, the
      reader is NOT expected to defend itself (trust-decay, not
      arbitration); the defense organ is future work and is hereby named,
      not smuggled in.
  **P47 SCORED (2026-08-05 night, results/knowledge_file.json, full on
  core).** ALL THREE CHECKS PASS. (a) beyond the bar in kind: the
  intact-file arm ends BETTER on C4 than pre-shock (−0.1137) while
  no-file forgets (+0.0224) — the frozen interface turns a foreign-domain
  shock into net home-domain improvement (exposure asymmetry noted: file
  arms get 32 extra C4-token batches; fresh-C4-replay arm is the v1
  control). (b) provenance 5/5 at spread coordinates — source
  traceability as a passing measurement. (c) knife-edge pass (0.494× vs
  0.5 bar, margin 0.006, n=1) whose decomposition is the finding: ~half
  the file benefit is content-in-order, ~half token-bag regularization;
  the smoke's tie at 6 replays separated at 32 — exposure-dependence as
  P37c/P33 predicted. Still unmeasured, named: content on the keyed-
  recall axis, and the arbitration/defense organ.

- **P48 — the pixel body (registered 2026-08-05, before any build).** The
  operating layer's first step off token streams, at minimum cost: a
  procedural first-person maze (deterministic from seed; R rooms joined by
  corridors), 8 deterministic walkers on fixed patrol scripts, egocentric
  24×24 render in 16 colors → 576 tokens per frame, fed through the
  UNCHANGED d=128 organism (B=8/K=64, POS gate values; a frame = 9 chunks;
  frame metrics are aggregates over its chunks). No RL, no reward — this
  tests perception learning and the surprise calculus on a simulated body,
  nothing else. Registered:
  (a) THE STACK RUNS UNCHANGED: per-token loss falls ≥30% below its
      ignition plateau; RSS flat (span < 0.3 GB post-warmup).
  (b) THE TRANSITION DETECTOR TRANSFERS TO PIXELS: post-ignition, frames
      entering a NEW room carry gate rate ≥2× the corridor-steady rate
      (≥20 entry events pooled). This is P43's transition law asked in a
      second modality.
  (c) PROVENANCE IN THE GAME WORLD: 5/5 sampled knowledge entries
      reproduce their frame BIT-EXACTLY by replaying the deterministic
      environment at the recorded (maze_seed, walker, step) coordinate —
      stricter than P47's text search, because the world is fully ours.
  (d) HABITUATION: a room's SECOND visit spikes less than its first
      (mean entry-surprise second < first by ≥20%) — familiarity as a
      measurable property of the same scalar.
  Falsifier for the modality claim: (b) or (d) absent while (a) holds —
  then the calculus learned pixels but did not organize them, and the
  transfer claim dies as registered.
  **P48 SCORED (2026-08-05 night, results/pixel_body.json, full + lane-
  instrument rerun on the x86 runner).** (a) PASS with room (drop 63.3%, RSS span
  0.0076 GB). (c) PASS 5/5 BIT-EXACT, twice — the harness is
  bit-deterministic on the x86 runner. (b)+(d) FALSIFIED, and the falsifier fires
  as registered: the transfer claim in its registered form is dead. The
  measured mechanism is sharper than the falsifier's wording, and both
  are recorded: the dilution rescue (batch-mean gate 8× blind to
  asynchronous single-walker events) was refuted by its own fix (per-lane
  entry/steady = 1.01), leaving the structural reading — a fully
  deterministic world under a fixed policy contains no underivable
  novelty after ignition, and a competent predictor silences every
  transition. P43's detector fired on a swap that was NOT derivable from
  within the stream; the maze's transitions are. The experiment conflated
  transition with novelty. Kept as the embodied-fleet lesson anyway: one
  shared gate across eight bodies dampens each body's signal 8×.

- **P49 — underivable novelty in the pixel world (registered 2026-08-05
  night, before the build).** Same world, one change: a seventh room is
  carved but SEALED from the start; at frame T* (mid-run, long after
  ignition, unpredictable from any history) its connecting wall opens,
  the wall-followers' circuits deform, and walkers encounter a room and a
  floor color that have never appeared in the stream. Registered:
  (a) per-lane surprise at first novel-room contact ≥ 3× that walker's
      steady mean, and the batch gate FIRES on the contact chunk;
  (b) habituation across subsequent encounters of the novel room ≥ 20%
      (the dynamic range P48d lacked, restored by genuine novelty);
  (c) the novel-room knowledge entries replay bit-exact from
      (maze_seed, walker, step) — provenance surviving a world MUTATION,
      which is the robotics case (the world changed; the record of when
      and where it changed is exact).
  Falsifier: (a) absent ⇒ the calculus does not detect underivable
  novelty in pixels, and the modality-transfer claim dies for real this
  time — no third framing will be registered.
  **P46 SCORED (2026-08-05 night, results/hub_n5.json, full on the second x86 runner).**
  (a) PASS: reader with union store beats reader without by +0.0313
  (bar ≥0.02, 60 replays) — the five-producer collective helps a stranger
  through the memory interface; with P47 the store now measurably carries
  BOTH axes (immunity and acceleration). (b) PASS, flipping P44: bounded-
  divergence merge-redistribute every 100 chunks reads 5.1537 vs best
  single arm 5.2542 — averaging BEATS its parts where the one-shot at
  full divergence lost to all of them. Weight fusion is a sync-frequency
  phenomenon; the divergence budget was P44's killer, not fusion itself.
  (c) the equal-compute ceiling CONFIRMED: same-instrument 5×S control 4.9585,
  itermerge trails by 0.195 — parallelism is a wall-clock purchase
  (5× speedup at ~0.2 nats), no token-efficiency free lunch, the
  embarrassment threshold stayed silent exactly as registered.

  **P49 SCORED (2026-08-05, results/pixel_p49.json, full 12k chunks on
  the x86 runner).** (a) FALSIFIED: median first-contact lane ratio 0.819 (bar
  ≥3×), contact-chunk gate rate 0.25 (bar: fires). Three of four
  contacting walkers were QUIETER at first sight of a never-streamed
  room than at their own steady state. The falsifier fires as
  registered: the gate calculus does not detect underivable novelty in
  pixels, and the modality-transfer claim is dead — no third framing,
  as promised. (b) unmeasurable, and the zero is the datum: ZERO
  knowledge entries harvested on first visits (novel_entries_visit1 = 0
  vs 695 on later visits) — no first-visit signal exists to habituate
  from. (c) PASS 5/5 BIT-EXACT with the unseal REPLAYED inside the
  provenance check — the world mutated mid-run and every entry still
  reconstructs from (maze_seed, walker, step, unseal_at). Provenance
  survives world mutation: the robotics clause, and the one that holds.
  Post-mortem, recorded without a new registration: the novel room at
  first sight is 1–3 cells of a 225-cell frame — token-sparse novelty
  under a 64-token lane MEAN. The harvest instrument (token-level NLL
  peaks) sees the room clearly (1,384 room entries by end of life); the
  gate instrument (lane mean) never does. After P48's 8× body dilution
  this is the 64× token dilution — embodied gates need foveal
  (token/location) resolution, not panoramic means. Text never hit this
  wall because text novelty is dense: whole documents switch topic at
  once, most tokens in the chunk carry the surprise. P48 full-rerun
  clauses in the same artifact stay consistent: (a) 61.2% drop PASS,
  (b) 0.973 FAIL, (d) 0.086 FAIL, (c) 5/5 PASS.

  **P45 DECISIVE 1 (2026-08-05, results/pos_rep50_d512_status.json +
  full chunk-trace comparison).** The queued 50M repeat (exact recipe,
  seed 42, d=512) ran 11 days after the original, different process,
  one mid-run HF stream reconnect. Result: **all 97,657 chunks
  identical on every recorded field** — s1/s2/s3 at print precision
  (6 decimals), every gate decision, every threshold; finals to the
  last digit (A2 5.0898, A3 5.170144, cum 0.1984, 19,375 fires). The
  differing status digests are an instrument artifact, not compute: the
  digest hashes wall-clock-driven eval lines (18 vs 15 at
  eval_every_s=900) — noted as a digest design flaw for cross-run use;
  the chunk-level comparison is authoritative. Verdict: same-seed
  variance at 50M is ZERO under undisturbed conditions — the rate is a
  deterministic function of (seed, width, recipe, stream). This
  sharpens P45(c) rather than contradicting it: the 2k forks were
  measured in PARALLEL forensics cells (co-load); the undisturbed
  production path reproduces exactly — the instability lives in the
  environment, not the calculus. The stream re-instantiated exactly
  through a reconnect (138,532 docs, same order): provenance at
  production scale. Width-vs-lottery now rides entirely on the seed
  axis — D2 (seed 43, d=512, 50M) is running.

- **P50 — the replay law decomposed (MS-C+I, registered 2026-08-05
  evening, BEFORE the build).** P39 measured catch-up replay ≈2× live
  CPU (2.01/2.04/2.22 across machines) and filed it as a candidate law.
  Code reading + the exp_d artifact suggest it is no law but a PRODUCT
  OF TWO ENVIRONMENTS: cpu_ratio 2.01 = wall_ratio 1.46 ×
  thread-inflation 1.38 (catch-up ran CPU 16.8s in 2.0s wall ≈ 8.4
  threads hot DESPITE set_num_threads(1); live idled its pool on
  stream waits), and the wall overhang matches ds.skip() re-downloading
  the organism's whole prior life (exp_d skip/T = 300/200 = 1.5; at a
  ~30% stream cost share, 1 + 1.5×0.31 ≈ 1.46). The cross-machine
  stability of "2×" would then be an artifact of every measurement
  sharing the same skip/T proportion — ratio = f(skip/T), not a
  constant. Registered, one harness (src/replay_law_run.py, the x86 runner),
  phase-instrumented (import/load, skip-to-first-doc, per-chunk stream,
  fwd, bwd, harvest — wall+cpu each):
  (a) DECOMPOSITION: cpu_ratio factors into wall_ratio ×
      inflation_ratio as measured, and phase timers attribute ≥80% of
      the wall overhang to named phases (skip + coldstart), <20%
      unexplained.
  (b) SKIP LINEARITY: at fixed T=200, wall overhang grows linearly in
      skip_docs (snapshot at 300/1500/3000 chunk-equivalents, r²≥0.9).
      Point bet: skip cost per doc ≈ live stream cost per doc (same
      download+decompress work, within ±50%).
  (c) COMPUTE PARITY: replay from a LOCAL token cache with
      single-thread enforced via environment (OMP/MKL=1 before import):
      cpu_ratio vs live ≤ 1.1 — replay COMPUTE is not more expensive
      than live; the 2× was environment (fast-forward + cpu-clock
      inflation), not calculus.
  (d) MÖBIUS RATE CONDITION (derived, then checked separately): from
      the measured components, B catches A iff
      T·(rate_live − rate_replay_cache) > skip_cost(life) + cold;
      the formula must predict the minimum cycle length for sync-debt
      convergence in the Möbius-stage geometry to ±20% — checked in a
      rate-matched Möbius run AFTER this scores.
  Falsifier chain: (b) near-zero skip slope AND (c) ratio ≥1.8 under
  cache+single-thread ⇒ replay IS intrinsically ~2× and the law stands
  → F7 gets the 2×-hardware replication clause; (c) alone ≥1.8 with
  (b) linear ⇒ both terms real, F7 gets both. Either way F7 gains a
  measured clause and P39(c) gets re-derived on the correct anchor.
  **P50 SCORED SAME EVENING (results/replay_law.json toy-cadence matrix,
  results/replay_law_prod.json production-cadence cell, both the x86 runner).**
  (a) PASS structurally, with the surprise being the SIZE of the
  inflation term: phase timers close the wall budget to ~100% in every
  cell (setup+load+stream+model+harvest ≈ main to 0.1s), but CPU-clock
  inflation is 13.6–15.5× wall (not the bet's 1.38×) and PIN-RESISTANT
  (OMP/MKL/torch pins change nothing on the 16-core machine) — the x86 runner's
  CPU clock measures fiction; every historical CPU-ratio there compared
  two fictions. (b) FALSIFIED, and the truth is better than the bet:
  skip cost is a STEP, not a line — 0.016s at 188 docs, 9.4s at 509,
  9.4s at 1056 (identical; reproduced at 10.3s in the prod cell) — a
  fixed fast-forward toll past a shard threshold, NOT proportional to
  the life replayed. Point bet (skip/doc ≈ stream/doc) dies with it.
  (c) PASS at production cadence, the verdict of the whole thread:
  cache-replay compute = 1.032× live (bar ≤1.1), stream-replay 1.098×,
  medians vs medians after warmup, parity bit-equal in every cell
  (6.020187 = 6.020187). THE 2× REPLAY LAW WAS NEVER A LAW — it was
  CPU-clock inflation × toy-cadence environment dominance × a fixed
  skip step. P39(c) re-derived: convergence failed against an
  instrument fiction; on wall medians replay tracks live at 3%.
  (d) LOCKED PREDICTION (before the run, queued behind D2 on the ARM machine):
  equilibrium sync-debt T∞ = rate_A·fix/(1 − rate_A·c_B); with P50
  parameters (fix ≈ 16s, c_B ≈ 3.7ms at stage cadence, rate_A = 20
  chunks/s) the Möbius stage should converge to T∞ ≈ 346 chunks of
  standing lag, band 280–420 (±20%), INDEPENDENT of cycle length. Two
  test levels registered: the formula against in-run-measured (fix,
  c_B), and the parameter transfer against the P50 values.

- **P51 — the arbitration organ v0: selective ingestion by measured
  benefit (registered 2026-08-05 evening, BEFORE the build).** The hub's
  named hard problem: the reader does NOT defend itself (MS1 conflict:
  trust decays, no arbitration), while the file interface CAN separate
  poison from nutrition post-hoc (P47c: shuffled control 0.494× the
  intact benefit). P51 closes the loop in-run: an ARBITER that spends a
  small probe budget per file and allocates the rest by measured
  benefit. Setup (the x86 runner, d128/B8/K64, the P46/P47 cadence): four
  producers at disjoint C4 offsets harvest frozen files à la P47; ONE
  file is poisoned by within-span token shuffling (the measured P47
  poison). A fresh consumer streams its own far-offset C4 budget in
  four arms from one deepcopied init: no-files control, NAIVE (equal
  replay dose from all four files), ARBITER (probe each file with ≤10%
  of the replay budget, measure per-file heldout delta on the shared
  C4-slice instrument, then dose only files with measured benefit),
  CLEAN-ORACLE (the three intact files, equal dose — the ceiling).
  Registered:
  (a) IDENTIFICATION: the arbiter's probe ranks the poisoned file LAST
      of four by measured benefit, both seeds (42, 43).
  (b) SELECTIVITY PAYS: arbiter final heldout beats naive by ≥ 0.005
      (the poison's dose-cost is measurable and avoidable in-run).
  (c) ORACLE GAP: arbiter lands within 0.01 of clean-oracle.
  (d) PRICE: the probe spends ≤10% of the replay budget.
  Falsifier chain, pre-committed: if NAIVE ≈ ORACLE (gap < 0.005), the
  shuffled poison is INERT at this dose — (b) becomes unscorable as
  registered, the finding is "dilution is the only cost of weak poison,"
  and the next attack is a STRONGER poison (adversarial spans, not
  shuffles) — to be registered as its own P, not smuggled in. If the
  probe misranks (a fails) while (b) passes, the benefit instrument is
  too noisy at probe dose — the arbiter needs repeated probes, and the
  price clause (d) is what breaks. Either failure names its mechanism.
  **P51 WITHDRAWN (2026-08-05, same evening, before any data).** The
  run was stopped in the smoke phase and produced no results. Reason,
  recorded verbatim in spirit: arbitration is APPLICATION-LAYER
  governance for an exchange layer that does not exist yet — building
  the lock before the chassis. The core primitives are the work; a
  defense organ becomes a real registration when there is a real
  exchange to defend. The registration text above stands as a design
  note for that later date; no clause was scored.

- **P52 — the organism enters a real game engine (registered 2026-08-05
  night, BEFORE the build).** David's call: launch a fresh reader into
  VizDoom and watch whether it learns the world over its life, with
  everything harvested into a frozen, replayable knowledge file. The
  substrate steps up from our own top-down maze (P48/P49) to a real 3D
  engine with a first-person body: scenario my_way_home (ego maze
  navigation), GRAY8 160×120, engine determinism VERIFIED before this
  registration (same seed + same actions → bit-identical frames, 30/30
  hashes). Eight bodies = eight independent game instances (per-episode
  re-seeding: seed(base+episode) makes every episode independently
  addressable — the provenance coordinate is (lane, episode, tic)).
  Frames pool to 15×16 tokens at 12 gray levels + separator pad = 256
  tokens = 4×K64 chunks through the UNCHANGED stack, d128/B8/K64.
  Registered:
  (a) IT LEARNS: streaming NLL over the last 10% of life sits ≥30%
      below the post-ignition plateau (chunks 100–600 mean), at flat
      RSS (span ≤0.1 GB after ignition).
  (b) IT KNOWS ITS OWN WORLD: end-of-life NLL on frozen frames from
      its OWN routes (seen mid-life) beats NLL on frames from a
      NEVER-SEEN route (fresh action seed, same map) — difference > 0.
  (c) PROVENANCE THROUGH A GAME ENGINE: 5 sampled knowledge entries
      reconstruct BIT-EXACT from (lane_seed, episode, tic) by fresh
      engine replay — the causal-file claim on third-party simulation
      software we did not write.
  (d) THE FILE CARRIES THE WORLD: a fresh reader dosed with the frozen
      file beats its no-file twin on the never-seen-route instrument by
      ≥ 0.02 — learning transferred through the artifact, not the
      weights.
  NOT registered, per the P49 falsifier's standing promise: no gate
  novelty-transfer clause on pixels — lane surprise runs logging-only.
  Falsifiers named: (a) fails ⇒ the tokenization (15×16@12 levels) is
  below the world's information floor for this reader — the next attack
  is the token map, not the claim; (c) fails ⇒ per-episode re-seeding
  does not isolate engine state — coordinate design is wrong and gets
  rebuilt before any other clause is read; (d) fails while (a)+(b)
  pass ⇒ the file-transfer loop needs the P47 dosing curve on this
  substrate — registered as its own follow-up, not patched in.
  **P52 INTERIM (same night): the (c) falsifier FIRED as registered,
  and its named consequence executes.** Full run: provenance 1/5 — and
  the one exact pick is the one with episode = 0, the smoke's regime
  (5/5 there, all episode-0). Isolation test on the engine confirms the
  mechanism to the letter: set_seed + new_episode on a RUNNING engine
  yields a DIFFERENT episode than the same seed on a FRESH engine
  (5/5 frame hashes diverge) — engine state survives new_episode, so
  per-episode re-seeding did not isolate. Per the registration, the
  coordinate design is rebuilt BEFORE any other clause is read: the
  life path now closes and re-creates the engine at every episode
  boundary (one factory for life and replay — identical by
  construction; ~10s total cost over the life). The (a)/(b)/(d)
  numbers of the failed run are recorded but UNREAD as clauses — (b)'s
  own-route reconstruction used the broken coordinates, so its frames
  were not the organism's own. Artifact of the fired falsifier:
  results/vizdoom_life.json (rss series additionally carries a
  KB-vs-bytes instrument bug on Linux, fixed in the same pass). Rerun
  in flight.
  **P52 SCORED (2026-08-05 night, results/vizdoom_life.json v2 after the
  registered coordinate rebuild).** (c) PASS: 5/5 BIT-EXACT across
  episodes {0,1,4,9,12} — provenance through third-party simulation
  software, with the life path and replay path sharing one engine
  factory. (d) PASS: the frozen file (24,660 entries, sha256
  17aadcc6…) carries the world — fresh reader with dosed file replay
  beats its no-file twin by 0.0562 on a never-seen route (bar 0.02).
  The refinery loop (P47) now holds through a game engine we did not
  write. (a) FALSIFIED at the bar: drop 13.5% vs 30% required, at flat
  RSS (span 0.011 GB) — the reader learns, and the 12-level 15×16
  token map exhausts quickly (NLL floor ~0.7 vs ~4–5 on text); per the
  registered falsifier the next attack is the TOKEN MAP (finer levels /
  resolution), not the claim — to be registered separately if pursued.
  (b) FALSIFIED AS INSTRUMENTED, and the instrument confound is named:
  own-route frames were drawn from the MIDDLE episode of each of 8
  lanes (8 different spawns, heterogeneous; NLL 1.258) while the
  fresh-route frames were one seed's episode-0 opening frames
  (homogeneous single spawn; NLL 0.259) — the populations differ in
  intrinsic difficulty, so the comparison measures population mismatch,
  not world knowledge. The clean version is position-matched (same
  episode indices and frame windows, only the seed differs); named as
  the follow-up design, not patched into this scoring.
  **P50(d) STAGE-1 NOTE + RE-LOCK (2026-08-05 late).** First parallel
  staging ran A to its 14,400-chunk budget with two complete cycles
  (catchup 3000 → 4500) before the cycle marks outran the budget: A ran
  at ~96 chunks/s against the 20/s budget assumption — the LEVEL-2
  parameter transfer is thereby already falsified in its rate_A input
  (the 346-chunk point assumed 20/s and 16s local fix; the real stage
  pays scp+remote-spawn fix over a real network). The in-memory cycle
  records (sync_debt_chunks) were lost to an orchestrator kill, so
  LEVEL 1 is re-locked BEFORE a clean rerun with the same formula and
  tonight's measured parameters: rate_A ≈ 96/s, fix ≈ 35–60s,
  c_B ≈ 3.5ms ⇒ T∞ = rate_A·fix/(1 − rate_A·c_B) ∈ [5100, 8700]
  chunks. The rerun (a-total 45,000, 4 cycles, window 500) must land
  its measured standing debt inside that band, with the in-run (fix,
  c_B, rate_A) closing the loop; outside the band, the formula — not
  the parameters — is what failed.

  **P45 DECISIVE 2 — THE WIDTH LAW STANDS (2026-08-05 night,
  results/pos_s43_50_d512_status.json).** Seed 43, d=512, the exact
  50M recipe: cumulative gate rate **0.1986** against seed 42's 0.1984
  — two seeds, the same rate to the third decimal, while the
  d128/d256 attractor sits 4.9 points away at 0.247 (a gap ~25×
  larger than the seed difference). The whole d512 profile reproduces
  across seeds, not just the rate: improvement ratio 0.9758 vs 0.9777,
  efficiency 0.3467 vs 0.3547 per million gradient tokens — still the
  best point of the width curve on its second seed. Verdict, both
  axes now closed: the 50M gate rate is DETERMINISTIC per (seed,
  width, recipe, stream) [D1: 97,657/97,657 chunks exact] and
  SEED-ROBUST at d512 [D2: Δ0.0002]. The early-horizon lottery is a
  TRANSIENT, not a fate — 2k-chunk rates fluctuate, the 50M cumulative
  rate converges to its width's attractor. Precision boundary,
  recorded: full seed-robustness is measured at d512; the 24.7% side
  rests on the two-width attractor (d128 and d256 landing on one rate
  independently) with one seed each at 50M. Run note: this run
  carries resume:true — externally killed at 68.2% and healed from
  its own checkpoint (the exp_a-measured exact-resume path); zero
  stream reconnects on the resumed leg.
  **P50(d) RE-LOCK #2 (2026-08-05, before run 4).** Run 3's records
  expose the staging truth: A (141/s) finished its 45k budget BEFORE
  cycle 0 pulled its snapshot — B then replayed the ENTIRE life in one
  catchup (45,000 chunks in 484.7s = 10.77 ms/chunk on the x86 runner
  at nice-19, HF streaming included), and cycles 1–3 idled against a
  finished A. The debt-0 readings are trivial (standing A), and the
  [5100, 8700] band was never confronted — the run measured the
  snapshot-teleport regime, not the chase. But run 3's own two rates
  arm the sharp test: rate_A ≈ 141/s (7.1 ms/chunk, ARM) versus
  c_B ≈ 10.77 ms/chunk (x86 at nice-19) gives rate_A·c_B ≈ 1.52 > 1 —
  the FORMULA now predicts the chase regime DIVERGES on this hardware
  pairing: each cycle's catchup must grow by a factor ≈ rate_A·c_B,
  locked band [1.3, 1.7] per cycle over ≥3 live cycles (run 4:
  a_total 150,000 so A outlives the cycles; in-run rates close the
  loop). If the growth factor lands in band, the rate condition is
  CONFIRMED in its divergent branch — and F7's two-regime lesson
  becomes complete and measured: chase replication requires
  rate_A·c_B < 1 (equal-or-faster follower hardware); snapshot-sync
  collapses debt regardless (run 3, cycle 0: one scp erased a 45k-chunk
  deficit; the transfer is the tilgung, not the compute).

- **P54 — CHIMERA v1: the F1 lock at production cadence (MS-D,
  registered 2026-08-05 night, BEFORE the build; pre-check of P33's
  artifacts completed first per the task's own rule).** P33's verdict
  carried three named liabilities: no cadence block in the artifact
  (pre-audit run — toy-weight suspicion), the reminder organ
  UNDECIDABLE at 2 fires (exposure 5× below the measured recurrence
  base rate), and clause (a) dose-confounded (chimera at 29% fewer
  gradient tokens under matched chunks). v1 removes all three: same
  five arms (chimera, r3 gate+sleep, r1 full, no_reminder,
  no_monitor), same MS3 shock protocol (C4→code→C4), at d128/B8/K64
  production cadence recorded in the artifact, ≥1,000 chunks per phase
  (expected reminder fires in the double digits from P33's own base
  rate), both dose axes reported (matched chunks as protocol,
  per-gradient-token normalization as metric). Registered:
  (a) THE RECOVERY ORGAN SURVIVES REAL CADENCE: chimera residual
      damage after recovery ≤ 0.05 while no_monitor reads ≥3× worse —
      the v0 headline and its in-run attribution reproduce off toy
      cadence.
  (b) THE REMINDER BECOMES DECIDABLE: ≥10 fires in the chimera arm;
      then minus-reminder loses ≥0.02 on at least one axis (organ
      earns its place) OR does not (organ falsified as an
      in-composition contributor at this exposure) — either outcome
      resolves P33's pre-registered UNDECIDABLE.
  (c) THE FORGETTING VERDICT REHABILITATES UNDER FAIR DOSE: normalized
      per gradient token, chimera beats r3 on ≥2 of the 3 continual
      axes (the v0 loss was the dose confound, not the composition).
  (d) NO SINGLE-ORGAN ARM DOMINATES chimera on all three axes
      (v0's (d), re-asked at production cadence).
  Falsifiers named: (a) fails ⇒ the dividend monitor's recovery effect
  was a toy-cadence artifact — F1's strongest composition claim loses
  its anchor and says so; (b) under 10 fires despite 1,000-chunk
  phases ⇒ the base-rate model is wrong and the reminder needs a
  recurrence-seeded protocol (its own P, not smuggled in); (c) fails
  ⇒ composition genuinely buys recovery at a forgetting price — the
  price gets stated as the law.
  **P54 AMENDED before the run (same hour, pre-check deepened):**
  chimera.py reads its cadence FROM the POS checkpoint config
  (B, K = cfg["batch"], cfg["chunk"]; the 40h-run snapshot is
  d128/B8/K64) — v0 already ran at production cadence, and the
  liability was the missing cadence BLOCK in the artifact
  (documentation), not toy weight. Clause (a) is therefore an
  EXPOSURE-reproduction clause (6.7× longer phases at the same
  cadence), not a cadence rescue; clauses (b), (c), (d) unchanged.
  v1 records the cadence block explicitly.
  **P50(d) SCORED FINAL (2026-08-05 night, run 4,
  results/moebius_rate_check4.json — a LIVING A this time).** The live
  debt series exists and the recursion is textbook: every cycle's
  catchup equals the previous cycle's debt exactly (10,000→catchup
  10,000; 19,000→catchup 19,000). Verdicts by clause level:
  DIVERGENCE DIRECTION — CONFIRMED: standing debt grew 10,000 → 19,000
  against a living A; the chase regime diverges on this hardware
  pairing exactly as rate_A·c_B > 1 requires. POINT BAND [1.3, 1.7] —
  FAILED at 1.90, with both causes named: the band was computed from
  run 3's rate_A ≈ 141/s while the in-run A ran at 200/s (the ARM
  machine had shed its co-load), and the asymptotic factor rate_A·c_B
  (in-run: 2.35) is not the finite-debt factor — the recursion carries
  a −debt−lockstep term the band ignored. THE RECURSION ITSELF —
  CONFIRMED IN-RUN TO 3%: debt₂ = rate_A·(transfer + c_B·(catchup₁ +
  lockstep)) − catchup₁ − lockstep = 19,580 predicted from in-run
  parameters vs 19,000 measured. The formula lives in its exact form;
  the factor band was the weaker phrasing of it. THE TWO-REGIME LAW,
  now fully measured: (1) CHASE replication requires rate_A·c_B < 1 —
  on unequal hardware (ARM source at 200/s vs x86 follower at
  11.8 ms/chunk under nice) it diverges at a computable rate; (2)
  SNAPSHOT-SYNC collapses any debt instantly — one scp erased a
  45,000-chunk deficit (run 3 cycle 0; run 4 cycle 0 replayed a whole
  45k life in 458s over the real network) — the transfer is the
  tilgung, not the compute. F7 gains both clauses with numbers.
  P39(d) footnote: debt-bounded reads PASS only against a finished A
  (trivial); against a living A the chase is divergent and parity is
  unmeasurable in this staging (B overtakes A's eval points) — recorded
  as the staging's design limit, not a parity failure. MS-C+I CLOSED.
  **P54 SCORED (2026-08-06 early, results/chimera_v1.json, full at
  1,000-chunk phases, cadence block d128/B8/K64 in the artifact).**
  4/4 PASS — F1 IS LOCKED. (a) chimera residual after recovery −0.040
  (BETTER than pre-shock) vs no_monitor +0.424 — the dividend monitor's
  recovery attribution survives 6.7× exposure with room. (b) THE
  REMINDER IS DECIDED: 12 injections (≥10, decidable for the first
  time), and minus-reminder loses 0.0278 on the plasticity axis
  (bar 0.02) with the injection instrument reading effect (mean NLL on
  reminded chunks 4.545) — P33's pre-registered UNDECIDABLE resolves to
  "the organ earns its place," narrowly and cleanly, on one axis.
  (c) THE DOSE CONFOUND WAS THE WHOLE v0 LOSS: chimera beats r3 RAW on
  all three axes (forgetting 0.381 vs 1.306, plasticity 0.672 vs 0.631,
  recovery −0.040 vs +0.599) at 41% FEWER gradient tokens (426,496 vs
  718,336) — normalization only widens it. (d) no arm dominates
  chimera; chimera fully dominates its own no_reminder ablation. The
  scaling detail that outranks the clauses: r3 degrades near-linearly
  with exposure (forgetting 0.189→1.306 at 6.7× phases) while chimera
  degrades sublinearly (0.259→0.381) — COMPOSITION IS THE STABILIZER.
  One process, every plasticity and memory decision from its own
  surprise, at production cadence, at decidable exposure, beating every
  single organ and every ablation on fewer gradients: F1's locking
  experiment is complete.

- **P56 — the width law's fourth point, and the attractor hardened
  (MS-T, registered 2026-08-06, BEFORE the launches).** The width law
  stands on three widths with full seed-robustness at d512; its named
  precision boundary is the 24.7% side (one seed per width at 50M).
  Three simultaneous runs, exact 50M recipe, all cadence-recorded:
  d1024/seed42 (the fourth curve point, solo on the ARM machine —
  ignition is the co-load-sensitive phase), d128/seed43 and d256/seed43
  (the attractor hardening, one x86 runner each). Registered:
  (a) EFFICIENCY MONOTONE: d1024's improvement per million gradient
      tokens ≥ d512's 0.3547 — selection keeps getting cheaper where
      gradients cost most.
  (b) THE RATE KEEPS FALLING: d1024 cumulative gate rate < d512's
      19.8% — the ignition-depth mechanism continues with width.
  (c) THE ATTRACTOR IS SEED-ROBUST: seed43 lands within ±0.5pp of
      24.7% at BOTH d128 and d256 — two widths, one rate, two seeds.
  Falsifiers: (a) fails ⇒ the efficiency curve has a peak and d512 is
  it — the law gets a maximum, stated as such; (b) fails ⇒ the
  depth→rate mechanism saturates, localize where; (c) fails ⇒ the
  24.7% attractor was seed accident and the curve needs seed averaging
  before any width claim below d512.

  **P56 SCORED (2026-08-06, results/gate_law_width_curve_q08.json,
  all three runs complete at 50,000,384 tokens, cadence in every
  status file).** The scoring found a config drift that rewrites the
  width-rate story, and the correction outranks every clause. THE
  DRIFT: the original width curve's note claimed one identical q=0.75
  recipe across d128/d256/d512 — the machine-written configs show the
  d512 run was launched at q=0.8, and the seed-43 hardening runs (and
  d1024) were launched at q=0.8 as well. The accident completes a
  2×4 q-width grid, and the grid separates the variables cleanly.
  THE RATE IS A DIAL, NOT AN EMERGENT: five q=0.8 runs across an 8×
  width range land inside 0.1981–0.1994 (0.13pp scatter; seed delta
  0.02pp), both q=0.75 runs land at 0.2472 — the 4.9pp "fall at d512"
  was the q flip, there never was a width→rate effect, and the old
  curve's load-bearing open question (why d512 fires less) dissolves.
  Clause verdicts: (a) FAIL, falsifier fires as pre-committed — at
  fixed q=0.8 the per-token efficiency falls monotonically with width
  (0.3696 → 0.3629 → 0.3547/0.3467 → 0.3305; d1024 misses the 0.3547
  bar): the registered "efficiency grows with width" was the q
  confound (a q=0.8 dose is fewer, more selective tokens — higher
  per-token improvement at ANY width). (b) formal inequality reads
  true (0.1981 < 0.1984) but is VOID as a width mechanism — scored as
  CORRECTED, rate ≈ (1−q) invariant to width, seed, and (per the
  grid) everything else measured. (c) NOT SCORABLE as registered —
  protocol drift: the seed-43 runs did not replicate the q=0.75
  attractor recipe, so the 24.7% side remains one-seed; the completing
  q=0.75/seed-43 pair is relaunched. What the drift measured instead
  is stronger than the registered clause: width-invariance of the
  rate at fixed dose. THE HEADLINE ABOVE ALL CLAUSES: at d1024 the
  improvement ratio CROSSES 1.0 — the gated arm ends at 5.4164
  against the full-gradient arm's 5.4464 (ratio 1.0092) on 19.8% of
  the gradient tokens. Selection is no longer a discount at d1024: it
  wins outright. The fixed-q ratio curve reads 0.9708 → 0.9892 →
  0.9777/0.9758 (two seeds agree — the d512 valley is real) → 1.0092,
  and the d128→d256 rise replicates at both q levels.

  **P56(c) COMPLETED (runs finished 2026-08-07, scored 2026-08-10;
  results/pos_s43q75_d128_status.json,
  results/pos_s43q75_d256_status.json).** The true q=0.75/seed-43
  pair closes the clause: d128 reads 0.2466, d256 reads 0.2478 —
  both within 0.06pp of seed-42's 0.2472 against the ±0.5pp bar.
  (c) PASS. The 24.7% line now stands on two widths × two seeds, so
  BOTH q levels are seed-hard and the dial law is complete on its
  grid. Profile reproduction beside the rate: at d256/q0.75 the seed
  pair reads ratio 0.9951 vs 0.9953 and efficiency 0.2936 vs 0.2937
  — reproduction to the third decimal; at d128/q0.75 ratio 0.9807 vs
  0.9729 (the narrowest width carries the largest ratio seed-delta,
  0.0078, while every RATE reproduces within 0.06pp — the rate is
  the hard invariant; the ratio softens with narrowness).

- **P55 — the file answers by key (MS-Q, registered 2026-08-06, BEFORE
  the build).** P47/P52 proved the frozen file transfers DIFFUSELY
  (global NLL gains to strangers). The end architecture claims more:
  tappable — targeted retrieval. Setup (knowledge_file_run substrate,
  d128/B8/K64, x86 runner, chained behind the s43_d128 width run):
  producer streams C4, harvests, freezes (sha256). Consumer twins from
  one init: WITH dosed file replay vs WITHOUT, then the KEYED PROBE —
  for N=100 sampled file entries, present the first half of the span
  (the key) and measure completion NLL on the second half (the value);
  plus a CONTROL probe on 100 never-harvested spans from the same
  stream region (matched length).
  (a) KEYED RECALL: with-file completion NLL on harvested values beats
      without-file by ≥ 0.05 (targeted, not diffuse — bar 2.5× the
      diffuse P47 effect).
  (b) SPECIFICITY: the with-file advantage on HARVESTED spans exceeds
      its advantage on control spans by ≥ 0.02 — the file carries THE
      ENTRIES, not just the domain.
  (c) mechanics: file frozen, sha recorded, both arms bit-comparable.
  Falsifiers: (a) fails ⇒ dosed replay stores no keyed content at this
  dose — the tappable claim needs an index-mediated read (the F4
  route), stated as the boundary between file-as-fertilizer and
  file-as-memory; (b) fails while (a) passes ⇒ the gain is domain
  adaptation, not entry storage — same boundary, other side.

  **P55 RUN 1 INSTRUMENT-NULL (2026-08-06, results/keyed_file_null_v1
  .json, full 1,500/1,500 chunks).** Zero probes: the harness filtered
  probe material at ≥ K+1 = 65 tokens while the harvest geometry caps
  every span at one chunk — 64 tokens (interior spikes give 33..64).
  The condition is unsatisfiable, both probe arms read a vacuous 0.0,
  clauses (a)/(b) are NOT scorable — a harness defect, not a verdict.
  What the run does carry: diffuse heldout gain +0.0356 (the file's
  fertility replicates a third time, on the P47 scale). Harness fixed
  the same hour — split at each span's own midpoint, controls
  length-matched per entry, minimum 33 tokens — fix verified on smoke
  (20/20 probes populated), full rerun in flight as v2. Clauses await
  the v2 artifact (results/keyed_file.json).

  **P55 SCORED (2026-08-06, results/keyed_file.json v2 after the
  probe-geometry fix, full 1,500/1,500 chunks, cadence block
  d128/B8/K64 q0.75 in the artifact).** 2/2 PASS + mechanics — THE
  FILE IS TAPPABLE. (a) KEYED RECALL +0.2643 (bar 0.05, 5.3× over):
  the with-file twin completes the file's own entries from their
  first half 0.26 nats better than its no-file twin — 7.4× the
  diffuse heldout effect (+0.0356); targeted, not diffuse. (b)
  SPECIFICITY +0.2182 (bar 0.02, 10.9× over): the advantage on
  harvested entries dwarfs the advantage on length-matched
  never-harvested spans from the same stream region (control gain
  +0.046) — the file carries THE ENTRIES, not just the domain. (c)
  mechanics hold: file frozen, sha256 87c81bcc0081…, and the
  producer's harvest is bit-identical across the null run and v2
  (same sha) — the file itself is deterministic. The falsifier
  boundary does not bite at this dose: dosed replay alone stores
  keyed, entry-level content; no index-mediated read is needed for
  recall at 1,500 consumer chunks. MS-Q complete: frozen file →
  dosed replay → keyed retrieval, measured end to end.

- **P57 — the composition curve: sublinearity over the exposure decade
  (MS-S, registered 2026-08-06, BEFORE the launch).** P54's stabilizer
  finding stands on two exposure points (150 / 1,000 chunk phases).
  The decade: phases 5,000 and 20,000, arms chimera + r3_replicate +
  r1_full (the ablations earned their verdicts; the curve needs the
  three regimes), same instruments, cadence recorded, chained on the
  second x86 runner behind the s43_d256 width run.
  (a) SUBLINEARITY IS A LAW: chimera forgetting vs phase length fits
      log-log slope < 0.8 over the four points (150, 1k, 5k, 20k),
      while r3's slope reads ≥ 0.9.
  (b) THE GAP WIDENS: chimera/r3 forgetting ratio at 20k ≤ its ratio
      at 1k (0.292) — composition's edge grows or holds with exposure.
  (c) RECOVERY SURVIVES THE DECADE: chimera residual after recovery
      ≤ 0.1 at 20k phases.
  Falsifiers: (a) fails ⇒ sublinearity was a two-point accident, the
  stabilizer claim reverts to the measured pair; (c) fails ⇒ the
  monitor's recovery has an exposure ceiling — locate it.

  **P57 SCORED (completed 2026-08-09, scored 2026-08-10,
  results/chimera_curve_5k.json + results/chimera_curve_20k.json,
  five arms at 5,000- and 20,000-chunk phases, cadence read from the
  POS checkpoint config).** The decade is measured and splits the
  clauses cleanly. (a) HALF-PASS, and the passing half is a law:
  chimera's forgetting fits log-log slope 0.295 over the four
  exposure points (150/1k/5k/20k: 0.259 → 0.381 → 0.752 → 1.027) —
  far under the 0.8 bar; sublinearity is a four-point law, not a
  two-point accident. The r3 contrast half fails in an unexpected
  direction: r3 does not stay near-linear, it SATURATES (1.306 →
  1.764 → 1.245; four-point slope 0.386) — forgetting has a ceiling
  and the fixed-schedule arm hits it early, sitting 21–72% above
  chimera throughout. (b) FAIL as registered: the 20k forgetting
  ratio reads 0.825 against ≤0.292 — the gap narrows because r3's
  forgetting hits its ceiling, not because chimera degrades; chimera
  remains the lowest-forgetting arm at every exposure (1.027 vs
  1.245 r3 vs 1.954 full-gradient — the firehose forgets most at 4×
  chimera's 7.8M gradient tokens). (c) PASS — RECOVERY SURVIVES THE
  DECADE: chimera residual +0.018 at 20k phases (bar 0.1), back to
  pre-shock level at 20× the P54 exposure, while the no_monitor
  ablation reads +0.651 — the dividend monitor's recovery
  attribution stands at every measured exposure. Finding beyond the
  clauses: the reminder organ's P54 edge VANISHES at 20k
  (no_reminder matches chimera on all three axes — forgetting 1.019
  vs 1.027, plasticity 0.988 vs 0.986, recovery +0.002 vs +0.018):
  the reminder is a small-exposure organ; its fixed dose does not
  scale with phase length, named as the next attack on that organ.
  Composition remains the stabilizer at the decade: lowest
  forgetting, near-zero recovery residual, quarter of the
  full-gradient budget.

- **P58 — surprise is a knowledge filter, or it is not (MS-R,
  registered 2026-08-06, BEFORE the build).** The distiller's ground
  question, stripped of any product: do surprise-selected positions
  point at NOVELTY, or merely at difficulty? Substrate WT-103 (dense
  relations, the P43 stream), organism streams 3,000 post-ignition
  chunks; SELECTION = the top-M chunk positions by gate surprise,
  CONTROL = M seeded-random post-ignition positions, both expanded to
  matched 128-token windows. The instrument is fully deterministic:
  a running registry of every token type and bigram the stream has
  shown so far; a window's novelty = its count of FIRST-EVER types and
  bigrams; its redundancy = its share of high-frequency (≥5 prior
  occurrences) bigrams.
  (a) SURPRISE POINTS AT NEW: median first-ever rate of surprise
      windows ≥ 1.5× the random windows'.
  (b) AND AWAY FROM OLD: median redundancy ≤ 0.8× the random windows'.
  (c) instrument validity: the verdicts hold against a second random
      seed (control-of-the-control).
  Falsifiers, pre-committed: (a) near 1.0× ⇒ surprise selects
  DIFFICULTY, not novelty — random sampling would feed a graph equally
  well, and F1-as-distillation-filter dies in this form, measured
  BEFORE anything is built on it. (b) fails alone ⇒ surprise finds new
  things amid old ones — filter useful for discovery, not dedup.
  Runs chained on the 16-core runner behind P55.

  **P58 SCORED (2026-08-06, results/surprise_filter.json, full 3,000
  chunks on the 16-core runner, cadence block d128/B8/K64 q0.75 in the
  artifact).** 3/3 PASS — SURPRISE POINTS AT NOVELTY, NOT DIFFICULTY.
  (a) first-ever rate of surprise windows 0.2656 vs random 0.1758 —
  ratio 1.511 (bar 1.5); on brand-new TYPES alone the gap is 7.0×
  (0.0547 vs 0.0078) — the gate lands on the stream's first encounters.
  (b) redundancy 0.2734 vs 0.3984 — ratio 0.686 (bar 0.8): selection
  points away from the already-frequent. (c) both verdicts hold against
  the second random seed (first-ever ratio 1.545, redundancy ratio
  0.648) — the instrument is not a seed artifact. The distiller's
  ground question is answered: gate surprise IS a knowledge filter —
  it selects windows where the stream shows new types and new bigram
  relations and skips the redundant, measured on a fully deterministic
  registry instrument with a control-of-the-control. The
  F1-as-distillation-filter claim survives its pre-committed falsifier.

- **P59 — the aged brain: does a 7-billion-token life gate, learn, and
  survive shocks differently? (MS-U, registered 2026-08-06, BEFORE the
  fork).** Today's determinism results turn the project's most
  expensive asset — the single uninterrupted multi-billion-token life —
  into an experimental substrate: its checkpoint can be forked
  (read-only copy; the living run is not touched) and the VETERAN
  measured against a FRESH 50M-token organism of identical
  configuration on an identical protocol. A new science axis: age.
  (a) THE RATE CARRIES AGE OR IT DOES NOT: the veteran's cumulative
      gate rate over a 2,000-chunk C4 continuation differs from the
      young organism's same-window rate by more than ±2pp (age shifts
      the surprise economy) — or it does not (the rate is a property
      of width+recipe alone, invariant to experience; either verdict
      is a law).
  (b) SHOCK: on the MS3 protocol (C4→code→C4, 1,000-chunk phases), the
      veteran's forgetting is LOWER than the young twin's (experience
      = immunity) — falsifier: HIGHER (age = ossification), and the
      direction is the finding.
  (c) PLASTICITY PRICE: the veteran's phase-2 plasticity within 25% of
      the young twin's (aging does not freeze learning) — outside ⇒
      the plasticity-decay curve becomes its own registered follow-up.
  Runs on the 16-core runner chained after P58; the lifetime host is
  never touched beyond one checkpoint copy.

  **P59 SCORED (2026-08-06 night, results/aged_brain.json, full:
  veteran forked at 7,442,664,960 tokens WITH optimizer state, young
  twin raised fresh in-harness at 49,999,872 tokens on the same
  vocabulary, cadence block d128/B8/K64 q0.75 in the artifact; the
  living run untouched beyond the one checkpoint copy).** The age axis
  opens with two laws and one price. (a) AGE DOES NOT SHIFT THE RATE:
  veteran 0.2970 vs young 0.2920 on the same 2,000-chunk fresh-gate C4
  probe — Δ0.50pp against the ±2pp bar. The either-way registration
  lands on INVARIANCE: 149× more lived experience moves the gate rate
  by half a point — the surprise economy is set by the recipe (per
  P56: by q), not by biography. The dial law gains its fourth
  invariance axis (width, seed, and now age). (b) EXPERIENCE IS
  IMMUNITY: the veteran forgets 32% less under the WT-103 shock
  (+0.0845 vs +0.1243) and returns BELOW its pre-shock loss after
  recovery (−0.0086 vs +0.0032) — entering the shock 0.167 nats ahead
  on C4 (4.4289 vs 4.5957). (c) THE PLASTICITY PRICE IS REAL AND
  SMALL: veteran phase-2 plasticity 0.7442 of the young's — outside
  the 25% band by 0.6pp, so the pre-committed follow-up fires: the
  plasticity-decay curve across age is registered as P66, and its
  ladder is already on disk (four archived mid-life snapshots of the
  909M organism plus its final state, plus the two lives measured
  here).

- **P60 — stranger verification: the hub's review mechanism as a
  measurement (MS-V, registered 2026-08-06, BEFORE the build).** Every
  provenance replay so far was run by the file's CREATOR. The end
  architecture's community review requires the opposite: a STRANGER on
  DIFFERENT hardware verifies a knowledge file by pure recomputation.
  (a) CROSS-ISA VERIFICATION: a file harvested on the ARM machine
      verifies 10/10 entries bit-exact on an x86 runner from
      coordinates alone (no creator state beyond the file), and the
      reverse direction likewise.
  (b) CONSENSUS FOR FREE: two independent replays of the same entries
      on the two ISAs agree bit-exactly with each other — majority
      verification needs no trust, only determinism.
  Falsifier: any cross-ISA mismatch localizes to a named layer
  (tokenizer, stream, arithmetic) — and that layer becomes the spec's
  normative anchor before any hub exists.

  **P60 SCORED (2026-08-06 night, results/stranger_verify.json
  [ARM→x86] + results/stranger_verify_arm.json [x86→ARM, file
  results/vizdoom_knowledge_x86.jsonl, 664 entries]).** The review
  mechanism works — and it caught exactly the kind of defect it
  exists to catch. FORWARD: the ARM-created file verifies 10/10
  bit-exact on x86 from coordinates alone, consensus 10/10. REVERSE:
  9/10 — and the falsifier fires WITH its demanded localization. The
  one divergent entry (lane 4205, episode 0, frame 14) differs in
  exactly ONE token at one position: x86 records gray level 2, both
  independent ARM replays read 1 — adjacent quantization bins, all 15
  other tokens identical, within-ISA consensus 10/10 on both sides.
  THE LAYER IS NAMED AND MEASURED: `tokenize_frame`'s float block-mean
  (80 uint8 pixels) followed by float floor-division (256.0/12) — a
  bin-boundary block flips on the 1-ULP difference between the two
  ISAs' vectorized reduction orders. Not the engine (frames replay
  identically), not the stream, not the tokenizer map: pure float
  arithmetic at a quantization edge. THE NORMATIVE ANCHOR, per the
  pre-commitment: pixel→token quantization must be integer-exact —
  level = (block_sum × LEVELS) // (80 × 256) on the exact uint8 sum —
  ISA-invariant by construction; recorded as a standing spec rule in
  DECISIONS.md (existing artifacts keep the measured v1 map). Net:
  cross-ISA verification 19/20 sampled entries bit-exact, the single
  miss localized to a named, fixable operation — a stranger on
  different silicon found a one-ULP float boundary by pure
  recomputation, which is the strongest possible demonstration of
  what coordinate-based review buys.
  **P59 AMENDED before the build (same morning):** the checkpoint
  adapter is measured trivial — the veteran's state_dict loads into
  the reference Organism class key- and shape-identical at V=5000
  (load + forward verified). Consequence for clause (b): the shock
  domain is WT-103 (the established P43/P47 shock substrate) instead
  of code, because the veteran lives in its own 5,000-type vocabulary
  and the code loader is coupled to the WT-2 stack; WT-103 streams
  tokenize cleanly through the veteran's own stoi, keeping both arms
  in the veteran's universe. Protocol otherwise unchanged; the young
  twin is raised fresh at 50M tokens on the SAME vocabulary.

- **P61 — the retrodiction organ, keyed v1 (MS-E, registered
  2026-08-06; full design in analysis/RETRO_SPEC_DRAFT.md, committed
  with this entry; BEFORE the build).** v0 (P41) died as an
  instrument: a bulk-histogram probe reads the marginal, not the
  memory. v1 measures the DECAY of the existing holographic keyed-read
  operation — the read is keyed by construction and cannot see the
  marginal — as a live per-chunk retention observable over the ladder
  H ∈ {2, 8, 32, 128} chunks, deliberately under the measured 16k MQAR
  wall so the ladder reads gradual decay, not an edge. Two mandatory
  controls: shuffled-key (kills "reads the prior"; the REAL−SHUFFLED
  contrast is load-bearing, the P41 amendment's lesson) and
  foreign-organism state (kills "reads a generic decay law"; the P60
  stranger logic turned inward). Lead decisions frozen at
  registration: Q1 — BOTH key substrates, in role division: a
  synthetic-MQAR arm as the instrument gate (proves the keyed read can
  read at all on this stack), the organic store-key arm as the
  measurement; Q2 — the value target is the span PREFIX (first K=64
  true tokens: rich, carrier-length, no summary and hence no
  marginal-readable channel); Q3 — the three-way replay-selector
  comparison (retro vs harvest-surprise vs 1/age) is its own follow-up
  registration after (d), not smuggled in. Clauses (a)–(d) exactly as
  drafted in the spec (meter: keyed-sees-signal ≥0.10 @ H8/H32 over
  2σ; two-regime backward knee; foreign state ≤0.02; actuator:
  measured-decay consolidation ≤ the dividend monitor's −0.040
  residual AND strictly better than its own shuffled-trigger control
  by ≥0.03 — the decisive loop control), every clause writing
  p_retro_<letter>_pass booleans per the standing scoring rules.
  Cost ~9 runs × ~3,000 chunks at d128/B8/K64, forking the POS
  checkpoint, chained behind the composition-curve arms.
  **P61 AMENDED before the full (2026-08-06; the four build findings
  are documented in results/RETRO_BUILD_NOTES.md and none was silently
  absorbed; process note: the builder drafted an amendment text
  directly into this file, which is outside its mandate — the draft
  was removed and this reviewed amendment replaces it).**
  (1) SUBSTRATE: the registration assumed the fork checkpoint carries
  the phase channel; measured, ckpt_359050240 is the scalar arm
  (use_phase=False) and the keyed-read operation does not exist on it.
  v1 therefore measures the keyed read on the holographic F3 stack
  (where the channel exists and trains), coupled to the POS store as
  the key/value source. A POS organism whose scan includes the phase
  channel is the truer form and becomes its own registration when
  built. (2) PRECONDITION: an untrained key channel sits at phi≈0 and
  cannot separate keys (measured contrast ~0.0); clauses (a)/(b) are
  read after the key channel is trained on the recall objective —
  training budget is part of the protocol, as with
  trained-with-consultation reads in MS1. (3) ORGANIC KEYS: this
  checkpoint's stored index is too id-sparse for a well-posed organic
  recall task, so the organic arm collects keys and span-prefix values
  live from the WT-2 stream during the run rather than from the frozen
  index; live collection also realizes Q2 (span prefix) literally.
  (4) SCORING DISCIPLINE, confirmed as coded: with the instrument gate
  closed, clauses (a)-(c) return None, not a trivially satisfied pass;
  p_retro_a_scorable records the gate state — the direct fix for the
  v0 vacuous-pass trap. Smoke (nice -19, 16.5s): instrument gate opens
  (MQAR contrast 0.50 at H=2, 0.18 at H=8 vs ~0.06 zeroed-null,
  train_acc 1.0); organic arm null on the frozen store as expected
  under (3); a_pass False / b_pass None / c_pass False / d_pass None,
  cadence block d128/B8/K64 present. Cost driver measured: evaluation
  of the upper ladder rungs (H=128 is an 8,192-token eval), not
  training; clause (d) runs as its own multi-arm actuator experiment
  (monitor / retro / retro_shuffled_trigger on the MS3 shock), anchored
  to the dividend-monitor residual −0.040215.

- **P62 — the dial law: rate ≈ 1−q across the regulator (MS-W,
  registered 2026-08-06 night, BEFORE the launches).** P56's correction
  found two q levels behaving as a dial (q0.75 → 0.2472 twice, q0.8 →
  0.1981–0.1994 five times). Two levels are an observation; a law needs
  the curve. Four fresh 50M runs at d128/seed42, identical recipe, only
  q varied: q ∈ {0.60, 0.70, 0.80, 0.90} (the 16-core x86 runner,
  1 thread each, nice; q0.75 exists as the anchor 0.2472). Registered:
  (a) THE DIAL TRACKS: every cumulative 50M rate lands within ±1.0pp of
      (1−q), and the rates are strictly monotone decreasing in q.
  (b) DOSE-RESPONSE: improvement per million gradient tokens rises
      monotonically with q across the five points (the P56 pair 0.2969
      → 0.3696 extends to a curve) — more selective tokens carry more
      improvement each.
  (c) mechanics: all four runs reach 50M, cadence recorded, rates read
      from the machine-written status files.
  Falsifiers: (a) fails ⇒ the dial has a floor or a nonlinearity —
  locate it; the two measured levels stay as the only calibrated
  settings. (b) fails ⇒ selectivity does not buy per-token value
  beyond some q — the dose-response has a knee, and the knee is the
  operating point.

- **P63 — the crossing is not a seed story (MS-X, registered 2026-08-06
  night, BEFORE the launch).** P56's headline — the gated arm beats the
  full-gradient arm at d1024 (ratio 1.0092, one fifth of the gradient
  tokens) — stands on one seed. The exact P56 d1024 recipe at seed 43,
  solo on the ARM machine (ignition is the co-load-sensitive phase):
  (a) THE CROSSING REPLICATES: seed43 improvement ratio > 1.0 at the
      50M anchor.
  (b) THE PROFILE REPRODUCES: seed43 gate rate within ±0.5pp of
      seed42's 0.1981, and efficiency within ±0.02 of 0.3305/M.
  Falsifiers: (a) fails with ratio ≥ 0.99 ⇒ the crossing sits inside
  seed noise at d1024 — state it as a boundary, the width trend
  itself is untouched; (a) fails below 0.99 ⇒ the d1024 point was a
  seed artifact and the curve needs seed pairs before any crossing
  claim.

  **P63 SCORED (run finished 2026-08-07, scored 2026-08-10,
  results/pos_d1024_s43_status.json + metrics, solo on the ARM
  machine).** 2/2 PASS — THE CROSSING IS NOT A SEED STORY. (a)
  seed-43 improvement ratio 1.0304 (seed 42: 1.0092): the gated arm
  beats the full-gradient arm by 0.096 nats on one fifth of the
  gradient tokens — the second seed lands ABOVE the first, the
  margin tripled. (b) the profile reproduces: gate rate 0.1978 vs
  0.1981 (Δ0.03pp, bar ±0.5pp), efficiency 0.3278 vs 0.3305
  (Δ0.0027, bar ±0.02). Mean d1024 ratio across seeds: 1.0198.
  Selection beating the firehose at width is a two-seed fact.

- **P64 — the filter earns its file, or it does not (MS-Y, registered
  2026-08-06 night, BEFORE the build).** The distiller bridge: P58
  proved the gate points at novelty; P55 proved the frozen file stores
  entries. The composed question — is surprise-HARVESTED content worth
  more per span once distilled into a file? Setup: the P55 substrate
  unchanged; two producers from one init on one stream — SURPRISE
  harvests at gated chunks (the P47/P55 recipe), RANDOM harvests at
  seeded-random chunks at the same rate with random span centers,
  span counts trimmed to match exactly; both files frozen (sha256).
  Three consumer twins from one init: dosed replay of file-S, of
  file-R, and without; identical budgets.
  (a) NOVELTY FERTILIZES: file-S diffuse heldout gain ≥ 1.5× file-R's.
  (b) STORAGE IS CONTENT-AGNOSTIC: BOTH files pass keyed recall of
      their own entries at the P55 bar (≥ 0.05) — replay stores what
      it is given; the filter decides what is worth giving.
  (c) mechanics: matched span counts, same dose, both shas recorded.
  Falsifiers, pre-committed: (a) fails near 1.0× ⇒ novelty selection
  does not add file value at this dose — the distiller can harvest
  ANYWHERE and the filter's value is elsewhere (dedup, coverage),
  measured before anything is built on it; (a) INVERTS (file-R wins)
  ⇒ surprise content is too hard to store efficiently — the filter
  needs a difficulty ceiling, and that ceiling is the next
  registration. (b) fails for file-R only ⇒ storage is NOT
  content-agnostic and surprise content has privileged replay
  dynamics — a mechanism finding bigger than the clause.

  **P64 SCORED (2026-08-06 night, results/filter_file.json, full
  1,500/1,500/1,500 chunks, 540 spans per file exactly matched, both
  shas recorded, cadence block d128/B8/K64 q0.75 in the artifact).**
  (a) INVERTED — the pre-committed falsifier fires, and the negative
  is stated as measured: the random-harvested file fertilizes BETTER
  than the surprise-harvested file on average-C4 heldout (+0.0617 vs
  +0.0356, a 1.7× inversion of the registered direction). Novelty
  selection does not buy diffuse transfer at this dose. (b) PASS both
  — and the split IS the finding: keyed recall of own entries reads
  +0.2643 (file-S) vs +0.2173 (file-R) — the surprise file stores its
  entries BETTER, which refutes the falsifier's own mechanism guess
  ("too hard to store efficiently"): storage of surprise content is
  fine, superior even. The composed reading: THE FILTER BUYS MEMORY,
  NOT FERTILIZER. Surprise selects content worth REMEMBERING (best
  entry-level keyed recall; P58's novelty); random spans match the
  eval's average distribution and therefore win the average-heldout
  race. The two file roles P47/P55 measured — fertilizer (diffuse)
  and memory (keyed) — are served by DIFFERENT harvest policies, and
  the gate serves the memory role. Next attack, named: a
  novelty-matched transfer eval (heldout restricted to rare/first-ever
  content classes, the P58 registry as the instrument) to test whether
  file-S wins on the content class the filter actually selects for —
  registered when launched. Producer determinism note: file-S's sha
  equals the P55 file's sha exactly (third bit-identical harvest of
  the same recipe).

- **P65 — the valley is not a seed story (registered 2026-08-06 night,
  BEFORE the launch).** The fixed-q ratio curve's one non-monotone
  feature — d256 (0.9892) above d512 (0.9777/0.9758) — rests on a
  single seed at d256/q0.8. One run closes it: d256/seed42/q0.8, exact
  50M recipe, second x86 runner:
  (a) THE POINT REPRODUCES: seed42 improvement ratio within ±0.005 of
      seed43's 0.9892 (the D2-measured seed delta at d512 was 0.0019).
  (b) THE VALLEY SURVIVES: seed42's d256 ratio > both d512 ratios —
      the dip at d512 is a width feature, not seed noise.
  (c) the rate stays on the dial: cumulative gate rate 0.1981–0.1994.
  Falsifiers: (a)/(b) fail ⇒ the d256 point carries seed variance the
  d512 pair does not — the fixed-q curve needs seed pairs at every
  width before any shape claim finer than the d1024 crossing.

  **P65 SCORED (run finished 2026-08-07, scored 2026-08-10,
  results/pos_d256_s42_q80_status.json + metrics).** (a) FAIL by one
  ten-thousandth: seed-42 ratio 0.9841 vs seed-43's 0.9892 —
  Δ0.0051 against the ±0.005 bar. The d256/q0.8 ratio seed-delta is
  ~2.7× the d512 pair's 0.0019, and the pre-committed reading
  applies in its narrow form: ratio-shape claims finer than the
  crossing need seed pairs per width. (b) PASS — THE VALLEY SURVIVES
  BOTH SEEDS: 0.9841 > both d512 ratios (0.9777/0.9758); d256 sits
  above d512 on either seed, so the dip at d512 is a width feature,
  not seed noise. (c) PASS: rate 0.1993, inside 0.1981–0.1994 —
  seven q=0.8 runs now sit inside 0.16pp across widths and seeds.

- **P66 — the plasticity-decay curve: what aging costs, across six
  ages (registered 2026-08-06 night, BEFORE the runs; the follow-up
  P59(c) fired).** P59 measured one pair: at 149× age, plasticity
  0.7442 of young. Six same-recipe ages exist on disk: the young twin
  (0.05B, from the P59 artifact), four archived mid-life snapshots of
  the 909M organism (0.18B / 0.24B / 0.36B / 0.74B,
  results/pos_snapshots/, A3 arm) plus its final state (0.91B,
  results/pos_ckpt.pt) — one life, five points — and the 7.44B
  veteran (P59 artifact) anchoring the far end. Each new age runs the
  EXACT P59 veteran-side protocol (fresh-gate 2,000-chunk rate probe,
  WT-103 shock at identical offsets and eval budgets), 16-core
  runner, chained behind P61.
  (a) MONOTONE PRICE: phase-2 plasticity decreases monotonically with
      age across the 0.05B–0.91B ladder.
  (b) NO FREEZE: the 7.44B far point's plasticity ≥ 0.6× the
      youngest's (the curve saturates rather than collapses; the
      functional-form read — plateau vs power-law — is made at
      scoring from the artifact's raw curve).
  (c) IMMUNITY GROWS WHERE PLASTICITY FALLS: forgetting decreases
      monotonically with age across the same ladder — P59(b) becomes
      a trend, not a pair.
  Cross-life caveat carried openly: the 7.44B point is a second life
  (same recipe, different stream history); clauses (a)/(c) are read
  on the one-life ladder plus the young point.
  Falsifiers: (a) non-monotone ⇒ plasticity is not an age function —
  find the covariate; (b) fails ⇒ the organism has a finite learning
  lifetime — state where the fitted curve crosses 0.5× and 0.1×;
  (c) fails ⇒ P59(b) was a pair accident.

- **P67 — the filter's file on the filter's turf (registered
  2026-08-10, BEFORE the build; the attack P64 named).** P64 measured
  the surprise file losing average-heldout transfer 1.7× while
  storing its own entries better. The registered question: does
  file-S win on NOVEL content — the class the gate selects for?
  Setup: the P64 producer pass regenerated (deterministic; shas must
  equal P64's), same three consumer twins; the EVAL is the new
  instrument: C4 heldout chunks stratified into terciles by
  first-ever-bigram rate against a registry accumulated over the
  producer stream (the P58 instrument applied to the eval),
  per-tercile heldout NLL per arm. Runs on the second x86 runner.
  (a) NOVEL TURF, FILTER WINS: in the TOP novelty tercile,
      gain_S ≥ gain_R.
  (b) GRADED: gain_S − gain_R increases monotonically from the low
      to the high tercile.
  (c) mechanics: file shas equal P64's, matched dose, equal tercile
      sizes.
  Falsifiers: (a) fails ⇒ the surprise file is not advantaged even
  on novel evals — the gate is a MEMORY organ only (P55/P64b) and
  fertilization is harvest-policy-free; the architecture assigns
  file-S to the keyed path and any-harvest to the diffuse path,
  measured before anything is built on the contrary assumption.
  (b) fails alone ⇒ the novelty advantage is a threshold, not a
  gradient — find the threshold.

  **P67 SCORED (2026-08-10, results/novelty_transfer.json, full
  1,500/1,500 chunks, file shas bit-equal to P64's — the fourth and
  fifth exact reproductions of the two producers).** 3/3 PASS — THE
  NOVELTY GRADIENT IS REAL AND STRICTLY MONOTONE. delta(S−R) runs
  −0.0530 (low tercile) → −0.0269 (mid) → +0.0015 (high): the random
  file's fertilizer advantage lives ENTIRELY in familiar content
  (3.5× at the low tercile, 0.0741 vs 0.0211), shrinks as the eval
  gets newer, and the surprise file crosses ahead exactly in the top
  novelty tercile (0.0616 vs 0.0601). (a) PASS on the crossing;
  (b) PASS on strict monotonicity — the 0.054-nat swing across
  terciles is the robust finding, the top-tercile edge itself is
  thin (+0.0015); (c) PASS (registry 167,185 bigrams, equal tercile
  sizes, tercile bounds 0.143/0.190). Composed with P64: the
  average-heldout inversion was an eval-distribution artifact,
  precisely as the P64 scoring suspected — the gate harvests for the
  FRONTIER; its file's transfer value is novelty-graded where the
  random file's is redundancy-weighted.

  **P64 SEED CHECK (2026-08-10, results/filter_file_s43.json, seed
  43, 556 matched spans).** The diffuse inversion REPLICATES: gain_R
  0.0735 vs gain_S 0.0381 — 1.9× on the second seed (1.7× on the
  first). Clause (b)'s finer ordering does NOT: keyed recall reads
  S +0.1983 vs R +0.2015, a tie (Δ0.003, against seed-42's S-ahead
  Δ0.047); both clear the 0.05 bar decisively on both seeds. The
  seed-42 "surprise stores BETTER" reading was seed-sized and is
  WITHDRAWN; storage is content-agnostic, full stop. The seed-robust
  facts across P64/P67: (1) random harvest wins average-heldout
  fertilization ~1.8×, (2) both files store their own entries
  equally well, (3) the surprise file's transfer is novelty-graded
  and crosses ahead on the newest content. THE FILTER BUYS THE
  FRONTIER — not storage superiority, not average fertilizer.

- **P68 — the crossing deepens, or it plateaus (registered
  2026-08-10, BEFORE the resume).** d128 crossed ratio 1.0 on the
  token axis at 909M; d1024 crosses at 50M on both seeds (P56/P63).
  The composed question: does width ACCELERATE the token-axis gain —
  is d1024's crossing the start of a widening lead or a plateau? The
  measured d1024/seed42 life resumes from its own checkpoint
  (results/pos_d1024_ckpt.pt; the exact-resume path is the
  D2-measured one) and streams 50M → 150M on the ARM machine, solo,
  identical recipe, q=0.8 explicit; the eval cadence writes the
  ratio trajectory.
  (a) THE LEAD GROWS: improvement ratio at 150M > 1.0092 (the 50M
      reading) — the crossing is a trajectory, not a point.
  (b) NO HUMP: the ratio at ~100M lies between the 50M and 150M
      readings.
  (c) the rate stays on the dial: cumulative gate rate at 150M
      within ±0.5pp of 0.1981.
  Falsifiers: (a) fails at ≤ 1.0 ⇒ the d1024 crossing is a
  50M-anchor artifact (A2's width-lag) and the anchor caveat becomes
  the finding; (a) fails inside (1.0, 1.0092] ⇒ plateau — the
  crossing holds but does not compound; (c) fails ⇒ the dial law
  has a token-horizon boundary — locate it.

- **P69 — the valley is a shape, or it is a q-artifact (registered
  2026-08-10, BEFORE the launch).** The fixed-q0.8 ratio curve has a
  measured shape: d128 < d512 < d256 < d1024, the d512 dip
  seed-robust (P65b). The q0.75 line has only d128/d256. One run —
  d512/seed42/q0.75, exact 50M recipe, second x86 runner — decides
  whether the shape belongs to the width curve or to the q level:
  (a) THE VALLEY IS q-INVARIANT: d512/q0.75 improvement ratio lands
      BELOW d256/q0.75's 0.9951 (the dip replicates at the second
      dial setting).
  (b) THE DIAL HOLDS: cumulative gate rate inside 0.2466–0.2478
      (the measured q0.75 band).
  (c) ORDERING: the ratio lands above d128/q0.75's seed-42 reading
      0.9729 (the full q0.8 ordering transfers).
  Falsifiers: (a) fails ⇒ the valley is a q0.8 artifact and the
  fixed-dose curve shape is dial-dependent — the width law's shape
  claims then need per-q measurement; (b) fails ⇒ the dial has a
  width×q interaction at d512 — locate it.

- **P70 — the curator pays in triplets, or it does not (registered
  2026-08-10, BEFORE the build; the LIVE-CAUSAL bridge experiment,
  analysis/LIVE_CAUSAL_SPEC.md §4).** P58 measured the gate's
  selection in tokens (first-ever types/bigrams); the builder loop
  needs it in the graph's own currency: VALIDATED TRIPLETS. Setup:
  the P58 protocol re-run carrying raw text alongside tokens — the
  organism streams WT-103, top-M surprise windows vs seeded-random
  windows (matched counts and sizes), both window sets run through
  the deterministic extractor + validation gate (vendor/fabel); no
  LLM anywhere in the loop.
  (a) YIELD: validated triplets per kilotoken from surprise windows
      ≥ 1.3× the random windows'.
  (b) NOVELTY IN GRAPH CURRENCY: the rate of triplets containing a
      FIRST-EVER entity (registry over the stream so far) ≥ 1.5×
      random's.
  (c) instrument: extraction is deterministic — byte-identical on a
      repeat pass over the same windows.
  Falsifiers, pre-committed: (a) fails near 1.0× ⇒ surprise selects
  novelty the causal extractor cannot harvest (novel ≠ causally
  structured) — the builder needs a second filter stage (connective
  density), named before built; (a) INVERTS ⇒ causal connectives
  live in familiar prose — the curator routes by content class
  instead of gating extraction; (b) fails ⇒ entity novelty and
  token novelty diverge — the registry instrument gains an entity
  mode.

  **P70 SCORED (2026-08-10, results/curator_yield.json, full 3,000
  chunks / 150 windows per arm on the second x86 runner, real fabel
  extractor + 14-step gate in-process, no LLM; instrument hardened
  in-run: a punctuation-free window reconstruction and a 0/0 vacuous
  pass were caught by the builder agent, and a Unicode case-folding
  misalignment (U+0130 expands under lower()) was caught by its own
  alignment assert and fixed via an index map — all before any full
  numbers existed).** THE REGISTERED FALSIFIER FIRES, near-1.0
  branch, at 10× the smoke sample. (a) FAIL: 18.13 vs 18.60
  validated triplets per kilotoken — ratio 0.974 against the 1.3
  bar. Surprise-selected text is NOT causally denser: extraction
  yield is essentially uniform across the novelty axis. (b) FAIL:
  first-ever-entity rate 0.529 vs 0.514 — ratio 1.03 against 1.5.
  The gate's token-level novelty (P58: 7× on first-ever types) does
  NOT pass through extraction into entity novelty. (c) PASS: double
  extraction byte-identical. THE COMPOSED BUILDER POLICY, measured
  across P58/P64/P67/P70: the gate is the STORAGE curator, not the
  extraction targeter — what it selects is worth REMEMBERING (keyed
  recall P55, novelty-graded transfer P67, low redundancy P58) but
  causal structure itself is uniformly distributed over the stream,
  so builder v0 extracts UNGATED (every window pays equally in
  triplet currency, ~18/kilotoken on WT-103) while the surprise
  signal governs the memory-file and dedup layer. Named next
  instruments, registered when launched: connective-density
  prefilter (the pre-committed (a) route) and — sharper — the
  NEW-EDGE rate against the growing graph (dedup currency; P58's
  0.69× redundancy predicts surprise wins there where entity
  novelty could not).

- **P71 — the live graph never rebuilds, measured (registered
  2026-08-10, BEFORE the measurement; engine built and unit-green,
  src/livecausal/infer.py, ca2b82a).** The LIVE-CAUSAL claims from
  the spec, as numbers on the 16-core x86 runner (seeded synthetic
  chain segments, ~50 records each):
  (a) EQUIVALENCE AT SCALE: a 40-segment graph built incrementally
      (on_append per segment) serializes bit-identically to the
      batch rebuild over the same store — canonical inferred-edge
      bytes equal, sha-comparable.
  (b) DELTA SCALING: appending one fixed-size segment onto graphs of
      5 → 80 segments (16× data growth) grows the append wall-time
      by ≤ 4× — cost tracks the delta's neighborhood, not the graph.
  (c) TRUNCATION IS A SCAN: dropping a mid-graph segment costs no
      more wall-time than one append at the same graph size, result
      bit-equal to the batch rebuild without that segment, zero
      closure recomputation of surviving edges.
  Falsifiers: (b) fails ⇒ the neighborhood term dominates earlier
  than the spec assumes — measure the density at which delta cost
  crosses batch cost and state it as the format's operating bound;
  (c) fails ⇒ invalidation needs an index (derivation → inverted
  index), named before built.

  **P71 SCORED (2026-08-10, results/livecausal_p71.json, run on the
  16-core x86 runner at OMP=1/nice 15, engine at ca2b82a plus an
  optional closure-counter flag whose default is byte-identical —
  both test suites green before and after).** 3/3 PASS — THE LIVE
  GRAPH NEVER REBUILDS, AS NUMBERS. (a) 40 segments incremental vs
  batch: 222 inferred edges on both paths, canonical-bytes sha256
  IDENTICAL (c6a96567…). (b) 16× data growth (5 → 80 segments), one
  fixed 50-record delta: append time grows 1.68× (bar ≤ 4; medians
  7.9ms → 13.3ms), and the delta's inferred-edge yield is constant
  (190) at every graph size — delta locality measured, not assumed.
  Scoring note carried openly: the workload is chain-shaped with a
  fixed anchor; the density at which the neighborhood term would
  dominate remains the format's operating bound per the registered
  falsifier and gets measured when real extraction graphs exist
  (P70/P72 output). (c) truncation: five repetitions, ZERO closure
  computations on drop (instrumented at the only two closure sites),
  5/5 bit-equal to the batch rebuild without the dropped segment,
  drop median 8.4ms ≤ append median 10.6ms. The three mechanics the
  LIVE-CAUSAL spec §2 claims are now measured mechanics.