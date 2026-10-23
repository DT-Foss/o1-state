# Unification: GSSM-Selective as an Input-Controlled Reproducing Kernel over the Linear-SSM Family

**Project:** GSSM — *From Markov Chains to Minkowski Space* (Foss 2026)
**Disclosed:** 2026-06-26 · PolyForm Noncommercial 1.0.0
**Companion derivation:** `analysis/RKHS_CHARACTERIZATION.md`
**Scan integration + benchmarks:** `src/parallel_scan_integration.py` (results `src/parallel_scan_integration_results.json`)

> **Scope.** The one real result is the RKHS readout (§2). The kernel is a data-dependent, time-inhomogeneous (causal Volterra) object — explicitly **not** a fixed-Mercer "unikernel," and no kernel-ridge shortcut replaces BPTT for the selective model. This section states the readout at full strength, draws that boundary precisely, and adds the measured parallel-scan result that turns the algorithmic half of the claim from aspirational into verified.

---

## 1. The corrected headline claim

**The GSSM-Selective scan is an exact, closed-form reproducing-kernel readout in rapidity space, computed by the same associative affine prefix scan that defines the entire linear-SSM family — and its kernel is a data-dependent, time-inhomogeneous (causal Volterra) kernel, not a fixed Mercer kernel.**

Concretely, with the feature map `φ(v) = log(1 − v²)` and input-gated, forget-weighted coefficients `w_t[k] = α_k ∏_{j>k} γ_j`, the state is an inner product and the output is a fixed nonlinear decode:

$$
z_t \;=\; \langle w_t,\ \phi(\bar v_{0:t})\rangle, \qquad s_t \;=\; \sqrt{1 - \exp z_t}\ \in[0,1).
$$

The induced **per-path time-time Gram**

$$
K^{(x)}(s,t) \;=\; \sum_{k\le \min(s,t)} \alpha_k^2\,\Gamma_{k\to s}\,\Gamma_{k\to t},
\qquad \Gamma_{k\to t}=\textstyle\prod_{j=k+1}^{t}\gamma_j,
$$

is **PSD for every input** (it is the Gram `W Wᵀ` of explicit real weight vectors), and it reduces **exactly** to the stationary exponential Mercer kernel `γ^{|s−t|} = e^{−|s−t|/τ}`, `τ = −1/log γ`, **iff the forget gate is input-independent** — the LTI regime that contains S5 and LRU as the complex-diagonal case.

**What is *not* claimed (and was, falsely, before):** there is **no fixed kernel `k(v,v′)` in raw velocity space** that reproduces the scan across inputs, because the kernel *weights* `α_k(x)`, `γ_j(x)` and the *features* `φ(\bar v_k(x))` are functions of the same `x`. "Replace BPTT with ridge regression on a fixed Mercer kernel" is therefore **false for the selective model** and legal only when the gates are frozen input-independent. The reproducing property is genuine but **conditional on the gate path** — a controlled/Volterra-kernel situation, not a kernel-machine situation.

This is simultaneously **stronger** than the hype (it is an exact structural identity with an explicit feature map, a boundedness-guaranteeing decoder `φ^{-1}`, and a clean specialization to Mamba/S5/LRU) and **narrower** (it tells you exactly when the kernel-ridge shortcut is legal and why selectivity forbids it).

---

## 2. The exact kernel and its PSD status — three Gram objects, three statuses

Conflating these three objects is the easy mistake. Keeping them separate is the contribution.

| Object | Definition | Status | Evidence |
|---|---|---|---|
| **(1) Per-path time-time Gram** `K^{(x)}(s,t)` | Gram of weight vectors `{w_t}` for a *fixed* input path `x` | **PSD unconditionally**: `Σ c_s c_t K^{(x)}(s,t) = ‖Σ_t c_t w_t‖² ≥ 0` | min eig `+0.42` (data-dependent γ path); structural (`K=WWᵀ`), **no counterexample possible** — min eig `≥ −1e-12` over 40 000 adversarial paths incl. γ at [0,1] extremes and signed α |
| **(2) Constant-gate stationary kernel** `c·γ^{|s−t|}` | `K^{(x)}` collapsed at `γ_t≡γ`, `α_t≡α` | **strictly Mercer-PD** (Bochner): DTFT is the strictly positive Poisson kernel `(1−γ²)/(1−2γcosω+γ²) > 0` for `γ∈(0,1)` | min eig `+0.32`; Poisson kernel min `≈0.176 > 0` |
| **(3) Cross-input Gram** | Gram of final states across `N` different inputs | PSD as numbers (min eig `−9.7e-17 ≈ 0`) but **buys nothing** — both `w` and `Φ` depend on `x`; it is a trivial Gram of real vectors, **not** a fixed Mercer kernel in `v` | two paths with identical `φ_k` but different gate context give different `z_T` (`−1.38` vs `−0.44`) — concretely **no fixed `k(v,v′)`** |

**Net PSD statement.** PSD per path always; Mercer/Bochner-strictly-PD iff gates are time-constant; no fixed cross-input Mercer kernel because the gates depend on `x`.

### Precision caveats (so the framing inherits no loose terms)

The following items are cosmetic and corrected here in the framing rather than left to misread:

- **"Mercer" is used loosely.** The per-path object is a finite-dimensional Gram on `ℝ^{T+1}` (the Riesz representer of a bounded linear functional), not Mercer in the continuous-PD-kernel / eigendecomposition sense on a compact domain. We say **"per-path Gram / reproducing readout,"** reserving "Mercer/Bochner-PD" for the constant-gate stationary kernel (2), where it is literally correct.
- **`φ` is not itself a PD kernel.** `φ(v)=log(1−v²)` is a rank-1 nonlinear *coordinate*; the PSD-ness comes from the outer-product memory weighting `w_t w_tᵀ`, **not** from a Moore–Aronszajn kernel `k(v,v′)=φ(v)φ(v′)`. We do not claim universality of any velocity-space kernel.
- **The readout is outside the RKHS norm.** Linearity and the inner-product structure live in `z`-space; `s_t = √(1−e^{z_t})` is a fixed nonlinear decode. Any kernel-regression statement is about `z`, with `s` recovered by the bijection `φ^{-1}`.
- **`[0,1]` is `[0, 1+5e-7]` in code.** The pure math map is exactly `[0,1)` for `z≤0`; the implementation's `+EPS` inside the sqrt lets `s` reach `1.0000005`. Negligible, stated for precision.
- **Code-comment drift (cosmetic).** The executed code logs the *gated* velocity `log(1−(g·v)²)`; the inline comment / docstring still say the ungated `log(1−v²)`. The derivation already tracks the executed (gated) code; only the source comment is stale. *(Suggested one-line code fix; not load-bearing for the theory.)*
- **`≥ 0` vs `> 0` typo in §3.2** of the companion doc: for `γ∈(0,1)` the Poisson kernel is strictly positive, so "strictly PD" is the correct reading.

Numerically the inner-product form reproduces the *actual code recurrence* (gated `\bar v`, eps, clamp) to **8e-9**, and the rapidity identity `φ(tanh ξ) = −2 log cosh ξ` to **1e-14**.

---

## 3. The measured parallel-scan result (depth, identity, wall-time)

The scan is the same associative affine operator for every model in the family (§4), so it is parallelizable by prefix scan — `O(log T)` *depth* instead of `O(T)`. We integrated the parallel (Hillis–Steele doubling) scan into the reference `SelectiveRapiditySqrtScanLayer` via a call-time monkeypatch (covering both the causal and the forward+reverse non-causal paths) **with zero edits to the frozen reference file**, and measured it. The measured identity is *tighter* than the claim — the parallel scan matches the loop to machine precision in fp64.

**Function identity — the parallel scan is provably the same function as the sequential loop, autograd included:**

| Quantity | float32 (MPS) | float64 (CPU) |
|---|---|---|
| Forward max\|Δ\|, causal | 1.49e-7 | 1.67e-16 |
| Forward max\|Δ\|, bidirectional | 8.94e-8 | — |
| Gradient max\|Δ\| (all params + input grad), causal | 1.91e-6 | 3.55e-15 |
| Gradient max\|Δ\|, bidirectional | 5.72e-6 | — |

The float32 deltas are pure FP-reassociation noise from the doubling scan's summation order; in float64 the two scans **and their autograd** agree to machine precision. `torch.autograd.gradcheck` **passes** on the parallel scan for `T ∈ {1,2,3,7,8,17,32,33}`; par-vs-seq grads w.r.t. both `α` and `γ` agree to **3.55e-15** for all `T` up to 1000. The gradient test is the real test — autograd genuinely flows through the slice/cat/mul/add graph, not forward-only. Edge cases (`T=1`, `T=2`, non-power-of-two `T`) all hold.

**Training identity — swapping the scan changes nothing the model learns:** same seed, fixed offline synthetic tensor, 12 steps: both curves start at **3.930505** and end at **3.119214** (bit-identical to print precision), max \|Δloss\| over the whole curve = **4.77e-7** (≪ the 1e-4 bar). Extended to **120 steps** (10×): drift stays within **1.9e-5**, peaks mid-run, then **shrinks** — no compounding split.

**The timing story.** The `O(log T)` depth advantage converts to wall-time on parallel hardware — 4.2×–5.0× on MPS — and is a wash on CPU, where the tight loop wins. Ship parallel on GPU/MPS, sequential on CPU:

| T | MPS seq → par | MPS speedup | CPU seq → par | CPU result |
|---|---|---|---|---|
| 128 | 4.19 → 0.95 ms | **4.40× par** | 0.63 → 0.74 ms | 0.86× (seq wins) |
| 512 | 14.17 → 3.40 ms | **4.17× par** | 1.67 → 3.17 ms | 0.53× (seq wins) |
| 1024 | 26.82 → 6.02 ms | **4.45× par** | 4.45 → 11.23 ms | 0.40× (seq wins) |
| 2048 | 57.44 → 11.47 ms | **5.01× par** | 9.14 → 32.24 ms | 0.28× (seq wins) |

On MPS the parallel scan wins **4.2×–5.0×** across `T`; on CPU it **loses at every size**, worsening with `T`, because the doubling scan does `O(T log T)` total work plus per-step concat allocations that a tight sequential loop avoids. **Takeaway: ship the parallel scan on GPU/MPS, keep the sequential loop as the CPU fallback.** The kernel statement of §1–2 is independent of which scan computes it. Two precision notes: the parallel path is a couple of bits *noisier* than sequential in fp32 (tree reassociation builds larger intermediate products — biased toward the parallel side, magnitude ≤5e-4 at the adversarial `γ=0.999`/`T=4096` corner, zero algorithmic difference confirmed in fp64); and the forward-only timing above is the **conservative** case — including the backward pass, the parallel scan wins even on CPU.

The O(log T) parallel scan is integrated, function- and gradient-identical to the loop in fp64 (3.55e-15), training-identical (max |Δloss| 4.77e-7 over 12 steps), and 4.2×–5.0× faster on MPS — measured, not promised. The remaining note is on *deployment* (the trained model still runs the sequential loop) and on the work-efficient Blelloch variant (built, but its `index_copy` scatter is less MPS-friendly than the doubling scan's slice/concat form that wins here).

---

## 4. Mamba, S5, LRU as special cases — we generalize the linear-SSM family

Every modern linear/selective SSM is a first-order recurrence `h_t = A_t h_{t-1} + B_t u_t` carrying `(A_t, B_t)` under the **identical** affine operator

$$
(A_2,B_2)\otimes(A_1,B_1) = (A_2 A_1,\ A_2 B_1 + B_2), \qquad \text{identity } (1,0),
$$

which `parallel_scan.py` verifies in the real case to **8.88e-16 in float64** (**4.77e-7 in the default float32**, FP-summation noise, zero algorithmic difference) and which reproduces the sequential recurrence in the complex-diagonal case to **~1e-15 in float64** (S5/LRU taxonomy verified; measured band 5e-16–2.3e-15). The models differ only in **three parametric switches**: the state algebra of `A`, whether `A_t` is input-dependent, and the input map for `B_t`. The affine-prefix-scan reproducing-kernel view is their common generalization.

| Model | State algebra | `A_t` (forget) | `B_t` (drive) | `A_t` input-dep? | Kernel type |
|---|---|---|---|---|---|
| **GSSM-Selective** | real **scalar** `∈(0,1)` | `γ_t = σ(W_γ x_t)` | `α_t φ(\bar v_t)`, `φ=log(1−·²)` | **yes** | data-dep. time-inhomogeneous Volterra, **nonlinear feature φ**, bounded readout `φ^{-1}` |
| **Mamba (S6)** | real diagonal `∈(0,1)` | `\bar A_t = exp(Δ_t A)` | `Δ_t \bar B u_t` (linear) | **yes** | data-dep. time-inhomogeneous — **also not a fixed Mercer kernel**, same reason |
| **S5** | **complex** diagonal | `exp(ΔΛ)` (time-const) | `Δ B u_t` | no | **fixed Mercer** (LTI), complex-exponential kernel |
| **LRU** | **complex** diagonal `e^{−ν+iθ}` | `diag(λ)` (time-const) | `B u_t` | no | **fixed Mercer** (LTI), complex-exponential kernel |

- **GSSM is to Mamba as `φ` is to identity.** Both are input-gated, hence both time-inhomogeneous — **Mamba is not a fixed Mercer kernel either**, for exactly the reason GSSM isn't. GSSM swaps Mamba's linear feature `B_t u_t` for the log-complement rapidity feature `α_t φ(\bar v_t)` and adds a closed-form decoder `φ^{-1}` that **guarantees `s_t ∈ [0,1]`**; Mamba has no such boundedness.
- **S5 / LRU are the LTI Mercer case.** Their `A` is time-constant and input-independent, so they collapse to the genuine fixed complex-exponential kernel. **GSSM's constant-γ limit `γ^{|s−t|}` is the real, scalar, 1-pole sibling** of that kernel.
- **The dictionary:** "selective" ⇔ "`A_t` input-dependent" ⇔ "kernel is time-inhomogeneous, not Mercer." This single equivalence organizes the whole family.

---

## 5. What the characterization carries — and what it deliberately does not

The boundary is the contribution: the reproducing-kernel readout is exact, and exactly two things it is *often assumed* to imply do not hold.

| Statement | Status | Why |
|---|---|---|
| **GSSM-Selective is a reproducing-kernel readout** `z_t=⟨w_t,φ(\bar v_{0:t})⟩`, `s_t=√(1−exp z_t)` | **EXACT** | Verified to 8e-9 against executed code (5.5e-17 against the idealized recurrence, float64; 4.77e-7 in float32, FP-summation noise); rapidity feature `φ(tanh ξ)=−2 log cosh ξ` verified to 1e-14. |
| **The per-path Gram is PSD** | **EXACT** | Structural `K=WWᵀ`; min eig `≥ −1e-12` over 40 000 adversarial paths. |
| **"GSSM is a *fixed* Mercer 'unikernel'; BPTT = ridge regression"** | **DOES NOT HOLD** | Gates depend on `x`, so there is no fixed `k(v,v′)`; two inputs with identical `φ_k` give different `z_T` (−1.38 vs −0.44). The kernel-ridge shortcut is legal only in the constant-gate LTI limit, which the architecture is built to leave. |
| **"the kernel view solves recall / unlocks capacity"** | **DOES NOT HOLD** | The RKHS view *explains* the 14% MQAR limit (bounded scalar functional ⇒ bounded pair-lookup capacity) but does not rescue it; the capacity limit is structural. |
| **Fast parallel-scan training** | **MEASURED** | Integrated, function/grad/training-identical, 4.2×–5.0× on MPS, loses on CPU. Aspirational → benchmarked. |

The split is clean: the **RKHS readout is real and exact**; the **fixed-Mercer / capacity-unlock readings do not survive selectivity**, and the parallel-scan speedup is now measured rather than asserted.

---

## 6. Open items (not yet closed)

1. **The lead falsifiable claim is not yet experimentally closed.** *Predicted:* freeze `W_γ, W_α` so the gates are constant, and a BPTT-trained Selective model's read map must coincide to numerical tolerance with the geometric Toeplitz convolution `γ^{|s−t|}` / kernel-ridge closed-form. `constant_gamma_closed_form` already reproduces the *sequential scan* to 5e-17 in float64 (4.77e-7 in float32, FP-summation noise) — but the open claim is that the **learned constant-gate optimum equals the kernel solution.** This is the single most important experiment to run; if the learned read map does not match, the characterization is wrong.
2. **No exact associative recall from the scalar channel — a capacity limit.** Pure-Selective tops out at **14% MQAR** (and the double dissociation SSAS=100% vs PPAP=16% holds); the RKHS view explains *why* (a rank-controlled scalar functional has bounded capacity for pair lookups) but does **not** rescue recall. One attention layer fixes it in the hybrid; the standalone scalar limit is structural.
3. **Deployment gap on the parallel scan.** The scan is integrated and verified, but the *trained* model still runs the `O(T)` sequential loop; "constant inference state, no KV-cache" is claimed, fast parallel-scan *training* is now demonstrated but not yet the deployed default. The work-efficient Blelloch variant exists but its `index_copy` scatter underperforms the doubling scan on MPS.
4. **Stationarity holds only in the constant-gate limit.** Bochner's spectral PD guarantee certifies positive-definiteness only when gates are input-independent; for selective gates the kernel is non-stationary. PSD survives (Gram of real vectors), but the spectral/Fourier PD characterization does not apply.
5. **Cosmetic code/doc fixes** (non-load-bearing): align the source comment/docstring to the executed gated `log(1−(g·v)²)`; note the implementation's `s ∈ [0, 1+5e-7]` vs the mathematical `[0,1)`; change the §3.2 `≥0` to `>0`.

---

### Verification log (numbers cited above)

- Inner-product form vs executed code recurrence (gated `\bar v`, eps, clamp): **8e-9**; vs idealized recurrence: **5.5e-17 (float64); 4.77e-7 (float32, FP-summation noise, zero algorithmic difference)**.
- Rapidity identity `φ(tanh ξ)=−2 log cosh ξ`: **1e-14**.
- Per-path Gram min eig: **+0.42** (data-dep), **≥ −1e-12** over 40 000 adversarial paths.
- Constant-γ stationary kernel min eig **+0.32**; Poisson kernel min **≈0.176**.
- Cross-input Gram min eig **−9.7e-17 ≈ 0** (trivial real Gram, not a fixed kernel); distinct-context counterexample `z_T = −1.38` vs `−0.44`.
- Complex-diagonal (S5/LRU) affine operator vs sequential recurrence: **4e-16**.
- Parallel scan: fwd `1.49e-7`/`8.94e-8` (fp32 causal/bidir), grad `1.91e-6`/`5.72e-6` (fp32); fwd `1.67e-16`, grad `3.55e-15` (fp64); gradcheck passes `T∈{1,2,3,7,8,17,32,33}`.
- Training identity: start `3.930505` → end `3.119214` both scans, max \|Δloss\| **4.77e-7** (12 steps), within **1.9e-5** over 120 steps.
- Speed (MPS): **4.40×/4.17×/4.45×/5.01×** par at `T=128/512/1024/2048`; CPU: **0.86×/0.53×/0.40×/0.28×** (seq wins).
