"""
Vortex Algebra Module — Integrated Z₃/Z₉ Language Architecture
================================================================
NOT a feature list. An integrated system where all 5 components
work together from the start.

Architecture:
  ┌─────────────────────────────────────────────┐
  │  p-Adic Hierarchy (3-adic weighted scales)  │
  │  ┌─────────┬─────────┬─────────┐           │
  │  │ Fiber 1 │ Fiber 2 │ Fiber 3 │  ← FiberRouter
  │  │ (Char)  │ (Struct)│ (Ctx)   │           │
  │  └────┬────┴────┬────┴────┬────┘           │
  │       │  ×3 Bridge (tunable)  │             │
  │       └─────────┬─────────────┘             │
  │         Galois Consensus (6 heads)          │
  │                 │                           │
  │         Bernoulli Perturbation              │
  │         (×2 mod 9, temperature)             │
  │                 │                           │
  │         Online Adaptation                   │
  │         (every output → feedback)           │
  └─────────────────────────────────────────────┘

Key parameters:
  - bridge_strength: ×3 coupling between fibers (THE critical param)
  - temperature: Bernoulli shift intensity (0=deterministic, 1=chaotic)
  - scale_weights: 3-adic (1, 1/3, 1/9) not equal

Reference: VORTEX_MATH_COMPLETE.md (50 tasks, 19 scripts)
T83 baseline: +1.7% on synthetic. Target: >5% on real text.
"""

import numpy as np
from collections import defaultdict
import math
import sys, os

# Import PPM from our language core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.language import PPMModel


# ════════════════════════════════════════════════════════════
# THE INTEGRATED SYSTEM
# ════════════════════════════════════════════════════════════

class VortexLanguageModel:
    """
    Integrated Vortex Language Model.

    Three PPM fibers at different feature levels, connected by
    tunable ×3 bridge, weighted by 3-adic hierarchy, with
    Bernoulli perturbation for generation and online adaptation
    feeding every output back into the model.

    This is NOT three separate PPMs averaged together.
    The fibers communicate through the bridge at a rate
    controlled by bridge_strength, and the consensus uses
    Galois-weighted geometric mean, not simple averaging.
    """

    VOWELS = set('aeiouAEIOU')
    CONSONANTS = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')

    def __init__(self, order=6, bridge_strength=0.15, temperature=0.0):
        """
        Args:
            order: PPM context depth
            bridge_strength: ×3 coupling between fibers [0, 1]
                0.0 = fibers completely isolated
                1.0 = fibers fully coupled (equivalent to single PPM)
                Sweet spot expected at 0.1-0.3
            temperature: Bernoulli shift intensity [0, 1]
                0.0 = deterministic (take mode)
                1.0 = full chaotic exploration
        """
        self.order = order
        self.bridge_strength = bridge_strength
        self.temperature = temperature

        # Three fibers with different feature extractions
        # Fiber 1 {1,4,7}: Raw character PPM (fine-grained)
        self.fiber_char = PPMModel(max_order=order)
        # Fiber 2 {2,5,8}: Structural pattern PPM (V/C/S/D/P classes)
        self.fiber_struct = PPMModel(max_order=order * 2)
        # Fiber 3 {3,6,9}: Context-window PPM (trigram classes)
        self.fiber_ctx = PPMModel(max_order=order)

        # 3-adic scale weights: local > mid > global
        # d(1,0)=1, d(3,0)=1/3, d(9,0)=1/9
        self._scale_weights = np.array([1.0, 1.0/3, 1.0/9])
        self._scale_weights /= self._scale_weights.sum()

        # Fiber consensus weights (adapted online)
        self.fiber_weights = np.array([0.6, 0.25, 0.15])

        # ×2 mod 9 Bernoulli shift table for perturbation
        self._bernoulli_table = self._build_bernoulli_table()
        # Main orbit for temporal modulation: {1,2,4,8,7,5}/9
        self._bernoulli_orbit = [1/9, 2/9, 4/9, 8/9, 7/9, 5/9]
        self._generation_step = 0

        # Online adaptation state
        self._char_history = []
        self._n_updates = 0

    def _build_bernoulli_table(self):
        """Build ×2 mod 9 lookup for Bernoulli perturbation."""
        # Maps position mod 9 to shifted position
        # ×2: 1→2, 2→4, 3→6, 4→8, 5→1, 6→3, 7→5, 8→7, 9→9
        table = {}
        for i in range(1, 10):
            table[i] = (2 * i) % 9 or 9
        return table

    def _char_class(self, ch):
        """Classify character into structural class."""
        if ch in self.VOWELS:
            return 'V'
        elif ch in self.CONSONANTS:
            return 'C'
        elif ch == ' ':
            return 'S'
        elif ch.isdigit():
            return 'D'
        elif ch == '\n':
            return 'N'
        else:
            return 'P'

    def _trigram_class(self, c1, c2, c3):
        """Three-character structural class."""
        return self._char_class(c1) + self._char_class(c2) + self._char_class(c3)

    def train(self, text):
        """
        Train all three fibers on text.

        Online adaptation means training continues during inference,
        so this is just the initial bootstrap.
        """
        chars = list(text)

        # Fiber 1: Raw characters
        self.fiber_char.train(chars)

        # Fiber 2: Structural pattern
        struct_seq = [self._char_class(ch) for ch in chars]
        self.fiber_struct.train(struct_seq)

        # Fiber 3: Trigram context classes
        if len(chars) >= 3:
            trigram_seq = []
            for i in range(len(chars) - 2):
                trigram_seq.append(self._trigram_class(chars[i], chars[i+1], chars[i+2]))
            self.fiber_ctx.train(trigram_seq)

        self._char_history = chars[-self.order * 3:]  # Keep recent history

    def predict(self, context):
        """
        Predict next character using integrated fiber system.

        1. Each fiber predicts independently
        2. ×3 bridge mixes fiber predictions (bridge_strength controls how much)
        3. Galois consensus combines (weighted geometric mean)
        4. Bernoulli perturbation adds controlled chaos

        Args:
            context: list of preceding characters

        Returns:
            dict {char: probability}
        """
        # ── Fiber 1: Character-level prediction ──
        p1 = self.fiber_char.predict(context)

        # ── Fiber 2: Structure-level prediction → project to char probs ──
        struct_ctx = [self._char_class(ch) for ch in context]
        p2_class = self.fiber_struct.predict(struct_ctx)

        p2 = {}
        for ch in p1:
            cls = self._char_class(ch)
            class_prob = p2_class.get(cls, 1e-10)
            p2[ch] = class_prob
        p2_total = sum(p2.values())
        if p2_total > 0:
            p2 = {ch: p / p2_total for ch, p in p2.items()}

        # ── Fiber 3: Trigram context → project to char probs ──
        if len(context) >= 2:
            trigram_ctx = []
            for i in range(max(0, len(context) - self.order), len(context) - 2):
                trigram_ctx.append(
                    self._trigram_class(context[i], context[i+1], context[i+2])
                    if i + 2 < len(context) else 'SSS')
            p3_class = self.fiber_ctx.predict(trigram_ctx)

            p3 = {}
            if len(context) >= 2:
                last_two_cls = self._char_class(context[-2]) + self._char_class(context[-1])
                for ch in p1:
                    target_cls = last_two_cls + self._char_class(ch)
                    class_prob = p3_class.get(target_cls, 1e-10)
                    p3[ch] = class_prob
                p3_total = sum(p3.values())
                if p3_total > 0:
                    p3 = {ch: p / p3_total for ch, p in p3.items()}
            else:
                p3 = p1
        else:
            p3 = p1

        # ── ×3 Bridge: controlled inter-fiber mixing ──
        # With bridge_strength β:
        #   p_mixed_i = (1-β)*p_i + β*mean(p_j for j≠i)
        all_fibers = [p1, p2, p3]
        all_symbols = set()
        for fp in all_fibers:
            all_symbols.update(fp.keys())

        if self.bridge_strength > 0:
            bridged = []
            for i, fp in enumerate(all_fibers):
                others = [all_fibers[j] for j in range(3) if j != i]
                mixed = {}
                for ch in all_symbols:
                    own = fp.get(ch, 1e-10)
                    other_avg = np.mean([o.get(ch, 1e-10) for o in others])
                    mixed[ch] = ((1 - self.bridge_strength) * own +
                                self.bridge_strength * other_avg)
                bridged.append(mixed)
            all_fibers = bridged

        # ── Galois Consensus: 3-adic weighted geometric mean ──
        consensus = {}
        for ch in all_symbols:
            log_p = 0.0
            for i, fp in enumerate(all_fibers):
                p = max(fp.get(ch, 1e-10), 1e-10)
                # 3-adic weighting: fiber 1 (chars) gets weight 1,
                # fiber 2 (struct) gets 1/3, fiber 3 (ctx) gets 1/9
                w = self._scale_weights[i]
                log_p += w * math.log(p)
            consensus[ch] = math.exp(log_p)

        # ── Bernoulli Perturbation: ×2 mod 9 chaos ──
        if self.temperature > 0:
            consensus = self._bernoulli_perturb(consensus)

        # Normalize
        total = sum(consensus.values())
        if total > 0:
            consensus = {ch: p / total for ch, p in consensus.items()}

        return consensus

    def _bernoulli_perturb(self, probs):
        """
        Apply Bernoulli shift as TEMPORAL temperature modulation.

        Instead of permuting symbol ranks (which destroys coherence),
        ×2 mod 9 controls the EFFECTIVE TEMPERATURE at each generation step.

        The main orbit {1,2,4,8,7,5}/9 cycles through 6 different
        temperature levels as text is generated:
          Step 0: T_eff = T * 1/9  (conservative)
          Step 1: T_eff = T * 2/9  (slightly more)
          Step 2: T_eff = T * 4/9  (moderate)
          Step 3: T_eff = T * 8/9  (exploratory)
          Step 4: T_eff = T * 7/9  (pulling back)
          Step 5: T_eff = T * 5/9  (mid-range)
          ... then repeats

        This creates RHYTHMIC variation in exploration — some positions
        in the sentence explore more (word boundaries, new clauses),
        others stay conservative (mid-word spelling). The algebraic
        structure ensures the rhythm is deterministic and reproducible.

        Contrast with softmax temperature: constant T everywhere.
        Bernoulli-shift: structured temporal variation.
        """
        if self.temperature <= 0:
            return probs

        # Get current position in the ×2 mod 9 orbit
        orbit_pos = self._generation_step % len(self._bernoulli_orbit)
        t_eff = self.temperature * self._bernoulli_orbit[orbit_pos]

        if t_eff <= 0:
            return probs

        # Apply softmax temperature scaling with orbit-modulated T
        symbols = list(probs.keys())
        log_p = np.array([math.log(max(probs[s], 1e-10)) for s in symbols])

        # Temperature: divide log-probs by (1 + t_eff) to flatten
        # t_eff=0 → no change, t_eff=1 → fully flat
        effective_temp = 1.0 / (1.0 + t_eff * 3.0)  # Scale so T=1 gives meaningful flattening
        log_p_scaled = log_p * effective_temp

        # Normalize in log space
        log_p_scaled -= log_p_scaled.max()
        weights = np.exp(log_p_scaled)
        weights /= weights.sum()

        return {s: float(w) for s, w in zip(symbols, weights)}

    def update(self, context, symbol):
        """
        Online adaptation: update ALL fibers with new observation.

        This is the architectural principle: every output feeds back
        into every subsystem. The system learns from itself as it works.
        """
        # Fiber 1: Character update
        self.fiber_char.update(context, symbol)

        # Fiber 2: Structure update
        struct_ctx = [self._char_class(ch) for ch in context]
        self.fiber_struct.update(struct_ctx, self._char_class(symbol))

        # Fiber 3: Trigram update
        full_ctx = list(context) + [symbol]
        if len(full_ctx) >= 3:
            trigram_ctx = []
            for i in range(max(0, len(context) - self.order), len(context) - 2):
                if i + 2 < len(context):
                    trigram_ctx.append(
                        self._trigram_class(context[i], context[i+1], context[i+2]))
            if len(context) >= 2:
                target = self._trigram_class(context[-2], context[-1], symbol)
                self.fiber_ctx.update(trigram_ctx, target)

        self._char_history.append(symbol)
        if len(self._char_history) > self.order * 3:
            self._char_history = self._char_history[-self.order * 3:]

        self._n_updates += 1

    def bits_per_char(self, text, online_adapt=True):
        """
        Compute BPC on text with optional online adaptation.

        This is the standard evaluation metric.
        """
        chars = list(text)
        total_bits = 0.0
        context = []

        for ch in chars:
            probs = self.predict(context)
            p = max(probs.get(ch, 1e-10), 1e-10)
            total_bits -= math.log2(p)

            if online_adapt:
                self.update(context, ch)

            context.append(ch)

        return total_bits / max(len(chars), 1)

    def generate(self, seed, n_chars, rng=None):
        """Generate text using Bernoulli-controlled temperature."""
        if rng is None:
            rng = np.random.RandomState()

        context = list(seed)
        generated = []
        self._generation_step = 0

        for _ in range(n_chars):
            probs = self.predict(context)
            if not probs:
                break

            symbols = list(probs.keys())
            weights = np.array([probs[s] for s in symbols])
            weights /= weights.sum()

            next_ch = symbols[rng.choice(len(symbols), p=weights)]
            generated.append(next_ch)

            # Online adapt during generation
            self.update(context, next_ch)
            context.append(next_ch)
            self._generation_step += 1

        return ''.join(generated)

    def adapt_bridge_strength(self, validation_text, search_range=None):
        """
        Find optimal bridge strength on validation text.

        This is THE critical parameter. Too weak = fragmented.
        Too strong = no fiber advantage.

        Returns:
            (best_bridge_strength, best_bpc, all_results)
        """
        if search_range is None:
            search_range = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]

        val_chars = list(validation_text[:3000])
        results = []

        for bs in search_range:
            self.bridge_strength = bs

            # Quick eval (no online adapt for fair comparison)
            total_bits = 0.0
            ctx = []
            for ch in val_chars:
                probs = self.predict(ctx)
                p = max(probs.get(ch, 1e-10), 1e-10)
                total_bits -= math.log2(p)
                ctx.append(ch)

            bpc = total_bits / len(val_chars)
            results.append((bs, bpc))

        # Find best
        best_bs, best_bpc = min(results, key=lambda x: x[1])
        self.bridge_strength = best_bs

        return best_bs, best_bpc, results


# ════════════════════════════════════════════════════════════
# SYMMETRY DETECTOR (for consensus orchestrator)
# ════════════════════════════════════════════════════════════

class SymmetryDetector:
    """
    Detect optimal lifting group for a graph.

    T27: Symmetry-matched lifting beats mismatched by 64%.
    """

    def __init__(self, adjacency):
        self.A = adjacency
        self.n = adjacency.shape[0]
        self._eigenvalues = None

    @property
    def eigenvalues(self):
        if self._eigenvalues is None:
            self._eigenvalues = np.sort(np.linalg.eigvalsh(self.A))[::-1]
        return self._eigenvalues

    def z2_indicator(self):
        eigs = self.eigenvalues
        if abs(eigs[0]) < 1e-10:
            return 0.0
        return abs(eigs[-1]) / abs(eigs[0])

    def z3_indicator(self):
        A3 = self.A @ self.A @ self.A
        n_triangles = np.trace(A3) / 6
        n_edges = np.sum(self.A) / 2
        if n_edges < 3:
            return 0.0
        max_tri = n_edges * (n_edges - 1) / (3 * self.n) if self.n > 0 else 1
        return min(n_triangles / max(max_tri, 1), 1.0)

    def bottleneck_indicator(self):
        D = np.diag(self.A.sum(axis=1))
        L = D - self.A
        eigs = np.sort(np.linalg.eigvalsh(L))
        fiedler = eigs[1] if len(eigs) > 1 else 0
        d_max = self.A.sum(axis=1).max()
        return fiedler / max(d_max, 1)

    def recommend_lifting(self):
        z2 = self.z2_indicator()
        z3 = self.z3_indicator()
        bn = self.bottleneck_indicator()

        indicators = {'Z2': z2, 'Z3': z3, 'bottleneck': bn}

        scores = {
            'Z2': z2 * 0.7 + (1 - z3) * 0.3,
            'Z3': z3 * 0.8 + (1 - z2) * 0.2,
            'Z4': (1 - bn) * 0.5 + z2 * 0.25 + z3 * 0.25,
        }

        best = max(scores, key=lambda k: scores[k])
        order = {'Z2': 2, 'Z3': 3, 'Z4': 4}[best]
        conf = scores[best] / max(sum(scores.values()), 1e-10)

        return best, order, conf, indicators
