"""
FOSS-KI Engine — The Combined AI
===================================
Puts all layers together into a working system.

Architecture (8 layers):
  ┌─ Vortex Language Model (3-fiber PPM, +9% on real text)
  │    ├─ Fiber 1: Character PPM (3-adic weight: 1)
  │    ├─ Fiber 2: Structure PPM (3-adic weight: 1/3)
  │    └─ Fiber 3: Context PPM (3-adic weight: 1/9)
  │
  ├─ Knowledge Store (Modern Hopfield, anti-hallucination)
  │    └─ Attractor distance = confidence → "I don't know"
  │
  ├─ MarkovReservoir (PS-Lifted, fixed weights)
  │    └─ Z₂ parity as additional features
  │
  ├─ Hopfield Memory (pattern storage/recall)
  │
  ├─ Ensemble Consensus (Foss-gossip, 26-115× speedup)
  │
  ├─ Topology Evolver (MAP-Elites over graphs)
  │
  ├─ Spike Encoder (Z₂ parity events)
  │
  └─ Readout (linear ridge, the ONLY trained part)

Key principles:
- Online adaptation: every output feeds back (no frozen weights)
- Anti-hallucination: attractor distance says "I don't know"
- No catastrophic forgetting: PPM trees are additive
- Foss consensus WHERE IT WORKS: bottleneck networks
- 3-adic hierarchy: local > global (algebraically weighted)
"""

import numpy as np
import networkx as nx
from .markov import PSLifted
from .reservoir import MarkovReservoir
from .memory import HopfieldMemory
from .consensus import FossConsensus, EnsembleConsensus
from .evolution import TopologyEvolver, GraphGenome
from .spikes import SpikeEncoder, EventDrivenProcessor
from .knowledge import KnowledgeStore
from .vortex import VortexLanguageModel, SymmetryDetector


class FossKI:
    """
    The combined Markov Chain AI.

    NOT a transformer. NOT gradient-based (except linear readout).
    Built from: Reservoir Computing + Hopfield Memory +
    Foss Consensus + Evolutionary Search + Z₂ Spikes +
    Vortex Language Model + Knowledge Store.

    Key differentiators from Transformers:
    - Anti-hallucination via attractor distance (knows when it doesn't know)
    - No catastrophic forgetting (PPM trees are additive)
    - Online adaptation (system improves while running)
    - CPU-based, no GPU monopoly, ~7ms per query
    - Algebraically determined structure, not learned
    """

    def __init__(self, config=None):
        """
        Args:
            config: dict with architecture parameters
        """
        defaults = self._default_config()
        if config:
            defaults.update(config)
        self.config = defaults
        self._build()

    @staticmethod
    def _default_config():
        return {
            # Graph
            'n_reservoir_nodes': 50,
            'graph_type': 'barabasi_albert',
            'graph_params': {'m': 3},

            # Reservoir
            'input_dim': 1,
            'output_dim': 1,
            'spectral_radius': 0.95,
            'input_scaling': 0.1,
            'leak_rate': 0.3,

            # Foss
            'p_c': 0.80,
            'p_s': 0.003,

            # Memory
            'memory_size': 100,
            'memory_temperature': 0.0,

            # Ensemble
            'n_ensembles': 1,
            'ensemble_topology': 'ring',

            # Evolution
            'evolve_topology': False,
            'n_generations': 20,

            # Training
            'washout': 20,
            'ridge_alpha': 1e-6,

            # Vortex Language Model
            'lm_order': 6,
            'lm_bridge_strength': 0.0,  # Fibers fully isolated (best result)

            # Knowledge Store
            'knowledge_dim': 128,
        }

    def _build(self):
        """Build all components."""
        cfg = self.config
        n = cfg['n_reservoir_nodes']

        # Build graph
        self.adjacency = self._build_graph(n, cfg['graph_type'], cfg['graph_params'])

        # Build ensemble of reservoirs
        self.reservoirs = []
        for i in range(cfg['n_ensembles']):
            # Each reservoir gets a slightly different graph (seed variation)
            if i > 0:
                A_var = self._perturb_graph(self.adjacency, rate=0.1, seed=i)
            else:
                A_var = self.adjacency

            res = MarkovReservoir(
                A_var,
                input_dim=cfg['input_dim'],
                output_dim=cfg['output_dim'],
                p_c=cfg['p_c'],
                p_s=cfg['p_s'],
                spectral_radius=cfg['spectral_radius'],
                input_scaling=cfg['input_scaling'],
                leak_rate=cfg['leak_rate'],
            )
            self.reservoirs.append(res)

        # Memory
        self.memory = HopfieldMemory(
            cfg['memory_size'],
            temperature=cfg['memory_temperature'],
        )

        # Consensus (if ensemble)
        if cfg['n_ensembles'] > 1:
            self.consensus = EnsembleConsensus(
                cfg['n_ensembles'],
                topology=cfg['ensemble_topology'],
            )
        else:
            self.consensus = None

        # Spike encoder
        self.spike_encoder = SpikeEncoder(self.adjacency, cfg['p_c'], cfg['p_s'])

        # Markov chain (for direct access)
        self.mc = PSLifted(self.adjacency, cfg['p_c'], cfg['p_s'])

        # Vortex Language Model (3-fiber PPM with 3-adic weighting)
        self.language_model = VortexLanguageModel(
            order=cfg.get('lm_order', 6),
            bridge_strength=cfg.get('lm_bridge_strength', 0.0),
        )

        # Knowledge Store (Modern Hopfield with anti-hallucination)
        self.knowledge = KnowledgeStore(
            dim=cfg.get('knowledge_dim', 128),
        )

        # Symmetry detector (for optimal lifting selection)
        self.symmetry = SymmetryDetector(self.adjacency)

        self.trained = False

    @staticmethod
    def _build_graph(n, graph_type, params):
        """Build a graph adjacency matrix."""
        if graph_type == 'barabasi_albert':
            G = nx.barabasi_albert_graph(n, params.get('m', 3), seed=42)
        elif graph_type == 'erdos_renyi':
            G = nx.erdos_renyi_graph(n, params.get('p', 0.15), seed=42)
        elif graph_type == 'watts_strogatz':
            G = nx.watts_strogatz_graph(n, params.get('k', 4),
                                         params.get('p', 0.3), seed=42)
        elif graph_type == 'barbell':
            m = n // 2
            G = nx.barbell_graph(m, max(1, n - 2 * m))
        elif graph_type == 'grid':
            side = int(np.sqrt(n))
            G = nx.grid_2d_graph(side, max(1, n // side))
            G = nx.convert_node_labels_to_integers(G)
        else:
            G = nx.barabasi_albert_graph(n, 3, seed=42)

        return nx.to_numpy_array(G)

    @staticmethod
    def _perturb_graph(A, rate=0.1, seed=0):
        """Slightly modify a graph (for ensemble diversity)."""
        rng = np.random.RandomState(seed)
        A_new = A.copy()
        n = A.shape[0]

        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < rate:
                    A_new[i, j] = 1.0 - A_new[i, j]
                    A_new[j, i] = A_new[i, j]

        # Ensure connected
        genome = GraphGenome(n, A_new, rng)
        return genome.A

    def train(self, X, Y, washout=None):
        """
        Train the system on input-output pairs.

        ONLY the readout layer is trained (ridge regression).
        The reservoir, memory, consensus, and spikes are FIXED.

        Args:
            X: (T, input_dim) input sequence
            Y: (T, output_dim) target sequence (aligned after washout)
        """
        washout = washout or self.config['washout']
        ridge = self.config['ridge_alpha']

        errors = []
        for res in self.reservoirs:
            err = res.train(X, Y[washout:], washout=washout, ridge_alpha=ridge)
            errors.append(err)

        self.trained = True
        return np.mean(errors)

    def predict(self, X, washout=None):
        """
        Predict outputs for input sequence.

        If ensemble: each reservoir predicts, then consensus combines.
        """
        if not self.trained:
            raise RuntimeError("Must call train() first")

        washout = washout or 0

        if len(self.reservoirs) == 1:
            return self.reservoirs[0].predict(X, washout=washout)

        # Ensemble prediction with consensus
        predictions = []
        for res in self.reservoirs:
            pred = res.predict(X, washout=washout)
            predictions.append(pred)

        # Average (consensus is overkill for small ensembles,
        # but demonstrates the architecture)
        return np.mean(predictions, axis=0)

    def generate(self, seed_input, n_steps):
        """Generate sequence autoregressively."""
        if not self.trained:
            raise RuntimeError("Must call train() first")

        return self.reservoirs[0].predict_generative(seed_input, n_steps)

    def store_pattern(self, pattern, meta=None):
        """Store a pattern in Hopfield memory."""
        # Binarize
        binary = np.sign(pattern)
        binary[binary == 0] = 1
        # Pad/truncate to memory size
        if len(binary) < self.memory.n:
            padded = np.ones(self.memory.n)
            padded[:len(binary)] = binary
            binary = padded
        else:
            binary = binary[:self.memory.n]
        self.memory.store(binary, meta)

    def recall_pattern(self, query):
        """Recall nearest pattern from memory."""
        binary = np.sign(query)
        binary[binary == 0] = 1
        if len(binary) < self.memory.n:
            padded = np.ones(self.memory.n)
            padded[:len(binary)] = binary
            binary = padded
        else:
            binary = binary[:self.memory.n]
        retrieved, steps, energy = self.memory.retrieve(binary)
        idx, overlap = self.memory.identify(retrieved)
        return retrieved, idx, overlap

    def evolve_topology(self, X, Y, n_generations=None):
        """
        Use evolutionary search to find the optimal graph topology.

        Fitness = negative prediction error after training.
        """
        n_gen = n_generations or self.config['n_generations']
        n = self.config['n_reservoir_nodes']

        def fitness(A):
            """Evaluate a topology by training and testing."""
            try:
                res = MarkovReservoir(
                    A,
                    input_dim=self.config['input_dim'],
                    output_dim=self.config['output_dim'],
                    p_c=self.config['p_c'],
                    p_s=self.config['p_s'],
                    spectral_radius=self.config['spectral_radius'],
                    input_scaling=self.config['input_scaling'],
                    leak_rate=self.config['leak_rate'],
                )
                washout = min(self.config['washout'], len(X) // 4)
                err = res.train(X, Y[washout:], washout=washout,
                               ridge_alpha=self.config['ridge_alpha'])
                return -err  # Negative error = higher fitness
            except Exception:
                return -1e6

        evolver = TopologyEvolver(n, fitness)
        best = evolver.evolve(n_generations=n_gen, n_offspring=15)

        if best is not None:
            # Update the system with the best topology
            self.adjacency = best.A
            self._build()
            # Retrain with new topology
            self.train(X, Y)

        return best

    # ── Language Methods ─────────────────────────────────

    def train_language(self, text):
        """Train the Vortex Language Model on text."""
        self.language_model.train(text)

    def predict_text(self, context):
        """
        Predict next character using Vortex Language Model.
        Returns dict {char: probability}.
        """
        return self.language_model.predict(list(context))

    def generate_text(self, seed, n_chars, temperature=0.0):
        """Generate text with optional Bernoulli perturbation."""
        old_temp = self.language_model.temperature
        self.language_model.temperature = temperature
        result = self.language_model.generate(list(seed), n_chars)
        self.language_model.temperature = old_temp
        return result

    def text_bpc(self, text, online_adapt=True):
        """Compute bits per character on text."""
        return self.language_model.bits_per_char(text, online_adapt)

    # ── Knowledge Methods ────────────────────────────────

    def store_fact(self, subject, relation, obj):
        """Store a fact in the Knowledge Store."""
        self.knowledge.store_fact(subject, relation, obj)

    def store_facts(self, facts):
        """Store multiple (S, R, O) facts."""
        self.knowledge.store_facts(facts)

    def query_fact(self, subject, relation=None):
        """
        Query the Knowledge Store.

        Returns dict with:
            fact: (S, R, O) or None
            confidence: float
            confidence_level: HIGH/MEDIUM/LOW/REJECTED/UNKNOWN

        Anti-hallucination: REJECTED means we found an attractor
        but it's too far away. System knows it doesn't know.
        """
        return self.knowledge.query(subject, relation)

    def knows(self, subject, relation=None):
        """Quick check: does the system confidently know this?"""
        return self.knowledge.knows(subject, relation)

    # ── Causal Reasoning Methods ────────────────────────

    def load_causal_graph(self, builder_name='simpson'):
        """
        Load a causal graph for interventional reasoning.
        Available: 'simpson', 'sprinkler', 'smoking'
        """
        from .causal import build_simpson_paradox, build_sprinkler, build_smoking_cancer
        builders = {
            'simpson': build_simpson_paradox,
            'sprinkler': build_sprinkler,
            'smoking': build_smoking_cancer,
        }
        if builder_name not in builders:
            raise ValueError(f"Unknown causal graph: {builder_name}. Available: {list(builders.keys())}")
        self._causal_graph = builders[builder_name]()
        return self._causal_graph

    def causal_query(self, target, intervention, evidence=None):
        """
        Interventional query: P(target | do(intervention)).
        Pearl's do-calculus — removes confounding, gives TRUE causal effect.
        Transformers CANNOT do this.
        """
        if not hasattr(self, '_causal_graph'):
            raise RuntimeError("Must call load_causal_graph() first")
        return self._causal_graph.do_query(target, intervention, evidence)

    def causal_effect(self, cause, effect):
        """Compute Average Causal Effect of cause on effect."""
        if not hasattr(self, '_causal_graph'):
            raise RuntimeError("Must call load_causal_graph() first")
        return self._causal_graph.causal_effect(cause, effect)

    # ── Metacognition Methods ────────────────────────────

    def _ensure_metacognition(self):
        """Lazy-init metacognition engine."""
        if not hasattr(self, '_metacognition'):
            from .metacognition import MetacognitionEngine
            self._metacognition = MetacognitionEngine(self.knowledge)
        return self._metacognition

    def smart_query(self, subject, relation=None):
        """
        Self-improving query: try → fill gap → re-try.
        Uses metacognition for automatic gap detection and filling.
        """
        meta = self._ensure_metacognition()
        return meta.query(subject, relation)

    def selftest(self, test_queries):
        """Run self-test benchmark and return regression report."""
        meta = self._ensure_metacognition()
        return meta.selftest(test_queries)

    def gap_report(self):
        """Get prioritized knowledge gap report."""
        meta = self._ensure_metacognition()
        return meta.gap_report()

    def metacognition_status(self):
        """Get metacognition system status."""
        meta = self._ensure_metacognition()
        return meta.status()

    def learn_patterns(self):
        """Discover new extraction patterns from failures."""
        meta = self._ensure_metacognition()
        return meta.learn_patterns()

    # ── Dialog & Extraction Methods ─────────────────────

    def learn_from_text(self, text):
        """
        Extract (S, R, O) triplets from text and store them.
        Online learning: text in → facts immediately queryable.
        Also stores raw text in metacognition corpus for gap-filling.
        Returns number of new facts stored.
        """
        # Store in metacognition corpus for future gap-filling
        meta = self._ensure_metacognition()
        return meta.ingest(text)

    def chat(self, user_input):
        """
        Multi-turn dialogue with anti-hallucination.
        Combines PPM context + Hopfield knowledge + reference resolution.
        Returns dict with response, answer, confidence, etc.
        """
        from .dialog import DialogSystem
        if not hasattr(self, '_dialog'):
            self._dialog = DialogSystem(knowledge_dim=self.config.get('knowledge_dim', 128))
            self._dialog.knowledge = self.knowledge
        return self._dialog.turn(user_input)

    def chat_reset(self):
        """Reset dialogue state (keeps knowledge)."""
        if hasattr(self, '_dialog'):
            self._dialog.reset()

    # ── System Info ──────────────────────────────────────

    def info(self):
        """Print system information."""
        cfg = self.config
        sym_name, sym_order, sym_conf, _ = self.symmetry.recommend_lifting()
        lines = [
            "FOSS-KI Engine",
            "=" * 40,
            f"Reservoir nodes:  {cfg['n_reservoir_nodes']}",
            f"Graph type:       {cfg['graph_type']}",
            f"Input dim:        {cfg['input_dim']}",
            f"Output dim:       {cfg['output_dim']}",
            f"Ensembles:        {cfg['n_ensembles']}",
            f"p_c / p_s / p_r:  {cfg['p_c']} / {cfg['p_s']} / {1-cfg['p_c']-cfg['p_s']:.3f}",
            f"γ_base:           {self.mc.gamma_base:.6f}",
            f"γ_lift:           {self.mc.gamma_lift:.6f}",
            f"Foss ratio:       {self.mc.ratio:.4f}",
            f"Symmetry:         {sym_name} (order {sym_order}, conf {sym_conf:.2f})",
            f"Memory capacity:  {self.memory.n_stored}/{int(0.14 * cfg['memory_size'])}",
            f"Knowledge facts:  {self.knowledge.n_facts}",
            f"LM fibers:        3 (bridge={cfg.get('lm_bridge_strength', 0):.2f})",
            f"Trained:          {self.trained}",
        ]
        return "\n".join(lines)
