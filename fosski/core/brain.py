"""
Brain Snapshot — Unified State for the Entire FOSS-KI System
==============================================================
S14: David's directive — the "Gehirn" as one atomic file.

A Brain contains the COMPLETE cognitive state:
  - Hopfield Knowledge Store (facts + vectors + thresholds)
  - PPM Language Model (context tree + vocab + statistics)
  - Reasoning Engine (rules + common knowledge config)
  - Router Config (fiber weights, escalation thresholds)
  - Metacognition State (gap log, corpus, selftest history)
  - System Config (dimensions, encoder type, etc.)

Save → one file. Load → instant clone with identical knowledge,
language patterns, reasoning rules, and confidence thresholds.

Transfer a Brain to another machine → that instance has the
same knowledge immediately. This is the foundation for:
  - S15: Swarm-Spawn (each agent gets a Brain)
  - S16: Swarm-Consensus (agents share partial Brains)
  - S17: Federated Sync (merge Brains periodically)
  - S18: Superposition Retrieval (Brain state = attractor landscape)

Format: MessagePack + zlib compressed binary.
Fallback: JSON for debugging/inspection.
"""

import json
import os
import time
import zlib
import hashlib
from typing import Dict, Any, Optional
from pathlib import Path


class BrainSnapshot:
    """
    Atomic snapshot of the entire FOSS-KI cognitive state.

    Save/load/transfer the complete system as one file.
    Every subsystem's state is captured and can be restored
    to produce an identical clone.
    """

    VERSION = 1  # Bump when format changes
    MAGIC = b'FOSS-KI-BRAIN'

    def __init__(self):
        self.state = {
            'version': self.VERSION,
            'timestamp': None,
            'domain': None,      # For swarm specialization
            'agent_id': None,    # Unique agent identifier
            'knowledge': None,   # KnowledgeStore state
            'language': None,    # PPM state
            'reasoning': None,   # ReasoningEngine state
            'router': None,      # Router config
            'metacognition': None,  # Gap log, corpus
            'config': None,      # System config
        }

    @classmethod
    def capture(cls, engine, domain=None, agent_id=None) -> 'BrainSnapshot':
        """
        Capture the complete state of a FossKI engine.

        Args:
            engine: FossKI instance (from engine.py)
            domain: optional domain label (e.g., 'medicine', 'law')
            agent_id: optional unique identifier

        Returns:
            BrainSnapshot with all state captured
        """
        snap = cls()
        snap.state['timestamp'] = time.time()
        snap.state['domain'] = domain
        snap.state['agent_id'] = agent_id or _generate_id()

        # 1. Knowledge Store → facts + config
        snap.state['knowledge'] = _capture_knowledge(engine.knowledge)

        # 2. PPM Language Model → context tree
        if hasattr(engine, 'language_model') and engine.language_model is not None:
            snap.state['language'] = _capture_language(engine.language_model)

        # 3. Reasoning Engine → rules + common knowledge config
        if hasattr(engine, '_reasoning') and engine._reasoning is not None:
            snap.state['reasoning'] = _capture_reasoning(engine._reasoning)

        # 4. Router config
        snap.state['router'] = {
            'confidence_threshold': 0.5,
        }

        # 5. Metacognition state
        if hasattr(engine, '_metacognition'):
            snap.state['metacognition'] = _capture_metacognition(engine._metacognition)

        # 6. System config
        snap.state['config'] = dict(engine.config)

        return snap

    @classmethod
    def capture_from_parts(cls, knowledge_store=None, lm_model=None,
                           reasoning_engine=None, config=None,
                           domain=None, agent_id=None) -> 'BrainSnapshot':
        """
        Capture from individual components (without full engine).
        Useful for building specialized agent brains.
        """
        snap = cls()
        snap.state['timestamp'] = time.time()
        snap.state['domain'] = domain
        snap.state['agent_id'] = agent_id or _generate_id()

        if knowledge_store:
            snap.state['knowledge'] = _capture_knowledge(knowledge_store)
        if lm_model:
            snap.state['language'] = _capture_language(lm_model)
        if reasoning_engine:
            snap.state['reasoning'] = _capture_reasoning(reasoning_engine)
        if config:
            snap.state['config'] = dict(config)

        return snap

    def restore(self, engine=None):
        """
        Restore Brain state into a FossKI engine.

        If engine is None, creates a new one.
        Returns the engine with all state restored.
        """
        if engine is None:
            from .engine import FossKI
            config = self.state.get('config') or {}
            engine = FossKI(config=config)

        # 1. Knowledge
        if self.state['knowledge']:
            _restore_knowledge(engine.knowledge, self.state['knowledge'])

        # 2. Language Model
        if self.state['language']:
            _restore_language(engine, self.state['language'])

        # 3. Reasoning (restored lazily when first accessed)
        if self.state['reasoning']:
            engine._brain_reasoning_state = self.state['reasoning']

        return engine

    def restore_parts(self):
        """
        Restore individual components without a full engine.
        Returns dict of component instances.
        """
        parts = {}

        if self.state['knowledge']:
            from .knowledge import KnowledgeStore
            from .normalizer import EntityNormalizer
            ks_state = self.state['knowledge']
            ks = KnowledgeStore(
                dim=ks_state['dim'],
                encoder=ks_state.get('encoder', 'ngram'),
                normalizer=EntityNormalizer(),
            )
            ks.store_facts_fast([tuple(f) for f in ks_state['facts']])
            parts['knowledge'] = ks

        if self.state['language']:
            from .language import PPMModel
            lang_state = self.state['language']
            ppm = PPMModel(max_order=lang_state['max_order'],
                           vocab=lang_state.get('vocab'))
            ppm._total_symbols = lang_state.get('total_symbols', 0)
            ppm.root = ppm._dict_to_node(lang_state['tree'])
            parts['language'] = ppm

        if self.state['reasoning']:
            from .reasoning import ReasoningEngine
            ks = parts.get('knowledge')
            engine = ReasoningEngine(knowledge_store=ks)
            if self.state['reasoning'].get('rules'):
                engine.rules = self.state['reasoning']['rules']
            parts['reasoning'] = engine

        return parts

    # ── Serialization ─────────────────────────────────────

    def save(self, path: str, compressed: bool = True):
        """
        Save Brain to file. Atomic write: temp file + rename.
        Never leaves a corrupted half-written brain on disk.

        Args:
            path: file path (.brain or .brain.json)
            compressed: if True, use zlib compression (default)
        """
        import tempfile
        data = json.dumps(self.state, ensure_ascii=False, sort_keys=True)
        dir_path = os.path.dirname(path) or '.'
        os.makedirs(dir_path, exist_ok=True)

        if compressed and not path.endswith('.json'):
            payload = zlib.compress(data.encode('utf-8'), level=6)
            fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix='.brain.tmp')
            try:
                with os.fdopen(fd, 'wb') as f:
                    f.write(self.MAGIC)
                    f.write(self.VERSION.to_bytes(2, 'big'))
                    f.write(len(payload).to_bytes(8, 'big'))
                    f.write(payload)
                os.replace(tmp_path, path)  # Atomic on POSIX
            except Exception:
                os.unlink(tmp_path)
                raise
        else:
            fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix='.brain.json.tmp')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(self.state, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, path)
            except Exception:
                os.unlink(tmp_path)
                raise

    @classmethod
    def load(cls, path: str) -> 'BrainSnapshot':
        """Load Brain from file (auto-detects format)."""
        snap = cls()

        with open(path, 'rb') as f:
            header = f.read(len(cls.MAGIC))

            if header == cls.MAGIC:
                # Binary compressed format
                version = int.from_bytes(f.read(2), 'big')
                if version > cls.VERSION:
                    raise ValueError(f"Brain version {version} > supported {cls.VERSION}")
                size = int.from_bytes(f.read(8), 'big')
                payload = f.read(size)
                data = zlib.decompress(payload).decode('utf-8')
                snap.state = json.loads(data)
            else:
                # JSON format
                f.seek(0)
                snap.state = json.loads(f.read().decode('utf-8'))

        return snap

    # ── Brain Info ────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of Brain contents."""
        s = self.state
        lines = [
            f"FOSS-KI Brain v{s.get('version', '?')}",
            f"  Agent:    {s.get('agent_id', '?')}",
            f"  Domain:   {s.get('domain', 'general')}",
            f"  Created:  {_fmt_time(s.get('timestamp'))}",
        ]

        if s.get('knowledge'):
            n_facts = len(s['knowledge'].get('facts', []))
            dim = s['knowledge'].get('dim', '?')
            lines.append(f"  Facts:    {n_facts} (dim={dim})")

        if s.get('language'):
            vocab_size = len(s['language'].get('vocab', []))
            order = s['language'].get('max_order', '?')
            total = s['language'].get('total_symbols', 0)
            lines.append(f"  PPM:      order={order}, vocab={vocab_size}, "
                         f"trained on {total} symbols")

        if s.get('reasoning'):
            n_rules = len(s['reasoning'].get('rules', []))
            lines.append(f"  Rules:    {n_rules}")

        if s.get('metacognition'):
            n_gaps = len(s['metacognition'].get('gaps', []))
            n_corpus = s['metacognition'].get('corpus_count', 0)
            lines.append(f"  Meta:     {n_gaps} gaps, {n_corpus} corpus entries")

        return "\n".join(lines)

    def fingerprint(self) -> str:
        """SHA-256 hash of the Brain state (for integrity/dedup)."""
        data = json.dumps(self.state, sort_keys=True).encode('utf-8')
        return hashlib.sha256(data).hexdigest()[:16]

    @property
    def n_facts(self) -> int:
        if self.state.get('knowledge'):
            return len(self.state['knowledge'].get('facts', []))
        return 0

    @property
    def domain(self) -> str:
        return self.state.get('domain', 'general')

    @property
    def agent_id(self) -> str:
        return self.state.get('agent_id', 'unknown')


# ── Capture helpers ───────────────────────────────────────

def _capture_knowledge(ks) -> Dict[str, Any]:
    """Capture KnowledgeStore state."""
    return {
        'dim': ks.dim,
        'encoder': getattr(ks, '_encoder_type', 'ngram'),
        'facts': list(ks.facts),
        'synonyms': dict(getattr(ks, '_synonyms', {})),
    }


def _capture_language(lm) -> Dict[str, Any]:
    """Capture PPM, VortexLanguageModel, or FossLanguageModel state."""
    # FossLanguageModel (new — replaces PPM)
    from .foss_lm import FossLanguageModel
    if isinstance(lm, FossLanguageModel):
        return {
            'type': 'foss_lm',
            'vocab': sorted(lm.vocab),
            'char_orders': lm.CHAR_ORDERS,
            'momentum_decay': lm._momentum_decay,
        }

    # Check if it's a PPMModel directly
    from .language import PPMModel
    if isinstance(lm, PPMModel):
        return {
            'type': 'ppm',
            'max_order': lm.max_order,
            'vocab': sorted(lm.vocab),
            'total_symbols': lm._total_symbols,
            'tree': lm._node_to_dict(lm.root),
        }

    # VortexLanguageModel has fibers with PPMs
    from .vortex import VortexLanguageModel
    if isinstance(lm, VortexLanguageModel):
        fibers = []
        for fiber in lm.fibers:
            fibers.append({
                'max_order': fiber.max_order,
                'vocab': sorted(fiber.vocab),
                'total_symbols': fiber._total_symbols,
                'tree': fiber._node_to_dict(fiber.root),
            })
        return {
            'type': 'vortex',
            'order': lm.order,
            'bridge_strength': lm.bridge_strength,
            'fibers': fibers,
        }

    return None


def _capture_reasoning(re) -> Dict[str, Any]:
    """Capture ReasoningEngine state."""
    return {
        'rules': list(getattr(re, 'rules', [])),
        'has_knowledge': re.knowledge is not None,
    }


def _capture_metacognition(meta) -> Dict[str, Any]:
    """Capture MetacognitionEngine state."""
    state = {
        'gaps': list(meta.filler.gap_log.gaps),
        'corpus_count': sum(len(v) for v in meta.filler._corpus.values()),
    }
    # Capture corpus texts for transfer
    corpus = {}
    for key, texts in meta.filler._corpus.items():
        corpus[key] = texts[:50]  # Cap at 50 per entity to limit size
    state['corpus'] = corpus
    return state


# ── Restore helpers ───────────────────────────────────────

def _restore_knowledge(ks, state: Dict):
    """Restore KnowledgeStore from captured state."""
    # Clear existing facts
    ks.facts = []
    ks._sr_vectors = []
    ks._o_vectors = []
    ks._sr_sim_stats = None

    # Restore facts with lazy Hopfield (vectors built on demand at Tier 3)
    facts = [tuple(f) for f in state.get('facts', [])]
    if facts:
        ks.store_facts_fast(facts)

    # Restore synonyms
    synonyms = state.get('synonyms', {})
    if synonyms and hasattr(ks, '_synonyms'):
        ks._synonyms = dict(synonyms)


def _restore_language(engine, state: Dict):
    """Restore language model from captured state."""
    if state.get('type') == 'foss_lm':
        from .foss_lm import FossLanguageModel
        flm = FossLanguageModel(
            momentum_decay=state.get('momentum_decay', 0.95)
        )
        flm.vocab = set(state.get('vocab', []))
        engine.language_model = flm
        return

    if state.get('type') == 'ppm':
        from .language import PPMModel
        ppm = PPMModel(max_order=state['max_order'],
                       vocab=state.get('vocab'))
        ppm._total_symbols = state.get('total_symbols', 0)
        ppm.root = ppm._dict_to_node(state['tree'])
        # Store as accessible attribute
        engine._brain_ppm = ppm

    elif state.get('type') == 'vortex':
        from .vortex import VortexLanguageModel
        vlm = VortexLanguageModel(
            order=state.get('order', 6),
            bridge_strength=state.get('bridge_strength', 0.0),
        )
        for i, fiber_state in enumerate(state.get('fibers', [])):
            if i < len(vlm.fibers):
                fiber = vlm.fibers[i]
                fiber.vocab = set(fiber_state.get('vocab', []))
                fiber._total_symbols = fiber_state.get('total_symbols', 0)
                fiber.root = fiber._dict_to_node(fiber_state['tree'])
        engine.language_model = vlm


# ── Utility ───────────────────────────────────────────────

def _generate_id() -> str:
    """Generate a short unique agent ID."""
    import random
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    return 'agent-' + ''.join(random.choices(chars, k=8))


def _fmt_time(ts) -> str:
    """Format timestamp for display."""
    if ts is None:
        return 'unknown'
    import datetime
    dt = datetime.datetime.fromtimestamp(ts)
    return dt.strftime('%Y-%m-%d %H:%M:%S')


# ── Brain Merge (for Federated Sync, S17) ────────────────

def merge_brains(brains: list, strategy='union') -> BrainSnapshot:
    """
    Merge multiple Brain snapshots into one.

    Strategies:
        'union': combine all facts from all brains (no dedup by content)
        'intersection': keep only facts all brains agree on
        'weighted': weight by brain's fact count (larger brain = more authority)

    This is the foundation for Federated Knowledge Sync (S17).
    """
    if not brains:
        return BrainSnapshot()

    merged = BrainSnapshot()
    merged.state['timestamp'] = time.time()
    merged.state['agent_id'] = _generate_id()
    merged.state['domain'] = 'merged'

    if strategy == 'union':
        # Collect all facts, deduplicate
        all_facts = set()
        all_synonyms = {}
        for brain in brains:
            ks = brain.state.get('knowledge', {})
            for fact in ks.get('facts', []):
                all_facts.add(tuple(fact))
            for k, v in ks.get('synonyms', {}).items():
                all_synonyms[k] = v

        # Use max dim from all brains
        dims = [b.state.get('knowledge', {}).get('dim', 128) for b in brains]
        max_dim = max(dims) if dims else 128

        merged.state['knowledge'] = {
            'dim': max_dim,
            'encoder': brains[0].state.get('knowledge', {}).get('encoder', 'ngram'),
            'facts': [list(f) for f in sorted(all_facts)],
            'synonyms': all_synonyms,
        }

    elif strategy == 'intersection':
        # Only keep facts that appear in ALL brains
        fact_sets = []
        for brain in brains:
            ks = brain.state.get('knowledge', {})
            facts = {tuple(f) for f in ks.get('facts', [])}
            fact_sets.append(facts)

        common = fact_sets[0]
        for fs in fact_sets[1:]:
            common &= fs

        merged.state['knowledge'] = {
            'dim': brains[0].state.get('knowledge', {}).get('dim', 128),
            'encoder': brains[0].state.get('knowledge', {}).get('encoder', 'ngram'),
            'facts': [list(f) for f in sorted(common)],
            'synonyms': {},
        }

    # Merge language models: use the one with most training data
    lang_states = [b.state.get('language') for b in brains if b.state.get('language')]
    if lang_states:
        best = max(lang_states, key=lambda l: l.get('total_symbols', 0))
        merged.state['language'] = best

    # Merge reasoning: union of all rules
    all_rules = []
    for brain in brains:
        rs = brain.state.get('reasoning') or {}
        all_rules.extend(rs.get('rules', []))
    merged.state['reasoning'] = {'rules': list(set(all_rules)), 'has_knowledge': True}

    # Config: use first brain's config
    merged.state['config'] = brains[0].state.get('config')

    return merged
