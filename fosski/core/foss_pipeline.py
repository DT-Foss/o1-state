"""
Foss Pipeline — Integrated Intelligence from All Components
============================================================
NOT a sequential fallback chain. All components run in parallel,
then Foss Consensus combines their answers.

Architecture (matching the 6-layer design from CLAUDE.md):

  Query → [Attention] → weighted tokens
                ↓
        ┌───────┼───────┐
        ↓       ↓       ↓
   Reservoir  Hopfield  KBStore    ← parallel retrieval
        ↓       ↓       ↓
        └───────┼───────┘
                ↓
         [Consensus]   ← Foss-Gossip on barbell graph
                ↓
         [Z₂ Novelty]  ← confidence gating
                ↓
           Answer

For "why" questions: Causal engine + MultiHop reasoning.
For novel queries (high Z₂ disagreement): lower confidence.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from .tokenutils import clean_token, find_token_id, find_token_embedding


class FossPipeline:
    """Orchestrate all FOSS-KI components into one coherent system.

    Components (injected):
      - reservoir: ReservoirLM (ESN with Foss topology)
      - emb_store: EmbeddingStore (Qwen3 512d projected embeddings)
      - attention: ExtractedAttention (Transformer Q·K^T importance)
      - ricci: RicciAttention (geometric O(n) importance)
      - hopfield: ModernHopfieldLayer (associative memory)
      - consensus: FossConsensus (PS-Lifted gossip on barbell)
      - knowledge: KnowledgeStore (fact store with attractor confidence)
      - raw_emb: raw 2048d embeddings (for attention computation)
      - lm_head: (151936, 2048) matrix for vocabulary decoding
      - id2token: token ID → string mapping
    """

    STOP_WORDS = frozenset(
        'the a an is are was were be been being have has had '
        'do does did will would shall should can could may might must '
        'of in on at to for with by from as into through during '
        'before after above below between under about against '
        'without within along across behind beyond near among '
        'towards upon what which who whom how where when why'.split()
    )

    def __init__(self):
        # Components — set via configure()
        self.reservoir = None
        self.emb_store = None
        self.attention = None
        self.ricci = None
        self.hopfield = None
        self.consensus = None
        self.knowledge = None
        self.raw_emb = None
        self.lm_head = None       # (151936, 2048) vocab decoder
        self.lm_head_normed = None  # pre-normalized for cosine sim
        self.id2token = None       # token ID → string
        self.multi_hop = None      # MultiHopReasoner
        self.commonsense = None    # CommonSenseEngine
        self.causal_dag = None     # CausalDAG for interventional reasoning
        self.residual_hopfield = None  # Pre-contextualized residuals
        self.situation_memory = None   # Anti-hallucination memory (D1)
        self.experience_pool = None    # Online learning pool (D3)
        self.online_ridge = None       # Online readout learning (D5)
        self.multi_timescale = None    # Fast/Medium/Slow reservoir (D4)
        self._ds_mixed_embs = None     # DS layer mixed embeddings (temp per query)

    def configure(self, **kwargs):
        """Set components. Call after constructing all modules."""
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)

        # Build reverse lookup for token IDs (needed for robust embedding lookup)
        if self.id2token is not None:
            self._token2id = {v: k for k, v in self.id2token.items()}

        # Pre-normalize lm_head for cosine similarity
        if self.lm_head is not None:
            norms = np.linalg.norm(self.lm_head, axis=1, keepdims=True)
            self.lm_head_normed = self.lm_head / np.clip(norms, 1e-8, None)

        # Load Extracted Attention (Qwen3 Q·K^T importance weighting)
        if self.attention is None:
            try:
                from .extracted_attention import ExtractedAttention
                attn = ExtractedAttention()
                if attn.available:
                    self.attention = attn
            except Exception:
                pass

        # Load Residual Hopfield (pre-contextualized embeddings)
        try:
            from .residual_hopfield import ResidualHopfield
            self.residual_hopfield = ResidualHopfield()
            if not self.residual_hopfield.available:
                self.residual_hopfield = None
        except Exception:
            self.residual_hopfield = None

        # Build Hopfield Template Bank from KB facts (Modern Hopfield, 512d)
        if self.hopfield is None and self.knowledge and self.emb_store:
            try:
                from .hopfield_bank import build_hopfield_bank
                self.hopfield = build_hopfield_bank(
                    self.emb_store, self.knowledge, self.commonsense,
                    max_patterns=2000)
            except Exception:
                self.hopfield = None

        # Load MLP Fact Vectors (198 stolen transformer knowledge vectors)
        self._mlp_facts = None
        self._mlp_words = []
        self._mlp_matrix_normed = None
        try:
            import os
            mlp_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data', 'qwen3_mlp_facts.npz')
            if os.path.exists(mlp_path):
                mlp_data = np.load(mlp_path)
                self._mlp_words = list(mlp_data.files)
                if self._mlp_words:
                    vecs = np.array([mlp_data[w].astype(np.float32)
                                     for w in self._mlp_words])
                    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                    self._mlp_matrix_normed = vecs / np.clip(norms, 1e-8, None)
                    self._mlp_facts = mlp_data
        except Exception:
            self._mlp_facts = None

        # Build FAISS index for fast NN search (replaces brute-force cosine)
        self._nn_index = None
        if self.raw_emb is not None:
            try:
                from .nn_index import NNIndex
                self._nn_index = NNIndex(self.raw_emb, self.id2token, mode='auto', n_trees=10)
            except Exception:
                self._nn_index = None

        # Situation Memory — anti-hallucination (D1)
        try:
            from .situation_memory import SituationMemory
            self.situation_memory = SituationMemory(
                dim=self.emb_store.dim if self.emb_store else 512,
                max_size=500, novelty_threshold=0.3)
        except Exception:
            self.situation_memory = None

        # Experience Pool — online learning (D3)
        try:
            from .experience_pool import ExperiencePool
            self.experience_pool = ExperiencePool(
                dim=self.emb_store.dim if self.emb_store else 512)
        except Exception:
            self.experience_pool = None

        # Multi-Timescale Reservoir — Fast/Medium/Slow novelty detector (D4)
        try:
            from .multi_timescale import MultiTimescaleReservoir
            dim = self.emb_store.dim if self.emb_store else 512
            self.multi_timescale = MultiTimescaleReservoir(input_dim=dim, seed=42)
        except Exception:
            self.multi_timescale = None

        # Online Ridge — learns during inference (D5)
        try:
            from .online_ridge import OnlineRidge
            expanded_dim = (self.emb_store.dim * 2) if self.emb_store else 1024
            output_dim = self.raw_emb.shape[1] if self.raw_emb is not None else 2048
            self.online_ridge = OnlineRidge(
                input_dim=expanded_dim, output_dim=output_dim)
        except Exception:
            self.online_ridge = None

        # Build Causal DAG from KB + ConceptNet
        self._build_causal_dag()

        # Build Foss Consensus on barbell topology
        self._build_consensus()

        # Diagnostics health check (F9/F10)
        self._health = {}
        if self.reservoir and self.reservoir.W_out is not None:
            try:
                from .diagnostics import reservoir_health
                self._health = reservoir_health(self.reservoir)
            except Exception:
                pass

    def _build_consensus(self):
        """Build FossConsensus on a barbell graph.

        3 nodes per cluster, 1 bridge:
          [Reservoir]─[Reservoir]─┐
                                  ├─ bridge
          [KB]────────[KB]────────┘

        PS-Lifted gives 26x speedup on this topology (T391).
        """
        try:
            from .consensus import FossConsensus
            # Barbell(6,1): two 3-cliques connected by 1 bridge
            n = 6
            adj = np.zeros((n, n))
            # Cluster 1 (reservoir side)
            for i in range(3):
                for j in range(3):
                    if i != j:
                        adj[i, j] = 1.0
            # Cluster 2 (KB side)
            for i in range(3, 6):
                for j in range(3, 6):
                    if i != j:
                        adj[i, j] = 1.0
            # Bridge
            adj[2, 3] = adj[3, 2] = 1.0

            self.consensus = FossConsensus(adj)
        except Exception:
            self.consensus = None

    def query(self, question: str) -> Dict[str, Any]:
        """Run the full pipeline on a question.

        Returns:
            {
                'answer': str or None,
                'confidence': float,
                'sources': list of (source_name, answer, confidence),
                'novelty': float (Z₂ disagreement),
                'attention_weights': dict of token→weight,
                'trace': list of strings,
            }
        """
        trace = []
        sources = []

        # Step 0: Tokenize and filter
        clean = question.lower()
        for ch in '?.,!;:"\'-()[]{}':
            clean = clean.replace(ch, ' ')
        all_words = clean.split()
        content_words = [w for w in all_words if w not in self.STOP_WORDS and len(w) > 1]

        if not content_words:
            return self._empty_result("No content words found")

        trace.append(f"Content words: {content_words}")

        # Step 1: Attention — compute token importance (3 sources)
        attn_weights = {}
        raw_embeddings = None

        # 1a. Extracted Transformer Attention (Q·K^T)
        if self.attention and self.attention.available and self.raw_emb is not None:
            raw_embeddings = self._get_raw_embeddings(all_words)
            weights = self.attention.compute_importance(raw_embeddings, temperature=1.0)
            for w, wt in zip(all_words, weights):
                attn_weights[w] = float(wt)
            trace.append(f"Extracted Attention: top={self._top_tokens(all_words, weights)}")

        # 1b. Ricci geometric attention (fallback)
        if self.ricci and raw_embeddings is None:
            proj_embs = self._get_proj_embeddings(all_words)
            if proj_embs:
                ricci_w, curv = self.ricci.attend(proj_embs)
                if not attn_weights:
                    for w, wt in zip(all_words, ricci_w):
                        attn_weights[w] = float(wt)
                trace.append(f"Ricci Attention: top={self._top_tokens(all_words, ricci_w)}")

        # 1c. Z₂ Disagreement Attention — tokens that surprise the reservoir
        # Feed each content word and measure pos/neg divergence.
        # High Z₂ = novel/important token → boost its attention weight.
        if self.reservoir and self.emb_store:
            z2_weights = self._z2_attention(content_words)
            if z2_weights:
                # Blend Z₂ with existing attention (additive boost)
                for w, z2w in z2_weights.items():
                    if w in attn_weights:
                        attn_weights[w] = attn_weights[w] * 0.7 + z2w * 0.3
                    else:
                        attn_weights[w] = z2w
                z2_top = sorted(z2_weights.items(), key=lambda x: -x[1])[:3]
                trace.append(f"Z₂ Attention: {', '.join(f'{w}({v:.2f})' for w,v in z2_top)}")

        # Step 1d: DS Layer Stack — cross-token information integration (F7/FB1)
        if self.emb_store and len(content_words) >= 3:
            try:
                from .ds_layers import DSLayerStack
                ds = DSLayerStack(n_layers=3, sinkhorn_iters=1)
                word_embs = []
                for w in content_words:
                    v = self.emb_store.encode(w)
                    if v is not None:
                        word_embs.append(v)
                if len(word_embs) >= 3:
                    emb_matrix = np.array(word_embs, dtype=np.float32)
                    mixed = ds.forward(emb_matrix)
                    # Replace content word embeddings with DS-mixed versions
                    # Store for later retrieval methods
                    self._ds_mixed_embs = mixed
                    trace.append(f"DS Layers: {len(word_embs)} tokens × 3 layers")
            except Exception:
                pass

        # Step 1e: Experience Pool priming — retrieve near-duplicate past experiences (D3)
        # Threshold 0.85 = near-duplicate ONLY. Lower causes cross-contamination:
        # e.g. "born einstein" gets "1564" from "Shakespeare born" at sim=0.7.
        if self.experience_pool and self.experience_pool.size > 0 and self.emb_store:
            query_emb = self._query_embedding(content_words)
            if query_emb is not None:
                past = self.experience_pool.retrieve(query_emb, top_k=1)
                for past_answer, past_sim in past:
                    if past_sim > 0.85:
                        sources.append(('experience', past_answer, past_sim * 0.5))
                        trace.append(f"Experience: '{past_answer}' (sim={past_sim:.2f})")

        # Step 2: Parallel retrieval
        novelty = 0.0

        # 2a. Reservoir ESN
        if self.reservoir and self.emb_store and self.reservoir.W_out is not None:
            res_answer, res_conf, res_novelty = self._reservoir_retrieve(
                content_words, attn_weights, raw_embeddings, all_words)
            novelty = res_novelty
            if res_answer:
                sources.append(('reservoir', res_answer, res_conf))
                trace.append(f"Reservoir: '{res_answer}' (conf={res_conf:.2f}, novelty={res_novelty:.3f})")

        # 2b. Hopfield Memory
        if self.hopfield:
            hop_answer, hop_conf = self._hopfield_retrieve(content_words)
            if hop_answer:
                sources.append(('hopfield', hop_answer, hop_conf))
                trace.append(f"Hopfield: '{hop_answer}' (conf={hop_conf:.2f})")

        # 2c. Knowledge Store
        if self.knowledge:
            kb_answer, kb_conf = self._kb_retrieve(question)
            if kb_answer:
                sources.append(('knowledge', kb_answer, kb_conf))
                trace.append(f"KB: '{kb_answer}' (conf={kb_conf:.2f})")

        # 2d. Causal DAG interventional reasoning ("what if X disappeared?")
        if self.causal_dag:
            dag_answer = self.causal_dag.answer_intervention(question)
            if dag_answer:
                sources.append(('causal_dag', dag_answer, 0.7))
                trace.append(f"CausalDAG: '{dag_answer[:60]}...'")

        # 2e. Residual Hopfield — pre-contextualized transformer states
        if self.residual_hopfield and self.emb_store:
            res_hop_answer, res_hop_conf = self._residual_hopfield_retrieve(content_words)
            if res_hop_answer:
                sources.append(('residual_hopfield', res_hop_answer, res_hop_conf))
                trace.append(f"ResidualHop: '{res_hop_answer}' (conf={res_hop_conf:.2f})")

        # 2f. MLP Fact Memory — stolen transformer knowledge vectors
        if self._mlp_matrix_normed is not None and self.emb_store:
            mlp_answer, mlp_conf = self._mlp_retrieve(content_words)
            if mlp_answer:
                sources.append(('mlp_facts', mlp_answer, mlp_conf))
                trace.append(f"MLP: '{mlp_answer}' (conf={mlp_conf:.2f})")

        # 2g. Multi-Hop Reasoning (WHY/HOW/WHAT-IF questions)
        import re as _re
        is_causal = bool(_re.match(
            r'(?:why|how|what\s+(?:would|will)\s+happen)',
            question.lower().strip()))

        if is_causal and self.multi_hop:
            hop_result = self.multi_hop.reason(question)
            if hop_result.get('answered') and hop_result.get('chain'):
                chain = hop_result['chain']
                # Build paragraph from chain
                sentences = []
                for s, r, o in chain:
                    s_clean = s.replace('_', ' ')
                    o_clean = o.replace('_', ' ')
                    if r == 'causes':
                        sentences.append(f"{s_clean} causes {o_clean}")
                    elif r == 'needs':
                        sentences.append(f"{s_clean} needs {o_clean}")
                    elif r == 'is_a':
                        sentences.append(f"{s_clean} is a type of {o_clean}")
                    else:
                        sentences.append(f"{s_clean} {r.replace('_', ' ')} {o_clean}")

                chain_answer = '. '.join(s.capitalize() for s in sentences) + '.'
                # Use pre-built answer if MultiHop provided one
                if hop_result.get('answer'):
                    chain_answer = hop_result['answer']

                sources.append(('multi_hop', chain_answer,
                                min(1.0, 0.5 + 0.15 * len(chain))))
                trace.append(f"MultiHop({hop_result.get('method','?')}, "
                             f"{len(chain)} hops): '{chain_answer[:60]}...'")

        # Step 3: Consensus — combine answers
        if not sources:
            return self._empty_result("No sources returned answers", trace=trace,
                                      novelty=novelty, attn=attn_weights)

        answer, confidence = self._consensus_combine(sources)
        trace.append(f"Consensus: '{answer}' (conf={confidence:.2f})")

        # Step 3b: CASI Confidence Penalty — detect systematic overconfidence (F5)
        if len(sources) >= 4:
            from .casi_gate import CASIGenerationGate
            casi_gate = CASIGenerationGate()
            source_confs = np.array([s[2] for s in sources], dtype=np.float64)
            penalty = casi_gate.confidence_penalty(source_confs)
            if penalty > 0.05:
                confidence *= (1.0 - penalty)
                trace.append(f"CASI penalty: {penalty:.2f} → conf={confidence:.2f}")

        # Step 4: Z₂ Novelty gating
        if novelty > 0.5:
            confidence *= 0.7  # Novel input = less certain
            trace.append(f"Z₂ Novelty high ({novelty:.3f}) → confidence reduced")

        # Step 4b: Multi-Timescale novelty — fast/slow mismatch (D4)
        if self.multi_timescale and self.emb_store:
            try:
                self.multi_timescale.reset()
                zero = np.zeros(self.emb_store.dim, dtype=np.float32)
                for w in content_words:
                    v = self.emb_store.encode(w)
                    self.multi_timescale.step(v if v is not None else zero)
                # Extra mixing steps
                for _ in range(3):
                    self.multi_timescale.step(zero)
                mt_mismatch = self.multi_timescale.z2_mismatch()
                if mt_mismatch > 0.3:
                    confidence *= max(0.5, 1.0 - mt_mismatch)
                    trace.append(f"MultiTimescale mismatch={mt_mismatch:.2f} → conf={confidence:.2f}")
            except Exception:
                pass

        # Step 5: Situation Memory — anti-hallucination (D1)
        if self.situation_memory and self.emb_store:
            query_emb = self._query_embedding(content_words)
            if query_emb is not None:
                # Only check novelty if memory has enough entries to be meaningful
                if self.situation_memory.size >= 10:
                    is_novel, sim = self.situation_memory.is_novel(query_emb)
                    if is_novel and confidence > 0.5:
                        confidence *= 0.5  # Novel = less confident
                        trace.append(f"SituationMemory: novel (sim={sim:.2f}) → conf halved")
                # Always store for future queries
                self.situation_memory.store(query_emb, answer, confidence)

        # Step 5b: Cerebellum Reasoning Loop — iterative refinement (D2)
        if confidence < 0.4 and len(content_words) >= 2:
            try:
                from .cerebellum import CerebellumLoop
                cerebellum = CerebellumLoop(max_iterations=3, min_improvement=0.1)

                def _requery(augmented_words):
                    """Re-run retrieval with augmented input."""
                    sub_sources = []
                    if self.hopfield:
                        a, c = self._hopfield_retrieve(augmented_words)
                        if a:
                            sub_sources.append(('hopfield', a, c))
                    if self.knowledge:
                        a, c = self._kb_retrieve(' '.join(augmented_words))
                        if a:
                            sub_sources.append(('knowledge', a, c))
                    if not sub_sources:
                        return None, 0.0
                    return self._consensus_combine(sub_sources)

                ref_answer, ref_conf, n_iters = cerebellum.refine(
                    answer, confidence, _requery, content_words, self.emb_store)
                if ref_conf > confidence:
                    answer = ref_answer
                    confidence = ref_conf
                    trace.append(f"Cerebellum: {n_iters} iters → '{answer}' (conf={confidence:.2f})")
            except Exception:
                pass

        # Step 6: Experience Pool — prime future queries (D3)
        if self.experience_pool and self.emb_store and confidence > 0.5:
            query_emb = self._query_embedding(content_words)
            if query_emb is not None:
                self.experience_pool.store(query_emb, answer, confidence)

        # Step 7: Online Ridge — learn from successful queries (D5)
        if self.online_ridge and self.reservoir and confidence > 0.6:
            try:
                state = self.reservoir.state_pos  # Current reservoir state
                if state is not None:
                    expanded = np.concatenate([state, state ** 2])  # Match _expand_features
                    # Target: raw embedding of answer word
                    target = find_token_embedding(
                        answer.split()[0], self._token2id, self.raw_emb)
                    if target is not None:
                        self.online_ridge.update(expanded, target)
            except Exception:
                pass

        return {
            'answer': answer,
            'confidence': confidence,
            'sources': sources,
            'novelty': novelty,
            'attention_weights': attn_weights,
            'trace': trace,
        }

    def health(self) -> Dict[str, Any]:
        """Return health diagnostics for all components."""
        result = dict(self._health) if self._health else {}
        result['hopfield_patterns'] = self.hopfield.n_patterns if self.hopfield else 0
        result['kb_facts'] = len(self.knowledge.facts) if self.knowledge else 0
        result['situation_memory_size'] = self.situation_memory.size if self.situation_memory else 0
        result['experience_pool_size'] = self.experience_pool.size if self.experience_pool else 0
        result['online_ridge_updates'] = self.online_ridge.n_updates if self.online_ridge else 0
        result['nn_index'] = self._nn_index.backend if self._nn_index else 'none'
        return result

    # ================================================================
    # Retrieval methods
    # ================================================================

    def _query_embedding(self, content_words):
        """Build average 512d embedding for content words."""
        vecs = []
        for w in content_words:
            v = self.emb_store.encode(w)
            if v is not None:
                vecs.append(v)
        if not vecs:
            return None
        return np.mean(vecs, axis=0).astype(np.float32)

    def _reservoir_retrieve(self, content_words, attn_weights, raw_embs, all_words):
        """Retrieve via Reservoir ESN with attention-weighted input."""
        self.reservoir.reset_state()
        zero = np.zeros(self.emb_store.dim, dtype=np.float32)

        for w in content_words:
            v = self.emb_store.encode(w)
            if v is None:
                v = zero
            # Scale by attention if available
            if attn_weights and w in attn_weights:
                scale = attn_weights[w] * len(all_words)  # normalize
                v = v * max(scale, 0.1)  # don't zero out
            self.reservoir.step(v)

        # Mixing steps
        for _ in range(5):
            self.reservoir.step(zero)

        state = (self.reservoir.state_pos + self.reservoir.state_neg) / 2.0
        novelty = float(np.mean(np.abs(self.reservoir.state_pos - self.reservoir.state_neg)))

        pred = self.reservoir.predict(state)
        if pred is None:
            return None, 0.0, novelty

        # Find nearest token
        neighbors = self._nearest_raw(pred, top_k=5)
        for token, sim in neighbors:
            clean = token.replace('Ġ', '').replace('ĉ', ' ').strip()
            if clean and len(clean) > 1 and clean[0].isalpha():
                confidence = min(1.0, max(0.0, sim))
                return clean, confidence, novelty

        return None, 0.0, novelty

    def _hopfield_retrieve(self, content_words):
        """Retrieve via Hopfield associative memory.

        Supports both HopfieldTemplateBank (Qwen3 512d, returns list of
        (value_dict, score)) and GloVe HopfieldSequenceMemory (returns
        dict of word → score).
        """
        if not hasattr(self.hopfield, 'retrieve'):
            return None, 0.0

        results = self.hopfield.retrieve(content_words, top_k=3)
        if not results:
            return None, 0.0

        # HopfieldTemplateBank: list of (value_dict, score)
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, tuple) and len(first) == 2:
                val, score = first
                if isinstance(val, dict):
                    answer = val.get('answer', '')
                    if answer:
                        return str(answer), min(1.0, float(score) * 10)
                # GloVe-based: (pattern, metadata, score)
                if len(first) == 3:
                    _, meta, score = first
                    answer = meta.get('answer', meta.get('object', ''))
                    return answer if answer else None, float(score)

        # GloVe HopfieldSequenceMemory: dict of word → score
        if isinstance(results, dict):
            if not results:
                return None, 0.0
            best_word = max(results, key=results.get)
            return best_word, min(1.0, results[best_word])

        return None, 0.0

    def _kb_retrieve(self, question):
        """Retrieve via KnowledgeStore fact match.

        Parses question into (subject, relation) and queries KB.
        Uses question patterns to extract the right structure.
        """
        if not hasattr(self.knowledge, 'query'):
            return None, 0.0

        q = question.lower().strip().rstrip('?').strip()

        # Question patterns → (subject, relation)
        import re
        patterns = [
            # "What is the X of Y" → subject=Y, relation=X
            (r'what (?:is|are) the (.+?) of (.+)', lambda m: (m.group(2), m.group(1))),
            # "What is X made of" → subject=X, relation=made
            (r'what (?:is|are) (.+?) made of', lambda m: (m.group(1), 'made')),
            # "What X does Y have" → subject=Y, relation=X
            (r'what (.+?) does (.+?) have', lambda m: (m.group(2), m.group(1))),
            # "Who wrote/discovered X" → subject=X, relation=author/discoverer
            (r'who (?:wrote|created|invented|founded) (.+)', lambda m: (m.group(1), 'author')),
            (r'who (?:discovered) (.+)', lambda m: (m.group(1), 'discovered')),
            # "Where is X" → subject=X, relation=location
            (r'where is (.+)', lambda m: (m.group(1), 'location')),
            # "When was X born" → subject=X, relation=born
            (r'when was (.+?) born', lambda m: (m.group(1), 'born')),
            # "When was X" → subject=X, relation=date
            (r'when (?:was|is|did) (.+)', lambda m: (m.group(1), 'date')),
            # "X born" → subject=X, relation=born
            (r'(\w+)\s+born', lambda m: (m.group(1), 'born')),
            # "X made of" / "X composed of" → subject=X, relation=made
            (r'(\w+)\s+(?:made|composed)\s+of', lambda m: (m.group(1), 'made')),
            # "How many X does Y have" → subject=Y, relation=X
            (r'how many (.+?) (?:does|do|did) (.+?) have', lambda m: (m.group(2), m.group(1))),
            # "How many X has Y" → subject=Y, relation=X
            (r'how many (.+?) (?:has|have) (.+)', lambda m: (m.group(2), m.group(1))),
            # "largest X" / "biggest X" → subject=X, relation=largest
            (r'(?:what is the )?(?:largest|biggest) (.+)', lambda m: (m.group(1), 'largest')),
            # "speed of X" → subject=X, relation=speed
            (r'(?:what is the )?speed of (.+)', lambda m: (m.group(1), 'speed')),
        ]

        subject = None
        relation = None
        for pattern, extractor in patterns:
            m = re.search(pattern, q)
            if m:
                subject, relation = extractor(m)
                subject = subject.strip()
                relation = relation.strip()
                break

        # Fallback: extract content words
        if not subject:
            clean = q
            for ch in '.,!;:"\'-()[]{}':
                clean = clean.replace(ch, ' ')
            words = [w for w in clean.split() if w not in self.STOP_WORDS and len(w) > 1]
            if not words:
                return None, 0.0
            # Last content word is usually the subject
            subject = words[-1]
            relation = words[0] if len(words) > 1 else None

        # Relation alias map — covers all common query→KB relation mismatches
        aliases = {
            'author': ['author', 'wrote', 'written_by', 'created_by', 'creator'],
            'capital': ['capital', 'capital_city', 'capital_of'],
            'location': ['location', 'located_in', 'in', 'country'],
            'formula': ['chemical_formula', 'formula', 'composed_of', 'made_of'],
            'color': ['color', 'colour', 'has_color', 'has_property'],
            'language': ['language', 'official_language', 'speaks'],
            'size': ['largest', 'biggest', 'size', 'description'],
            'speed': ['speed', 'velocity', 'rate'],
            'number': ['legs', 'count', 'number_of'],
            'born': ['born', 'birth', 'date_of_birth', 'born_in'],
            'died': ['died', 'death', 'date_of_death'],
            'discovered': ['discovered', 'discoverer', 'discovered_by', 'discovery', 'description'],
            'made': ['made', 'composed_of', 'made_of', 'formula', 'has_part'],
            'largest': ['description', 'has_property', 'size'],
        }

        # Try subject with case variants (KB stores both "France" and "france")
        subject_variants = [subject]
        if subject:
            subject_variants = list(dict.fromkeys([
                subject, subject.capitalize(), subject.lower(), subject.upper(),
            ]))

        # Collect all relation variants to try
        rel_variants = [relation] if relation else []
        if relation:
            for alias_group in aliases.values():
                if relation in alias_group:
                    rel_variants = list(dict.fromkeys(alias_group))
                    break

        # Try (subject_variant, relation_variant) — Tier 1/2 only, NO Hopfield
        for subj in subject_variants:
            for rel in rel_variants:
                result = self.knowledge.query(subject=subj, relation=rel, max_tier=2)
                if result and result.get('fact') and result.get('confidence', 0) > 0.5:
                    fact = result['fact']
                    answer = fact[2] if len(fact) >= 3 else str(fact)
                    return answer, result['confidence']

        # Superlative queries: "largest planet" → scan descriptions for keyword match
        if relation in ('largest', 'smallest', 'fastest', 'tallest', 'longest',
                        'biggest', 'highest', 'deepest'):
            # Extract the noun being asked about (e.g., "planet" from "largest planet solar system")
            search_words = [w for w in (subject or '').split()
                           if w not in self.STOP_WORDS and len(w) > 2]
            if search_words:
                target_word = search_words[0]  # "planet"
                # Scan all "description" facts for one mentioning "largest {target}"
                for i, fact in enumerate(self.knowledge.facts):
                    if (len(fact) >= 3 and fact[1].lower() == 'description'
                            and relation in fact[2].lower()
                            and target_word in fact[2].lower()):
                        return fact[0], 0.95  # Return the subject (e.g., "Jupiter")

        # Subject-only: scan all facts for this entity, return best relation match
        for subj in subject_variants:
            indices = self.knowledge._entity_subject_index.get(subj.lower().strip(), [])
            if not indices:
                indices = self.knowledge._entity_subject_index.get(subj, [])
            if indices and relation:
                # Score each fact by relation relevance
                for idx in indices:
                    fact = self.knowledge.facts[idx]
                    fact_rel = fact[1].lower() if len(fact) >= 3 else ''
                    if fact_rel in rel_variants or fact_rel == relation:
                        answer = fact[2] if len(fact) >= 3 else str(fact)
                        return answer, 0.9

        return None, 0.0

    def _residual_hopfield_retrieve(self, content_words):
        """Retrieve via pre-contextualized transformer residual states.

        Query = average of content word embeddings → project to 2048d →
        softmax attention over stored Layer-18 residuals.
        The residuals encode "Paris-after-10-layers-already-knows-France".
        """
        vecs = []
        for w in content_words:
            v = self.emb_store.encode(w)
            if v is not None:
                vecs.append(v)
        if not vecs:
            return None, 0.0

        query_512 = np.mean(vecs, axis=0).astype(np.float32)
        results = self.residual_hopfield.retrieve_from_512d(query_512, layer='layer_18', top_k=3)
        if not results:
            return None, 0.0

        best_word, best_score = results[0]
        # Filter: skip if the best match is just one of the query words
        query_set = set(w.lower() for w in content_words)
        for word, score in results:
            if word.lower() not in query_set and score > 0.01:
                return word, min(1.0, score * 5.0)

        return None, 0.0

    def _mlp_retrieve(self, content_words):
        """Retrieve via MLP Fact Vectors — stolen transformer knowledge.

        198 entity vectors extracted from Qwen3 MLP layers.
        These encode the transformer's "knowledge" about entities
        (Germany, Jupiter, etc.) in 2048d space.

        Query = average content word raw embeddings → cosine vs MLP vectors.
        If query words match an MLP entity, return it with high confidence.
        """
        # Build query in 2048d raw space
        vecs = []
        for w in content_words:
            v = find_token_embedding(w, self._token2id, self.raw_emb)
            if v is not None:
                vecs.append(v)
        if not vecs:
            return None, 0.0

        query = np.mean(vecs, axis=0).astype(np.float32)
        q_norm = np.linalg.norm(query)
        if q_norm < 1e-8:
            return None, 0.0
        query /= q_norm

        # Cosine similarity against all 198 MLP fact vectors
        sims = self._mlp_matrix_normed @ query
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < 0.3:
            return None, 0.0

        best_word = self._mlp_words[best_idx]
        # Skip if it's just echoing a query word
        query_set = set(w.lower() for w in content_words)
        if best_word.lower() in query_set:
            # Return second-best
            sims[best_idx] = -1
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim < 0.3:
                return None, 0.0
            best_word = self._mlp_words[best_idx]
            if best_word.lower() in query_set:
                return None, 0.0

        return best_word, min(1.0, best_sim)

    # ================================================================
    # Consensus
    # ================================================================

    def _consensus_combine(self, sources):
        """Combine multiple answers via Foss Consensus Ensemble.

        6 source types map to nodes on a barbell graph:
          Cluster 1 (pattern-based): reservoir, hopfield, autoregressive
          Cluster 2 (knowledge-based): knowledge, multi_hop, causal_dag

        PS-Lifted gossip converges 26× faster than standard (T391).
        This IS our depth replacement — instead of 28 sequential layers,
        we run 6 parallel sources and combine via topology-accelerated consensus.

        Cross-cluster agreement (reservoir agrees with KB) gets the
        highest confidence — that's like deep layers confirming shallow ones.

        Sources: list of (name, answer, confidence)
        """
        if len(sources) == 1:
            return sources[0][1], sources[0][2]

        # Group by answer (case-insensitive)
        answer_groups = {}
        for name, answer, conf in sources:
            key = answer.lower().strip()[:100]  # Truncate for grouping
            if key not in answer_groups:
                answer_groups[key] = {'answer': answer, 'total_conf': 0.0,
                                       'count': 0, 'sources': set()}
            answer_groups[key]['total_conf'] += conf
            answer_groups[key]['count'] += 1
            answer_groups[key]['sources'].add(name)

        # Source type classification
        PATTERN_SOURCES = {'reservoir', 'hopfield', 'autoregressive', 'lm_head',
                           'residual_hopfield', 'mlp_facts', 'experience'}
        KNOWLEDGE_SOURCES = {'knowledge', 'multi_hop', 'causal_dag', 'commonsense'}

        # Foss Consensus on barbell (6 nodes: 3+3)
        if self.consensus and len(answer_groups) >= 2:
            try:
                groups_sorted = sorted(answer_groups.values(),
                                       key=lambda g: g['total_conf'],
                                       reverse=True)

                # Assign confidence to barbell nodes
                # Cluster 1 = pattern sources, Cluster 2 = knowledge sources
                values = np.zeros(6)
                weights = np.zeros(6)

                for g in groups_sorted:
                    has_pattern = bool(g['sources'] & PATTERN_SOURCES)
                    has_knowledge = bool(g['sources'] & KNOWLEDGE_SOURCES)
                    if has_pattern:
                        # Distribute across cluster 1
                        for i in range(3):
                            values[i] += g['total_conf']
                            weights[i] += g['count']
                    if has_knowledge:
                        # Distribute across cluster 2
                        for i in range(3, 6):
                            values[i] += g['total_conf']
                            weights[i] += g['count']

                # If only one cluster has data, spread to both
                if np.sum(values[:3]) == 0:
                    values[:3] = values[3:6]
                elif np.sum(values[3:]) == 0:
                    values[3:] = values[:3]

                # Run Foss consensus
                converged, steps, speedup = self.consensus.average_consensus(values)

                # Consensus spread = agreement metric
                spread = float(np.std(converged))
                consensus_mean = float(converged.mean())

                # Cross-cluster agreement: both clusters converge to similar value
                cluster1_mean = float(np.mean(converged[:3]))
                cluster2_mean = float(np.mean(converged[3:]))
                cross_agreement = 1.0 - min(1.0, abs(cluster1_mean - cluster2_mean))

                # Final confidence: consensus mean + cross-agreement boost
                final_conf = min(1.0, consensus_mean + 0.2 * cross_agreement)

                return groups_sorted[0]['answer'], final_conf
            except Exception:
                pass

        # Fallback: weighted combination
        best = max(answer_groups.values(),
                   key=lambda g: g['total_conf'] * (1 + 0.3 * (g['count'] - 1)))

        base_conf = best['total_conf'] / best['count']
        agreement_boost = min(0.3, 0.15 * (best['count'] - 1))
        final_conf = min(1.0, base_conf + agreement_boost)

        # Cross-source agreement boost
        has_pattern = bool(best['sources'] & PATTERN_SOURCES)
        has_knowledge = bool(best['sources'] & KNOWLEDGE_SOURCES)
        if has_pattern and has_knowledge and best['count'] >= 2:
            final_conf = min(1.0, final_conf + 0.15)

        return best['answer'], final_conf

    # ================================================================
    # lm_head Generation
    # ================================================================

    def _reservoir_generate(self, content_words, top_k=10):
        """Reservoir state → prediction → nearest vocabulary tokens.

        Returns list of (token_string, similarity) sorted by similarity.
        """
        if (self.reservoir is None or self.emb_store is None
                or self.reservoir.W_out is None):
            return []

        # Encode through reservoir
        self.reservoir.reset_state()
        zero = np.zeros(self.emb_store.dim, dtype=np.float32)
        for w in content_words:
            v = self.emb_store.encode(w)
            if v is None:
                v = zero
            self.reservoir.step(v)
        for _ in range(3):
            self.reservoir.step(zero)
        state = (self.reservoir.state_pos + self.reservoir.state_neg) / 2.0

        # Predict embedding via trained readout
        pred = self.reservoir.predict(state)
        if pred is None:
            return []

        # Find nearest tokens (works for both 512d and 2048d predictions)
        neighbors = self._nearest_raw(pred, top_k=top_k)
        results = []
        for token, sim in neighbors:
            clean = token.replace('Ġ', '').replace('ĉ', ' ').strip()
            if clean and len(clean) > 1 and clean[0].isalpha():
                results.append((clean, float(sim)))
        return results

    def generate(self, question: str, top_k=5) -> Dict[str, Any]:
        """Full generation pipeline: question → reservoir → lm_head → answer tokens.

        Combines KB retrieval (exact facts) with lm_head generation
        (associative vocabulary search) via consensus.
        """
        result = self.query(question)

        # Also get lm_head generation candidates
        clean = question.lower()
        for ch in '?.,!;:"\'-()[]{}':
            clean = clean.replace(ch, ' ')
        content_words = [w for w in clean.split()
                         if w not in self.STOP_WORDS and len(w) > 1]

        gen_tokens = self._reservoir_generate(content_words, top_k=top_k)
        result['generated_tokens'] = gen_tokens

        # If pipeline had no answer but generation did, use generation
        if not result['answer'] and gen_tokens:
            result['answer'] = gen_tokens[0][0]
            result['confidence'] = gen_tokens[0][1]
            result['sources'].append(('lm_head', gen_tokens[0][0], gen_tokens[0][1]))

        return result

    def generate_sequence(self, question: str, max_tokens=12,
                          temperature: float = 0.7, top_k: int = 8,
                          rep_penalty: float = 1.5) -> Dict[str, Any]:
        """Autoregressive generation: reservoir unrolling with score fusion.

        For each output position:
          1. Feed last predicted token embedding INTO reservoir
          2. Predict next embedding from reservoir state
          3. Find nearest vocabulary tokens
          4. Apply repetition penalty + temperature sampling
          5. CASI gate: stop if structure collapses
          6. Pick token, feed back, repeat

        Args:
            temperature: sampling temperature (0 = greedy, 1 = uniform, 0.7 = balanced)
            top_k: number of candidates for sampling
            rep_penalty: multiplicative penalty for seen tokens (1.0 = none, 2.0 = strong)

        Returns query result with 'generated_sequence' added.
        """
        result = self.query(question)

        if (self.reservoir is None or self.emb_store is None
                or self.reservoir.W_out is None):
            result['generated_sequence'] = []
            return result

        # Encode question into reservoir state
        clean = question.lower()
        for ch in '?.,!;:"\'-()[]{}':
            clean = clean.replace(ch, ' ')
        content_words = [w for w in clean.split()
                         if w not in self.STOP_WORDS and len(w) > 1]

        self.reservoir.reset_state()
        zero = np.zeros(self.emb_store.dim, dtype=np.float32)
        for w in content_words:
            v = self.emb_store.encode(w)
            if v is None:
                v = zero
            self.reservoir.step(v)

        # Mixing steps to let context settle
        for _ in range(3):
            self.reservoir.step(zero)

        # CASI generation gate — stops when quality drops
        try:
            from .casi_gate import CASIGenerationGate
            casi_gate = CASIGenerationGate(threshold=1.3, window=4)
        except Exception:
            casi_gate = None

        # Autoregressive loop with temperature sampling + repetition penalty
        generated = []
        seen_tokens = {}  # token → count (for frequency penalty)

        for step in range(max_tokens):
            state = (self.reservoir.state_pos + self.reservoir.state_neg) / 2.0
            novelty = float(np.mean(np.abs(
                self.reservoir.state_pos - self.reservoir.state_neg)))

            # Predict next embedding
            pred = self.reservoir.predict(state)
            if pred is None:
                break

            # Find nearest tokens (fetch extra for sampling pool)
            neighbors = self._nearest_raw(pred, top_k=top_k * 5)

            # Build candidate pool: clean tokens with scores
            candidates = []
            for token, sim in neighbors:
                if (token and len(token) > 1
                        and token[0].isalpha()
                        and token.lower() not in self.STOP_WORDS):
                    candidates.append((token, sim))
                    if len(candidates) >= top_k * 3:
                        break

            if not candidates:
                break

            # Apply repetition penalty
            scored = []
            for token, sim in candidates:
                penalty = rep_penalty ** seen_tokens.get(token.lower(), 0)
                scored.append((token, sim / penalty))

            # Temperature sampling over top-k
            scored.sort(key=lambda x: -x[1])
            top_cands = scored[:top_k]

            if temperature <= 0.01:
                # Greedy
                best_token, best_sim = top_cands[0]
            else:
                # Softmax sampling with temperature
                sims_arr = np.array([s for _, s in top_cands], dtype=np.float32)
                logits = sims_arr / temperature
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()
                idx = np.random.choice(len(top_cands), p=probs)
                best_token, best_sim = top_cands[idx]

            # Get original (unpenalized) similarity for quality check
            orig_sim = dict([(t, s) for t, s in candidates]).get(best_token, best_sim)
            if orig_sim < 0.02:
                break  # Quality too low, stop generating

            # Z₂ novelty spike → stop (reservoir in uncharted territory)
            # Relaxed: only stop on extreme novelty after 5+ tokens
            if novelty > 0.95 and step > 5:
                break

            # CASI gate: monitor structural quality (only after 6+ tokens)
            if casi_gate and step > 6:
                casi_gate.add_token(orig_sim)
                should_stop, casi_score = casi_gate.should_stop()
                if should_stop:
                    break  # Structure collapsed → stop

            generated.append((best_token, orig_sim, novelty))
            seen_tokens[best_token.lower()] = seen_tokens.get(best_token.lower(), 0) + 1

            # Feed predicted token back into reservoir (autoregressive)
            # Reservoir expects 512d (projected) input, not 2048d raw
            best_emb = None
            if self.emb_store is not None:
                best_emb = self.emb_store.encode(best_token)
            if best_emb is None and self.raw_emb is not None and hasattr(self, '_token2id'):
                # Fallback: get raw 2048d and project to 512d via SVD
                tid = find_token_id(best_token, self._token2id)
                if tid is not None:
                    raw_vec = self.raw_emb[tid].astype(np.float32)
                    # Project 2048d → 512d (emb_store handles this)
                    if hasattr(self.emb_store, 'project'):
                        best_emb = self.emb_store.project(raw_vec)
            if best_emb is not None:
                self.reservoir.step(best_emb)
            else:
                self.reservoir.step(zero)

        result['generated_sequence'] = generated

        # If query had no answer, use generated sequence
        if not result['answer'] and generated:
            # Pick highest-similarity token as the answer
            best = max(generated, key=lambda x: x[1])
            result['answer'] = best[0]
            result['confidence'] = best[1]
            result['sources'].append(('autoregressive', best[0], best[1]))

        return result

    # ================================================================
    # Helpers
    # ================================================================

    def _nearest_raw(self, vec, top_k=10, filter_garbage=True):
        """Find nearest tokens in raw 2048d embedding space.

        Used when reservoir outputs 2048d predictions (trained with raw targets).
        Falls back to emb_store.nearest() for 512d predictions.

        If ResidualHopfield is available, enriches the prediction with
        pre-contextualized transformer representations (stolen depth).

        filter_garbage: skip unicode, special tokens, and non-alphabetic tokens.
        """
        if vec.shape[0] == self.emb_store.dim:
            # 512d prediction → use projected space
            return self.emb_store.nearest(vec, top_k=top_k)

        # Enrich 2048d prediction with residual patterns (stolen transformer depth)
        if self.residual_hopfield is not None:
            vec = self.residual_hopfield.enrich_prediction(vec, alpha=0.2)

        if self.raw_emb is None or self.id2token is None:
            return []

        # Use FAISS index if available (11× faster than brute-force)
        if hasattr(self, '_nn_index') and self._nn_index is not None:
            raw_results = self._nn_index.search(vec, top_k=top_k * 10 if filter_garbage else top_k)
            if not filter_garbage:
                return [(clean_token(self.id2token.get(tid, '?')), sim)
                        for tid, sim in raw_results[:top_k]]
            # Fall through to filter logic below with FAISS results
            top_idx = [tid for tid, _ in raw_results]
            top_sims = {tid: sim for tid, sim in raw_results}
        else:
            # Brute-force fallback
            vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
            if not hasattr(self, '_raw_emb_normed') or self._raw_emb_normed is None:
                norms = np.linalg.norm(self.raw_emb, axis=1, keepdims=True)
                self._raw_emb_normed = self.raw_emb / np.clip(norms, 1e-8, None)

            sims = self._raw_emb_normed @ vec_norm

            if not filter_garbage:
                top_idx_arr = np.argpartition(sims, -top_k)[-top_k:]
                top_idx_arr = top_idx_arr[np.argsort(sims[top_idx_arr])[::-1]]
                return [(clean_token(self.id2token.get(int(i), '?')), float(sims[i])) for i in top_idx_arr]

            fetch_k = top_k * 10
            top_idx_arr = np.argpartition(sims, -fetch_k)[-fetch_k:]
            top_idx_arr = top_idx_arr[np.argsort(sims[top_idx_arr])[::-1]]
            top_idx = [int(i) for i in top_idx_arr]
            top_sims = {int(i): float(sims[i]) for i in top_idx_arr}

        results = []
        seen_clean = set()
        for i in top_idx:
            token = self.id2token.get(int(i), '?')
            cleaned = clean_token(token)
            # Skip: empty, too short, unicode, special tokens, duplicates
            if (not cleaned or len(cleaned) < 2
                    or not cleaned[0].isascii() or not cleaned[0].isalpha()
                    or token.startswith('<|') or token.startswith('ð')
                    or any(ord(c) > 127 for c in cleaned)
                    # Skip camelCase/PascalCase code tokens
                    or (any(c.isupper() for c in cleaned[1:]) and any(c.islower() for c in cleaned))
                    # Skip tokens with digits mixed in
                    or any(c.isdigit() for c in cleaned)
                    # Skip very long tokens (likely code identifiers)
                    or len(cleaned) > 15
                    # Skip duplicates (ĠFrance and France → same word)
                    or cleaned.lower() in seen_clean):
                continue
            seen_clean.add(cleaned.lower())
            sim_score = top_sims.get(i, 0.0) if isinstance(top_sims, dict) else float(top_sims)
            results.append((cleaned, sim_score))
            if len(results) >= top_k:
                break
        return results

    def _build_causal_dag(self):
        """Build CausalDAG from KB + ConceptNet relations."""
        try:
            from .causal_dag import CausalDAG
            self.causal_dag = CausalDAG()
            if self.knowledge:
                self.causal_dag.build_from_kb(self.knowledge)
            if self.commonsense:
                self.causal_dag.build_from_commonsense(self.commonsense)
            stats = self.causal_dag.stats()
            if stats['nodes'] == 0:
                self.causal_dag = None
        except Exception:
            self.causal_dag = None

    def _z2_attention(self, content_words):
        """Compute Z₂ disagreement per token as attention signal.

        Feeds each word into a fresh reservoir and measures how much
        pos/neg copies diverge. High divergence = novel/surprising token
        = more important for the answer.

        Returns dict: word → importance (0-1 normalized).
        """
        if not content_words:
            return {}

        zero = np.zeros(self.emb_store.dim, dtype=np.float32)
        novelties = {}

        for w in content_words:
            v = self.emb_store.encode(w)
            if v is None:
                novelties[w] = 0.0
                continue
            # Fresh state for each word
            self.reservoir.reset_state()
            self.reservoir.step(v)
            # One mixing step
            self.reservoir.step(zero)
            nov = float(np.mean(np.abs(
                self.reservoir.state_pos - self.reservoir.state_neg)))
            novelties[w] = nov

        # Normalize to 0-1
        vals = list(novelties.values())
        if not vals:
            return {}
        vmin, vmax = min(vals), max(vals)
        spread = vmax - vmin
        if spread < 1e-8:
            return {w: 1.0 / len(content_words) for w in content_words}

        return {w: (v - vmin) / spread for w, v in novelties.items()}

    def _get_raw_embeddings(self, words):
        """Get raw 2048d embeddings for attention computation."""
        result = []
        for w in words:
            found = False
            for v in [w, 'Ġ'+w, w.capitalize(), 'Ġ'+w.capitalize(),
                      w.lower(), 'Ġ'+w.lower()]:
                tid = self.emb_store._token2id.get(v)
                if tid is not None:
                    result.append(self.raw_emb[tid])
                    found = True
                    break
            if not found:
                result.append(np.zeros(self.raw_emb.shape[1], dtype=np.float32))
        return result

    def _get_proj_embeddings(self, words):
        """Get 512d projected embeddings for Ricci attention."""
        result = []
        for w in words:
            v = self.emb_store.encode(w)
            if v is not None:
                result.append(v)
            else:
                result.append(np.zeros(self.emb_store.dim, dtype=np.float32))
        return result

    def _top_tokens(self, words, weights, n=3):
        """Format top-n tokens by weight."""
        pairs = sorted(zip(words, weights), key=lambda x: -x[1])
        return ", ".join(f"{w}({wt:.2f})" for w, wt in pairs[:n])

    def naturalize(self, question, result):
        """Convert pipeline result into a natural language answer.

        Uses the question structure + fact to generate a fluent response.
        No LLM — just templates + the extracted information.
        """
        if not result or not result.get('answer'):
            return None

        answer = result['answer']
        q = question.lower().strip().rstrip('?').strip()

        # Clean up answer (remove BPE artifacts, underscores)
        answer_clean = answer.replace('_', ' ').replace('Ġ', '').strip()
        if answer_clean and answer_clean[0].islower():
            # Don't capitalize if it looks like a formula or code
            if not any(c in answer_clean for c in '0123456789+=/'):
                pass  # keep as-is, capitalize in template

        import re

        # Template patterns — MORE SPECIFIC FIRST
        templates = [
            # Specific patterns first
            (r'what language is spoken in (.+)',
             lambda m, a: f"The language spoken in {m.group(1).title()} is {a}."),
            (r'what (?:is|are) the (.+?) of (.+)',
             lambda m, a: f"The {m.group(1)} of {m.group(2).title()} is {a}."),
            (r'who (?:wrote|created) (.+)',
             lambda m, a: f"{m.group(1).title()} was written by {a}."),
            (r'who (?:invented|discovered) (.+)',
             lambda m, a: f"{m.group(1).title()} was discovered by {a}."),
            (r'who (?:founded) (.+)',
             lambda m, a: f"{m.group(1).title()} was founded by {a}."),
            (r'where is (.+)',
             lambda m, a: f"{m.group(1).title()} is located in {a}."),
            (r'when (?:was|did) (.+)',
             lambda m, a: f"{m.group(1).title()} was {a}."),
            (r'how many (.+?) (?:does|do) (.+?) have',
             lambda m, a: f"{m.group(2).title()} has {a} {m.group(1)}."),
            (r'what (?:is|are) (.+)',
             lambda m, a: f"{m.group(1).title()} is {a}."),
            (r'is (.+?) (?:a|an|the) (.+)',
             lambda m, a: f"{'Yes' if a.lower() in ('true','yes') else 'No'}, {m.group(1)} is {a}."),
        ]

        for pattern, formatter in templates:
            m = re.search(pattern, q)
            if m:
                return formatter(m, answer_clean)

        # Default: just capitalize
        return f"{answer_clean[0].upper()}{answer_clean[1:]}." if answer_clean else None

    def _empty_result(self, reason, trace=None, novelty=0.0, attn=None):
        return {
            'answer': None,
            'confidence': 0.0,
            'sources': [],
            'novelty': novelty,
            'attention_weights': attn or {},
            'trace': (trace or []) + [f"Empty: {reason}"],
        }
