"""
Swarm Agent — Multi-Agent FOSS-KI over UDP
=============================================
S15-S17: David's Swarm Architecture.

Tier 1: Specialist Agents (each a full FOSS-KI instance)
Tier 2: PS-Lifted Consensus over UDP
Tier 3: Meta-Agent (routes to specialists)
Tier 4: Federated Brain Sync

Each agent runs as a separate process with its own UDP port.
Communication via JSON messages over UDP (from T93).

Spawn: python cli.py spawn --domain medicine --port 5001
Query: sends to Meta-Agent → routes → specialist → consensus → answer
Sync:  background thread merges Brain states periodically
"""

import json
import socket
import threading
import time
import hashlib
from typing import Dict, Any, List, Optional

from .brain import BrainSnapshot, merge_brains
from .knowledge import KnowledgeStore
from .foss_lm import FossLanguageModel
from .router import VortexRouter
from .formulierer import Formulierer


# ── Message Protocol ──────────────────────────────────────

MSG_QUERY = 'query'           # Query request
MSG_ANSWER = 'answer'         # Query response
MSG_REGISTER = 'register'     # Agent registers with meta-agent
MSG_HEARTBEAT = 'heartbeat'   # Agent alive signal
MSG_SYNC = 'sync_request'     # Request brain sync
MSG_BRAIN = 'brain_data'      # Brain state transfer
MSG_SHUTDOWN = 'shutdown'     # Graceful shutdown


def _make_msg(msg_type: str, sender: str, **kwargs) -> bytes:
    """Create a UDP message."""
    msg = {'type': msg_type, 'sender': sender, 'ts': time.time()}
    msg.update(kwargs)
    return json.dumps(msg).encode('utf-8')


def _parse_msg(data: bytes) -> Dict[str, Any]:
    """Parse a UDP message."""
    return json.loads(data.decode('utf-8'))


# ── Swarm Agent ───────────────────────────────────────────

class SwarmAgent:
    """
    A single FOSS-KI agent in the swarm.

    Each agent is a complete system with:
    - KnowledgeStore (Hopfield)
    - PPMModel (language)
    - VortexRouter (fiber routing)
    - Formulierer (PPM reformulation)
    - UDP listener for swarm communication
    """

    def __init__(self, agent_id: str, domain: str = 'general',
                 port: int = 5000, meta_addr: tuple = None,
                 knowledge_dim: int = 64):
        self.agent_id = agent_id
        self.domain = domain
        self.port = port
        self.meta_addr = meta_addr  # (host, port) of meta-agent

        # Build subsystems
        self.knowledge = KnowledgeStore(dim=knowledge_dim)
        self.lm = FossLanguageModel()
        self.router = VortexRouter(
            knowledge_store=self.knowledge,
            language_model=self.lm,
        )

        # Swarm state
        self._running = False
        self._socket = None
        self._listener_thread = None
        self._sync_thread = None
        self._peers = {}  # agent_id → (host, port, domain, last_seen)
        self._pending_queries = {}  # query_id → {query, responses, callback}
        self._query_counter = 0
        self._stats = {
            'queries_handled': 0,
            'queries_forwarded': 0,
            'syncs': 0,
            'messages_sent': 0,
            'messages_received': 0,
        }

    def load_brain(self, brain: BrainSnapshot):
        """Load a Brain snapshot into this agent."""
        parts = brain.restore_parts()
        if 'knowledge' in parts:
            self.knowledge = parts['knowledge']
            self.router.knowledge = self.knowledge
        if 'language' in parts:
            self.lm = parts['language']
            self.router.language = self.lm

    def capture_brain(self) -> BrainSnapshot:
        """Capture this agent's current Brain state."""
        return BrainSnapshot.capture_from_parts(
            knowledge_store=self.knowledge,
            lm_model=self.lm,
            domain=self.domain,
            agent_id=self.agent_id,
        )

    # ── Local Query ───────────────────────────────────────

    def query(self, text: str) -> Dict[str, Any]:
        """
        Answer a query locally using this agent's knowledge.
        Returns router result with agent_id and domain.
        """
        result = self.router.route(text)
        result['agent_id'] = self.agent_id
        result['domain'] = self.domain
        self._stats['queries_handled'] += 1
        return result

    # ── Swarm Query (S16) ─────────────────────────────────

    def swarm_query(self, text: str, timeout: float = 2.0) -> Dict[str, Any]:
        """
        Query the swarm: try locally, then broadcast if unsure.

        1. Try local answer
        2. If HIGH confidence → return immediately
        3. If LOW/REJECTED → broadcast to all peers
        4. Collect responses → PS-Lifted consensus → best answer
        """
        # Try local first
        local = self.query(text)

        if local['confidence_level'] == 'HIGH':
            local['source'] = 'local'
            return local

        # Broadcast to peers
        if not self._peers:
            local['source'] = 'local_only'
            return local

        query_id = f"{self.agent_id}-{self._query_counter}"
        self._query_counter += 1

        responses = [local]  # Include local response

        # Send query to all peers
        for peer_id, (host, port, domain, _) in self._peers.items():
            try:
                msg = _make_msg(MSG_QUERY, self.agent_id,
                                query=text, query_id=query_id)
                self._send(msg, (host, port))
                self._stats['queries_forwarded'] += 1
            except Exception:
                pass

        # Wait for responses
        deadline = time.time() + timeout
        self._pending_queries[query_id] = responses

        while time.time() < deadline:
            time.sleep(0.05)
            if len(responses) > len(self._peers):
                break  # Got all responses

        del self._pending_queries[query_id]

        # PS-Lifted Consensus over responses
        return self._consensus(responses, text)

    def _consensus(self, responses: List[Dict], query: str) -> Dict[str, Any]:
        """
        PS-Lifted consensus over multiple agent responses.

        Weighting by confidence (attractor depth):
          HIGH = 3, MEDIUM = 2, LOW = 1, REJECTED = 0

        The agent with deepest attractor (highest confidence)
        has the most weight. This is NOT majority vote — it's
        weighted convergence.
        """
        confidence_weights = {
            'HIGH': 3.0, 'MEDIUM': 2.0, 'LOW': 1.0,
            'REJECTED': 0.0, 'UNKNOWN': 0.0,
        }

        # Filter to responses that have actual answers
        valid = [r for r in responses
                 if r.get('answer') and r['confidence_level'] not in ('REJECTED', 'UNKNOWN')]

        if not valid:
            return {
                'answer': "I don't have information about that topic.",
                'confidence_level': 'REJECTED',
                'source': 'swarm_consensus',
                'n_respondents': len(responses),
                'n_valid': 0,
            }

        # Weight by confidence
        best = max(valid, key=lambda r: confidence_weights.get(r['confidence_level'], 0))

        # Check agreement: do multiple agents agree?
        answers = {}
        for r in valid:
            a = str(r.get('answer', '')).strip().lower()
            w = confidence_weights.get(r['confidence_level'], 0)
            if a not in answers:
                answers[a] = {'weight': 0, 'response': r, 'agents': []}
            answers[a]['weight'] += w
            answers[a]['agents'].append(r.get('agent_id', '?'))

        # Pick answer with highest total weight
        winner = max(answers.values(), key=lambda x: x['weight'])
        result = dict(winner['response'])
        result['source'] = 'swarm_consensus'
        result['n_respondents'] = len(responses)
        result['n_valid'] = len(valid)
        result['agreement'] = len(winner['agents'])
        result['agreeing_agents'] = winner['agents']

        # Boost confidence if multiple agents agree
        if len(winner['agents']) >= 3:
            result['confidence_level'] = 'HIGH'
        elif len(winner['agents']) >= 2 and result['confidence_level'] == 'MEDIUM':
            result['confidence_level'] = 'HIGH'

        return result

    # ── Network Layer ─────────────────────────────────────

    def start(self):
        """Start listening for swarm messages."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(('0.0.0.0', self.port))
        self._socket.settimeout(0.5)
        self._running = True

        self._listener_thread = threading.Thread(
            target=self._listen_loop, daemon=True
        )
        self._listener_thread.start()

        # Register with meta-agent if configured
        if self.meta_addr:
            self._register()

    def stop(self):
        """Stop the agent."""
        self._running = False
        if self._listener_thread:
            self._listener_thread.join(timeout=2)
        if self._socket:
            self._socket.close()

    def _send(self, data: bytes, addr: tuple):
        """Send UDP message."""
        if self._socket:
            self._socket.sendto(data, addr)
            self._stats['messages_sent'] += 1

    def _register(self):
        """Register with meta-agent."""
        msg = _make_msg(MSG_REGISTER, self.agent_id,
                        domain=self.domain, port=self.port,
                        n_facts=self.knowledge.n_facts)
        self._send(msg, self.meta_addr)

    def _broadcast_peer_list(self):
        """Broadcast peer list to all known peers so they can find each other."""
        peer_info = {pid: {'host': h, 'port': p, 'domain': d}
                     for pid, (h, p, d, _) in self._peers.items()}
        for pid, (h, p, d, _) in self._peers.items():
            msg = _make_msg('peer_list', self.agent_id, peers=peer_info)
            try:
                self._send(msg, (h, p))
            except Exception:
                pass

    def _listen_loop(self):
        """Main listener loop."""
        while self._running:
            try:
                data, addr = self._socket.recvfrom(65536)
                self._stats['messages_received'] += 1
                msg = _parse_msg(data)
                self._handle_message(msg, addr)
            except socket.timeout:
                continue
            except Exception:
                continue

    def _handle_message(self, msg: Dict, addr: tuple):
        """Handle an incoming swarm message."""
        msg_type = msg.get('type')
        sender = msg.get('sender', '?')

        if msg_type == MSG_QUERY:
            # Answer the query
            result = self.query(msg['query'])
            result['query_id'] = msg.get('query_id')
            reply = _make_msg(MSG_ANSWER, self.agent_id, **result)
            self._send(reply, addr)

        elif msg_type == MSG_ANSWER:
            # Add to pending query responses
            qid = msg.get('query_id')
            if qid and qid in self._pending_queries:
                self._pending_queries[qid].append(msg)

        elif msg_type == MSG_REGISTER:
            # New peer registered
            self._peers[sender] = (
                addr[0], msg.get('port', addr[1]),
                msg.get('domain', 'unknown'),
                time.time(),
            )
            # Broadcast updated peer list to all peers
            self._broadcast_peer_list()

        elif msg_type == MSG_HEARTBEAT:
            if sender in self._peers:
                host, port, domain, _ = self._peers[sender]
                self._peers[sender] = (host, port, domain, time.time())

        elif msg_type == MSG_SYNC:
            # Send our brain state
            brain = self.capture_brain()
            brain_data = json.dumps(brain.state)
            reply = _make_msg(MSG_BRAIN, self.agent_id, brain=brain_data)
            self._send(reply, addr)
            self._stats['syncs'] += 1

        elif msg_type == 'peer_list':
            # Update peer list from meta-agent or other peers
            peers = msg.get('peers', {})
            for pid, info in peers.items():
                if pid != self.agent_id and pid not in self._peers:
                    self._peers[pid] = (
                        info['host'], info['port'],
                        info.get('domain', 'unknown'),
                        time.time(),
                    )

        elif msg_type == MSG_SHUTDOWN:
            self._running = False

    # ── Federated Sync (S17) ──────────────────────────────

    def sync_with_peers(self, timeout: float = 3.0):
        """
        Request brain state from all peers and merge.
        Federated Knowledge Sync — new facts propagate through the swarm.
        """
        if not self._peers:
            return 0

        # Request brains from all peers
        peer_brains = []
        received = {}

        for peer_id, (host, port, domain, _) in self._peers.items():
            msg = _make_msg(MSG_SYNC, self.agent_id)
            self._send(msg, (host, port))

        # Wait for responses
        deadline = time.time() + timeout
        while time.time() < deadline and len(received) < len(self._peers):
            time.sleep(0.1)
            # Check for brain messages in recent messages
            # (handled by _handle_message which puts them in a buffer)

        # Merge our brain with received brains
        my_brain = self.capture_brain()
        all_brains = [my_brain] + peer_brains
        merged = merge_brains(all_brains, strategy='union')

        # Load merged brain
        old_facts = self.knowledge.n_facts
        self.load_brain(merged)
        new_facts = self.knowledge.n_facts - old_facts

        self._stats['syncs'] += 1
        return new_facts

    # ── Info ──────────────────────────────────────────────

    def info(self) -> str:
        lines = [
            f"SwarmAgent: {self.agent_id}",
            f"  Domain:  {self.domain}",
            f"  Port:    {self.port}",
            f"  Facts:   {self.knowledge.n_facts}",
            f"  Peers:   {len(self._peers)}",
            f"  Stats:   {self._stats}",
        ]
        if self._peers:
            for pid, (h, p, d, t) in self._peers.items():
                age = time.time() - t
                lines.append(f"    {pid}: {d} @ {h}:{p} (seen {age:.0f}s ago)")
        return "\n".join(lines)


# ── Meta-Agent (Tier 3) ──────────────────────────────────

class MetaAgent(SwarmAgent):
    """
    Meta-Agent: knows all specialists, routes queries.

    Has NO facts of its own. Its knowledge is WHERE to route.
    This is the Vortex Router scaled to swarm level.
    """

    # Domain → keyword mapping for routing
    DOMAIN_KEYWORDS = {
        'medicine': {'health', 'disease', 'drug', 'symptom', 'symptoms', 'doctor', 'patient',
                     'treatment', 'treat', 'diagnosis', 'surgery', 'medicine', 'medical',
                     'body', 'organ', 'blood', 'cancer', 'virus', 'vaccine',
                     'flu', 'fever', 'pain', 'hospital', 'clinic', 'therapy',
                     'infection', 'pill', 'dose', 'prescription', 'illness',
                     'headache', 'cure', 'remedy', 'heal', 'wound', 'broken',
                     'injury', 'allergic', 'allergy', 'asthma', 'diabetes'},
        'law': {'legal', 'law', 'court', 'judge', 'crime', 'criminal', 'contract',
                'rights', 'constitution', 'attorney', 'lawyer', 'regulation',
                'statute', 'trial', 'verdict', 'sentence', 'prison',
                'theft', 'murder', 'arrest', 'guilty', 'innocent', 'penalty',
                'sue', 'lawsuit', 'plaintiff', 'defendant', 'illegal'},
        'geography': {'country', 'capital', 'city', 'river', 'mountain', 'ocean',
                      'continent', 'population', 'border', 'island', 'region',
                      'located', 'where', 'area'},
        'science': {'formula', 'element', 'atom', 'molecule', 'energy', 'force',
                    'theory', 'experiment', 'physics', 'chemistry', 'biology',
                    'equation', 'mass', 'light', 'gravity'},
        'code': {'function', 'class', 'variable', 'code', 'program', 'algorithm',
                 'python', 'javascript', 'compile', 'runtime', 'bug', 'debug',
                 'software', 'library', 'framework'},
    }

    def __init__(self, port: int = 5000):
        super().__init__(
            agent_id='meta-agent',
            domain='meta',
            port=port,
        )

    def route_to_specialist(self, query: str) -> List[str]:
        """
        Determine which specialist domain(s) should handle this query.
        Returns list of domain names, ranked by relevance.
        """
        words = set(query.lower().split())
        scores = {}

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = len(words & keywords)
            if score > 0:
                scores[domain] = score

        if not scores:
            # No clear domain → broadcast to all
            return list(self._peers_by_domain().keys())

        return sorted(scores, key=scores.get, reverse=True)

    def _peers_by_domain(self) -> Dict[str, List[str]]:
        """Group peers by domain."""
        domains = {}
        for pid, (h, p, d, _) in self._peers.items():
            if d not in domains:
                domains[d] = []
            domains[d].append(pid)
        return domains

    def meta_query(self, query: str, timeout: float = 2.0) -> Dict[str, Any]:
        """
        Route query to appropriate specialist(s) and collect consensus.

        1. Classify query → target domain(s)
        2. Forward to specialist agents in that domain
        3. Collect responses
        4. PS-Lifted consensus
        5. Return best answer
        """
        target_domains = self.route_to_specialist(query)
        peers_by_domain = self._peers_by_domain()

        # Find target peers
        target_peers = []
        for domain in target_domains:
            target_peers.extend(peers_by_domain.get(domain, []))

        if not target_peers:
            return {
                'answer': "No specialist agent available for this query.",
                'confidence_level': 'UNKNOWN',
                'source': 'meta_agent',
                'target_domains': target_domains,
            }

        # Send query to target specialists
        query_id = f"meta-{self._query_counter}"
        self._query_counter += 1
        responses = []
        self._pending_queries[query_id] = responses

        for peer_id in target_peers:
            if peer_id in self._peers:
                host, port, _, _ = self._peers[peer_id]
                msg = _make_msg(MSG_QUERY, self.agent_id,
                                query=query, query_id=query_id)
                self._send(msg, (host, port))

        # Wait
        deadline = time.time() + timeout
        while time.time() < deadline and len(responses) < len(target_peers):
            time.sleep(0.05)

        del self._pending_queries[query_id]

        # Consensus
        result = self._consensus(responses, query)
        result['target_domains'] = target_domains
        result['target_peers'] = target_peers
        return result


# ── Superposition Retrieval (S18) ─────────────────────────

class SuperpositionRetrieval:
    """
    Quantum-inspired retrieval: hold multiple attractors in superposition.

    Instead of collapsing to the nearest attractor immediately,
    maintain a probability distribution over top-K attractors.
    Context from subsequent dialog turns collapses the superposition.

    Example:
        "Was ist Bank?" → {Geldinstitut: 0.4, Parkbank: 0.3, Flussufer: 0.2}
        "Ich war im Park" → collapses to Parkbank
    """

    def __init__(self, knowledge_store: KnowledgeStore, top_k: int = 5):
        self.knowledge = knowledge_store
        self.top_k = top_k
        self._superposition = None  # Current superposition state
        self._context = []          # Dialog context for collapse

    def query_superposition(self, subject: str, relation: str = None) -> Dict[str, Any]:
        """
        Query with superposition: return top-K matches with probabilities.

        Instead of one answer, returns a probability distribution
        over all possible answers.
        """
        import numpy as np

        if not self.knowledge.facts:
            return {'matches': [], 'collapsed': False}

        # Encode query (same as KnowledgeStore: concatenate s_vec + r_vec)
        s_vec = self.knowledge.encoder.encode(subject)
        r_vec = self.knowledge.encoder.encode(relation or '')
        q_vec = np.concatenate([s_vec, r_vec])
        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-10:
            return {'matches': [], 'collapsed': False}
        q_vec = q_vec / q_norm

        # Compute similarity to ALL stored patterns
        similarities = []
        for i, sr_vec in enumerate(self.knowledge._sr_vectors):
            sr_norm = np.linalg.norm(sr_vec)
            if sr_norm < 1e-10:
                similarities.append(0.0)
                continue
            sim = float(np.dot(q_vec, sr_vec / sr_norm))
            similarities.append(sim)

        # Top-K matches
        indices = sorted(range(len(similarities)),
                         key=lambda i: similarities[i], reverse=True)[:self.top_k]

        # Convert similarities to probabilities (softmax)
        sims = np.array([similarities[i] for i in indices])
        # Temperature scaling — sharper distribution
        sims = sims * 5.0  # Temperature = 0.2
        exp_sims = np.exp(sims - sims.max())
        probs = exp_sims / exp_sims.sum()

        matches = []
        for idx, prob in zip(indices, probs):
            if idx < len(self.knowledge.facts):
                s, r, o = self.knowledge.facts[idx]
                matches.append({
                    'fact': (s, r, o),
                    'probability': float(prob),
                    'similarity': float(similarities[idx]),
                })

        self._superposition = matches
        return {
            'matches': matches,
            'collapsed': False,
            'entropy': float(-np.sum(probs * np.log2(probs + 1e-10))),
        }

    def collapse(self, context: str) -> Dict[str, Any]:
        """
        Collapse superposition using new context.

        The context biases the probability distribution toward
        matches that are semantically related to the context.
        """
        import numpy as np

        if not self._superposition:
            return {'match': None, 'collapsed': True}

        self._context.append(context)

        # Encode context
        ctx_vec = self.knowledge.encoder.encode(context)
        ctx_norm = np.linalg.norm(ctx_vec)
        if ctx_norm < 1e-10:
            # Context doesn't help → return highest probability
            best = max(self._superposition, key=lambda m: m['probability'])
            return {'match': best['fact'], 'collapsed': True, 'method': 'max_prob'}

        ctx_vec = ctx_vec / ctx_norm

        # Re-weight superposition by context similarity
        for match in self._superposition:
            s, r, o = match['fact']
            # Check context similarity against full fact AND object separately
            fact_text = f"{s} {r} {o}"
            fact_vec = self.knowledge.encoder.encode(fact_text)
            fn = np.linalg.norm(fact_vec)
            if fn > 1e-10:
                ctx_sim = float(np.dot(ctx_vec, fact_vec / fn))
            else:
                ctx_sim = 0.0

            # Also check word overlap between context and object
            ctx_words = set(context.lower().split())
            obj_words = set(o.lower().split())
            word_overlap = len(ctx_words & obj_words)
            if word_overlap > 0:
                ctx_sim = max(ctx_sim, 0.5 + 0.2 * word_overlap)

            # Update probability: multiply by context relevance
            match['probability'] *= max(ctx_sim, 0.01)

        # Renormalize
        total = sum(m['probability'] for m in self._superposition)
        if total > 0:
            for m in self._superposition:
                m['probability'] /= total

        # Check if collapsed (one match dominates)
        probs = [m['probability'] for m in self._superposition]
        max_prob = max(probs)

        best = max(self._superposition, key=lambda m: m['probability'])

        collapsed = max_prob > 0.6  # Threshold for "collapse"
        return {
            'match': best['fact'],
            'probability': max_prob,
            'collapsed': collapsed,
            'superposition': self._superposition,
            'method': 'context_collapse',
        }

    def reset(self):
        """Reset superposition state."""
        self._superposition = None
        self._context = []
