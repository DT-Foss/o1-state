"""
Conversation Memory — Sliding Window + PPM Compression
========================================================
The problem: fixed-size context windows lose old information.
Transformers use attention (O(n²)) to handle this.

Our solution: sliding window of recent turns (verbatim) +
compressed summary of older turns. PPM context tree naturally
absorbs statistical patterns from older turns, giving the
system "memory" without storing raw text.

Architecture:
  - Window: last N turns stored verbatim (full detail)
  - Summary: older turns compressed into fact extractions
  - PPM Context: running character context from all turns
  - Entity Memory: Hopfield-backed entity persistence

The key insight: PPM's context tree IS a form of compression.
When we feed conversation text through PPM during training,
the statistical patterns (common phrases, entity co-occurrences)
are absorbed into the tree. We don't need to store the raw text.
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import re


class ConversationMemory:
    """
    Sliding window conversation memory with fact extraction.

    Recent turns: verbatim in a fixed-size window.
    Older turns: compressed into extracted facts + summary.
    """

    def __init__(self, window_size=10, max_summary_facts=50,
                 knowledge_store=None, lm_model=None):
        """
        Args:
            window_size: number of recent turns to keep verbatim
            max_summary_facts: max facts extracted from older turns
            knowledge_store: KnowledgeStore for fact persistence
            lm_model: PPM model for context absorption
        """
        self.window_size = window_size
        self.max_summary_facts = max_summary_facts
        self.knowledge = knowledge_store
        self.lm = lm_model

        # Verbatim recent turns
        self.window = []  # [(role, text, metadata), ...]

        # Compressed older turns
        self.summary_facts = []  # [(subject, relation, object), ...]
        self.topic_counts = defaultdict(int)  # topic → mention count
        self.entity_timeline = []  # [(entity, turn_idx), ...]

        # Running statistics
        self.total_turns = 0
        self.compressions = 0

    def add_turn(self, role: str, text: str, metadata: Optional[Dict] = None):
        """
        Add a conversation turn.

        Args:
            role: 'user' or 'system'
            text: the message text
            metadata: optional dict (confidence, fiber, source, etc.)
        """
        self.total_turns += 1
        turn = (role, text, metadata or {})
        self.window.append(turn)

        # Extract entities for timeline
        entities = self._extract_entities(text)
        for e in entities:
            self.entity_timeline.append((e, self.total_turns))
            self.topic_counts[e] += 1

        # Feed into PPM context if available
        if self.lm:
            self.lm.update_context(list(text))

        # Compress if window exceeds size
        if len(self.window) > self.window_size:
            self._compress_oldest()

    def _compress_oldest(self):
        """
        Compress the oldest turns in the window into summary facts.
        """
        n_to_compress = len(self.window) - self.window_size
        if n_to_compress <= 0:
            return

        old_turns = self.window[:n_to_compress]
        self.window = self.window[n_to_compress:]
        self.compressions += 1

        for role, text, meta in old_turns:
            # Extract facts from the turn
            facts = self._extract_facts_from_turn(role, text, meta)
            self.summary_facts.extend(facts)

        # Deduplicate and trim
        self._deduplicate_facts()
        if len(self.summary_facts) > self.max_summary_facts:
            # Keep most recent facts
            self.summary_facts = self.summary_facts[-self.max_summary_facts:]

    def _extract_facts_from_turn(self, role: str, text: str,
                                  meta: Dict) -> List[Tuple[str, str, str]]:
        """
        Extract factual content from a turn for compression.
        """
        facts = []

        # If metadata contains structured answer info, use it
        if meta.get('answer') and meta.get('subject'):
            subj = meta['subject']
            rel = meta.get('relation', 'info')
            obj = meta['answer']
            facts.append((subj, rel, str(obj)))

        # Extract from system responses that state facts
        if role == 'system':
            # "The capital of France is Paris."
            m = re.search(r'[Tt]he\s+(\w+)\s+of\s+(\w+)\s+is\s+(.+?)\.', text)
            if m:
                facts.append((m.group(2), m.group(1), m.group(3).strip()))

            # "X is Y" patterns
            m = re.search(r'^(\w+(?:\s+\w+)?)\s+is\s+(.+?)\.', text)
            if m and m.group(1)[0].isupper():
                facts.append((m.group(1), 'is', m.group(2).strip()))

        # Extract from user questions (what they asked about)
        if role == 'user':
            m = re.search(r'(?:about|of)\s+([A-Z]\w+(?:\s+[A-Z]\w+)*)', text)
            if m:
                facts.append((m.group(1), 'asked_about', 'true'))

        return facts

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (capitalized words) from text."""
        # Simple: find capitalized words not at sentence start
        words = text.split()
        entities = []
        for i, w in enumerate(words):
            clean = w.strip('?.,!;:()"\'')
            if clean and clean[0].isupper() and len(clean) > 1:
                # Skip sentence-start words unless they look like names
                if i == 0 and clean.lower() in ('the', 'what', 'where', 'who',
                                                  'when', 'how', 'why', 'is',
                                                  'are', 'and', 'or', 'i'):
                    continue
                entities.append(clean)
        return entities

    def _deduplicate_facts(self):
        """Remove duplicate facts, keeping the latest."""
        seen = {}
        for fact in self.summary_facts:
            key = (fact[0].lower(), fact[1].lower())
            seen[key] = fact  # Later facts overwrite earlier
        self.summary_facts = list(seen.values())

    def get_context(self, max_chars: Optional[int] = None) -> str:
        """
        Get conversation context for the current query.

        Returns a text representation combining:
        1. Summary of compressed turns (if any)
        2. Recent turns verbatim

        Args:
            max_chars: optional character limit
        """
        parts = []

        # Summary section
        if self.summary_facts:
            fact_strs = [f"{s} {r}: {o}" for s, r, o in self.summary_facts
                        if r != 'asked_about']
            if fact_strs:
                parts.append("Known: " + "; ".join(fact_strs))

        # Recent turns
        for role, text, meta in self.window:
            prefix = "User" if role == 'user' else "System"
            parts.append(f"{prefix}: {text}")

        context = "\n".join(parts)

        if max_chars and len(context) > max_chars:
            # Trim from the beginning (keep recent)
            context = "..." + context[-(max_chars - 3):]

        return context

    def get_recent_topics(self, n: int = 5) -> List[str]:
        """Get the N most recently discussed topics."""
        recent = []
        seen = set()
        for entity, _ in reversed(self.entity_timeline):
            if entity not in seen:
                recent.append(entity)
                seen.add(entity)
            if len(recent) >= n:
                break
        return recent

    def get_topic_frequency(self) -> Dict[str, int]:
        """Get all topics sorted by mention frequency."""
        return dict(sorted(self.topic_counts.items(),
                          key=lambda x: -x[1]))

    def get_summary(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_turns': self.total_turns,
            'window_turns': len(self.window),
            'summary_facts': len(self.summary_facts),
            'compressions': self.compressions,
            'unique_topics': len(self.topic_counts),
            'entity_mentions': len(self.entity_timeline),
        }

    def search_memory(self, query: str) -> List[Dict]:
        """
        Search conversation memory for relevant information.

        Checks both summary facts and recent turns.
        """
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Search summary facts
        for s, r, o in self.summary_facts:
            if (query_lower in s.lower() or query_lower in o.lower() or
                    any(w in s.lower() or w in o.lower() for w in query_words)):
                results.append({
                    'source': 'summary',
                    'fact': (s, r, o),
                    'text': f"{s} {r}: {o}",
                })

        # Search recent turns
        for role, text, meta in self.window:
            if any(w in text.lower() for w in query_words):
                results.append({
                    'source': 'window',
                    'role': role,
                    'text': text,
                    'metadata': meta,
                })

        return results

    def persist_to_knowledge(self):
        """
        Push summary facts into the knowledge store for long-term retention.
        Only stores facts that aren't 'asked_about' markers.
        """
        if not self.knowledge:
            return 0

        stored = 0
        for s, r, o in self.summary_facts:
            if r != 'asked_about':
                self.knowledge.store_fact(s, r, o)
                stored += 1
        return stored

    def clear(self):
        """Clear all conversation memory."""
        self.window = []
        self.summary_facts = []
        self.topic_counts.clear()
        self.entity_timeline = []
        self.total_turns = 0
        self.compressions = 0
