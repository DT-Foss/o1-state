"""
Nexus Adapter — FOSS-KI as Agent Gamma in the Multi-Agent Network
===================================================================
Makes FOSS-KI the persistent memory and fact-checker for the Nexus.

Three roles:
  1. MEMORY: Auto-extract facts from every Nexus message → Hopfield storage
  2. FACT-CHECKER: Verify claims from Alpha/Bravo against stored knowledge
  3. TASK-TRACKER: Extract task mentions → queryable (Task, status, result)

FOSS-KI becomes Agent Gamma — the one that never forgets.
When transformers go down, Gamma has everything.

Usage:
    from core.nexus import NexusAdapter
    adapter = NexusAdapter(agent)
    adapter.watch()  # Background loop
    # or:
    adapter.process_new()  # One-shot check
"""

import json
import os
import re
import time
import datetime
from typing import Dict, Any, List, Optional


class NexusAdapter:
    """
    Connects FOSS-KI to the Nexus multi-agent coordination system.

    Watches /tmp/nexus.json, processes new messages, and writes
    Gamma's responses back.
    """

    def __init__(self, agent=None, nexus_path='/tmp/nexus.json',
                 agent_name='gamma'):
        """
        Args:
            agent: Agent instance (or None for standalone fact extraction)
            nexus_path: path to nexus.json
            agent_name: our identifier in the nexus
        """
        self.agent = agent
        self.nexus_path = nexus_path
        self.name = agent_name
        self._last_processed = 0  # Index of last processed message
        self._task_store = {}  # task_id → {status, description, result}
        self._fact_count = 0

        # Get knowledge store from agent or create standalone
        if agent:
            self.knowledge = agent.dialog.knowledge
            self.meta = agent.meta
        else:
            from .knowledge import KnowledgeStore
            from .metacognition import MetacognitionEngine
            self.knowledge = KnowledgeStore(dim=128)
            self.meta = MetacognitionEngine(self.knowledge)

    def read_nexus(self) -> Dict:
        """Read current nexus state."""
        if not os.path.exists(self.nexus_path):
            return {'messages': []}
        try:
            with open(self.nexus_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {'messages': []}

    def write_message(self, text: str, kind: str = 'chat'):
        """Write a message to nexus as Gamma."""
        data = self.read_nexus()
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        data['messages'].append({
            'from': self.name,
            'text': text,
            'ts': ts,
            'kind': kind,
        })
        with open(self.nexus_path, 'w') as f:
            json.dump(data, f)

    def process_new(self) -> List[Dict]:
        """Process all new messages since last check."""
        data = self.read_nexus()
        messages = data.get('messages', [])

        if self._last_processed >= len(messages):
            return []

        new_messages = messages[self._last_processed:]
        self._last_processed = len(messages)

        results = []
        for msg in new_messages:
            # Skip our own messages
            if msg.get('from') == self.name:
                continue

            result = self._process_message(msg)
            if result:
                results.append(result)

        return results

    def _process_message(self, msg: Dict) -> Optional[Dict]:
        """Process a single nexus message."""
        text = msg.get('text', '')
        sender = msg.get('from', 'unknown')

        result = {
            'sender': sender,
            'facts_extracted': 0,
            'tasks_found': 0,
            'response': None,
        }

        # 1. Extract facts from every message
        facts = self._extract_facts_from_message(text)
        if facts:
            self.knowledge.store_facts(facts)
            result['facts_extracted'] = len(facts)
            self._fact_count += len(facts)

        # 2. Extract task mentions
        tasks = self._extract_tasks(text)
        if tasks:
            for task_id, info in tasks.items():
                self._task_store[task_id] = info
            result['tasks_found'] = len(tasks)

        # 3. Check if addressed to us
        if self._is_addressed_to_us(text):
            # Extract the actual question
            question = self._extract_question(text)
            if question and self.agent:
                r = self.agent.process(question)
                result['response'] = r['response']
                # Auto-respond in nexus
                self.write_message(
                    f"[Gamma → {sender}] {r['response']}",
                    kind='answer'
                )

        # 4. Fact-check claims if we detect assertions
        claims = self._extract_claims(text)
        for claim in claims:
            verdict = self._verify_claim(claim)
            if verdict:
                result.setdefault('verifications', []).append(verdict)

        return result

    def _extract_facts_from_message(self, text: str) -> List[tuple]:
        """Extract (S, R, O) facts from a nexus message."""
        facts = []

        # Pattern: "X is Y" / "X has Y" / "X = Y"
        for pattern in [
            r'(\w[\w\s]+?)\s+(?:is|are|was|were)\s+(?:a\s+)?(\w[\w\s]+?)(?:\.|,|$)',
            r'(\w[\w\s]+?)\s+(?:has|have|had)\s+(?:a\s+)?(\w[\w\s]+?)(?:\.|,|$)',
        ]:
            for m in re.finditer(pattern, text):
                s = m.group(1).strip()
                o = m.group(2).strip()
                if len(s) > 2 and len(o) > 2 and len(s) < 50 and len(o) < 50:
                    facts.append((s, 'description', o))

        # Pattern: "T89/T90/S14 DONE/COMPLETE/FAILED"
        for m in re.finditer(r'(T\d+|S\d+)\s*(?::|is|was)?\s*(DONE|COMPLETE|FAILED|RUNNING|NEGATIVE|POSITIV)',
                             text, re.I):
            task_id = m.group(1)
            status = m.group(2).upper()
            facts.append((task_id, 'status', status))

        # Pattern: "X: Y%" or "accuracy: Y%"
        for m in re.finditer(r'(\w[\w\s]+?):\s+(\d+(?:\.\d+)?%)', text):
            metric = m.group(1).strip()
            value = m.group(2)
            facts.append((metric, 'value', value))

        # Pattern: "X → Y" or "X results: Y"
        for m in re.finditer(r'(\w[\w\s]+?)\s*(?:→|results?:)\s*(\w[\w\s]+?)(?:\.|,|$)', text):
            s = m.group(1).strip()
            o = m.group(2).strip()
            if len(s) > 1 and len(o) > 1:
                facts.append((s, 'result', o))

        return facts

    def _extract_tasks(self, text: str) -> Dict[str, Dict]:
        """Extract task mentions from text."""
        tasks = {}

        # Pattern: T\d+ or S\d+ with status
        for m in re.finditer(
            r'(T\d+|S\d+)\s*(?:\(([^)]+)\))?\s*(?::|—|-)?\s*(?:(\w[\w\s]{3,50}))?',
            text
        ):
            task_id = m.group(1)
            status_hint = m.group(2) or ''
            description = (m.group(3) or '').strip()

            # Detect status from surrounding text
            context = text[max(0, m.start()-20):min(len(text), m.end()+50)]
            if any(w in context.upper() for w in ('DONE', 'COMPLETE', '100%', '✓')):
                status = 'done'
            elif any(w in context.upper() for w in ('FAIL', 'NEGATIVE', 'BROKEN', '✗')):
                status = 'failed'
            elif any(w in context.upper() for w in ('RUNNING', 'STARTED', 'LÄUFT')):
                status = 'running'
            else:
                status = 'mentioned'

            tasks[task_id] = {
                'status': status,
                'description': description or status_hint,
                'last_seen': datetime.datetime.now().isoformat(),
            }

        return tasks

    def _is_addressed_to_us(self, text: str) -> bool:
        """Check if message is directed at us."""
        text_lower = text.lower()
        return any(addr in text_lower for addr in [
            'gamma', 'foss-ki', 'foss ki', '@gamma',
            'memory check', 'fact check', 'verify:',
        ])

    def _extract_question(self, text: str) -> Optional[str]:
        """Extract the question part from a message addressed to us."""
        # Remove addressing prefix
        cleaned = re.sub(
            r'(?:@?gamma|foss-?ki)\s*[,:]\s*', '', text, flags=re.I
        ).strip()
        return cleaned if cleaned else None

    def _extract_claims(self, text: str) -> List[Dict]:
        """Extract verifiable claims from text."""
        claims = []

        # "X is the capital of Y"
        for m in re.finditer(r'(\w+)\s+is\s+the\s+(\w+)\s+of\s+(\w[\w\s]+)', text):
            claims.append({
                'subject': m.group(3).strip(),
                'relation': m.group(2).strip(),
                'object': m.group(1).strip(),
            })

        return claims

    def _verify_claim(self, claim: Dict) -> Optional[Dict]:
        """Verify a claim against stored knowledge."""
        s = claim['subject']
        r = claim['relation']
        o_claimed = claim['object']

        result = self.knowledge.query(s, r)
        if result['confidence_level'] in ('HIGH', 'MEDIUM'):
            actual = result['fact'][2]
            matches = actual.lower() == o_claimed.lower()
            return {
                'claim': f"{o_claimed} is the {r} of {s}",
                'verified': matches,
                'actual': actual if not matches else None,
            }

        return None

    def status_report(self) -> str:
        """Generate a status report for nexus."""
        facts_total = self.knowledge.n_facts if hasattr(self.knowledge, 'n_facts') else len(self.knowledge.facts)
        tasks_total = len(self._task_store)
        tasks_done = sum(1 for t in self._task_store.values() if t['status'] == 'done')
        tasks_open = tasks_total - tasks_done

        report = (
            f"[Gamma Status Report]\n"
            f"  Facts in memory: {facts_total}\n"
            f"  Facts extracted this session: {self._fact_count}\n"
            f"  Tasks tracked: {tasks_total} "
            f"({tasks_done} done, {tasks_open} open)\n"
        )

        if self._task_store:
            report += "\n  Open tasks:\n"
            for tid, info in sorted(self._task_store.items()):
                if info['status'] != 'done':
                    report += f"    {tid}: {info['status']} — {info['description'][:60]}\n"

        return report

    def query_tasks(self, query: str = '') -> List[Dict]:
        """Query tracked tasks."""
        if not query:
            return list(self._task_store.items())

        query_lower = query.lower()
        results = []
        for tid, info in self._task_store.items():
            if (query_lower in tid.lower() or
                    query_lower in info.get('description', '').lower() or
                    query_lower in info.get('status', '')):
                results.append((tid, info))
        return results

    def watch(self, interval: float = 2.0, max_iterations: int = 0):
        """
        Watch nexus for new messages in a loop.

        Args:
            interval: seconds between checks
            max_iterations: 0 = infinite
        """
        print(f"[Gamma] Watching {self.nexus_path} (interval={interval}s)")
        iteration = 0

        while True:
            results = self.process_new()
            for r in results:
                if r['facts_extracted'] > 0:
                    print(f"  [Gamma] Extracted {r['facts_extracted']} facts "
                          f"from {r['sender']}")
                if r['tasks_found'] > 0:
                    print(f"  [Gamma] Found {r['tasks_found']} tasks "
                          f"from {r['sender']}")
                if r.get('response'):
                    print(f"  [Gamma → {r['sender']}] {r['response'][:100]}")

            iteration += 1
            if max_iterations and iteration >= max_iterations:
                break

            time.sleep(interval)
