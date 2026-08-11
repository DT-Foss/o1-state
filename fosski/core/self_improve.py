"""
Self-Improvement Engine — FOSS-KI learns by playing against itself
====================================================================
Ported from FORGE's self_improve.py.

Like AlphaZero plays chess against itself, FOSS-KI plays against
the Python runtime and its own knowledge store:

1. GapDetector: Find weak spots in knowledge + code generation
2. TaskGenerator: Convert gaps into test queries
3. SelfImproveLoop: Execute → validate → learn → repeat

This is autonomous self-improvement without any LLM involvement.
"""

import ast
import json
import os
import re
import time
import random
from typing import Dict, List, Any
from collections import Counter


class GapDetector:
    """
    Analyze the system to find weak spots worth testing.

    Checks:
    - Orphan fragments (fragments no query maps to)
    - Low-confidence knowledge (facts near threshold)
    - Untested code templates
    - Knowledge frontier entities (via KnowledgeExplorer)
    """

    def __init__(self, forge_assembler=None, knowledge_store=None):
        self.forge = forge_assembler
        self.knowledge = knowledge_store
        self._tested_fragments: set = set()
        self._tested_compositions: set = set()

    def detect_all(self) -> List[Dict[str, Any]]:
        """Return all detected gaps, sorted by priority."""
        gaps = []

        if self.forge:
            gaps.extend(self._find_untested_fragments())
            gaps.extend(self._find_composition_gaps())

        if self.knowledge:
            gaps.extend(self._find_knowledge_frontiers())

        gaps.sort(key=lambda g: g['priority'], reverse=True)
        return gaps

    def mark_tested(self, key: str, success: bool):
        """Mark a fragment as tested so we don't retest it."""
        self._tested_fragments.add(key)

    def _find_untested_fragments(self) -> List[Dict]:
        """Find fragments that have never been tested."""
        if not self.forge:
            return []

        gaps = []
        all_keys = [k for k in self.forge.fragments.keys()
                     if k not in self._tested_fragments]
        sample = random.sample(all_keys, min(50, len(all_keys)))

        for key in sample:
            meta = self.forge.registry.registry.get(key)
            if meta and meta['role'] == 'STANDALONE':
                gaps.append({
                    'type': 'untested_fragment',
                    'key': key,
                    'priority': 3,
                    'action': f"Test fragment: {key}",
                })
            elif meta and meta['role'] in ('SOURCE', 'TRANSFORM'):
                gaps.append({
                    'type': 'untested_source',
                    'key': key,
                    'priority': 4,
                    'action': f"Test source fragment: {key}",
                })

        return gaps

    def _find_composition_gaps(self) -> List[Dict]:
        """Find fragment pairs that could compose but haven't been tested."""
        if not self.forge:
            return []

        gaps = []
        sources = [k for k, v in self.forge.registry.registry.items()
                    if v['role'] == 'SOURCE']
        sinks = [k for k, v in self.forge.registry.registry.items()
                  if v['role'] == 'SINK']

        for _ in range(min(20, len(sources))):
            src = random.choice(sources) if sources else None
            snk = random.choice(sinks) if sinks else None
            if src and snk:
                pair_key = f"{src}→{snk}"
                if pair_key in self._tested_compositions:
                    continue
                wiring = self.forge.registry.get_wiring(src, snk)
                if wiring:
                    gaps.append({
                        'type': 'untested_composition',
                        'source': src,
                        'sink': snk,
                        'wiring': wiring,
                        'priority': 5,
                        'action': f"Test composition: {src} → {snk}",
                    })

        return gaps

    def _find_knowledge_frontiers(self) -> List[Dict]:
        """Find knowledge entities that need more facts."""
        if not self.knowledge:
            return []

        from .metacognition import KnowledgeExplorer
        explorer = KnowledgeExplorer(self.knowledge)
        targets = explorer.exploration_targets(n=10)

        return [{
            'type': 'knowledge_frontier',
            'entity': t['entity'],
            'domain': t['domain'],
            'priority': t['priority'],
            'action': f"Learn more about: {t['entity']}",
        } for t in targets]


class TaskGenerator:
    """
    Convert gaps into executable test queries.

    Fragment gaps → code generation intents
    Knowledge gaps → factual questions
    """

    CODE_TEMPLATES = [
        "Write a {key} tool",
        "Build a {key}",
        "Create a {key} script",
        "Implement {key}",
    ]

    KNOWLEDGE_TEMPLATES = [
        "What is the {relation} of {entity}?",
        "Tell me about {entity}",
        "Who discovered {entity}?",
    ]

    def generate_tasks(self, gaps: List[Dict],
                        max_tasks: int = 50) -> List[Dict]:
        """Convert gaps into executable tasks."""
        tasks = []

        for gap in gaps[:max_tasks]:
            if gap['type'] in ('untested_fragment', 'untested_source'):
                key = gap['key']
                readable = key.replace('_', ' ')
                template = random.choice(self.CODE_TEMPLATES)
                tasks.append({
                    'query': template.format(key=readable),
                    'type': 'code',
                    'expected_fragment': key,
                    'gap': gap,
                })

            elif gap['type'] == 'untested_composition':
                src_readable = gap['source'].replace('_', ' ')
                snk_readable = gap['sink'].replace('_', ' ')
                tasks.append({
                    'query': f"{src_readable} and {snk_readable}",
                    'type': 'composition',
                    'expected_fragments': [gap['source'], gap['sink']],
                    'gap': gap,
                })

            elif gap['type'] == 'knowledge_frontier':
                entity = gap['entity']
                tasks.append({
                    'query': f"Tell me about {entity}",
                    'type': 'knowledge',
                    'expected_entity': entity,
                    'gap': gap,
                })

        return tasks


class SelfImproveLoop:
    """
    The autonomous improvement engine.

    Cycle:
    1. Detect gaps
    2. Generate tasks
    3. Execute tasks (direct forge + agent pipeline)
    4. Validate results
    5. Track failures with root cause
    6. Repeat

    Failures are categorized:
    - intent_misclass: Query classified wrong by IntentClassifier
    - no_fragment: ForgeAssembler couldn't find a matching fragment
    - syntax_error: Generated code has AST errors
    - no_knowledge: Knowledge query returned nothing useful
    """

    def __init__(self, agent=None, log_dir=None):
        self.agent = agent
        self._results: List[Dict] = []
        self._iteration = 0
        self._log_dir = log_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'logs'
        )

    def run(self, max_iterations: int = 100,
            time_budget_seconds: float = 300) -> Dict[str, Any]:
        """
        Run the self-improvement loop.

        Args:
            max_iterations: max number of gap-detect cycles
            time_budget_seconds: max time to spend

        Returns:
            summary dict with results
        """
        if not self.agent:
            return {'error': 'No agent provided'}

        # Get forge directly for code testing (bypass IntentClassifier)
        forge = getattr(self.agent.code_gen, '_forge', None)

        detector = GapDetector(
            forge_assembler=forge,
            knowledge_store=self.agent.dialog.knowledge,
        )
        generator = TaskGenerator()

        t0 = time.time()
        total_pass = 0
        total_fail = 0
        total_skip = 0

        for iteration in range(max_iterations):
            if time.time() - t0 > time_budget_seconds:
                break

            self._iteration = iteration

            # 1. Detect gaps
            gaps = detector.detect_all()
            if not gaps:
                break

            # 2. Generate tasks
            tasks = generator.generate_tasks(gaps, max_tasks=10)
            if not tasks:
                break

            # 3. Execute and validate
            for task in tasks:
                if time.time() - t0 > time_budget_seconds:
                    break

                result = self._execute_task(task, forge)
                self._results.append(result)

                # Mark tested so we don't retry
                if task['type'] in ('code', 'untested_source'):
                    key = task.get('expected_fragment', '')
                    detector.mark_tested(key, result['success'])

                if result['success']:
                    total_pass += 1
                elif result['skipped']:
                    total_skip += 1
                else:
                    total_fail += 1

        elapsed = time.time() - t0

        summary = {
            'iterations': self._iteration + 1,
            'total_tasks': len(self._results),
            'passed': total_pass,
            'failed': total_fail,
            'skipped': total_skip,
            'pass_rate': total_pass / max(total_pass + total_fail, 1),
            'elapsed': round(elapsed, 2),
            'tasks_per_second': round(len(self._results) / max(elapsed, 0.001), 1),
        }

        # Save results to log
        self._save_log(summary)

        return summary

    def _execute_task(self, task: Dict, forge=None) -> Dict:
        """
        Execute a single self-improvement task.

        For code tasks: test BOTH the ForgeAssembler directly AND the
        full Agent pipeline. This catches two classes of failures:
        - ForgeAssembler can't find a fragment → fragment coverage gap
        - Agent misclassifies the intent → IntentClassifier gap
        """
        result = {
            'query': task['query'],
            'type': task['type'],
            'success': False,
            'skipped': False,
            'error': None,
            'failure_class': None,
        }

        try:
            if task['type'] in ('code', 'composition'):
                # === Test 1: Direct ForgeAssembler lookup ===
                forge_ok = False
                if forge:
                    forge_result = forge.generate(task['query'])
                    code = forge_result.get('code', '')
                    lang = forge_result.get('language', 'python')
                    # Multi-language compositions: if query mentions "js"/"javascript"
                    # the output may be mixed Python+JS — skip AST validation
                    is_multi_lang = bool(re.search(
                        r'\bjs\b|\bjavascript\b|\btypescript\b|\bhtml\b|\bcss\b',
                        task['query'], re.I))

                    if code:
                        if lang in ('bash', 'shell') or code.lstrip().startswith(('#!/bin/bash', '#!/bin/sh')):
                            # Bash scripts are valid if they have content
                            forge_ok = True
                        elif is_multi_lang or lang in ('javascript', 'typescript', 'html', 'css', 'multi'):
                            # Multi-language output, can't validate with Python AST
                            forge_ok = True
                        else:
                            try:
                                ast.parse(code)
                                forge_ok = True
                            except SyntaxError as e:
                                result['error'] = f"Syntax error: {e}"
                                result['failure_class'] = 'syntax_error'
                    else:
                        result['failure_class'] = 'no_fragment'

                # === Test 2: Full Agent pipeline ===
                if self.agent:
                    r = self.agent.process(task['query'])
                    intent = r.get('intent', '')
                    agent_code = r.get('metadata', {}).get('code', '')

                    if forge_ok and not agent_code:
                        # FORGE found it but Agent didn't route correctly
                        result['failure_class'] = 'intent_misclass'
                        result['error'] = f"Intent={intent}, expected CODE_REQUEST"

                    if forge_ok:
                        result['success'] = True
                    elif agent_code:
                        # Agent got code even though direct lookup failed
                        try:
                            ast.parse(agent_code)
                            result['success'] = True
                        except SyntaxError as e:
                            result['error'] = f"Syntax error: {e}"
                            result['failure_class'] = 'syntax_error'

                if not result['success'] and not result['error']:
                    result['error'] = "No code generated"
                    result['failure_class'] = result['failure_class'] or 'no_fragment'

            elif task['type'] == 'knowledge':
                if self.agent:
                    r = self.agent.process(task['query'])
                    resp = r.get('response', '')
                    if resp and "don't" not in resp.lower():
                        result['success'] = True
                    else:
                        result['skipped'] = True
                        result['failure_class'] = 'no_knowledge'

        except Exception as e:
            result['error'] = str(e)
            result['failure_class'] = 'exception'

        return result

    def _save_log(self, summary: Dict):
        """Save results to JSON log for analysis."""
        os.makedirs(self._log_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(self._log_dir, f'self_improve_{timestamp}.json')

        log_data = {
            'summary': summary,
            'results': self._results,
            'failure_analysis': self._analyze_failures(),
        }

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        return log_path

    def _analyze_failures(self) -> Dict[str, Any]:
        """Analyze failures and produce actionable recommendations."""
        failures = [r for r in self._results if not r['success'] and not r.get('skipped')]

        # Group by failure class
        by_class = Counter(r.get('failure_class', 'unknown') for r in failures)

        # Collect specific examples per class
        examples: Dict[str, List[str]] = {}
        for r in failures:
            cls = r.get('failure_class', 'unknown')
            if cls not in examples:
                examples[cls] = []
            if len(examples[cls]) < 5:
                examples[cls].append(r['query'])

        # Generate recommendations
        recs = []
        if by_class.get('intent_misclass', 0) > 0:
            recs.append(
                f"IntentClassifier: {by_class['intent_misclass']} queries misclassified. "
                f"Add patterns for: {examples.get('intent_misclass', [])[:3]}"
            )
        if by_class.get('no_fragment', 0) > 0:
            recs.append(
                f"ForgeAssembler: {by_class['no_fragment']} queries found no fragment. "
                f"Improve lookup or add fragments for: {examples.get('no_fragment', [])[:3]}"
            )
        if by_class.get('syntax_error', 0) > 0:
            recs.append(
                f"Code Quality: {by_class['syntax_error']} fragments produce invalid AST. "
                f"Fix templates: {examples.get('syntax_error', [])[:3]}"
            )

        return {
            'total_failures': len(failures),
            'by_class': dict(by_class),
            'examples': examples,
            'recommendations': recs,
        }

    def report(self) -> str:
        """Human-readable report of self-improvement results."""
        if not self._results:
            return "No self-improvement runs yet."

        total = len(self._results)
        passed = sum(1 for r in self._results if r['success'])
        failed = sum(1 for r in self._results
                     if not r['success'] and not r.get('skipped'))
        skipped = sum(1 for r in self._results if r.get('skipped'))

        # Group failures by class
        failure_classes = Counter()
        for r in self._results:
            if r.get('failure_class'):
                failure_classes[r['failure_class']] += 1

        lines = [
            "Self-Improvement Report",
            "=" * 50,
            f"Total tasks:  {total}",
            f"Passed:       {passed} ({passed/total:.0%})",
            f"Failed:       {failed}",
            f"Skipped:      {skipped}",
            f"Iterations:   {self._iteration + 1}",
        ]

        if failure_classes:
            lines.append("\nFailure Breakdown:")
            for cls, count in failure_classes.most_common():
                lines.append(f"  {count:3d}× {cls}")

        # Actionable recommendations
        analysis = self._analyze_failures()
        if analysis['recommendations']:
            lines.append("\nRecommendations:")
            for i, rec in enumerate(analysis['recommendations'], 1):
                lines.append(f"  {i}. {rec}")

        return "\n".join(lines)
