"""
Apprentice — Meta-Learning from LLM Reasoning Patterns
========================================================
FOSS-KI learns HOW to think, not WHAT to answer.

This is NOT knowledge distillation. This is PROCESS extraction:
  - How does a good reasoner decompose a problem?
  - What patterns does good code follow?
  - How are explanations structured?
  - What's the reasoning chain for a math proof?

The Apprentice watches LLM responses (via Nexus or direct),
extracts structural patterns, and feeds them into FOSS-KI's
subsystems to improve their capabilities.

Pipeline:
  1. Observe: Get LLM response to a query
  2. Decompose: Split into structural elements
  3. Extract: Pull out reusable patterns
  4. Integrate: Feed patterns into the right subsystem
  5. Validate: Test that the new pattern actually helps

What gets learned:
  - Reasoning patterns → ChainOfThought decomposition rules
  - Code patterns → CodeGenerator templates + composition rules
  - Explanation patterns → TextGenerator templates
  - Problem decomposition → InstructionParser strategies
  - Answer structure → Formulierer response templates
"""

import re
import json
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter


class PatternType:
    """Types of extractable patterns."""
    REASONING = 'reasoning'      # Step-by-step logic
    CODE_STRUCTURE = 'code'      # Code organization patterns
    DECOMPOSITION = 'decomp'     # How to break problems apart
    EXPLANATION = 'explanation'  # How to explain things
    ERROR_HANDLING = 'error'     # How to handle edge cases
    COMPOSITION = 'composition'  # How to combine sub-solutions


class ExtractedPattern:
    """A reusable pattern extracted from an LLM response."""

    def __init__(self, pattern_type: str, template: str,
                 trigger: str, confidence: float = 1.0,
                 source: str = 'llm'):
        self.pattern_type = pattern_type
        self.template = template    # The generalized pattern
        self.trigger = trigger      # When to apply this pattern
        self.confidence = confidence
        self.source = source
        self.use_count = 0
        self.success_count = 0
        self.timestamp = time.time()

    def to_dict(self):
        return {
            'type': self.pattern_type,
            'template': self.template,
            'trigger': self.trigger,
            'confidence': self.confidence,
            'source': self.source,
            'use_count': self.use_count,
            'success_count': self.success_count,
        }

    @classmethod
    def from_dict(cls, d):
        p = cls(d['type'], d['template'], d['trigger'],
                d.get('confidence', 1.0), d.get('source', 'llm'))
        p.use_count = d.get('use_count', 0)
        p.success_count = d.get('success_count', 0)
        return p


class ReasoningExtractor:
    """
    Extract reasoning patterns from LLM responses.

    Detects:
    - Numbered steps ("1. First... 2. Then... 3. Finally...")
    - Conditional logic ("If X then Y, otherwise Z")
    - Analogies ("X is like Y because Z")
    - Decomposition ("This breaks down into A, B, and C")
    - Conclusions ("Therefore...", "This means...")
    """

    # Markers that indicate reasoning structure
    STEP_MARKERS = [
        r'(?:^|\n)\s*(\d+)\.\s+(.+)',              # "1. Do X"
        r'(?:^|\n)\s*(?:step|Step)\s+(\d+)[:\s]+(.+)',  # "Step 1: X"
        r'(?:^|\n)\s*(?:first|second|third|then|next|finally)[,:\s]+(.+)',
    ]

    LOGIC_MARKERS = [
        r'(?:if|when)\s+(.+?),?\s+(?:then|→)\s+(.+)',
        r'(?:because|since)\s+(.+?),\s+(.+)',
        r'(?:therefore|thus|hence|so)\s+(.+)',
        r'(.+?)\s+(?:implies|means|suggests)\s+(.+)',
    ]

    DECOMP_MARKERS = [
        r'(?:this|the problem|it)\s+(?:breaks down|decomposes|splits)\s+into\s+(.+)',
        r'(?:there are|we need)\s+(\d+)\s+(?:parts|steps|components|aspects)',
        r'(?:consider|let\'s look at)\s+(.+?)\s+(?:separately|individually)',
    ]

    def extract(self, text: str) -> List[ExtractedPattern]:
        """Extract reasoning patterns from text."""
        patterns = []

        # Extract step-by-step patterns
        steps = self._extract_steps(text)
        if steps:
            template = self._generalize_steps(steps)
            patterns.append(ExtractedPattern(
                PatternType.REASONING,
                template=template,
                trigger='multi_step_problem',
            ))

        # Extract conditional logic
        conditionals = self._extract_conditionals(text)
        for cond in conditionals:
            patterns.append(ExtractedPattern(
                PatternType.REASONING,
                template=cond,
                trigger='conditional_reasoning',
            ))

        # Extract decomposition patterns
        decomps = self._extract_decompositions(text)
        for decomp in decomps:
            patterns.append(ExtractedPattern(
                PatternType.DECOMPOSITION,
                template=decomp,
                trigger='complex_problem',
            ))

        return patterns

    def _extract_steps(self, text: str) -> List[str]:
        """Extract numbered steps from response."""
        steps = []
        for m in re.finditer(r'(?:^|\n)\s*(\d+)\.\s+(.+?)(?=\n\s*\d+\.|\n\n|$)',
                             text, re.DOTALL):
            step_text = m.group(2).strip()
            if len(step_text) > 10:
                steps.append(step_text)
        return steps

    def _generalize_steps(self, steps: List[str]) -> str:
        """
        Generalize specific steps into a reusable template.

        "1. Find the capital of France → Paris"
        "2. Look up the language of France → French"
        becomes: "1. Find {relation} of {entity} 2. Look up {relation2} of {entity}"
        """
        generalized = []
        for i, step in enumerate(steps, 1):
            # Replace specific entities with placeholders
            gen = re.sub(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
                         '{entity}', step)
            # Replace quoted strings
            gen = re.sub(r'"[^"]*"', '{value}', gen)
            # Replace numbers
            gen = re.sub(r'\b\d+\b', '{N}', gen)
            generalized.append(f"{i}. {gen}")
        return '\n'.join(generalized)

    def _extract_conditionals(self, text: str) -> List[str]:
        """Extract if-then-else patterns."""
        patterns = []
        for marker in self.LOGIC_MARKERS:
            for m in re.finditer(marker, text, re.I):
                patterns.append(m.group(0).strip())
        return patterns[:5]  # Cap at 5

    def _extract_decompositions(self, text: str) -> List[str]:
        """Extract problem decomposition patterns."""
        patterns = []
        for marker in self.DECOMP_MARKERS:
            for m in re.finditer(marker, text, re.I):
                patterns.append(m.group(0).strip())
        return patterns[:3]


class CodePatternExtractor:
    """
    Extract code patterns from LLM-generated code.

    Learns:
    - Function signatures and structures
    - Error handling patterns
    - Common idioms (list comprehensions, generators, decorators)
    - Code organization (imports, class structure, main guard)
    - Composition patterns (how functions call each other)
    """

    def extract(self, code: str) -> List[ExtractedPattern]:
        """Extract reusable code patterns."""
        patterns = []

        # Function structure patterns
        for func in self._extract_functions(code):
            patterns.append(ExtractedPattern(
                PatternType.CODE_STRUCTURE,
                template=func['template'],
                trigger=func['trigger'],
            ))

        # Error handling patterns
        for eh in self._extract_error_handling(code):
            patterns.append(ExtractedPattern(
                PatternType.ERROR_HANDLING,
                template=eh,
                trigger='error_handling',
            ))

        # Composition patterns (how functions work together)
        comp = self._extract_composition(code)
        if comp:
            patterns.append(ExtractedPattern(
                PatternType.COMPOSITION,
                template=comp,
                trigger='multi_function',
            ))

        return patterns

    def _extract_functions(self, code: str) -> List[Dict]:
        """Extract function templates with generalized signatures."""
        functions = []
        for m in re.finditer(
            r'def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*(\w+))?\s*:',
            code
        ):
            name = m.group(1)
            params = m.group(2)
            ret = m.group(3)

            # Get the function body (rough extraction)
            start = m.end()
            body_lines = []
            for line in code[start:].split('\n'):
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    break
                if line.strip():
                    body_lines.append(line)

            # Generalize: what kind of function is this?
            trigger = 'function'
            if 'for ' in '\n'.join(body_lines):
                trigger = 'loop_function'
            if 'if ' in '\n'.join(body_lines):
                trigger = 'conditional_function'
            if 'return [' in '\n'.join(body_lines):
                trigger = 'list_builder'
            if 'yield ' in '\n'.join(body_lines):
                trigger = 'generator'
            if name.startswith('test_'):
                trigger = 'test_function'

            # Template: generalize the function
            param_count = len([p for p in params.split(',') if p.strip()])
            template = f"def {{name}}({params})"
            if ret:
                template += f" -> {ret}"
            template += f":\n    # {trigger} with {param_count} params"

            functions.append({
                'template': template,
                'trigger': trigger,
                'name': name,
            })

        return functions

    def _extract_error_handling(self, code: str) -> List[str]:
        """Extract try/except patterns."""
        patterns = []
        for m in re.finditer(
            r'try:\s*\n(.*?)except\s+(\w+(?:\s+as\s+\w+)?)\s*:\s*\n(.*?)(?=\n\S|\Z)',
            code, re.DOTALL
        ):
            exc_type = m.group(2).split()[0]
            # Generalize
            patterns.append(f"try: {{operation}} except {exc_type}: {{fallback}}")
        return patterns

    def _extract_composition(self, code: str) -> Optional[str]:
        """Extract how functions call each other (call graph skeleton)."""
        # Find all function definitions
        funcs = re.findall(r'def\s+(\w+)\s*\(', code)
        if len(funcs) < 2:
            return None

        # Find call relationships
        calls = []
        for func in funcs:
            # Find what this function calls
            func_start = code.find(f'def {func}(')
            if func_start < 0:
                continue
            # Rough body extraction
            body_start = code.find(':', func_start) + 1
            next_def = code.find('\ndef ', body_start)
            body = code[body_start:next_def] if next_def > 0 else code[body_start:]

            for other in funcs:
                if other != func and f'{other}(' in body:
                    calls.append(f"{func} → {other}")

        if calls:
            return "Composition: " + ', '.join(calls)
        return None


class ExplanationExtractor:
    """
    Extract explanation patterns from LLM text.

    Learns:
    - How to start an explanation
    - How to use analogies
    - How to structure depth (overview → detail → summary)
    - How to use examples
    """

    def extract(self, text: str) -> List[ExtractedPattern]:
        """Extract explanation structure patterns."""
        patterns = []

        # Detect analogy usage
        analogies = re.findall(
            r'(?:like|similar to|think of it as|imagine)\s+(.+?)(?:\.|,)',
            text, re.I
        )
        for analogy in analogies[:3]:
            patterns.append(ExtractedPattern(
                PatternType.EXPLANATION,
                template=f"Think of it as {analogy}",
                trigger='analogy',
            ))

        # Detect example usage
        examples = re.findall(
            r'(?:for example|e\.g\.|for instance|consider)\s*[,:]?\s*(.+?)(?:\.|$)',
            text, re.I
        )
        for ex in examples[:3]:
            patterns.append(ExtractedPattern(
                PatternType.EXPLANATION,
                template=f"For example, {{example}}",
                trigger='example',
            ))

        # Detect overview-detail structure
        if re.search(r'(?:in short|to summarize|in summary|the key)', text, re.I):
            patterns.append(ExtractedPattern(
                PatternType.EXPLANATION,
                template="overview → detail → summary",
                trigger='structured_explanation',
            ))

        return patterns


class PatternStore:
    """
    Persistent storage for learned patterns.

    Patterns are stored by type and ranked by success rate.
    Old unused patterns decay over time.
    """

    def __init__(self, store_path=None):
        if store_path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            store_path = os.path.join(base, 'data', 'learned_patterns.json')
        self.store_path = store_path
        self.patterns = self._load()

    def _load(self) -> Dict[str, List[ExtractedPattern]]:
        """Load patterns from disk."""
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, 'r') as f:
                    data = json.load(f)
                result = {}
                for ptype, plist in data.items():
                    result[ptype] = [ExtractedPattern.from_dict(p) for p in plist]
                return result
            except (json.JSONDecodeError, KeyError):
                pass
        return {}

    def save(self):
        """Persist patterns to disk."""
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        data = {}
        for ptype, plist in self.patterns.items():
            data[ptype] = [p.to_dict() for p in plist]
        with open(self.store_path, 'w') as f:
            json.dump(data, f, indent=2)

    def add(self, pattern: ExtractedPattern):
        """Add a pattern, avoiding duplicates."""
        ptype = pattern.pattern_type
        if ptype not in self.patterns:
            self.patterns[ptype] = []

        # Check for duplicate (same trigger + similar template)
        for existing in self.patterns[ptype]:
            if (existing.trigger == pattern.trigger and
                    self._similarity(existing.template, pattern.template) > 0.8):
                # Update confidence instead of adding duplicate
                existing.confidence = max(existing.confidence,
                                          pattern.confidence)
                existing.use_count += 1
                return

        self.patterns[ptype].append(pattern)

    def get(self, pattern_type: str, trigger: str = None,
            top_n: int = 5) -> List[ExtractedPattern]:
        """Get best patterns for a type/trigger."""
        candidates = self.patterns.get(pattern_type, [])
        if trigger:
            candidates = [p for p in candidates
                          if p.trigger == trigger or trigger in p.trigger]

        # Rank by success rate, then confidence
        def score(p):
            if p.use_count > 0:
                return p.success_count / p.use_count
            return p.confidence
        candidates.sort(key=score, reverse=True)
        return candidates[:top_n]

    def record_use(self, pattern: ExtractedPattern, success: bool):
        """Record that a pattern was used and whether it helped."""
        pattern.use_count += 1
        if success:
            pattern.success_count += 1

    def _similarity(self, a: str, b: str) -> float:
        """Simple word-overlap similarity."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        overlap = len(words_a & words_b)
        return overlap / max(len(words_a), len(words_b))

    @property
    def total_patterns(self) -> int:
        return sum(len(v) for v in self.patterns.values())

    def stats(self) -> Dict[str, int]:
        """Pattern counts by type."""
        return {ptype: len(plist) for ptype, plist in self.patterns.items()}


class Apprentice:
    """
    The Meta-Learning Controller.

    Orchestrates the full learning cycle:
    1. Pose a question/task to an LLM (via Nexus or API)
    2. Receive the response
    3. Extract structural patterns (NOT content)
    4. Store reusable patterns
    5. Integrate into FOSS-KI subsystems

    The Apprentice makes FOSS-KI SMARTER over time
    without becoming dependent on the LLM.
    Once a pattern is learned, the LLM is no longer needed for it.
    """

    def __init__(self, pattern_store=None):
        self.store = pattern_store or PatternStore()
        self.reasoning_ext = ReasoningExtractor()
        self.code_ext = CodePatternExtractor()
        self.explanation_ext = ExplanationExtractor()
        self._learning_log = []

    def learn_from_response(self, query: str, response: str,
                            response_type: str = 'auto') -> Dict[str, Any]:
        """
        Core learning method. Takes an LLM response and extracts patterns.

        Args:
            query: what was asked
            response: the LLM's full response
            response_type: 'auto', 'code', 'reasoning', 'explanation'

        Returns:
            dict with:
                patterns_extracted: int
                by_type: dict of counts
                details: list of extracted patterns
        """
        if response_type == 'auto':
            response_type = self._detect_type(query, response)

        all_patterns = []

        # Always try reasoning extraction
        reasoning_patterns = self.reasoning_ext.extract(response)
        all_patterns.extend(reasoning_patterns)

        # Type-specific extraction
        if response_type == 'code' or '```' in response or 'def ' in response:
            # Extract code from markdown blocks or raw
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```',
                                      response, re.DOTALL)
            if not code_blocks:
                code_blocks = [response]
            for code in code_blocks:
                code_patterns = self.code_ext.extract(code)
                all_patterns.extend(code_patterns)

        if response_type in ('explanation', 'auto'):
            explanation_patterns = self.explanation_ext.extract(response)
            all_patterns.extend(explanation_patterns)

        # Store all patterns
        for p in all_patterns:
            self.store.add(p)

        # Save
        if all_patterns:
            self.store.save()

        # Log
        result = {
            'patterns_extracted': len(all_patterns),
            'by_type': Counter(p.pattern_type for p in all_patterns),
            'details': [p.to_dict() for p in all_patterns],
        }
        self._learning_log.append({
            'query': query[:100],
            'type': response_type,
            'patterns': result['patterns_extracted'],
            'timestamp': time.time(),
        })

        return result

    def learn_from_nexus(self, nexus_path='/tmp/nexus.json',
                         agents=None) -> Dict[str, Any]:
        """
        Learn from all LLM messages in Nexus.

        Watches Alpha/Bravo messages and extracts patterns
        from their responses.
        """
        if agents is None:
            agents = {'alpha', 'bravo'}

        if not os.path.exists(nexus_path):
            return {'messages_processed': 0, 'patterns': 0}

        try:
            with open(nexus_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {'messages_processed': 0, 'patterns': 0}

        total_patterns = 0
        messages_processed = 0

        for msg in data.get('messages', []):
            if msg.get('from') not in agents:
                continue

            text = msg.get('text', '')
            if len(text) < 50:  # Skip short messages
                continue

            result = self.learn_from_response(
                query=f"nexus:{msg.get('from')}",
                response=text,
                response_type='auto',
            )
            total_patterns += result['patterns_extracted']
            messages_processed += 1

        return {
            'messages_processed': messages_processed,
            'patterns': total_patterns,
            'store_total': self.store.total_patterns,
        }

    def suggest_questions(self, knowledge_store=None,
                          n=10) -> List[Dict[str, str]]:
        """
        Generate questions that FOSS-KI should ask an LLM
        to learn the most useful patterns.

        Prioritizes:
        1. Areas where FOSS-KI fails most (from gap log)
        2. Pattern types with fewest examples
        3. High-value capabilities (code, reasoning)
        """
        questions = []

        # Pattern type gaps — what do we have fewest patterns for?
        stats = self.store.stats()
        type_priorities = [
            (PatternType.CODE_STRUCTURE, 'code'),
            (PatternType.REASONING, 'reasoning'),
            (PatternType.DECOMPOSITION, 'decomp'),
            (PatternType.COMPOSITION, 'composition'),
            (PatternType.EXPLANATION, 'explanation'),
        ]

        for ptype, label in type_priorities:
            count = stats.get(ptype, 0)
            if count < 10:  # Need more of this type
                questions.extend(
                    self._generate_learning_questions(label, 3)
                )

        # Knowledge gaps → questions about reasoning
        if knowledge_store:
            from .metacognition import KnowledgeExplorer
            explorer = KnowledgeExplorer(knowledge_store)
            targets = explorer.exploration_targets(n=5)
            for t in targets:
                questions.append({
                    'question': f"Explain {t['entity']} in detail: "
                                f"what it is, how it works, and why it matters.",
                    'target_type': 'explanation',
                    'reason': f"Frontier entity (degree={t['degree']})",
                })

        return questions[:n]

    def _generate_learning_questions(self, category: str,
                                      n: int) -> List[Dict[str, str]]:
        """Generate questions optimized for pattern extraction."""
        templates = {
            'code': [
                "Write a Python function that {task}. Include error handling and docstring.",
                "Refactor this code to be more Pythonic: {code}",
                "Write a class that implements {pattern} with proper __init__, __repr__, and methods.",
            ],
            'reasoning': [
                "Walk me through step by step how to {task}.",
                "What are the trade-offs between {A} and {B}?",
                "If {premise}, what follows logically and why?",
            ],
            'decomp': [
                "Break down the problem of {task} into sub-problems.",
                "What are the components of a {system}?",
                "How would you architect a solution for {problem}?",
            ],
            'composition': [
                "Design a pipeline that takes {input} and produces {output}.",
                "How do these components work together: {list}?",
                "Show me how to compose {A} with {B} to achieve {goal}.",
            ],
            'explanation': [
                "Explain {concept} as if I were a beginner.",
                "What is {concept} and why does it matter?",
                "Give me an analogy for {concept}.",
            ],
        }

        result = []
        for template in templates.get(category, [])[:n]:
            result.append({
                'question': template,
                'target_type': category,
                'reason': f"Need more {category} patterns "
                          f"(have {self.store.stats().get(category, 0)})",
            })
        return result

    def _detect_type(self, query: str, response: str) -> str:
        """Auto-detect response type."""
        q_lower = query.lower()
        if any(w in q_lower for w in ('write', 'code', 'function', 'implement')):
            return 'code'
        if any(w in q_lower for w in ('explain', 'why', 'how does')):
            return 'explanation'
        if any(w in q_lower for w in ('step by step', 'reason', 'prove', 'derive')):
            return 'reasoning'
        if '```' in response or 'def ' in response:
            return 'code'
        return 'auto'

    def status(self) -> str:
        """Human-readable status report."""
        stats = self.store.stats()
        total = self.store.total_patterns
        sessions = len(self._learning_log)

        lines = [
            "Apprentice Status",
            "=" * 40,
            f"Total patterns learned: {total}",
            f"Learning sessions: {sessions}",
            "",
            "Patterns by type:",
        ]

        for ptype, count in sorted(stats.items()):
            bar = '█' * min(count, 20)
            lines.append(f"  {ptype:15s} {bar} ({count})")

        if sessions > 0:
            last = self._learning_log[-1]
            lines.append(f"\nLast session: {last['type']}, "
                         f"{last['patterns']} patterns extracted")

        return "\n".join(lines)
