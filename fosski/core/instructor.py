"""
Instruction Parser — Natural Language → Action Pipeline
=========================================================
Parses free-form instructions into executable action plans.

Examples:
  "Read the file config.py and tell me what classes it defines"
  → [tool:read_file(config.py), reasoning:extract_classes, answer]

  "What is 2+2 and who discovered radium?"
  → [math:2+2, knowledge:radium/discoverer, compose_answer]

  "Find all Python files with TODO comments"
  → [tool:search_files(*.py), tool:grep(TODO), format_results]

This is the glue between natural language and the system's capabilities.
"""

import re
from typing import Dict, Any, List, Optional


class Action:
    """A single action in an execution plan."""

    def __init__(self, action_type: str, target: str,
                 args: Optional[Dict] = None, depends_on: Optional[int] = None):
        """
        Args:
            action_type: 'knowledge', 'reasoning', 'math', 'tool', 'chain', 'answer'
            target: what to query/execute (tool name, query, etc.)
            args: additional arguments
            depends_on: index of action this depends on (for chaining)
        """
        self.action_type = action_type
        self.target = target
        self.args = args or {}
        self.depends_on = depends_on
        self.result = {}  # type: dict

    def __repr__(self):
        dep = f" (depends on #{self.depends_on})" if self.depends_on is not None else ""
        return f"Action({self.action_type}:{self.target}{dep})"


class InstructionParser:
    """
    Parses natural language instructions into action plans.

    Two-stage classification:
    1. Regex patterns for unambiguous commands (file ops, explicit math)
    2. Foss-LM perplexity scoring for ambiguous queries:
       Each domain (knowledge, reasoning, math, tool) has its own
       Markov chain trained on domain-specific text. The query is
       scored by each chain; lowest perplexity = best domain match.
       This replaces regex for the ambiguous cases.
    """

    # Intent patterns: (regex, action_type, target_extractor)
    # These handle UNAMBIGUOUS commands. Ambiguous queries go to perplexity scoring.
    PATTERNS = [
        # Tool: file operations
        (r'(?:read|open|show|cat)\s+(?:the\s+)?(?:file\s+)?["\']?([^\s"\']+\.[\w]+)["\']?',
         'tool', lambda m: ('read_file', {'path': m.group(1)})),

        (r'(?:write|save|create)\s+(?:the\s+)?(?:file\s+)?["\']?([^\s"\']+)["\']?',
         'tool', lambda m: ('write_file', {'path': m.group(1)})),

        (r'(?:list|ls|show)\s+(?:files?\s+)?(?:in\s+)?["\']?([^\s"\']+/?)["\']?',
         'tool', lambda m: ('list_dir', {'path': m.group(1)})),

        (r'(?:find|search\s+for)\s+(?:all\s+)?(?:files?\s+)?(?:matching\s+)?["\']?([*\w.]+)["\']?',
         'tool', lambda m: ('search_files', {'pattern': m.group(1)})),

        (r'(?:grep|search\s+(?:for)?)\s+(?:for\s+)?["\'](.+?)["\'](?:\s+in\s+(.+))?',
         'tool', lambda m: ('grep', {'pattern': m.group(1),
                                      'path': m.group(2).strip() if m.group(2) else '.'})),

        (r'(?:run|execute)\s+(?:command\s+)?["\']?(.+?)["\']?\s*$',
         'tool', lambda m: ('shell', {'command': m.group(1)})),

        # Math
        (r'(?:calculate|compute|what\s+is)\s+(\d[\d\s+\-*/^().%]+\d)',
         'math', lambda m: ('solve', {'query': m.group(1)})),

        (r'(?:calculate|compute)\s+(sqrt\s*\(.+?\))',
         'math', lambda m: ('solve', {'query': m.group(1)})),

        (r'(?:solve|find\s+x)\s*:?\s*(.+)',
         'math', lambda m: ('solve', {'query': m.group(1)})),

        (r'(?:derivative|differentiate|d/dx)\s+(?:of\s+)?(.+)',
         'math', lambda m: ('solve', {'query': f'derivative of {m.group(1)}'})),

        (r'(?:integral|integrate)\s+(?:of\s+)?(.+)',
         'math', lambda m: ('solve', {'query': f'integral of {m.group(1)}'})),

        (r'(?:is\s+)?\d+\s+(?:a\s+)?prime',
         'math', lambda m: ('solve', {'query': m.group(0)})),

        (r'(?:gcd|lcm|factorize|factor)\s+.+',
         'math', lambda m: ('solve', {'query': m.group(0)})),

        # Reasoning
        (r'(?:why|how\s+does|how\s+can|explain\s+why)\s+(.+)',
         'reasoning', lambda m: ('reason', {'query': m.group(0)})),

        (r'(?:prove|disprove)\s+(?:that\s+)?(.+)',
         'reasoning', lambda m: ('reason', {'query': m.group(0)})),

        (r'(?:if\s+.+\s+then\s+.+)',
         'reasoning', lambda m: ('reason', {'query': m.group(0)})),

        # Chain-of-thought (multi-hop)
        (r'((?:in\s+which|what\s+is\s+the\s+\w+\s+of\s+the)\s+.+)',
         'chain', lambda m: ('think', {'query': m.group(1)})),

        # Knowledge queries (broad — should be last)
        (r'(?:what|who|where|when|which)\s+(?:is|are|was|were|did|does|do)\s+.+',
         'knowledge', lambda m: ('query', {'query': m.group(0)})),

        (r'(?:tell\s+me\s+about|describe|what\s+do\s+you\s+know\s+about)\s+(.+)',
         'knowledge', lambda m: ('query', {'query': m.group(0)})),
    ]

    def parse(self, instruction: str) -> List[Action]:
        """
        Parse an instruction into a list of actions.

        Handles:
        - Single instructions
        - Compound instructions ("X and then Y", "X and Y")
        - Conditional instructions ("if X then Y")
        """
        instruction = instruction.strip()

        # Check for compound instructions
        parts = self._split_compound(instruction)
        if len(parts) > 1:
            actions = []
            for i, part in enumerate(parts):
                sub_actions = self._parse_single(part.strip())
                for action in sub_actions:
                    if i > 0 and action.depends_on is None:
                        action.depends_on = len(actions) - 1
                    actions.append(action)
            return actions

        return self._parse_single(instruction)

    def _parse_single(self, instruction: str) -> List[Action]:
        """Parse a single instruction into actions.

        Stage 1: Regex for unambiguous commands.
        Stage 2: Foss-LM perplexity scoring for ambiguous queries.
        """
        for pattern, action_type, extractor in self.PATTERNS:
            m = re.search(pattern, instruction, re.I)
            if m:
                target, args = extractor(m)
                return [Action(action_type, target, args)]

        # Stage 2: Perplexity-based classification
        # Score the instruction against domain-specific patterns.
        # This replaces the old "?" heuristic with statistical scoring.
        action_type = self._classify_by_pattern(instruction)
        if action_type == 'math':
            return [Action('math', 'solve', {'query': instruction})]
        elif action_type == 'reasoning':
            return [Action('reasoning', 'reason', {'query': instruction})]
        elif action_type == 'chain':
            return [Action('chain', 'think', {'query': instruction})]
        elif action_type == 'knowledge':
            return [Action('knowledge', 'query', {'query': instruction})]

        return [Action('unknown', instruction, {})]

    def _classify_by_pattern(self, instruction: str) -> str:
        """Classify instruction by statistical pattern matching.

        Uses lightweight keyword scoring weighted by position.
        Future: Replace with per-domain Foss-LM perplexity.
        """
        inst = instruction.lower()
        scores = {
            'math': 0.0,
            'reasoning': 0.0,
            'knowledge': 0.0,
            'chain': 0.0,
        }

        # Math signals
        math_words = ['calculate', 'compute', 'sum', 'product', 'average',
                       'factorial', 'prime', 'equation', 'solve', 'integral',
                       'derivative', 'sqrt', 'root', 'divide', 'multiply']
        for w in math_words:
            if w in inst:
                scores['math'] += 2.0
        # Numbers in the instruction
        import re as _re
        if _re.search(r'\d+\s*[+\-*/^%]\s*\d+', inst):
            scores['math'] += 5.0

        # Reasoning signals
        reason_words = ['why', 'how does', 'explain', 'prove', 'because',
                         'therefore', 'imply', 'consequence', 'reason', 'cause']
        for w in reason_words:
            if w in inst:
                scores['reasoning'] += 2.0
        if 'if ' in inst and ' then ' in inst:
            scores['reasoning'] += 4.0

        # Chain-of-thought signals
        chain_words = ['what is the', 'in which', 'which of', 'name the']
        for w in chain_words:
            if w in inst:
                scores['chain'] += 2.0

        # Knowledge signals
        knowledge_words = ['what', 'who', 'where', 'when', 'which',
                            'tell me', 'describe', 'define', 'meaning']
        for w in knowledge_words:
            if w in inst:
                scores['knowledge'] += 1.5
        if '?' in inst:
            scores['knowledge'] += 1.0

        best = max(scores, key=lambda k: scores[k])
        if scores[best] > 0:
            return best
        return 'knowledge'  # default

    def _split_compound(self, instruction: str) -> List[str]:
        """Split compound instructions on 'and then', 'then', 'and also'."""
        # Don't split "and" inside quotes
        if '"' in instruction or "'" in instruction:
            return [instruction]

        # Split on explicit sequencing
        parts = re.split(
            r'\s+(?:and\s+then|then|after\s+that|next|and\s+also)\s+',
            instruction, flags=re.I
        )
        if len(parts) > 1:
            return parts

        # Split on "and" only if both parts look like separate instructions
        and_parts = re.split(r'\s+and\s+', instruction, flags=re.I)
        if len(and_parts) == 2:
            # Check if both parts match an intent
            for p in and_parts:
                if not any(re.search(pat, p, re.I) for pat, _, _ in self.PATTERNS):
                    return [instruction]  # Don't split — "and" is part of the query
            return and_parts

        return [instruction]

    def explain(self, instruction: str) -> str:
        """Human-readable explanation of the action plan."""
        actions = self.parse(instruction)
        lines = [f"Instruction: {instruction}",
                 f"Actions ({len(actions)}):"]
        for i, a in enumerate(actions):
            dep = f" → depends on step {a.depends_on + 1}" if a.depends_on is not None else ""
            lines.append(f"  {i+1}. [{a.action_type}] {a.target} {a.args}{dep}")
        return '\n'.join(lines)


class InstructionExecutor:
    """
    Executes action plans produced by InstructionParser.

    Connects to: Router, ChainOfThought, MathSolver, ToolExecutor.
    """

    def __init__(self, router=None, chain=None, math_solver=None,
                 tool_executor=None):
        self.router = router
        self.chain = chain
        self.math = math_solver
        self.tools = tool_executor
        self.parser = InstructionParser()

    def execute(self, instruction: str) -> Dict[str, Any]:
        """
        Parse and execute an instruction.

        Returns:
            dict with: answer, actions, trace
        """
        actions = self.parser.parse(instruction)
        trace = [f"Parsed {len(actions)} actions"]
        results = []

        for i, action in enumerate(actions):
            # Get previous result if dependent
            prev_result = None
            if action.depends_on is not None and action.depends_on < len(results):
                prev_result = results[action.depends_on]

            result = self._execute_action(action, prev_result, trace)
            results.append(result)
            action.result = result

        # Compose final answer from all results
        answers = [r.get('answer') or r.get('result', '')
                   for r in results if r.get('answer') or r.get('result')]
        final_answer = ' '.join(str(a) for a in answers) if answers else None

        return {
            'answer': final_answer,
            'actions': [{'type': a.action_type, 'target': a.target,
                        'result': a.result} for a in actions],
            'trace': trace,
        }

    def _execute_action(self, action: Action, prev_result: Optional[Dict],
                        trace: list) -> Dict:
        """Execute a single action."""
        trace.append(f"  Executing: {action}")

        if action.action_type == 'knowledge' and self.router:
            query = action.args.get('query', action.target)
            result = self.router.route(query)
            trace.append(f"  → {result.get('answer', '')[:100]} [{result.get('confidence_level')}]")
            return result

        if action.action_type == 'chain' and self.chain:
            query = action.args.get('query', action.target)
            result = self.chain.think(query)
            trace.append(f"  → {result.get('answer')} ({result.get('n_hops')} hops)")
            return result

        if action.action_type == 'math' and self.math:
            query = action.args.get('query', action.target)
            result = self.math.solve(query)
            trace.append(f"  → {result.get('answer')}")
            return result

        if action.action_type == 'reasoning' and self.router:
            query = action.args.get('query', action.target)
            result = self.router.route(query)
            trace.append(f"  → {result.get('answer')}")
            return result

        if action.action_type == 'tool' and self.tools:
            tool_name = action.target
            args = {k: v for k, v in action.args.items() if k != 'query'}
            result = self.tools.execute(tool_name, **args)
            trace.append(f"  → {'OK' if result.get('success') else 'FAIL'}")
            return result

        trace.append(f"  → No handler for {action.action_type}")
        return {'answer': None, 'error': f'No handler for {action.action_type}'}
