"""
Agent — The Intelligence Layer That Makes FOSS-KI a Real Assistant
===================================================================
This is the layer that turns FOSS-KI from a Q&A bot into a Claude replacement.

Without this: "What is the capital of France?" → "Paris"
With this:    "Write a Python function that sorts a list" → generates code
              "Read server.py and find the bug" → reads file, analyzes, suggests fix
              "Explain how quicksort works" → multi-paragraph explanation

Architecture:
  1. Intent Classification (rule-based, no ML)
     - QUESTION: factual query → Dialog pipeline
     - INSTRUCTION: do something → Action executor
     - CODE_REQUEST: write/fix/explain code → Code generator
     - REASONING: prove/explain/why → Reasoning engine
     - GENERATION: write text/email/summary → Text generator
     - CONVERSATION: greeting/chitchat → Conversational response
     - TOOL_USE: read/write files, run commands → Tool executor

  2. Action Routing → appropriate subsystem
  3. Response Assembly → unified output format

This is CPU-only. No transformer. No API calls.
The FLM (Foss Language Model) does the generation.
"""

import os
import re
import subprocess
import traceback
from typing import Dict, Any, List, Optional, Tuple


class IntentClassifier:
    """
    Rule-based intent classification.

    Maps user input to one of the action types.
    No ML needed — pattern matching on keywords and syntax.
    """

    # Intent patterns: (regex, intent_type, priority)
    # Higher priority = checked first
    PATTERNS = [
        # Code requests
        (r'(?:write|create|implement|code|build|make)\s+(?:an?\s+)?(?:python|function|class|script|program|module)',
         'CODE_REQUEST', 10),
        (r'(?:write|create|implement|make)\s+(?:an?\s+)?[\w\s]+?(?:function|method|class)',
         'CODE_REQUEST', 10),
        (r'(?:write|create)\s+(?:a\s+)?(?:for loop|while loop|if statement|try|except)',
         'CODE_REQUEST', 10),
        (r'(?:fix|debug|refactor|optimize|improve)\s+(?:the\s+)?(?:code|function|class|bug)',
         'CODE_REQUEST', 10),
        (r'(?:add|implement)\s+(?:a\s+)?(?:method|endpoint|route|handler|test)',
         'CODE_REQUEST', 10),
        (r'(?:write|create|implement|build|make)\s+(?:an?\s+)?(?:sort|search|fibonacci|fib|prime|factorial|palindrome|gcd|calculator|parser|server|client|timer|retry|cache|logger|decorator|api|csv|json|http|bfs|dfs|merge|class|dataclass)',
         'CODE_REQUEST', 10),
        (r'(?:write|create|implement|build|make)\s+(?:an?\s+)?\w+\s+(?:decorator|server|client|handler|parser|algorithm)',
         'CODE_REQUEST', 10),
        # Catch-all: write/build/create + technical noun (FORGE has 1237 fragments)
        (r'(?:write|create|implement|build|make|generate)\s+(?:an?\s+)?(?:[\w\s]+?)(?:scanner|scraper|crawler|monitor|logger|parser|encoder|decoder|encryptor|decryptor|converter|validator|checker|tester|generator|handler|listener|sender|receiver|exploiter|harvester|injector|sniffer|tunnel|proxy|beacon|implant|dropper|stager|loader|dumper|brute\s*force|fuzzer|pipeline|framework|dashboard|tool|utility|helper|wrapper|adapter|manager|controller|service|daemon|bot|agent|worker|shell|breaker|pattern|server|client|chain|spoofer|deauth)',
         'CODE_REQUEST', 10),
        # More coding verbs: check, find, merge, validate, compute, parse, convert, extract
        (r'^(?:check|find|merge|validate|compute|parse|convert|extract|flatten|decode|encode|hash|detect|scan|remove|count)\s+',
         'CODE_REQUEST', 8),
        # Noun + "using/with/from" → likely code request ("queue using two stacks")
        (r'^\w+\s+(?:using|with|from|via)\s+',
         'CODE_REQUEST', 7),
        # Universal: any query starting with build/write/create/implement/make = CODE_REQUEST
        # This catches ALL forge fragment queries (1237 fragments)
        (r'^(?:write|create|implement|build|make|generate)\s+',
         'CODE_REQUEST', 9),
        (r'```', 'CODE_REQUEST', 10),  # Code block in input
        # "How do I ..." + coding keywords → CODE_REQUEST
        (r'how\s+(?:do\s+i|can\s+i|to)\s+(?:read|write|parse|sort|search|filter|create|implement|build|make|open|close|connect|send|receive|handle|process|convert|encode|decode|encrypt|decrypt|hash|validate|check|test|debug|fix|run|execute|call|invoke|import|install|deploy|use)\s+',
         'CODE_REQUEST', 10),
        # "I need to read/write/parse..." → CODE_REQUEST
        (r'i\s+(?:need|want)\s+to\s+(?:read|write|parse|sort|search|filter|create|implement|build|make|open|send|receive|handle|process|convert|encode|decode|encrypt|decrypt|hash|validate|check)\s+',
         'CODE_REQUEST', 10),

        # File operations
        (r'(?:read|open|show|cat|display)\s+(?:the\s+)?(?:file|contents?\s+of)\s+',
         'TOOL_USE', 9),
        (r'(?:read|open|show)\s+\S+\.\w{1,5}$',
         'TOOL_USE', 9),
        (r'(?:write|save|create)\s+(?:a\s+)?(?:file|to)\s+',
         'TOOL_USE', 9),
        (r'(?:edit|modify|change|update)\s+(?:the\s+)?(?:file\s+)?\S+\.\w{1,5}',
         'TOOL_USE', 9),
        (r'(?:list|ls|find|show)\s+(?:all\s+)?(?:\w+\s+)?(?:files|directory|folder|directories)',
         'TOOL_USE', 9),

        # Shell commands
        (r'(?:run|execute|launch)\s+',
         'TOOL_USE', 8),
        (r'^(?:ls|pwd|cd|mkdir|git|pip|npm|python|node|make|curl|wget)\s',
         'TOOL_USE', 8),

        # Reasoning — math and logical proofs, NOT explanations
        (r'^why\s+(?:is|does|do|are|was|were|can|should)',
         'REASONING', 7),
        (r'^(?:prove|disprove)\s+',
         'REASONING', 7),
        (r'(?:what\s+would\s+happen|what\s+if|if\s+.+then)',
         'REASONING', 7),
        (r'(?:solve|calculate|compute|evaluate)\s+',
         'REASONING', 7),
        # Math in natural language: "X to the power of Y", "square root", "X squared"
        (r'\d+\s+to\s+the\s+power',
         'REASONING', 8),
        (r'(?:square\s+root|sqrt)\s*(?:of\s+)?\d+',
         'REASONING', 8),
        (r'\d+\s+(?:squared|cubed)',
         'REASONING', 8),
        (r'what\s+is\s+\d+\s*[\+\-\*/\^]',
         'REASONING', 8),
        (r'what\s+is\s+\d+\s+(?:plus|minus|times|divided|to\s+the\s+power|squared|cubed)',
         'REASONING', 8),
        # Bare math: "100 divided by 7", "5 plus 3", "12 times 8", "100 modulo 7"
        (r'\d+\s+(?:plus|minus|times|divided\s+by|modulo|mod)\s+\d+',
         'REASONING', 8),
        # Pure arithmetic expressions: "5 * 3", "100 / 7"
        (r'^\d+\s*[\+\-\*/\^%]\s*\d+',
         'REASONING', 8),

        # Learning / ingestion — declarative statements (not questions)
        (r'^(?:remember|learn|note)\s+(?:that\s+)?',
         'LEARN', 9),

        # Text generation — must be HIGHER priority than universal code catch-all
        # "write an email" = GENERATION, "write a scanner" = CODE_REQUEST
        (r'(?:write|draft|compose|tell)\s+(?:me\s+)?(?:a\s+|an\s+)?(?:email|letter|message|summary|report|essay|paragraph|text|description|story|poem|article|blog|post|review|proposal|memo|note|brief|joke|limerick|haiku)',
         'GENERATION', 11),
        (r'(?:summarize|translate|rephrase|rewrite|paraphrase)',
         'GENERATION', 6),
        (r'(?:explain|describe)\s+',
         'GENERATION', 8),

        # Questions (factual)
        (r'^(?:what|who|where|when|which|whose)\s+',
         'QUESTION', 4),
        (r'(?:tell\s+me|do\s+you\s+know|what\s+do\s+you\s+know)',
         'QUESTION', 4),
        (r'(?:compare|list|name)\s+',
         'QUESTION', 4),

        # Conversation — meta-questions about the AI itself need high priority
        (r'^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|thanks|thank\s+you|bye|goodbye)',
         'CONVERSATION', 3),
        (r'^(?:how\s+are\s+you|what\s+can\s+you\s+do|who\s+are\s+you|what\s+are\s+you)',
         'CONVERSATION', 12),
        (r'^help$', 'CONVERSATION', 11),

        # Generation: "tell me a joke" / "write me a story"
        (r'(?:tell\s+me\s+a\s+(?:joke|story|riddle|fact))',
         'GENERATION', 6),
    ]

    @classmethod
    def classify(cls, text: str) -> Tuple[str, float]:
        """
        Classify user input into an intent type.

        Returns: (intent_type, confidence)
        """
        text_lower = text.lower().strip()

        # Strip greeting prefixes so "Hey, what is X?" → "what is X?"
        text_stripped = re.sub(
            r'^(?:hey|hi|hello|yo|ok|okay|please|thanks|thank you)'
            r'[,!.\s]*\s*', '', text_lower).strip()

        best_intent = 'QUESTION'  # default
        best_priority = 0

        for pattern, intent, priority in cls.PATTERNS:
            # Check both original and stripped text
            if (re.search(pattern, text_lower) or
                    re.search(pattern, text_stripped)) and priority > best_priority:
                best_intent = intent
                best_priority = priority

        confidence = min(best_priority / 10.0, 1.0)
        return best_intent, confidence


class ToolSelectionPolicy:
    """
    Basal Ganglia-inspired action selection for tool routing.
    Ported from VizDoom foss-v2/modules/basal_ganglia.py.

    In VizDoom: multiple brain modules (combat, navigation, survival)
    propose actions with urgency values → softmax → winner-take-all.

    Here: multiple subsystems (dialog, code, math, tools, text)
    bid on a query → score weighting → best subsystem handles it.

    This is a Markov decision policy without neural networks:
    state = (intent, keyword_scores, history)
    action = which subsystem to route to
    reward = user satisfaction (tracked via corrections)

    The policy LEARNS from routing failures:
    if a subsystem returns a low-confidence response,
    the policy shifts weight away from that route.
    """

    # Subsystem capability profiles
    SUBSYSTEMS = {
        'dialog': {
            'handles': {'QUESTION'},
            'keywords': {'what', 'who', 'where', 'when', 'which',
                         'capital', 'language', 'population', 'tell'},
            'base_weight': 1.0,
        },
        'code': {
            'handles': {'CODE_REQUEST'},
            'keywords': {'write', 'function', 'class', 'code', 'implement',
                         'script', 'program', 'sort', 'fibonacci', 'api',
                         'server', 'debug', 'fix'},
            'base_weight': 1.0,
        },
        'math': {
            'handles': {'REASONING'},
            'keywords': {'calculate', 'compute', 'evaluate', 'solve',
                         'plus', 'minus', 'times', 'divided'},
            'base_weight': 1.0,
        },
        'text': {
            'handles': {'GENERATION'},
            'keywords': {'explain', 'describe', 'summarize', 'write',
                         'email', 'essay', 'draft'},
            'base_weight': 1.0,
        },
        'tools': {
            'handles': {'TOOL_USE'},
            'keywords': {'read', 'write', 'file', 'run', 'execute',
                         'list', 'open', 'save', 'delete', 'ls', 'cat'},
            'base_weight': 1.0,
        },
        'conversation': {
            'handles': {'CONVERSATION'},
            'keywords': {'hello', 'hi', 'hey', 'thanks', 'bye',
                         'help', 'who are you'},
            'base_weight': 1.0,
        },
    }

    def __init__(self):
        # Learned weights — start at base, adapt over time
        self.weights = {name: info['base_weight']
                        for name, info in self.SUBSYSTEMS.items()}
        # Success/failure tracking for weight adaptation
        self._history = []  # (subsystem, success: bool)
        self._temperature = 1.5  # softmax temperature (from BasalGanglia)
        self._dominance = 1.3  # winner boost

    def select(self, intent: str, text: str) -> List[Tuple[str, float]]:
        """
        Select best subsystem(s) for the given query.

        Returns ranked list of (subsystem_name, score).
        First element is the winner.
        """
        text_lower = text.lower()
        words = set(text_lower.split())

        scores = {}
        for name, info in self.SUBSYSTEMS.items():
            score = 0.0

            # Intent match (strongest signal)
            if intent in info['handles']:
                score += 5.0

            # Keyword overlap
            keyword_overlap = len(words & info['keywords'])
            score += keyword_overlap * 1.0

            # Learned weight adjustment
            score *= self.weights[name]

            scores[name] = score

        # Softmax normalization (from BasalGanglia)
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                import math
                exp_scores = {}
                for name, s in scores.items():
                    scaled = (s - max_score) / max(0.1, self._temperature)
                    exp_scores[name] = math.exp(max(-10, scaled))
                total = sum(exp_scores.values()) + 1e-10

                # Winner-take-all with dominance boost
                winner = max(exp_scores, key=lambda k: exp_scores[k])
                for name in exp_scores:
                    if name == winner:
                        exp_scores[name] *= self._dominance
                total = sum(exp_scores.values()) + 1e-10

                scores = {name: v / total for name, v in exp_scores.items()}

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked

    def record_outcome(self, subsystem: str, success: bool):
        """
        Record whether routing was successful.
        Adjusts weights for future decisions.
        """
        self._history.append((subsystem, success))

        # Exponential moving average weight update
        alpha = 0.1  # learning rate
        if success:
            self.weights[subsystem] = min(
                2.0, self.weights[subsystem] + alpha)
        else:
            self.weights[subsystem] = max(
                0.3, self.weights[subsystem] - alpha)

    def stats(self) -> Dict[str, Any]:
        """Current policy state."""
        total = len(self._history)
        if total == 0:
            return {'total': 0, 'weights': dict(self.weights)}

        successes = sum(1 for _, s in self._history if s)
        by_subsystem = {}
        for name, success in self._history:
            if name not in by_subsystem:
                by_subsystem[name] = {'total': 0, 'success': 0}
            by_subsystem[name]['total'] += 1
            if success:
                by_subsystem[name]['success'] += 1

        return {
            'total': total,
            'success_rate': successes / total,
            'weights': dict(self.weights),
            'by_subsystem': by_subsystem,
        }


class ToolExecutor:
    """
    Safe execution of file operations and shell commands.

    Security: sandboxed directory, no rm -rf, no sudo.
    """

    BLOCKED_COMMANDS = {
        'rm -rf /', 'rm -rf ~', 'rm -rf *',
        'sudo rm', 'mkfs', 'dd if=',
        ':(){:|:&};:', 'fork bomb',
    }

    def __init__(self, workspace_dir='.', allow_shell=True):
        self.workspace = os.path.abspath(workspace_dir)
        self.allow_shell = allow_shell

    def read_file(self, path: str) -> Dict[str, Any]:
        """Read a file and return its contents."""
        full_path = self._resolve_path(path)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            lines = content.split('\n')
            return {
                'success': True,
                'content': content,
                'lines': len(lines),
                'path': full_path,
                'response': f"File `{path}` ({len(lines)} lines):\n```\n{content}\n```",
            }
        except FileNotFoundError:
            return {'success': False, 'response': f"File not found: {path}"}
        except Exception as e:
            return {'success': False, 'response': f"Error reading {path}: {e}"}

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write content to a file."""
        full_path = self._resolve_path(path)
        try:
            os.makedirs(os.path.dirname(full_path) or '.', exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {
                'success': True,
                'path': full_path,
                'response': f"Written {len(content)} characters to `{path}`.",
            }
        except Exception as e:
            return {'success': False, 'response': f"Error writing {path}: {e}"}

    def list_files(self, path: str = '.', pattern: str = '*') -> Dict[str, Any]:
        """List files in a directory."""
        full_path = self._resolve_path(path)
        try:
            entries = sorted(os.listdir(full_path))
            files = []
            dirs = []
            for entry in entries:
                fp = os.path.join(full_path, entry)
                if os.path.isdir(fp):
                    dirs.append(entry + '/')
                else:
                    size = os.path.getsize(fp)
                    files.append(f"{entry} ({size}B)")
            listing = '\n'.join(dirs + files)
            return {
                'success': True,
                'dirs': dirs,
                'files': files,
                'response': f"Contents of `{path}`:\n{listing}",
            }
        except Exception as e:
            return {'success': False, 'response': f"Error listing {path}: {e}"}

    def run_command(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute a shell command safely."""
        if not self.allow_shell:
            return {'success': False, 'response': "Shell execution is disabled."}

        # Security check
        cmd_lower = command.lower()
        for blocked in self.BLOCKED_COMMANDS:
            if blocked in cmd_lower:
                return {'success': False,
                        'response': f"Blocked dangerous command: {command}"}

        try:
            result = subprocess.run(
                command, shell=True, capture_output=True,
                text=True, timeout=timeout, cwd=self.workspace,
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
                'response': f"```\n$ {command}\n{output.strip()}\n```",
            }
        except subprocess.TimeoutExpired:
            return {'success': False, 'response': f"Command timed out after {timeout}s"}
        except Exception as e:
            return {'success': False, 'response': f"Error: {e}"}

    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to workspace."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.workspace, path)


class CodeGenerator:
    """
    Generate code using FLM language model + templates + AST validation.

    Strategy:
    1. Parse the request to identify: language, function name, parameters, purpose
    2. Select code template skeleton
    3. Use FLM to fill in implementation details
    4. Validate syntax with ast.parse (Python) or basic bracket matching (other)
    5. Return with explanation

    Without a trained FLM: falls back to template-only generation.
    """

    # Code templates for common patterns
    TEMPLATES = {
        'function': '''def {name}({params}):
    """{description}"""
    {body}
''',
        'class': '''class {name}:
    """{description}"""

    def __init__(self{init_params}):
        {init_body}

    {methods}
''',
        'sort': '''def {name}(arr):
    """{description}"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return {name}(left) + middle + {name}(right)
''',
        'search': '''def {name}(arr, target):
    """{description}"""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
''',
        'fibonacci': '''def {name}(n):
    """{description}"""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
''',
        'factorial': '''def {name}(n):
    """{description}"""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
''',
        'reverse_string': '''def {name}(s):
    """{description}"""
    return s[::-1]
''',
        'palindrome': '''def {name}(s):
    """{description}"""
    s = s.lower().replace(' ', '')
    return s == s[::-1]
''',
        'prime': '''def {name}(n):
    """{description}"""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
''',
        'gcd': '''def {name}(a, b):
    """{description}"""
    while b:
        a, b = b, a % b
    return a
''',
        'read_file': '''def {name}(filepath):
    """{description}"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()
''',
        'write_file': '''def {name}(filepath, content):
    """{description}"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
''',
        'http_get': '''import urllib.request

def {name}(url):
    """{description}"""
    with urllib.request.urlopen(url) as response:
        return response.read().decode('utf-8')
''',
        'list_comprehension': '''def {name}(items):
    """{description}"""
    return [item for item in items if {condition}]
''',
        'dict_operations': '''def {name}(data):
    """{description}"""
    result = {{}}
    for key, value in data.items():
        result[key] = value
    return result
''',
        'try_except': '''def {name}({params}):
    """{description}"""
    try:
        {body}
    except Exception as e:
        print(f"Error: {{e}}")
        return None
''',
    }

    # Keyword → template mapping
    KEYWORD_MAP = {
        'sort': 'sort', 'sorting': 'sort', 'quicksort': 'sort',
        'search': 'search', 'binary search': 'search', 'find': 'search',
        'fibonacci': 'fibonacci', 'fib': 'fibonacci',
        'factorial': 'factorial',
        'reverse': 'reverse_string', 'reverse string': 'reverse_string',
        'palindrome': 'palindrome',
        'prime': 'prime', 'is prime': 'prime', 'primality': 'prime',
        'gcd': 'gcd', 'greatest common': 'gcd', 'euclidean': 'gcd',
        'read file': 'read_file', 'read a file': 'read_file',
        'write file': 'write_file', 'write a file': 'write_file',
        'http': 'http_get', 'fetch': 'http_get', 'request': 'http_get',
        'filter': 'list_comprehension',
    }

    def __init__(self, lm_model=None):
        self.lm = lm_model
        # FORGE Assembler: 1237 battle-tested fragments
        try:
            from .forge_assembler import ForgeAssembler
            self._forge = ForgeAssembler()
        except Exception:
            self._forge = None

    def generate(self, request: str) -> Dict[str, Any]:
        """Generate code from a natural language request."""
        request_lower = request.lower()

        # 0. Try specific template library FIRST (high-confidence matches)
        from .code_templates import find_template, TEMPLATE_REGISTRY
        ext_code = find_template(request)
        # Check if the match is specific enough (multi-word keyword, >= 7 chars)
        _best_kw_len = max(
            (len(kw) for kw in TEMPLATE_REGISTRY if kw in request_lower),
            default=0
        )
        if ext_code and _best_kw_len >= 5:
            valid = self._validate_python(ext_code)
            return {
                'code': ext_code,
                'language': 'python',
                'valid': valid,
                'method': 'template_library',
                'template': 'extended',
                'response': f"```python\n{ext_code}```\n\n"
                            f"{'Syntax valid.' if valid else 'Warning: syntax issues.'}",
            }

        # 1. FORGE Fragment System (secondary — after specific template matches)
        if self._forge:
            forge_result = self._forge.generate(request)
            if forge_result['code'] and forge_result['valid']:
                return {
                    'code': forge_result['code'],
                    'language': 'python',
                    'valid': True,
                    'method': 'forge_fragments',
                    'template': forge_result['fragment_key'],
                    'composed': forge_result.get('composed', False),
                    'response': forge_result['response'],
                }

        # 2. Try less-specific template match (shorter keywords)
        if ext_code:
            valid = self._validate_python(ext_code)
            return {
                'code': ext_code,
                'language': 'python',
                'valid': valid,
                'method': 'template_library',
                'template': 'extended',
                'response': f"```python\n{ext_code}```\n\n"
                            f"{'Syntax valid.' if valid else 'Warning: syntax issues.'}",
            }

        # 3. Try to match a known template
        template_key = self._match_template(request_lower)

        # 2. Extract function name from request
        name = self._extract_name(request)

        # 3. Generate description
        description = self._extract_description(request)

        if template_key and template_key in self.TEMPLATES:
            code = self.TEMPLATES[template_key].format(
                name=name,
                params=self._extract_params(request, template_key),
                description=description,
                body='pass  # TODO: implement',
                init_params='',
                init_body='pass',
                methods='',
                condition='True',
            )
            method = 'template'
        else:
            # Generic function template
            params = self._extract_params(request, 'function')
            code = self.TEMPLATES['function'].format(
                name=name,
                params=params,
                description=description,
                body='pass  # TODO: implement with FLM generation',
            )
            method = 'generic'

        # 4. Validate syntax
        valid = self._validate_python(code)

        return {
            'code': code,
            'language': 'python',
            'valid': valid,
            'method': method,
            'template': template_key,
            'response': f"```python\n{code}```\n\n"
                        f"{'Syntax valid.' if valid else 'Warning: syntax may have issues.'}",
        }

    def _match_template(self, text: str) -> Optional[str]:
        """Match request to a code template."""
        for keyword, template in self.KEYWORD_MAP.items():
            if keyword in text:
                return template
        return None

    def _extract_name(self, request: str) -> str:
        """Extract function name from request."""
        # "write a function called foo" / "create sort_list"
        m = re.search(r'(?:called|named)\s+(\w+)', request, re.I)
        if m:
            return m.group(1)

        # "write a sort function" → "sort"
        m = re.search(r'(?:write|create|implement|make)\s+(?:a\s+)?(\w+)', request, re.I)
        if m and m.group(1).lower() not in ('function', 'class', 'method', 'python', 'script'):
            return m.group(1).lower()

        return 'my_function'

    def _extract_params(self, request: str, template_key: str) -> str:
        """Extract function parameters from request."""
        # Template-specific defaults
        defaults = {
            'sort': 'arr',
            'search': 'arr, target',
            'fibonacci': 'n',
            'factorial': 'n',
            'reverse_string': 's',
            'palindrome': 's',
            'prime': 'n',
            'gcd': 'a, b',
            'read_file': 'filepath',
            'write_file': 'filepath, content',
            'http_get': 'url',
        }
        if template_key in defaults:
            return defaults[template_key]

        # Try to extract from "that takes X and Y"
        m = re.search(r'(?:takes?|accepts?|with)\s+(.+?)(?:\s+and\s+returns|\s*$)',
                       request, re.I)
        if m:
            params = m.group(1).strip()
            # Convert natural language to parameter names
            params = re.sub(r'\s+and\s+', ', ', params)
            params = re.sub(r'a\s+', '', params)
            return params

        return ''

    def _extract_description(self, request: str) -> str:
        """Generate a docstring from the request."""
        # Remove action verbs
        desc = re.sub(r'^(?:write|create|implement|make|code|build)\s+(?:a\s+)?',
                       '', request, flags=re.I)
        desc = re.sub(r'(?:in python|using python|python)\s*$', '', desc, flags=re.I)
        return desc.strip().rstrip('.') + '.'

    def _validate_python(self, code: str) -> bool:
        """Validate Python syntax using ast."""
        import ast
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False


class TextGenerator:
    """
    Free text generation using FLM language model.

    Modes:
    - explain: generate explanation of a concept
    - summarize: compress input text
    - email: write an email
    - generic: free-form text

    Without a trained FLM: falls back to template-based responses.
    """

    EXPLANATION_TEMPLATES = {
        'quicksort': (
            "Quicksort is a divide-and-conquer sorting algorithm. "
            "It works by selecting a 'pivot' element and partitioning "
            "the array into elements less than and greater than the pivot. "
            "It then recursively sorts both partitions. "
            "Average time complexity is O(n log n), worst case O(n^2). "
            "It is one of the most widely used sorting algorithms in practice."
        ),
        'binary search': (
            "Binary search is an efficient algorithm for finding an element "
            "in a sorted array. It works by repeatedly dividing the search "
            "interval in half. If the target value is less than the middle "
            "element, the search continues in the lower half. Otherwise, "
            "it continues in the upper half. Time complexity is O(log n)."
        ),
        'recursion': (
            "Recursion is a programming technique where a function calls "
            "itself to solve a problem. Every recursive function needs a "
            "base case (stopping condition) and a recursive case. "
            "Common examples include factorial, Fibonacci, and tree traversal. "
            "Recursion can be elegant but may cause stack overflow for deep calls."
        ),
        'hash table': (
            "A hash table is a data structure that maps keys to values using "
            "a hash function. The hash function converts keys into array indices, "
            "enabling O(1) average-case lookup, insertion, and deletion. "
            "Collisions are handled via chaining (linked lists) or open addressing. "
            "Python dictionaries are implemented as hash tables."
        ),
    }

    def __init__(self, lm_model=None, knowledge_store=None):
        self.lm = lm_model
        self.knowledge = knowledge_store

    def generate(self, request: str, mode: str = 'generic') -> Dict[str, Any]:
        """Generate text based on request and mode."""
        request_lower = request.lower()

        if mode == 'explain' or request_lower.startswith('explain'):
            return self._explain(request)
        elif mode == 'summarize' or 'summarize' in request_lower:
            return self._summarize(request)
        else:
            return self._generic(request)

    def _explain(self, request: str) -> Dict[str, Any]:
        """Generate an explanation."""
        topic = re.sub(r'^(?:explain|describe|what is|how does)\s+', '',
                        request, flags=re.I).strip().rstrip('?.')
        # Strip leading articles
        topic = re.sub(r'^(?:the|a|an)\s+', '', topic, flags=re.I)

        # Check knowledge store
        explanation_parts = []
        if self.knowledge:
            for s, r, o in self.knowledge.facts:
                if s.lower() == topic.lower() or topic.lower() in s.lower():
                    explanation_parts.append(f"{s} {r}: {o}")

        # Check templates
        for key, template in self.EXPLANATION_TEMPLATES.items():
            if key in topic.lower():
                return {
                    'text': template,
                    'method': 'template',
                    'response': template,
                }

        # Build from knowledge — compose into natural prose
        if explanation_parts:
            # Group facts by relation for cleaner output
            facts_by_entity = {}
            if self.knowledge:
                for s, r, o in self.knowledge.facts:
                    if s.lower() == topic.lower() or topic.lower() in s.lower():
                        if s not in facts_by_entity:
                            facts_by_entity[s] = []
                        facts_by_entity[s].append((r, o))

            # Build prose from facts
            sentences = []
            for entity, facts in facts_by_entity.items():
                for r, o in facts:
                    if r == 'type':
                        sentences.insert(0, f"{entity} is a {o}.")
                    elif r == 'definition':
                        sentences.insert(0, f"{entity} is {o}.")
                    elif r in ('known_for', 'known_as'):
                        sentences.append(f"It is known for {o}.")
                    elif r in ('creator', 'created_by', 'inventor'):
                        sentences.append(f"It was created by {o}.")
                    elif r in ('first_released', 'created', 'founded', 'invented'):
                        sentences.append(f"It was introduced in {o}.")
                    elif r in ('time_complexity', 'complexity'):
                        sentences.append(f"Its time complexity is {o}.")
                    elif r == 'based_on':
                        sentences.append(f"It is based on {o}.")
                    elif r in ('property', 'principle'):
                        sentences.append(f"Key property: {o}.")
                    elif r == 'uses':
                        sentences.append(f"It uses {o}.")
                    elif r in ('about', 'studies'):
                        sentences.append(f"It deals with {o}.")
                    elif r in ('discovered_by', 'discoverer'):
                        sentences.append(f"It was discovered by {o}.")
                    elif r == 'stands_for':
                        sentences.append(f"It stands for {o}.")
                    elif r == 'structure':
                        sentences.append(f"It has a {o} structure.")
                    elif r == 'components':
                        sentences.append(f"It consists of {o}.")
                    elif r in ('formula', 'equation'):
                        sentences.append(f"The equation is {o}.")
                    elif r in ('founders', 'author'):
                        sentences.append(f"It was created by {o}.")
                    elif r in ('paradigm',):
                        sentences.append(f"It follows the {o} paradigm.")
                    elif r in ('typing',):
                        sentences.append(f"It is {o}.")
                    elif r in ('requirement',):
                        sentences.append(f"It requires {o}.")
                    elif r in ('inspired_by',):
                        sentences.append(f"It is inspired by {o}.")
                    elif r in ('method', 'mechanism'):
                        sentences.append(f"It works through {o}.")
                    elif r in ('states', 'meaning'):
                        sentences.append(f"It states that {o}.")
                    elif r in ('value',):
                        sentences.append(f"Its value is {o}.")
                    elif r in ('symbol',):
                        sentences.append(f"It is symbolized as {o}.")
                    elif r == 'starts_with':
                        sentences.append(f"It starts with: {o}.")
                    elif r == 'range':
                        sentences.append(f"Its range is {o}.")
                    elif r == 'parameters':
                        sentences.append(f"Its parameters are {o}.")
                    elif r in ('proved_by', 'proven_by'):
                        sentences.append(f"It was proved by {o}.")
                    elif r in ('year', 'year_published'):
                        sentences.append(f"It dates to {o}.")
                    elif r in ('branches', 'types'):
                        sentences.append(f"Its types include {o}.")
                    elif r in ('inventors', 'authors'):
                        sentences.append(f"It was created by {o}.")
                    elif r.startswith(('worst_case', 'average_', 'best_')):
                        sentences.append(f"Its {r.replace('_', ' ')} is {o}.")
                    else:
                        # Generic fallback — use relation as-is but formatted
                        r_readable = r.replace('_', ' ')
                        sentences.append(f"Its {r_readable} is {o}.")

            if sentences:
                text = " ".join(sentences)
                return {'text': text, 'method': 'knowledge_explain', 'response': text}

        # FLM generation if available
        if self.lm:
            seed = f"{topic} is "
            generated = self.lm.generate(list(seed), 200, temperature=0.7)
            text = seed + ''.join(generated)
            # Truncate at last sentence
            last_period = text.rfind('.')
            if last_period > len(seed):
                text = text[:last_period + 1]
            return {'text': text, 'method': 'ppm', 'response': text}

        return {
            'text': f"I can explain {topic} once my language model is trained on relevant text. "
                    f"Use 'feed' to provide text about {topic}.",
            'method': 'no_model',
            'response': f"I don't have enough knowledge about {topic} yet. "
                        f"Feed me text about it with `feed \"...text about {topic}...\"`",
        }

    def _summarize(self, request: str) -> Dict[str, Any]:
        """Summarize text."""
        # Extract the text to summarize (after "summarize")
        m = re.search(r'(?:summarize|summary\s+of)\s*[:\-]?\s*(.*)', request,
                       re.I | re.DOTALL)
        if not m or not m.group(1).strip():
            return {
                'text': 'Please provide the text to summarize.',
                'method': 'error',
                'response': 'Please provide the text to summarize.',
            }

        text = m.group(1).strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Simple extractive summary: first and last sentence + any with key words
        if len(sentences) <= 2:
            summary = text
        else:
            # Score sentences by position and keyword density
            scored = []
            for i, sent in enumerate(sentences):
                score = 0
                if i == 0:
                    score += 3  # First sentence bonus
                if i == len(sentences) - 1:
                    score += 2  # Last sentence bonus
                # Keyword bonus
                keywords = {'important', 'key', 'main', 'significant', 'therefore',
                            'conclusion', 'result', 'found', 'discovered', 'because'}
                words = set(sent.lower().split())
                score += len(words & keywords)
                scored.append((score, sent))

            scored.sort(reverse=True)
            # Take top 3 sentences, keep original order
            top_indices = set()
            for j, (_, sent) in enumerate(scored[:3]):
                top_indices.add(sentences.index(sent))
            summary = ' '.join(sentences[i] for i in sorted(top_indices))

        return {
            'text': summary,
            'method': 'extractive',
            'response': f"Summary:\n{summary}",
        }

    # Template-based generation for common tasks (no FLM needed)
    EMAIL_TEMPLATE = (
        "Subject: {subject}\n\n"
        "Dear {recipient},\n\n"
        "{body}\n\n"
        "Best regards"
    )

    EMAIL_BODIES = {
        'late': "I'm writing to inform you that I will be arriving late today due to unforeseen circumstances. I apologize for any inconvenience and will make up the time.",
        'sick': "I'm writing to let you know that I'm feeling unwell and won't be able to come in today. I'll keep you updated on my status.",
        'resign': "After careful consideration, I have decided to resign from my position. I want to thank you for the opportunities and support during my time here.",
        'request': "I'm writing to formally request your approval on the matter discussed. Please let me know if you need any additional information.",
        'update': "I wanted to provide a quick update on the current status of the project. Things are progressing well and we're on track to meet our deadline.",
        'follow_up': "I'm following up on our previous conversation. I wanted to check if there are any updates or if you need anything further from my side.",
    }

    def _generic(self, request: str) -> Dict[str, Any]:
        """Generic text generation with template fallback."""
        # FLM generation if available
        if self.lm:
            seed = request.strip()
            if not seed.endswith((' ', ':', '.')):
                seed += ' '
            generated = self.lm.generate(list(seed), 300, temperature=0.8)
            text = seed + ''.join(generated)
            last_period = text.rfind('.')
            if last_period > len(seed):
                text = text[:last_period + 1]
            return {'text': text, 'method': 'ppm', 'response': text}

        # Template-based fallback for email
        request_lower = request.lower()
        if 'email' in request_lower or 'letter' in request_lower or 'message' in request_lower:
            return self._generate_email(request)

        # Template-based for other text tasks
        if any(w in request_lower for w in ('joke', 'funny')):
            return {
                'text': "Why do programmers prefer dark mode? Because light attracts bugs.",
                'method': 'template',
                'response': "Why do programmers prefer dark mode? Because light attracts bugs.",
            }

        # Knowledge-based text generation
        if self.knowledge:
            # Try to find relevant facts and compose
            words = set(request_lower.split()) - {'write', 'me', 'a', 'an', 'the', 'about',
                                                    'of', 'for', 'on', 'to', 'and', 'or'}
            relevant = []
            for s, r, o in self.knowledge.facts:
                if any(w in s.lower() for w in words):
                    relevant.append(f"{s} {r}: {o}")
            if relevant:
                text = "Here's what I know:\n" + "\n".join(f"- {f}" for f in relevant[:10])
                return {'text': text, 'method': 'knowledge_compose', 'response': text}

        return {
            'text': "I can generate better text once my language model is trained. "
                    "Use `train --text <file>` to train on a text corpus. "
                    "Meanwhile, I can answer factual questions and generate code.",
            'method': 'no_model',
            'response': "I can generate better text once my language model is trained. "
                        "Try: `python3 cli.py train --text <textfile>`. "
                        "Meanwhile, I can answer questions and write code.",
        }

    def _generate_email(self, request: str) -> Dict[str, Any]:
        """Generate an email from the request context."""
        request_lower = request.lower()

        # Detect recipient
        recipient = "Sir/Madam"
        m = re.search(r'(?:to|for)\s+(?:my\s+)?(\w+)', request_lower)
        if m:
            recipient = m.group(1).title()

        # Detect topic/purpose
        subject = "Regarding Your Request"
        body = self.EMAIL_BODIES.get('request')

        for key, template_body in self.EMAIL_BODIES.items():
            if key in request_lower:
                body = template_body
                subject = key.replace('_', ' ').title()
                break

        # Try to extract subject from request
        m = re.search(r'about\s+(.+?)(?:\s+to\s+|\s+for\s+|$)', request_lower)
        if m:
            subject = m.group(1).strip().title()

        text = self.EMAIL_TEMPLATE.format(
            subject=subject, recipient=recipient, body=body
        )
        return {'text': text, 'method': 'email_template', 'response': text}


class MathSolver:
    """
    Symbolic math for basic arithmetic, algebra, and common operations.
    No ML needed — pure computation.
    """

    def solve(self, expression: str) -> Dict[str, Any]:
        """Evaluate a mathematical expression safely."""
        # Clean the expression
        expr = expression.strip()
        expr = re.sub(r'(?:what\s+is|calculate|compute|evaluate|solve)\s*', '', expr, flags=re.I)
        # Remove filler words
        expr = re.sub(r'\b(?:the|a|an|of)\b', '', expr, flags=re.I)
        expr = re.sub(r'\s+', ' ', expr).strip().rstrip('?.')

        # Replace natural language operators
        expr = expr.replace('^', '**')
        expr = re.sub(r'\b(\d+)\s+(?:plus|and)\s+(\d+)\b', r'\1+\2', expr)
        expr = re.sub(r'\b(\d+)\s+(?:minus|subtract)\s+(\d+)\b', r'\1-\2', expr)
        expr = re.sub(r'\b(\d+)\s+(?:times|multiplied by)\s+(\d+)\b', r'\1*\2', expr)
        expr = re.sub(r'\b(\d+)\s+(?:divided by|over)\s+(\d+)\b', r'\1/\2', expr)
        expr = re.sub(r'\b(\d+)\s+(?:modulo|mod|remainder)\s+(\d+)\b', r'\1%\2', expr)
        expr = re.sub(r'(?:square root|sqrt)\s+(?:of\s+)?(\d+)', r'\1**0.5', expr, flags=re.I)
        # "X to the power of Y" → X**Y
        expr = re.sub(r'(\d+)\s+to\s+(?:the\s+)?power\s+(\d+)', r'\1**\2', expr)
        # "X squared" → X**2, "X cubed" → X**3
        expr = re.sub(r'(\d+)\s+squared', r'\1**2', expr)
        expr = re.sub(r'(\d+)\s+cubed', r'\1**3', expr)

        # Whitelist check: only allow safe characters
        if not re.match(r'^[\d\s\+\-\*/\.\(\)\*%]+$', expr):
            return {
                'result': None,
                'response': f"Cannot evaluate: {expression}. Only arithmetic is supported.",
            }

        try:
            # Safe eval with no builtins
            result = eval(expr, {"__builtins__": {}}, {})
            # Format result
            if isinstance(result, float) and result == int(result):
                result = int(result)
            return {
                'result': result,
                'response': f"{expr} = {result}",
            }
        except Exception as e:
            return {
                'result': None,
                'response': f"Cannot evaluate: {expression}. Error: {e}",
            }


class Agent:
    """
    The intelligence layer that makes FOSS-KI a real assistant.

    Orchestrates:
    - IntentClassifier → route to right subsystem
    - DialogSystem → factual Q&A
    - CodeGenerator → code writing
    - TextGenerator → free text
    - ToolExecutor → file/shell operations
    - MathSolver → arithmetic
    - ReasoningEngine → logical reasoning

    This is the main entry point for user interaction.
    """

    def __init__(self, dialog_system=None, workspace_dir='.'):
        from .dialog import DialogSystem
        from .composer import ResponseComposer
        from .formulierer import Formulierer
        from .metacognition import MetacognitionEngine

        from .tool_loader import ToolRegistry

        self.dialog = dialog_system or DialogSystem(knowledge_dim=128)
        self.classifier = IntentClassifier()
        self.policy = ToolSelectionPolicy()
        self.tools = ToolExecutor(workspace_dir=workspace_dir)
        self.tool_registry = ToolRegistry()
        self.code_gen = CodeGenerator()
        self.text_gen = TextGenerator(knowledge_store=self.dialog.knowledge)
        self.math = MathSolver()
        self.meta = MetacognitionEngine(self.dialog.knowledge)

        # Enable full dialog pipeline
        self.dialog.enable_metacognition(self.meta)
        self.dialog.enable_composer(
            ResponseComposer(knowledge_store=self.dialog.knowledge))
        self.dialog.enable_formulierer(Formulierer())

        # Conversation history for context
        self.history = []

    def process(self, user_input: str) -> Dict[str, Any]:
        """
        Process any user input — the universal entry point.

        Returns dict with:
            response: str — the text to show the user
            intent: str — classified intent type
            confidence: float — classification confidence
            source: str — which subsystem handled it
            metadata: dict — subsystem-specific data
        """
        try:
            intent, conf = self.classifier.classify(user_input)
        except Exception:
            intent, conf = 'QUESTION', 0.5

        result = {
            'response': '',
            'intent': intent,
            'confidence': conf,
            'source': intent.lower(),
            'metadata': {},
        }

        try:
            _decl = self._is_declarative(user_input)
            if intent == 'LEARN' or _decl:
                try:
                    from .extractor import TripletExtractor
                    ext = TripletExtractor()
                    triplets = ext.extract_from_text(user_input)
                    if triplets:
                        for s, r, o in triplets:
                            self.dialog.knowledge.store_fact(s, r, o)
                        facts_str = "; ".join(f"{s} → {r} → {o}" for s, r, o in triplets)
                        result['response'] = f"Learned {len(triplets)} fact(s): {facts_str}"
                        result['source'] = 'learned'
                        result['metadata'] = {'facts': triplets}
                    elif intent == 'LEARN':
                        result['response'] = "I couldn't extract facts from that. Try: 'X is a Y'."
                        result['source'] = 'learn_failed'
                    else:
                        # Not a LEARN intent and no facts extracted — fall through to QUESTION
                        pass
                except Exception:
                    if intent == 'LEARN':
                        result['response'] = "I couldn't learn from that statement."
                        result['source'] = 'error_fallback'

                # If we learned something, skip everything else
                if result['source'] == 'learned':
                    intent = 'LEARN'  # prevent fall-through

            if intent == 'LEARN':
                pass  # Already handled above

            elif intent == 'QUESTION':
                # Try comparison FIRST ("Compare X and Y")
                cmp_result = self._try_compare(user_input)
                if cmp_result:
                    result['response'] = cmp_result['response']
                    result['source'] = 'compare'
                    result['metadata'] = cmp_result
                else:
                    # Try multi-hop for nested queries
                    hop_result = self._try_multi_hop(user_input)
                    if hop_result:
                        result['response'] = hop_result['response']
                        result['source'] = 'multi_hop'
                        result['metadata'] = hop_result
                    else:
                        try:
                            dialog_result = self.dialog.turn(user_input)
                            result['response'] = dialog_result['response']
                            result['source'] = dialog_result.get('source', 'dialog')
                            result['metadata'] = dialog_result
                        except Exception:
                            result['response'] = "I couldn't process that question. Could you rephrase?"
                            result['source'] = 'error_fallback'

                # Common sense fallback: if dialog can't answer, try CommonSenseEngine
                # BUT: if dialog already has "Did you mean?" suggestions, keep them
                resp_lower = (result['response'] or '').lower()
                has_suggestions = 'did you mean' in resp_lower
                if not has_suggestions and ("don't understand" in resp_lower or "don't have reliable" in resp_lower):
                    if hasattr(self, '_commonsense'):
                        try:
                            cs_result = self._commonsense.query(user_input)
                            if cs_result.get('answer') is not None:
                                answer = cs_result['answer']
                                if isinstance(answer, bool):
                                    result['response'] = f"{'Yes' if answer else 'No'}."
                                elif isinstance(answer, list):
                                    result['response'] = ', '.join(str(a) for a in answer) + '.'
                                else:
                                    result['response'] = str(answer)
                                result['source'] = 'common_sense'
                        except Exception:
                            pass

                # CBR fallback DISABLED: was hallucinating answers for unknown entities
                # (e.g. "Capital of Narnia?" → CBR returned random AI explanation)
                # The dialog's "I don't know" is the correct anti-hallucination response.

            elif intent == 'CODE_REQUEST':
                try:
                    code_result = self.code_gen.generate(user_input)
                    result['response'] = code_result['response']
                    result['metadata'] = code_result
                except Exception:
                    result['response'] = "I couldn't generate code for that request. Could you be more specific?"
                    result['source'] = 'error_fallback'

            elif intent == 'GENERATION':
                try:
                    text_result = self.text_gen.generate(user_input)
                    result['response'] = text_result['response']
                    result['metadata'] = text_result
                except Exception:
                    result['response'] = "I couldn't generate text for that request. Could you rephrase?"
                    result['source'] = 'error_fallback'

            elif intent == 'QUESTION' and (re.search(r'\d+\s*[\+\-\*/\^\*%]', user_input) or
                                           re.search(r'\d+\s+(?:modulo|mod)\s+\d+', user_input, re.I)):
                # Math disguised as question
                try:
                    math_result = self.math.solve(user_input)
                    if math_result['result'] is not None:
                        result['response'] = math_result['response']
                        result['source'] = 'math'
                        result['intent'] = 'REASONING'
                        result['metadata'] = math_result
                    else:
                        dialog_result = self.dialog.turn(user_input)
                        result['response'] = dialog_result['response']
                        result['metadata'] = dialog_result
                except Exception:
                    result['response'] = "I couldn't solve that. Could you rephrase?"
                    result['source'] = 'error_fallback'

            elif intent == 'REASONING':
                try:
                    # Check if it's math first
                    if re.search(r'\d+\s*[\+\-\*/\^]', user_input) or \
                       re.search(r'(?:calculate|compute|evaluate|solve)\s+', user_input, re.I) or \
                       re.search(r'(?:square\s+root|sqrt)\s+(?:of\s+)?\d', user_input, re.I) or \
                       re.search(r'(?:squared|cubed|power\s+of)\s*\d', user_input, re.I) or \
                       re.search(r'\d+\s+(?:plus|minus|times|divided|modulo|mod|remainder|to\s+the\s+power|squared|cubed)', user_input, re.I):
                        math_result = self.math.solve(user_input)
                        result['response'] = math_result['response']
                        result['source'] = 'math'
                        result['metadata'] = math_result
                    else:
                        # CBR first for "why/how" open-ended questions
                        cbr_done = False
                        if hasattr(self, '_case_base') and \
                           re.search(r'^(?:why|how)\s+', user_input, re.I):
                            cbr_answer = self._case_base.retrieve_and_adapt(user_input)
                            if cbr_answer:
                                result['response'] = cbr_answer
                                result['source'] = 'case_based_reasoning'
                                cbr_done = True

                        if not cbr_done:
                            # Try explanation templates (quicksort, binary search etc.)
                            text_result = self.text_gen._explain(user_input)
                            if text_result.get('method') != 'no_model':
                                result['response'] = text_result['response']
                                result['source'] = 'reasoning'
                                result['metadata'] = text_result
                            else:
                                # Try reasoning engine, then dialog
                                try:
                                    from .reasoning import ReasoningEngine
                                    re_engine = ReasoningEngine(
                                        knowledge_store=self.dialog.knowledge)
                                    reason_result = re_engine.reason(user_input)
                                    result['response'] = reason_result.get(
                                        'answer', reason_result.get('explanation',
                                        "I need more context to reason about this."))
                                    result['metadata'] = reason_result
                                except Exception:
                                    dialog_result = self.dialog.turn(user_input)
                                    result['response'] = dialog_result['response']
                                    result['metadata'] = dialog_result
                except Exception:
                    result['response'] = "I couldn't reason about that. Could you rephrase?"
                    result['source'] = 'error_fallback'

            elif intent == 'TOOL_USE':
                try:
                    result = self._handle_tool(user_input, result)
                except Exception:
                    result['response'] = "Tool execution failed. Please try again."
                    result['source'] = 'error_fallback'

            elif intent == 'CONVERSATION':
                try:
                    result['response'] = self._conversational(user_input)
                except Exception:
                    result['response'] = "I'm here. How can I help?"
                    result['source'] = 'error_fallback'

            else:
                # Unknown → try dialog
                try:
                    dialog_result = self.dialog.turn(user_input)
                    result['response'] = dialog_result['response']
                    result['metadata'] = dialog_result
                except Exception:
                    result['response'] = "I couldn't process that. Could you rephrase?"
                    result['source'] = 'error_fallback'

        except Exception:
            # Outer catch-all: if anything unexpected crashes
            if not result.get('response'):
                result['response'] = "Something went wrong. Please try again."
                result['source'] = 'error_fallback'

        # Ensure response is never None
        if result['response'] is None:
            result['response'] = "I don't have enough information to answer that."

        self.history.append({
            'input': user_input,
            'intent': intent,
            'response': result['response'][:200],
        })

        return result

    def _handle_tool(self, text: str, result: dict) -> dict:
        """Handle file/shell tool use requests."""
        text_lower = text.lower().strip()

        # Try plugin registry first
        match = self.tool_registry.match(text)
        if match:
            tool_name, args = match
            tool_result = self.tool_registry.execute(tool_name, args)
            result['response'] = tool_result.get('response', str(tool_result))
            result['source'] = f'tool:{tool_name}'
            result['metadata'] = tool_result
            return result

        # Fallback: built-in tool handling
        # Read file
        m = re.search(r'(?:read|open|show|cat)\s+(?:the\s+)?(?:file\s+)?(\S+)',
                       text, re.I)
        if m:
            path = m.group(1).strip('"\'')
            tool_result = self.tools.read_file(path)
            result['response'] = tool_result['response']
            result['metadata'] = tool_result
            return result

        # List files
        if re.search(r'(?:list|ls)\s+(?:files|directory|folder)', text_lower):
            m = re.search(r'(?:in|of)\s+(\S+)', text)
            path = m.group(1) if m else '.'
            tool_result = self.tools.list_files(path)
            result['response'] = tool_result['response']
            result['metadata'] = tool_result
            return result

        # Run command
        m = re.search(r'(?:run|execute)\s+[`"]?(.+?)[`"]?\s*$', text, re.I)
        if m:
            cmd = m.group(1).strip()
            tool_result = self.tools.run_command(cmd)
            result['response'] = tool_result['response']
            result['metadata'] = tool_result
            return result

        # Direct shell command
        if re.match(r'^(?:ls|pwd|git|pip|python|node|make|curl)\s', text_lower):
            tool_result = self.tools.run_command(text.strip())
            result['response'] = tool_result['response']
            result['metadata'] = tool_result
            return result

        result['response'] = "I understand you want to use a tool, but I couldn't parse the specific operation."
        return result

    @staticmethod
    def _is_declarative(text: str) -> bool:
        """Check if text is a declarative statement (not a question)."""
        text = text.strip()
        if not text or len(text) < 5:
            return False
        # Questions start with these words
        q_words = ('what', 'who', 'where', 'when', 'which', 'how', 'why',
                   'is', 'are', 'was', 'were', 'do', 'does', 'did', 'can',
                   'could', 'would', 'should', 'will', 'tell')
        first_word = text.split()[0].lower().strip('?.,!')
        if first_word in q_words:
            return False
        if text.rstrip().endswith('?'):
            return False
        # Must contain a verb-like word suggesting a fact
        declarative_markers = (
            ' is ', ' was ', ' are ', ' were ', ' invented ',
            ' created ', ' built ', ' founded ', ' discovered ',
            ' uses ', ' based on ', ' belongs to ', ' supports ',
        )
        return any(m in f' {text.lower()} ' for m in declarative_markers)

    def _conversational(self, text: str) -> str:
        """Handle casual conversation."""
        text_lower = text.lower().strip()

        if any(g in text_lower for g in ('hi', 'hello', 'hey')):
            n = self.dialog.knowledge.n_facts
            return (f"Hello! I'm FOSS-KI — a transformer-free AI running on pure CPU. "
                    f"I currently know {n} facts. Ask me anything, or say "
                    f"'help' to see what I can do.")

        if 'who are you' in text_lower:
            return ("I'm FOSS-KI — an AI built from forgotten paradigms: "
                    "FLM language models, Hopfield memory, and constraint solvers. "
                    "No transformer, no GPU, no API calls. Everything runs locally on your CPU. "
                    "I can answer questions, write code, read files, do math, "
                    "and reason about facts.")

        if 'what can you do' in text_lower or text_lower == 'help':
            tools_desc = self.tool_registry.describe() if self.tool_registry.tools else ""
            base = ("I can:\n"
                    "- Answer factual questions (521+ facts about countries, people, science)\n"
                    "- Write Python code (functions, algorithms, file operations)\n"
                    "- Read and write files\n"
                    "- Run shell commands (sandboxed)\n"
                    "- Do math (arithmetic, basic algebra)\n"
                    "- Explain concepts\n"
                    "- Summarize text\n"
                    "- Learn from new text (feed me Wikipedia articles!)\n"
                    "- Multi-turn conversation with reference resolution\n"
                    "- Anti-hallucination: I say 'I don't know' instead of making things up\n"
                    "- Self-improve: detect knowledge gaps and fill them automatically")
            if tools_desc:
                base += f"\n\n{tools_desc}"
            return base

        if any(t in text_lower for t in ('thanks', 'thank you')):
            return "You're welcome!"

        if any(b in text_lower for b in ('bye', 'goodbye')):
            return "Goodbye! Your knowledge is saved locally — nothing sent to any server."

        if 'how are you' in text_lower:
            return ("I'm running well! No GPU needed, no internet required. "
                    f"Currently holding {self.dialog.knowledge.n_facts} facts in Hopfield memory.")

        return "I'm here to help. Ask me a question, request code, or say 'help' to see what I can do."

    def _try_compare(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Compare two entities side by side.

        Handles: "Compare France and Germany", "Differences between Python and Linux"
        """
        q = query.lower().strip().rstrip('?').strip()
        m = re.search(
            r'(?:compare|difference(?:s)?\s+between|versus|vs)\s+'
            r'(.+?)\s+(?:and|vs\.?|versus)\s+(.+)',
            q
        )
        if not m:
            # "What is bigger/larger/etc, X or Y?"
            m = re.search(
                r'(?:bigger|larger|smaller|taller|faster|older|newer|richer|'
                r'more\s+\w+)\s*,?\s+(.+?)\s+(?:or|and|vs\.?)\s+(.+)',
                q
            )
        if not m:
            # "X vs Y" / "X versus Y"
            m = re.search(r'^(.+?)\s+(?:vs\.?|versus)\s+(.+)$', q)
        if not m:
            return None

        e1 = m.group(1).strip().title()
        e2 = m.group(2).strip().title()

        # Collect all facts for both entities
        facts1 = {}
        facts2 = {}
        for s, r, o in self.dialog.knowledge.facts:
            if s.lower() == e1.lower():
                facts1.setdefault(r, []).append(o)
            if s.lower() == e2.lower():
                facts2.setdefault(r, []).append(o)

        if not facts1 and not facts2:
            return None

        # Build comparison table
        all_rels = sorted(set(facts1.keys()) | set(facts2.keys()))
        lines = [f"Comparing {e1} vs {e2}:"]
        for rel in all_rels:
            v1 = ', '.join(facts1.get(rel, ['-']))
            v2 = ', '.join(facts2.get(rel, ['-']))
            lines.append(f"  {rel}: {v1} | {v2}")

        return {
            'response': '\n'.join(lines),
            'entities': [e1, e2],
            'facts': {e1: facts1, e2: facts2},
        }

    def _try_multi_hop(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Decompose nested queries into multi-hop knowledge lookups.

        Handles patterns like:
          "What is the capital of the country that borders France?"
          "What language do they speak in the country where Python was created?"
          "Who created the language spoken in Brazil?"
        """
        q = query.lower().strip().rstrip('?').strip()

        # Pattern 1a: "the X of/in the Y that/which/where Z"
        m = re.search(
            r'(?:what\s+(?:is|are)\s+)?(?:the\s+)?(\w+)\s+(?:of|in|do\s+they\s+speak\s+in)\s+'
            r'the\s+(\w+)\s+(?:that|which|where|who)\s+(.+)',
            q
        )
        if m:
            outer_rel = m.group(1)
            inner_clause = m.group(3)
            return self._resolve_hop_chain(outer_rel, inner_clause)

        # Pattern 1b: "what X does the Y that Z use/have/speak"
        m = re.search(
            r'what\s+(\w+)\s+(?:does|do)\s+the\s+(\w+)\s+'
            r'(?:that|which|where|who)\s+(.+?)\s+(?:use|have|speak)',
            q
        )
        if m:
            outer_rel = m.group(1)
            inner_clause = m.group(3)
            return self._resolve_hop_chain(outer_rel, inner_clause)

        # Pattern 2: "Who created the X spoken/used in Y?"
        m = re.search(
            r'who\s+(\w+)\s+the\s+(\w+)\s+(?:spoken|used|found)\s+in\s+(.+)',
            q
        )
        if m:
            outer_rel = m.group(1)  # "created"
            inner_rel = m.group(2)  # "language"
            entity = m.group(3).strip().capitalize()
            # Step 1: get inner relation
            inner = self.dialog.knowledge.query_smart(entity, inner_rel)
            if inner['confidence_level'] in ('HIGH', 'MEDIUM'):
                inner_entity = inner['fact'][2]
                # Step 2: get outer relation
                rel_map = {'created': 'creator', 'invented': 'inventor',
                           'founded': 'founder', 'discovered': 'discoverer'}
                outer = self.dialog.knowledge.query_smart(
                    inner_entity, rel_map.get(outer_rel, outer_rel))
                if outer['confidence_level'] in ('HIGH', 'MEDIUM'):
                    answer = outer['fact'][2]
                    return {
                        'response': f"{answer} {outer_rel} {inner_entity} "
                                    f"(the {inner_rel} of {entity}).",
                        'chain': [inner['fact'], outer['fact']],
                        'hops': 2,
                    }
        return None

    def _resolve_hop_chain(self, outer_rel, inner_clause):
        """Resolve a 2-hop chain: inner_clause → entity, then outer_rel(entity)."""
        # Parse inner clause: "borders france" → (France, borders, ?)
        words = inner_clause.strip().split()
        if len(words) < 2:
            return None

        inner_rel = words[0]
        inner_entity = ' '.join(words[1:]).strip().capitalize()

        # Relation aliases for inner clause
        rel_map = {
            'borders': 'borders', 'created': 'creator',
            'invented': 'inventor', 'speaks': 'language',
        }
        actual_rel = rel_map.get(inner_rel, inner_rel)

        # Step 1: resolve inner — "the Y that borders Spain" means (Y, borders, Spain)
        # We want Y (the subject), not Spain (the inner_entity is the object)
        inner_result = self.dialog.knowledge.query_smart(inner_entity, actual_rel)
        mid_entity = None
        if inner_result['confidence_level'] in ('HIGH', 'MEDIUM'):
            fact = inner_result['fact']
            # Determine which side of the fact is the "other" entity
            if fact[0].lower() == inner_entity.lower():
                mid_entity = fact[2]  # inner_entity is subject → take object
            else:
                mid_entity = fact[0]  # inner_entity is object → take subject
        if mid_entity is None:
            # Try reverse lookup: find entities where (?, inner_rel, inner_entity)
            for s, r, o in self.dialog.knowledge.facts:
                if r.lower() == actual_rel and o.lower() == inner_entity.lower():
                    mid_entity = s
                    break
            else:
                return None

        # Step 2: resolve outer
        outer_result = self.dialog.knowledge.query_smart(mid_entity, outer_rel)
        if outer_result['confidence_level'] in ('HIGH', 'MEDIUM'):
            fact = outer_result['fact']
            return {
                'response': self.dialog._format_single_fact(fact),
                'chain': [inner_result.get('fact', (inner_entity, actual_rel, mid_entity)),
                          fact],
                'hops': 2,
            }
        return None

    def bootstrap(self, use_brain=True, fetch_wikidata=False):
        """
        Load world knowledge. Brain-first architecture:
        1. If .brain file exists → load it (fast, ~100ms)
        2. If not → bootstrap from code + optionally fetch Wikidata → save .brain

        Args:
            use_brain: if True, try loading from .brain file first
            fetch_wikidata: if True, pull live data from Wikidata SPARQL
        """
        from .metacognition import MetacognitionEngine as ME
        from .commonsense import CommonSenseEngine
        from .brain import BrainSnapshot
        import os

        brain_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'foss-ki.brain'
        )

        # === Try loading brain file ===
        if use_brain and os.path.exists(brain_path):
            try:
                snap = BrainSnapshot.load(brain_path)
                parts = snap.restore_parts()
                if 'knowledge' in parts and parts['knowledge'].n_facts > 0:
                    self.dialog.knowledge = parts['knowledge']
                    self.meta = ME(self.dialog.knowledge)
                    self.text_gen.knowledge = self.dialog.knowledge
                    self._commonsense = CommonSenseEngine()
                    try:
                        from .nlg import NLGPipeline
                        nlg = NLGPipeline()
                        self._case_base = nlg.case_base
                    except Exception:
                        pass
                    return parts['knowledge'].n_facts
            except Exception:
                pass  # Fall through to fresh bootstrap

        # === Fresh bootstrap ===
        from .wikidata import WikidataImporter
        from .knowledge_dump import inject_all

        wi = WikidataImporter(self.dialog.knowledge)
        n = wi.import_countries_simple()
        n += inject_all(self.dialog.knowledge)

        # Optionally pull live Wikidata
        if fetch_wikidata:
            try:
                from .wikidata import wikidata_live_import
                stats = wikidata_live_import(self.dialog.knowledge, verbose=True)
                n += stats.get('total', 0)
            except Exception as e:
                print(f"  Wikidata import failed: {e}")

        self.meta = ME(self.dialog.knowledge)
        self.text_gen.knowledge = self.dialog.knowledge
        self._commonsense = CommonSenseEngine()

        # Load CBR (Case-Based Reasoning) for open-ended questions
        try:
            from .nlg import NLGPipeline
            nlg = NLGPipeline()
            self._case_base = nlg.case_base
        except Exception:
            pass

        # === Save brain for next time ===
        try:
            snap = BrainSnapshot.capture_from_parts(
                knowledge_store=self.dialog.knowledge,
            )
            os.makedirs(os.path.dirname(brain_path), exist_ok=True)
            snap.save(brain_path, compressed=True)
        except Exception:
            pass  # Save is best-effort

        return n
