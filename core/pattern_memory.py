"""
Pattern Memory — Learn From Mistakes, Remember Fixes
=====================================================
THE critical missing piece in the self-improvement loop.

Without this, the loop is amnesic:
  Predict → Execute → Compare → Attribute → FIX ... → forget everything

With this:
  Predict → Execute → Compare → Attribute → FIX → STORE PATTERN → next time: instant fix

Two types of patterns stored:
  1. ERROR→FIX: "When I see NameError for 'x', check if x was defined before use"
  2. SUCCESS: "This code pattern works for this type of task"

Storage: JSON file, survives restarts. The system gets smarter over time.
"""

import json
import os
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict


class Pattern:
    """A single learned pattern (error→fix or success template)."""

    def __init__(self, pattern_type: str, trigger: str, response: str,
                 context: str = '', confidence: float = 1.0,
                 times_used: int = 0, times_worked: int = 0):
        self.pattern_type = pattern_type  # 'error_fix' or 'success'
        self.trigger = trigger            # What triggers this pattern
        self.response = response          # What to do when triggered
        self.context = context            # Additional context
        self.confidence = confidence      # How reliable (0-1)
        self.times_used = times_used
        self.times_worked = times_worked
        self.created = time.time()
        self.last_used = time.time()

    @property
    def success_rate(self) -> float:
        if self.times_used == 0:
            return self.confidence
        return self.times_worked / self.times_used

    def to_dict(self) -> Dict:
        return {
            'type': self.pattern_type,
            'trigger': self.trigger,
            'response': self.response,
            'context': self.context,
            'confidence': self.confidence,
            'times_used': self.times_used,
            'times_worked': self.times_worked,
            'created': self.created,
            'last_used': self.last_used,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'Pattern':
        p = cls(
            pattern_type=d.get('type', 'error_fix'),
            trigger=d.get('trigger', ''),
            response=d.get('response', ''),
            context=d.get('context', ''),
            confidence=d.get('confidence', 1.0),
            times_used=d.get('times_used', 0),
            times_worked=d.get('times_worked', 0),
        )
        p.created = d.get('created', time.time())
        p.last_used = d.get('last_used', time.time())
        return p


class PatternMemory:
    """
    Persistent memory for error→fix patterns and success templates.

    The system gets smarter over time:
    - First time seeing NameError → debug from scratch
    - After 3 NameErrors → instant fix: "check variable definition order"
    - After 10 NameErrors → high confidence, apply automatically

    Storage: JSON file at data/patterns.json
    """

    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 'patterns.json'
        )
        self.patterns: Dict[str, Pattern] = {}  # key → Pattern
        self._by_type: Dict[str, List[str]] = defaultdict(list)
        self._by_error: Dict[str, List[str]] = defaultdict(list)
        self._load()
        self._load_defaults()

    def _pattern_key(self, trigger: str, context: str = '') -> str:
        """Generate a unique key for a pattern."""
        raw = f"{trigger}:{context}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def store_error_fix(self, error_type: str, error_msg: str,
                        fix_description: str, code_context: str = '') -> str:
        """
        Store an error→fix pattern.

        Args:
            error_type: e.g. "NameError", "TypeError", "SyntaxError"
            error_msg: the error message
            fix_description: what fixed it
            code_context: code that caused the error (for matching)

        Returns:
            pattern key
        """
        trigger = f"{error_type}: {error_msg}"
        key = self._pattern_key(trigger, code_context)

        if key in self.patterns:
            # Reinforce existing pattern
            self.patterns[key].times_used += 1
            self.patterns[key].times_worked += 1
            self.patterns[key].confidence = min(1.0,
                self.patterns[key].confidence + 0.1)
            self.patterns[key].last_used = time.time()
        else:
            pattern = Pattern(
                pattern_type='error_fix',
                trigger=trigger,
                response=fix_description,
                context=code_context,
                confidence=0.5,  # Start cautious
                times_used=1,
                times_worked=1,
            )
            self.patterns[key] = pattern
            self._by_type['error_fix'].append(key)
            self._by_error[error_type].append(key)

        self._save()
        return key

    def store_success(self, task_type: str, code: str,
                      description: str = '') -> str:
        """
        Store a successful code pattern.

        Args:
            task_type: e.g. "sort", "filter", "parse_json"
            code: the code that worked
            description: what it does

        Returns:
            pattern key
        """
        trigger = f"task:{task_type}"
        key = self._pattern_key(trigger, code[:100])

        pattern = Pattern(
            pattern_type='success',
            trigger=trigger,
            response=code,
            context=description,
            confidence=0.7,
            times_used=1,
            times_worked=1,
        )
        self.patterns[key] = pattern
        self._by_type['success'].append(key)
        self._save()
        return key

    def lookup_fix(self, error_type: str, error_msg: str = '',
                   code: str = '') -> Optional[str]:
        """
        Look up a fix for a given error.

        Returns the fix description if found, None otherwise.
        Uses fuzzy matching on error type + message.
        """
        # Exact error type match first
        if error_type in self._by_error:
            candidates = []
            for key in self._by_error[error_type]:
                if key in self.patterns:
                    p = self.patterns[key]
                    # Score by relevance
                    score = p.confidence * p.success_rate
                    if error_msg and error_msg in p.trigger:
                        score *= 2.0  # Exact message match bonus
                    if code and p.context and p.context[:50] in code:
                        score *= 1.5  # Code context match bonus
                    candidates.append((score, p))

            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                best = candidates[0][1]
                best.times_used += 1
                best.last_used = time.time()
                self._save()
                return best.response

        # Fuzzy match: search all error_fix patterns
        trigger_search = f"{error_type}"
        for key, p in self.patterns.items():
            if p.pattern_type == 'error_fix' and trigger_search in p.trigger:
                if p.confidence >= 0.3:
                    p.times_used += 1
                    p.last_used = time.time()
                    self._save()
                    return p.response

        return None

    def lookup_template(self, task_type: str) -> Optional[str]:
        """Look up a successful code template for a task type."""
        trigger = f"task:{task_type}"
        best = None
        best_score = 0

        for key in self._by_type.get('success', []):
            if key in self.patterns:
                p = self.patterns[key]
                if task_type in p.trigger:
                    score = p.confidence * p.success_rate
                    if score > best_score:
                        best = p
                        best_score = score

        if best:
            best.times_used += 1
            best.last_used = time.time()
            self._save()
            return best.response
        return None

    def record_outcome(self, key: str, worked: bool):
        """Record whether a pattern's suggestion actually worked."""
        if key in self.patterns:
            p = self.patterns[key]
            p.times_used += 1
            if worked:
                p.times_worked += 1
                p.confidence = min(1.0, p.confidence + 0.05)
            else:
                p.confidence = max(0.0, p.confidence - 0.1)
            p.last_used = time.time()
            self._save()

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total = len(self.patterns)
        by_type = defaultdict(int)
        total_uses = 0
        total_successes = 0

        for p in self.patterns.values():
            by_type[p.pattern_type] += 1
            total_uses += p.times_used
            total_successes += p.times_worked

        return {
            'total_patterns': total,
            'by_type': dict(by_type),
            'total_uses': total_uses,
            'total_successes': total_successes,
            'overall_success_rate': total_successes / max(total_uses, 1),
            'high_confidence': sum(1 for p in self.patterns.values()
                                   if p.confidence >= 0.8),
            'low_confidence': sum(1 for p in self.patterns.values()
                                  if p.confidence < 0.3),
        }

    def prune(self, min_confidence: float = 0.1, max_age_days: int = 90):
        """Remove patterns that are consistently wrong or very old unused."""
        now = time.time()
        max_age = max_age_days * 86400
        to_remove = []

        for key, p in self.patterns.items():
            if p.confidence < min_confidence and p.times_used >= 3:
                to_remove.append(key)
            elif (now - p.last_used) > max_age and p.times_used <= 1:
                to_remove.append(key)

        for key in to_remove:
            p = self.patterns[key]
            # Clean up indexes
            if p.pattern_type in self._by_type:
                self._by_type[p.pattern_type] = [
                    k for k in self._by_type[p.pattern_type] if k != key]
            error_type = p.trigger.split(':')[0] if ':' in p.trigger else ''
            if error_type in self._by_error:
                self._by_error[error_type] = [
                    k for k in self._by_error[error_type] if k != key]
            del self.patterns[key]

        if to_remove:
            self._save()
        return len(to_remove)

    def _load(self):
        """Load patterns from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                for key, d in data.get('patterns', {}).items():
                    p = Pattern.from_dict(d)
                    self.patterns[key] = p
                    self._by_type[p.pattern_type].append(key)
                    if p.pattern_type == 'error_fix':
                        error_type = p.trigger.split(':')[0].strip()
                        self._by_error[error_type].append(key)
            except (json.JSONDecodeError, KeyError):
                pass

    def _save(self):
        """Save patterns to disk."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = {
            'version': 1,
            'updated': time.time(),
            'patterns': {k: p.to_dict() for k, p in self.patterns.items()},
        }
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def _load_defaults(self):
        """Pre-load common Python error→fix patterns."""
        defaults = [
            ('NameError', "name 'x' is not defined",
             "Variable used before assignment. Check: 1) Typo in name? "
             "2) Defined in different scope? 3) Missing import?"),
            ('TypeError', "unsupported operand type",
             "Type mismatch in operation. Check: 1) str + int? Add str() or int() cast. "
             "2) None value? Add None check. 3) Wrong variable?"),
            ('TypeError', "'NoneType' object is not",
             "Function returned None. Check: 1) Missing return statement? "
             "2) Function returns None on some paths? 3) Assignment vs comparison?"),
            ('IndexError', "list index out of range",
             "List access out of bounds. Check: 1) Off-by-one? Use len()-1. "
             "2) Empty list? Add 'if lst:' guard. 3) Wrong variable?"),
            ('KeyError', "",
             "Dict key not found. Check: 1) Typo in key? 2) Use .get(key, default). "
             "3) Key not yet added?"),
            ('AttributeError', "has no attribute",
             "Wrong attribute/method name. Check: 1) Typo? 2) Wrong type? "
             "3) Module not imported correctly?"),
            ('ValueError', "invalid literal for int()",
             "String can't be converted to int. Check: 1) Input has non-numeric chars? "
             "2) Strip whitespace first. 3) Use try/except."),
            ('SyntaxError', "unexpected EOF",
             "Incomplete code. Check: 1) Missing closing bracket/paren? "
             "2) Missing colon after if/for/def? 3) Unclosed string?"),
            ('IndentationError', "",
             "Indentation wrong. Check: 1) Mixed tabs and spaces? "
             "2) Inconsistent indent level? 3) Missing indent after colon?"),
            ('ImportError', "No module named",
             "Module not installed or wrong name. Check: 1) pip install <module>? "
             "2) Typo in module name? 3) Virtual env active?"),
            ('ZeroDivisionError', "division by zero",
             "Dividing by zero. Check: 1) Add 'if divisor != 0' guard. "
             "2) Use max(divisor, 1). 3) Wrong variable?"),
            ('RecursionError', "maximum recursion depth",
             "Infinite recursion. Check: 1) Missing/wrong base case? "
             "2) Recursive call doesn't reduce problem? 3) Mutual recursion?"),
            ('FileNotFoundError', "No such file",
             "File doesn't exist. Check: 1) Typo in path? 2) Wrong working directory? "
             "3) Use os.path.exists() first."),
            ('StopIteration', "",
             "Iterator exhausted. Check: 1) next() on empty iterator? "
             "2) Use 'for' loop instead of manual next(). 3) Iterator already consumed?"),
            ('UnicodeDecodeError', "",
             "Encoding mismatch. Check: 1) open(file, encoding='utf-8')? "
             "2) Binary file opened as text? 3) Use 'rb' mode for binary."),
        ]

        for error_type, msg, fix in defaults:
            trigger = f"{error_type}: {msg}" if msg else error_type
            key = self._pattern_key(trigger)
            if key not in self.patterns:
                self.patterns[key] = Pattern(
                    pattern_type='error_fix',
                    trigger=trigger,
                    response=fix,
                    confidence=0.9,  # High confidence — these are well-known
                    times_used=0,
                    times_worked=0,
                )
                self._by_type['error_fix'].append(key)
                self._by_error[error_type].append(key)
