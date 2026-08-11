"""
Regex Builder — Natural Language → Regular Expressions
========================================================
"Find all email addresses" → r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

Users describe what they want to match, FOSS-KI builds the regex.
Also: regex testing, explanation, and common pattern library.
"""

import re
from typing import Dict, Any, List, Optional, Tuple


# === Common Pattern Library ===

PATTERNS = {
    'email': {
        'pattern': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'description': 'Email address',
        'examples': ['user@example.com', 'name.last@domain.co.uk'],
    },
    'url': {
        'pattern': r'https?://[^\s<>"\']+',
        'description': 'URL (http/https)',
        'examples': ['https://example.com', 'http://test.org/page?q=1'],
    },
    'ipv4': {
        'pattern': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        'description': 'IPv4 address',
        'examples': ['192.168.1.1', '10.0.0.1'],
    },
    'ipv6': {
        'pattern': r'(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}',
        'description': 'IPv6 address (full form)',
        'examples': ['2001:0db8:85a3:0000:0000:8a2e:0370:7334'],
    },
    'phone': {
        'pattern': r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]{6,15}',
        'description': 'Phone number (international)',
        'examples': ['+1-555-123-4567', '(555) 123-4567'],
    },
    'date_iso': {
        'pattern': r'\d{4}-\d{2}-\d{2}',
        'description': 'Date (ISO 8601: YYYY-MM-DD)',
        'examples': ['2026-03-13', '2024-01-01'],
    },
    'date_eu': {
        'pattern': r'\d{1,2}[./]\d{1,2}[./]\d{2,4}',
        'description': 'Date (European: DD.MM.YYYY or DD/MM/YYYY)',
        'examples': ['13.03.2026', '01/12/2024'],
    },
    'time_24h': {
        'pattern': r'\b([01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\b',
        'description': 'Time (24h format)',
        'examples': ['14:30', '23:59:59'],
    },
    'hex_color': {
        'pattern': r'#(?:[0-9a-fA-F]{3}){1,2}\b',
        'description': 'Hex color code',
        'examples': ['#FF0000', '#abc'],
    },
    'mac_address': {
        'pattern': r'(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}',
        'description': 'MAC address',
        'examples': ['01:23:45:67:89:AB'],
    },
    'credit_card': {
        'pattern': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        'description': 'Credit card number (with optional spaces/dashes)',
        'examples': ['4111-1111-1111-1111', '4111 1111 1111 1111'],
    },
    'uuid': {
        'pattern': r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
        'description': 'UUID',
        'examples': ['550e8400-e29b-41d4-a716-446655440000'],
    },
    'python_import': {
        'pattern': r'^\s*(?:from\s+[\w.]+\s+)?import\s+[\w.,\s]+',
        'description': 'Python import statement',
        'examples': ['import os', 'from typing import Dict'],
    },
    'python_function': {
        'pattern': r'^\s*def\s+(\w+)\s*\(([^)]*)\)\s*(?:->.*?)?:',
        'description': 'Python function definition',
        'examples': ['def foo(x, y):', 'def bar() -> int:'],
    },
    'python_class': {
        'pattern': r'^\s*class\s+(\w+)(?:\(([^)]*)\))?\s*:',
        'description': 'Python class definition',
        'examples': ['class Foo:', 'class Bar(Base):'],
    },
    'html_tag': {
        'pattern': r'<([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>',
        'description': 'HTML opening tag',
        'examples': ['<div class="foo">', '<p>'],
    },
    'integer': {
        'pattern': r'-?\b\d+\b',
        'description': 'Integer (positive or negative)',
        'examples': ['42', '-7', '0'],
    },
    'float_number': {
        'pattern': r'-?\b\d+\.\d+\b',
        'description': 'Floating point number',
        'examples': ['3.14', '-0.5'],
    },
    'word': {
        'pattern': r'\b[a-zA-Z]+\b',
        'description': 'Single word (letters only)',
        'examples': ['hello', 'World'],
    },
    'sentence': {
        'pattern': r'[A-Z][^.!?]*[.!?]',
        'description': 'Sentence (starts with capital, ends with punctuation)',
        'examples': ['Hello world.', 'How are you?'],
    },
    'hashtag': {
        'pattern': r'#\w+',
        'description': 'Hashtag',
        'examples': ['#python', '#AI'],
    },
    'mention': {
        'pattern': r'@\w+',
        'description': 'Social media mention',
        'examples': ['@user', '@foss_ki'],
    },
    'file_path': {
        'pattern': r'(?:/[\w.-]+)+/?',
        'description': 'Unix file path',
        'examples': ['/home/user/file.txt', '/usr/local/bin/'],
    },
    'semver': {
        'pattern': r'\bv?\d+\.\d+\.\d+(?:-[\w.]+)?(?:\+[\w.]+)?\b',
        'description': 'Semantic version',
        'examples': ['v1.2.3', '2.0.0-beta.1'],
    },
}


# === NL → Regex Intent Mapping ===

NL_PATTERNS = [
    # Email
    (r'\b(?:email|e-mail|mail)\s*(?:address(?:es)?)?', 'email'),
    # URL
    (r'\b(?:url|link|website|web\s*address|http)', 'url'),
    # IP
    (r'\b(?:ip\s*(?:v4)?\s*address|ip)', 'ipv4'),
    (r'\b(?:ipv6)', 'ipv6'),
    # Phone
    (r'\b(?:phone|telephone|mobile)\s*(?:number)?', 'phone'),
    # Date
    (r'\b(?:date|datum)(?:\s+(?:iso|yyyy))?', 'date_iso'),
    (r'\b(?:european\s+date|eu\s+date|dd\.mm)', 'date_eu'),
    # Time
    (r'\b(?:time|uhrzeit|clock)', 'time_24h'),
    # Number
    (r'\b(?:float|decimal|floating)', 'float_number'),
    (r'\b(?:integer|number|digit|zahl)', 'integer'),
    # Color
    (r'\b(?:hex\s*color|color\s*code|farbe)', 'hex_color'),
    # MAC
    (r'\b(?:mac\s*address)', 'mac_address'),
    # Credit card
    (r'\b(?:credit\s*card|kreditkarte|card\s*number)', 'credit_card'),
    # UUID
    (r'\b(?:uuid|guid)', 'uuid'),
    # Code
    (r'\b(?:import\s*statement|python\s*import)', 'python_import'),
    (r'\b(?:function\s*def|python\s*function|def\s)', 'python_function'),
    (r'\b(?:class\s*def|python\s*class)', 'python_class'),
    # HTML
    (r'\b(?:html\s*tag|tag)', 'html_tag'),
    # Social
    (r'\b(?:hashtag)', 'hashtag'),
    (r'\b(?:mention|@\s*user)', 'mention'),
    # File
    (r'\b(?:file\s*path|path|pfad)', 'file_path'),
    # Version
    (r'\b(?:version|semver|semantic\s*version)', 'semver'),
    # Sentence
    (r'\b(?:sentence|satz)', 'sentence'),
    # Word
    (r'\b(?:word|wort)\b', 'word'),
]


class RegexBuilder:
    """
    Build, test, and explain regular expressions.

    Usage:
        rb = RegexBuilder()
        result = rb.from_natural_language("find all email addresses")
        matches = rb.test(r'\d+', "I have 3 cats and 2 dogs")
        explanation = rb.explain(r'\b\d{4}-\d{2}-\d{2}\b')
    """

    def __init__(self):
        self.patterns = dict(PATTERNS)
        self.history: List[Dict] = []

    def from_natural_language(self, description: str) -> Dict[str, Any]:
        """
        Convert a natural language description to a regex.

        Returns:
            dict with: pattern, name, description, examples, confidence
        """
        desc_lower = description.lower()

        # Try NL pattern matching
        for nl_pattern, pattern_name in NL_PATTERNS:
            if re.search(nl_pattern, desc_lower):
                info = self.patterns[pattern_name]
                return {
                    'pattern': info['pattern'],
                    'name': pattern_name,
                    'description': info['description'],
                    'examples': info['examples'],
                    'confidence': 0.9,
                }

        # Try keyword match against pattern descriptions
        best_match = None
        best_score = 0
        desc_words = set(desc_lower.split())

        for name, info in self.patterns.items():
            words = set(info['description'].lower().split()) | set(name.replace('_', ' ').split())
            overlap = len(desc_words & words)
            if overlap > best_score:
                best_score = overlap
                best_match = name

        if best_match and best_score >= 1:
            info = self.patterns[best_match]
            return {
                'pattern': info['pattern'],
                'name': best_match,
                'description': info['description'],
                'examples': info['examples'],
                'confidence': min(best_score / 3, 0.8),
            }

        # Build custom pattern from description
        custom = self._build_custom(description)
        if custom:
            return {
                'pattern': custom,
                'name': 'custom',
                'description': description,
                'examples': [],
                'confidence': 0.5,
            }

        return {
            'pattern': None,
            'name': None,
            'description': description,
            'confidence': 0.0,
            'error': 'Could not build regex from description',
        }

    def _build_custom(self, description: str) -> Optional[str]:
        """Try to build a custom regex from descriptive patterns."""
        desc = description.lower()

        # "N digits" or "N characters"
        m = re.search(r'(\d+)\s+(?:digit|number|char|character|letter)s?', desc)
        if m:
            n = m.group(1)
            if 'digit' in desc or 'number' in desc:
                return rf'\d{{{n}}}'
            elif 'letter' in desc:
                return rf'[a-zA-Z]{{{n}}}'
            else:
                return rf'.{{{n}}}'

        # "starts with X"
        m = re.search(r'starts?\s+with\s+["\']?(\w+)["\']?', desc)
        if m:
            return rf'^{re.escape(m.group(1))}'

        # "ends with X"
        m = re.search(r'ends?\s+with\s+["\']?(\w+)["\']?', desc)
        if m:
            return rf'{re.escape(m.group(1))}$'

        # "contains X"
        m = re.search(r'contains?\s+["\']?(\w+)["\']?', desc)
        if m:
            return rf'.*{re.escape(m.group(1))}.*'

        # "between X and Y"
        m = re.search(r'between\s+["\']?(.+?)["\']?\s+and\s+["\']?(.+?)["\']?', desc)
        if m:
            return rf'{re.escape(m.group(1))}.*?{re.escape(m.group(2))}'

        return None

    def test(self, pattern: str, text: str, flags: int = 0) -> Dict[str, Any]:
        """
        Test a regex pattern against text.

        Returns:
            dict with: matches, count, groups, positions
        """
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return {'error': str(e), 'valid': False}

        matches = []
        for m in compiled.finditer(text):
            match_info = {
                'match': m.group(),
                'start': m.start(),
                'end': m.end(),
            }
            if m.groups():
                match_info['groups'] = list(m.groups())
            matches.append(match_info)

        return {
            'valid': True,
            'matches': matches,
            'count': len(matches),
            'pattern': pattern,
        }

    def explain(self, pattern: str) -> str:
        """
        Human-readable explanation of a regex pattern.
        """
        explanations = {
            r'\d': 'any digit (0-9)',
            r'\w': 'any word character (a-z, A-Z, 0-9, _)',
            r'\s': 'any whitespace',
            r'\b': 'word boundary',
            r'.': 'any character',
            r'^': 'start of string',
            r'$': 'end of string',
            r'+': 'one or more',
            r'*': 'zero or more',
            r'?': 'zero or one (optional)',
        }

        parts = []
        i = 0
        while i < len(pattern):
            c = pattern[i]

            # Escape sequences
            if c == '\\' and i + 1 < len(pattern):
                seq = pattern[i:i+2]
                if seq in explanations:
                    parts.append(f"  {seq} → {explanations[seq]}")
                else:
                    parts.append(f"  {seq} → literal '{pattern[i+1]}'")
                i += 2
                continue

            # Quantifiers with braces
            if c == '{':
                end = pattern.find('}', i)
                if end > i:
                    quant = pattern[i:end+1]
                    nums = pattern[i+1:end]
                    if ',' in nums:
                        lo, hi = nums.split(',', 1)
                        parts.append(f"  {quant} → {lo.strip() or '0'} to {hi.strip() or '∞'} times")
                    else:
                        parts.append(f"  {quant} → exactly {nums} times")
                    i = end + 1
                    continue

            # Character classes
            if c == '[':
                end = pattern.find(']', i)
                if end > i:
                    cls = pattern[i:end+1]
                    parts.append(f"  {cls} → one of these characters")
                    i = end + 1
                    continue

            # Groups
            if c == '(':
                depth = 1
                j = i + 1
                while j < len(pattern) and depth > 0:
                    if pattern[j] == '(':
                        depth += 1
                    elif pattern[j] == ')':
                        depth -= 1
                    j += 1
                group = pattern[i:j]
                if group.startswith('(?:'):
                    parts.append(f"  {group} → non-capturing group")
                elif group.startswith('(?='):
                    parts.append(f"  {group} → lookahead")
                else:
                    parts.append(f"  {group} → capturing group")
                i = j
                continue

            # Single character explanations
            if c in explanations:
                parts.append(f"  {c} → {explanations[c]}")
            elif c not in '|)':
                parts.append(f"  {c} → literal '{c}'")

            i += 1

        return f"Pattern: {pattern}\n" + '\n'.join(parts) if parts else f"Pattern: {pattern}"

    def get_pattern(self, name: str) -> Optional[Dict]:
        """Get a named pattern from the library."""
        return self.patterns.get(name)

    def list_patterns(self) -> List[str]:
        """List all available pattern names."""
        return sorted(self.patterns.keys())

    def add_pattern(self, name: str, pattern: str, description: str,
                    examples: Optional[List[str]] = None):
        """Add a custom pattern to the library."""
        self.patterns[name] = {
            'pattern': pattern,
            'description': description,
            'examples': examples or [],
        }
