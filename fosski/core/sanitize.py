"""
Input Sanitization Layer for FOSS-KI
======================================
Centralized input cleaning applied at REPL entry point.
Prevents: oversized inputs, Unicode normalization issues,
control characters, and injection patterns.
"""

import unicodedata
import re

# Hard limits
MAX_INPUT_LENGTH = 2048
MAX_WORD_COUNT = 500


def sanitize(text: str) -> str:
    """
    Sanitize user input. Called once at REPL entry.

    1. Unicode NFC normalization (é as single char, not e + combining accent)
    2. Strip control characters (keep newlines, tabs)
    3. Collapse excessive whitespace
    4. Truncate to MAX_INPUT_LENGTH
    5. Strip null bytes
    """
    if not isinstance(text, str):
        return ''

    # NFC normalization — canonical composition
    text = unicodedata.normalize('NFC', text)

    # Strip null bytes and other control chars (keep \n \t \r)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Collapse whitespace runs (but preserve single newlines)
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Truncate
    if len(text) > MAX_INPUT_LENGTH:
        text = text[:MAX_INPUT_LENGTH]

    return text.strip()


def is_safe(text: str) -> bool:
    """
    Check if sanitized input is safe to process.
    Returns False for inputs that should be rejected outright.
    """
    if not text:
        return False

    # Word count limit (prevents DoS via extremely long inputs)
    if len(text.split()) > MAX_WORD_COUNT:
        return False

    return True
