"""
Creative Writing — Template-Based Text Generation
====================================================
Generates creative text forms using templates and word banks.
Not a neural network — uses structured templates + random selection.

Supported forms:
  - Haiku (5-7-5 syllables)
  - Limerick (AABBA rhyme)
  - Proverb generation
  - Pro/Con arguments
  - ELI5 (Explain Like I'm 5)
"""

import re
import random
from typing import Dict, Any, List, Tuple, Optional


# ============================================================
# Syllable Counter (approximate)
# ============================================================

def count_syllables(word: str) -> int:
    """Approximate syllable count for English words."""
    word = word.lower().strip()
    if not word:
        return 0
    # Special cases
    if len(word) <= 3:
        return 1

    count = 0
    vowels = 'aeiouy'
    prev_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    # Adjust for silent e
    if word.endswith('e') and count > 1:
        count -= 1
    # Adjust for -le endings
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        count += 1

    return max(1, count)


def count_line_syllables(line: str) -> int:
    """Count syllables in a line of text."""
    return sum(count_syllables(w) for w in line.split())


# ============================================================
# Word Banks by Topic
# ============================================================

TOPIC_WORDS = {
    'programming': {
        'nouns': ['code', 'bug', 'loop', 'function', 'data', 'server', 'screen',
                  'keyboard', 'logic', 'variable', 'stack', 'array', 'class',
                  'thread', 'process', 'file', 'error', 'syntax', 'algorithm'],
        'verbs': ['compile', 'debug', 'run', 'code', 'type', 'test', 'push',
                  'merge', 'build', 'deploy', 'parse', 'iterate', 'return'],
        'adjectives': ['clean', 'broken', 'fast', 'slow', 'recursive', 'abstract',
                       'elegant', 'complex', 'simple', 'infinite', 'logical'],
    },
    'nature': {
        'nouns': ['rain', 'wind', 'tree', 'leaf', 'river', 'mountain', 'ocean',
                  'flower', 'bird', 'cloud', 'sun', 'moon', 'star', 'snow',
                  'forest', 'meadow', 'stone', 'sky', 'dawn', 'dusk'],
        'verbs': ['falls', 'grows', 'flows', 'blooms', 'shines', 'rises',
                  'whispers', 'drifts', 'soars', 'rests', 'fades', 'glows'],
        'adjectives': ['gentle', 'wild', 'silent', 'ancient', 'golden', 'misty',
                       'deep', 'tall', 'bright', 'dark', 'warm', 'cold'],
    },
    'life': {
        'nouns': ['time', 'heart', 'dream', 'path', 'door', 'light', 'shadow',
                  'memory', 'journey', 'moment', 'silence', 'hope', 'love'],
        'verbs': ['passes', 'beats', 'opens', 'fades', 'shines', 'grows',
                  'changes', 'begins', 'ends', 'finds', 'seeks', 'waits'],
        'adjectives': ['brief', 'endless', 'quiet', 'bright', 'new', 'old',
                       'gentle', 'strong', 'fleeting', 'deep', 'true'],
    },
    'default': {
        'nouns': ['world', 'day', 'night', 'voice', 'hand', 'eye', 'mind',
                  'word', 'thought', 'truth', 'change', 'step'],
        'verbs': ['moves', 'speaks', 'turns', 'falls', 'rises', 'breaks',
                  'holds', 'finds', 'shows', 'knows', 'grows', 'stays'],
        'adjectives': ['still', 'swift', 'bright', 'dark', 'clear', 'soft',
                       'sharp', 'calm', 'bold', 'warm', 'free', 'vast'],
    },
}

# ============================================================
# Haiku Templates (5-7-5 syllable patterns)
# ============================================================

HAIKU_TEMPLATES = [
    # (5-syllable line, 7-syllable line, 5-syllable line)
    ("{adj} {noun} {verb}", "{noun} and {noun} together {verb}", "{adj} {noun} {verb}"),
    ("the {adj} {noun}", "{verb} across the {adj} {noun}", "in {adj} {noun}"),
    ("{noun} in the {noun}", "a {adj} {noun} {verb} here", "{noun} {verb} {adv}"),
    ("{adj} {noun} waits", "for the {noun} to {verb} again", "silence {verb}"),
    ("under {adj} {noun}", "{noun} and {noun} dance as one", "light {verb} gently"),
]


class CreativeWriter:
    """Generate creative text from templates."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def haiku(self, topic: str = '') -> str:
        """Generate a haiku about a topic."""
        words = self._get_words(topic)

        # Try multiple times to get syllable counts right
        for _ in range(20):
            line1 = self._generate_line(words, target_syllables=5)
            line2 = self._generate_line(words, target_syllables=7)
            line3 = self._generate_line(words, target_syllables=5)

            if (count_line_syllables(line1) == 5 and
                count_line_syllables(line2) == 7 and
                count_line_syllables(line3) == 5):
                return f"{line1}\n{line2}\n{line3}"

        # Fallback: use templates
        template = random.choice(HAIKU_TEMPLATES)
        lines = []
        for line_tmpl in template:
            line = line_tmpl.format(
                noun=random.choice(words['nouns']),
                verb=random.choice(words['verbs']),
                adj=random.choice(words['adjectives']),
                adv='softly',
            )
            lines.append(line)
        return '\n'.join(lines)

    def _generate_line(self, words: Dict[str, List[str]],
                       target_syllables: int) -> str:
        """Generate a line with target syllable count."""
        patterns = {
            5: [
                [('adjectives', 1), ('nouns', 1), ('verbs', 1)],
                [('nouns', 1), ('verbs', 1), ('adverb', 'softly')],
                [('the', 'the'), ('adjectives', 1), ('nouns', 1)],
            ],
            7: [
                [('nouns', 1), ('verbs', 1), ('adjectives', 1), ('nouns', 1)],
                [('adjectives', 1), ('nouns', 1), ('verbs', 1), ('adjectives', 1)],
                [('a', 'a'), ('adjectives', 1), ('nouns', 1), ('verbs', 1), ('nouns', 1)],
            ],
        }

        for pattern in patterns.get(target_syllables, patterns[5]):
            line_words = []
            for item in pattern:
                if isinstance(item[1], str):
                    line_words.append(item[1])
                else:
                    category = item[0]
                    if category in words:
                        line_words.append(random.choice(words[category]))
                    else:
                        line_words.append(random.choice(words['nouns']))

            line = ' '.join(line_words)
            if count_line_syllables(line) == target_syllables:
                return line

        # Just return something reasonable
        return ' '.join(random.choice(words['nouns']) for _ in range(target_syllables // 2 + 1))

    def pros_and_cons(self, topic: str, knowledge=None) -> str:
        """
        Generate a pro/con analysis.

        Uses CommonSenseEngine if available, otherwise uses generic templates.
        """
        topic_title = topic.strip().title()

        # Try to get domain-specific facts
        pros = []
        cons = []

        if knowledge:
            facts = knowledge._by_subject.get(topic.lower(), [])
            for r, o, w in facts:
                if r in ('causes', 'used_for', 'has_property'):
                    if w >= 3.0:
                        pros.append(o.replace('_', ' '))
                    elif w < 2.0:
                        cons.append(o.replace('_', ' '))

        # Generic template-based pros/cons
        generic_pros = [
            f"{topic_title} can lead to progress and innovation.",
            f"{topic_title} offers significant benefits when used responsibly.",
            f"Many people and organizations rely on {topic_title.lower()}.",
        ]
        generic_cons = [
            f"{topic_title} may have unintended consequences.",
            f"Over-reliance on {topic_title.lower()} can create problems.",
            f"Not everyone has equal access to {topic_title.lower()}.",
        ]

        # Build from knowledge or use generic
        pro_lines = []
        for p in (pros[:3] if pros else []):
            pro_lines.append(f"  + {p.capitalize()}")
        if len(pro_lines) < 3:
            for g in generic_pros[:3 - len(pro_lines)]:
                pro_lines.append(f"  + {g}")

        con_lines = []
        for c in (cons[:3] if cons else []):
            con_lines.append(f"  - {c.capitalize()}")
        if len(con_lines) < 3:
            for g in generic_cons[:3 - len(con_lines)]:
                con_lines.append(f"  - {g}")

        text = f"Pros and cons of {topic_title}:\n\n"
        text += "Pros:\n" + '\n'.join(pro_lines) + "\n\n"
        text += "Cons:\n" + '\n'.join(con_lines)

        return text

    def eli5(self, topic: str, knowledge=None) -> str:
        """
        Explain Like I'm 5 — simple explanation of a topic.
        """
        topic_lower = topic.lower().strip()
        topic_title = topic.strip().title()

        # Get facts if available
        facts = []
        if knowledge:
            for r, o, w in knowledge._by_subject.get(topic_lower, []):
                facts.append((r, o))

        # Build simple explanation
        explanation = f"Imagine {topic_title.lower()} is like... "

        # Use analogies based on type
        is_a = ''
        used_for = ''
        properties = []
        for r, o in facts:
            if r == 'is_a' or r == 'type':
                is_a = o.replace('_', ' ')
            elif r == 'used_for':
                used_for = o.replace('_', ' ')
            elif r == 'has_property':
                properties.append(o.replace('_', ' '))

        if is_a:
            explanation = f"{topic_title} is a kind of {is_a}. "
        else:
            explanation = f"Let me explain {topic_title.lower()}. "

        if used_for:
            explanation += f"People use it for {used_for}. "
        if properties:
            explanation += f"It is {', '.join(properties[:3])}. "

        # Add child-friendly closing
        explanation += f"Think of it as something that helps us understand the world better!"

        return explanation

    def _get_words(self, topic: str) -> Dict[str, List[str]]:
        """Get word bank for a topic."""
        topic_lower = topic.lower().strip()

        # Check for known topics
        for key in TOPIC_WORDS:
            if key in topic_lower or topic_lower in key:
                return TOPIC_WORDS[key]

        # Check for common mappings
        if any(w in topic_lower for w in ['code', 'program', 'software', 'computer', 'debug']):
            return TOPIC_WORDS['programming']
        if any(w in topic_lower for w in ['tree', 'flower', 'rain', 'sun', 'ocean', 'forest']):
            return TOPIC_WORDS['nature']
        if any(w in topic_lower for w in ['love', 'life', 'death', 'time', 'dream']):
            return TOPIC_WORDS['life']

        return TOPIC_WORDS['default']
