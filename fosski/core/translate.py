"""
Simple Translation — Dictionary-Based DE↔EN
==============================================
Basic word-by-word translation with phrase support.
Not a replacement for DeepL/Google — but handles common phrases.
"""

import re
from typing import Dict, List, Optional, Tuple


# Common word translations (EN → DE and DE → EN)
EN_DE = {
    # Greetings
    'hello': 'hallo', 'goodbye': 'auf wiedersehen', 'good morning': 'guten morgen',
    'good evening': 'guten abend', 'good night': 'gute nacht',
    'please': 'bitte', 'thank you': 'danke', 'thanks': 'danke',
    'yes': 'ja', 'no': 'nein', 'sorry': 'entschuldigung',
    'how are you': 'wie geht es dir', 'i am fine': 'mir geht es gut',
    'nice to meet you': 'freut mich',

    # Common words
    'the': 'der/die/das', 'a': 'ein/eine', 'is': 'ist', 'are': 'sind',
    'was': 'war', 'were': 'waren', 'have': 'haben', 'has': 'hat',
    'do': 'tun', 'does': 'tut', 'did': 'tat',
    'not': 'nicht', 'and': 'und', 'or': 'oder', 'but': 'aber',
    'with': 'mit', 'without': 'ohne', 'for': 'für', 'from': 'von',
    'to': 'zu', 'in': 'in', 'on': 'auf', 'at': 'an',
    'of': 'von', 'by': 'von', 'about': 'über',

    # Nouns
    'house': 'Haus', 'car': 'Auto', 'water': 'Wasser', 'food': 'Essen',
    'book': 'Buch', 'dog': 'Hund', 'cat': 'Katze', 'tree': 'Baum',
    'sun': 'Sonne', 'moon': 'Mond', 'star': 'Stern',
    'man': 'Mann', 'woman': 'Frau', 'child': 'Kind',
    'father': 'Vater', 'mother': 'Mutter', 'friend': 'Freund',
    'time': 'Zeit', 'day': 'Tag', 'night': 'Nacht',
    'year': 'Jahr', 'month': 'Monat', 'week': 'Woche',
    'city': 'Stadt', 'country': 'Land', 'world': 'Welt',
    'money': 'Geld', 'work': 'Arbeit', 'school': 'Schule',
    'computer': 'Computer', 'phone': 'Telefon', 'internet': 'Internet',
    'love': 'Liebe', 'life': 'Leben', 'death': 'Tod',
    'door': 'Tür', 'window': 'Fenster', 'table': 'Tisch',
    'bread': 'Brot', 'beer': 'Bier', 'coffee': 'Kaffee',

    # Verbs
    'eat': 'essen', 'drink': 'trinken', 'sleep': 'schlafen',
    'go': 'gehen', 'come': 'kommen', 'see': 'sehen',
    'know': 'wissen', 'think': 'denken', 'say': 'sagen',
    'want': 'wollen', 'can': 'können', 'must': 'müssen',
    'need': 'brauchen', 'give': 'geben', 'take': 'nehmen',
    'make': 'machen', 'find': 'finden', 'help': 'helfen',
    'read': 'lesen', 'write': 'schreiben', 'speak': 'sprechen',
    'learn': 'lernen', 'teach': 'lehren', 'understand': 'verstehen',
    'love': 'lieben', 'like': 'mögen', 'live': 'leben',
    'run': 'laufen', 'walk': 'gehen', 'swim': 'schwimmen',
    'fly': 'fliegen', 'play': 'spielen', 'sing': 'singen',

    # Adjectives
    'good': 'gut', 'bad': 'schlecht', 'big': 'groß', 'small': 'klein',
    'old': 'alt', 'new': 'neu', 'young': 'jung',
    'hot': 'heiß', 'cold': 'kalt', 'warm': 'warm',
    'fast': 'schnell', 'slow': 'langsam',
    'beautiful': 'schön', 'ugly': 'hässlich',
    'happy': 'glücklich', 'sad': 'traurig',
    'right': 'richtig', 'wrong': 'falsch',
    'true': 'wahr', 'false': 'falsch',
    'easy': 'einfach', 'hard': 'schwer',
    'long': 'lang', 'short': 'kurz',

    # Numbers
    'one': 'eins', 'two': 'zwei', 'three': 'drei', 'four': 'vier',
    'five': 'fünf', 'six': 'sechs', 'seven': 'sieben', 'eight': 'acht',
    'nine': 'neun', 'ten': 'zehn',

    # Question words
    'what': 'was', 'who': 'wer', 'where': 'wo', 'when': 'wann',
    'why': 'warum', 'how': 'wie', 'which': 'welcher',
}

# Build reverse dictionary
DE_EN = {}
for en, de in EN_DE.items():
    # Handle multiple translations (der/die/das)
    for variant in de.split('/'):
        DE_EN[variant.lower()] = en


class SimpleTranslator:
    """Dictionary-based EN↔DE translator."""

    def __init__(self):
        self.en_de = dict(EN_DE)
        self.de_en = dict(DE_EN)

    def translate(self, text: str, target: str = 'de') -> Dict[str, str]:
        """
        Translate text.

        Args:
            text: Input text
            target: 'de' for English→German, 'en' for German→English

        Returns:
            {'translation': str, 'source': str, 'target': str, 'coverage': float}
        """
        if target == 'de':
            return self._translate_en_to_de(text)
        elif target == 'en':
            return self._translate_de_to_en(text)
        else:
            return {'translation': text, 'source': '?', 'target': target, 'coverage': 0.0}

    def _translate_en_to_de(self, text: str) -> Dict[str, str]:
        """Translate English to German."""
        result_words = []
        words = text.split()
        translated = 0
        i = 0

        while i < len(words):
            # Try multi-word phrases first (longest match)
            found = False
            for length in range(min(4, len(words) - i), 0, -1):
                phrase = ' '.join(words[i:i+length]).lower().rstrip('.,!?')
                if phrase in self.en_de:
                    translation = self.en_de[phrase]
                    # Preserve capitalization
                    if words[i][0].isupper():
                        translation = translation[0].upper() + translation[1:]
                    result_words.append(translation)
                    translated += length
                    i += length
                    found = True
                    break

            if not found:
                result_words.append(words[i])  # Keep untranslated
                i += 1

        coverage = translated / len(words) if words else 0

        return {
            'translation': ' '.join(result_words),
            'source': 'en',
            'target': 'de',
            'coverage': round(coverage, 2),
        }

    def _translate_de_to_en(self, text: str) -> Dict[str, str]:
        """Translate German to English."""
        result_words = []
        words = text.split()
        translated = 0
        i = 0

        while i < len(words):
            found = False
            for length in range(min(4, len(words) - i), 0, -1):
                phrase = ' '.join(words[i:i+length]).lower().rstrip('.,!?')
                if phrase in self.de_en:
                    translation = self.de_en[phrase]
                    if words[i][0].isupper():
                        translation = translation[0].upper() + translation[1:]
                    result_words.append(translation)
                    translated += length
                    i += length
                    found = True
                    break

            if not found:
                result_words.append(words[i])
                i += 1

        coverage = translated / len(words) if words else 0

        return {
            'translation': ' '.join(result_words),
            'source': 'de',
            'target': 'en',
            'coverage': round(coverage, 2),
        }

    def detect_language(self, text: str) -> str:
        """Simple language detection based on word and phrase overlap."""
        text_lower = text.lower().strip()
        words = set(text_lower.split())

        en_score = sum(1 for w in words if w in self.en_de)
        de_score = sum(1 for w in words if w in self.de_en)

        # Also check multi-word phrases
        for phrase in self.de_en:
            if ' ' in phrase and phrase in text_lower:
                de_score += len(phrase.split())
        for phrase in self.en_de:
            if ' ' in phrase and phrase in text_lower:
                en_score += len(phrase.split())

        # German-specific letter detection
        if any(c in text for c in 'äöüßÄÖÜ'):
            de_score += 3

        if de_score > en_score:
            return 'de'
        return 'en'

    def lookup(self, word: str) -> Dict[str, str]:
        """Look up a single word in both directions."""
        result = {}
        w = word.lower().strip()

        if w in self.en_de:
            result['en→de'] = self.en_de[w]
        if w in self.de_en:
            result['de→en'] = self.de_en[w]

        return result

    def add_word(self, en: str, de: str):
        """Add a translation pair."""
        self.en_de[en.lower()] = de
        self.de_en[de.lower()] = en

    def stats(self) -> Dict[str, int]:
        return {
            'en_de_entries': len(self.en_de),
            'de_en_entries': len(self.de_en),
        }
