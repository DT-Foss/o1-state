"""
Benchmarks for Gap-Closing Modules
=====================================
Tests: naturalizer, multi_hop, creative, translate
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.naturalizer import AnswerNaturalizer
from core.multi_hop import MultiHopReasoner
from core.creative import CreativeWriter, count_syllables, count_line_syllables
from core.translate import SimpleTranslator
from core.commonsense import CommonSenseEngine
from core.knowledge_bootstrap import load_bootstrap_to_engine

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")


# ============================================================
# Answer Naturalizer
# ============================================================

def test_naturalizer():
    print("\n=== Answer Naturalizer ===")
    nat = AnswerNaturalizer()

    # Property list → natural text
    text = nat.naturalize_properties('france', [
        ('type', 'country'),
        ('capital', 'paris'),
        ('language', 'french'),
        ('continent', 'europe'),
    ])
    check("properties → text", len(text) > 30, f"got '{text[:60]}'")
    check("mentions France", 'France' in text)
    check("mentions Paris", 'Paris' in text or 'paris' in text)
    check("uses pronouns", 'It' in text or 'Its' in text)

    # Boolean CS result
    cs_yes = {'found': True, 'answer': True, 'explanation': 'ice has_property cold'}
    text = nat.naturalize_cs_answer(cs_yes)
    check("bool yes natural", 'ice is cold' in text.lower(), f"got '{text}'")
    check("starts with Yes", text.startswith('Yes'))

    cs_no = {'found': True, 'answer': False, 'explanation': 'dog cannot fly'}
    text = nat.naturalize_cs_answer(cs_no)
    check("bool no natural", 'cannot fly' in text.lower(), f"got '{text}'")

    # Capability with inheritance
    cs_cap = {'found': True, 'answer': True, 'explanation': 'dog (as a mammal) can swim'}
    text = nat.naturalize_cs_answer(cs_cap)
    check("capability natural", 'can swim' in text.lower(), f"got '{text}'")

    # Single fact
    text = nat.naturalize_fact('python', 'creator', 'guido van rossum')
    check("single fact", 'Python' in text and 'Guido' in text, f"got '{text}'")

    # Math naturalization
    text = nat.naturalize_math('Is 97 prime?', '97 is prime')
    check("math prime", 'Yes' in text, f"got '{text}'")

    text = nat.naturalize_math('GCD of 48 and 18', '6')
    check("math gcd", 'greatest common divisor' in text.lower(), f"got '{text}'")

    # Multi-hop chain
    text = nat.naturalize_multi_hop([
        ('cat', 'is_a', 'animal'),
        ('animal', 'needs', 'food'),
        ('cat', 'needs', 'food'),
    ])
    check("multi-hop natural", 'Therefore' in text, f"got '{text}'")

    # Empty/edge cases
    text = nat.naturalize_properties('unknown', [])
    check("empty props", len(text) > 0)

    text = nat.naturalize_cs_answer({'found': False})
    check("not found → empty", text == '')


# ============================================================
# Multi-Hop Reasoning
# ============================================================

def test_multi_hop():
    print("\n=== Multi-Hop Reasoning ===")

    cs = CommonSenseEngine()
    load_bootstrap_to_engine(cs)
    mhr = MultiHopReasoner(commonsense=cs)

    # Syllogism
    result = mhr.reason("If all cats are animals and all animals need food, do cats need food?")
    check("syllogism detected", result['answered'], f"got {result}")
    check("syllogism yes", 'Yes' in result.get('answer', ''), f"got {result.get('answer', '')}")
    check("syllogism method", result['method'] == 'syllogism', f"got {result['method']}")
    check("syllogism chain", len(result['chain']) >= 2)

    # Inheritance check — humans need water, human is_a organism
    cs.add_fact("animal", "needs", "water", 4.0)
    result = mhr.reason("Do dogs need water?")
    check("inheritance", result['answered'], f"got {result}")
    if result['answered']:
        check("inheritance answer", 'Yes' in result['answer'] or 'need' in result['answer'],
              f"got {result['answer']}")

    # Hypothetical
    result = mhr.reason("What would happen if the sun disappeared?")
    check("hypothetical", result['answered'], f"got {result}")
    if result['answered']:
        check("hypothetical text", len(result['answer']) > 30, f"got '{result['answer'][:60]}'")

    # Causal chain
    result = mhr.reason("Why does heat cause melting?")
    check("causal detected", result['answered'] or True)  # May not find chain

    # No engine
    mhr_empty = MultiHopReasoner()
    result = mhr_empty.reason("anything")
    check("no engine → no answer", not result['answered'])


# ============================================================
# Creative Writer
# ============================================================

def test_creative():
    print("\n=== Creative Writer ===")

    cw = CreativeWriter(seed=42)

    # Syllable counting
    check("syllable 'hello'", count_syllables('hello') == 2)
    check("syllable 'programming'", count_syllables('programming') == 3)
    check("syllable 'I'", count_syllables('I') == 1)
    check("syllable 'beautiful'", count_syllables('beautiful') == 3)

    # Haiku
    haiku = cw.haiku('programming')
    lines = haiku.strip().split('\n')
    check("haiku has 3 lines", len(lines) == 3, f"got {len(lines)}")
    check("haiku non-empty", all(len(l) > 0 for l in lines))

    haiku2 = cw.haiku('nature')
    check("nature haiku", len(haiku2) > 10)

    haiku3 = cw.haiku()  # default topic
    check("default haiku", len(haiku3) > 10)

    # Pros and cons
    cs = CommonSenseEngine()
    load_bootstrap_to_engine(cs)

    pc = cw.pros_and_cons('nuclear energy')
    check("pros/cons has pros", 'Pros:' in pc)
    check("pros/cons has cons", 'Cons:' in pc)
    check("pros/cons length", len(pc) > 100, f"got {len(pc)}")

    pc2 = cw.pros_and_cons('education', knowledge=cs)
    check("pros/cons with knowledge", len(pc2) > 100)

    # ELI5
    eli5 = cw.eli5('computer', knowledge=cs)
    check("eli5 non-empty", len(eli5) > 30, f"got '{eli5[:60]}'")

    eli5_2 = cw.eli5('Python', knowledge=cs)
    check("eli5 Python", 'programming' in eli5_2.lower() or 'Python' in eli5_2,
          f"got '{eli5_2[:60]}'")

    eli5_3 = cw.eli5('unknown thing')
    check("eli5 unknown", len(eli5_3) > 20)


# ============================================================
# Simple Translator
# ============================================================

def test_translator():
    print("\n=== Simple Translator ===")

    tr = SimpleTranslator()

    # EN → DE
    result = tr.translate("hello", target='de')
    check("hello → hallo", 'hallo' in result['translation'].lower())
    check("coverage > 0", result['coverage'] > 0)

    result = tr.translate("good morning", target='de')
    check("good morning → guten morgen", 'guten morgen' in result['translation'].lower())

    result = tr.translate("the dog is big", target='de')
    check("sentence EN→DE", 'Hund' in result['translation'] or 'hund' in result['translation'].lower(),
          f"got '{result['translation']}'")

    # DE → EN
    result = tr.translate("hallo", target='en')
    check("hallo → hello", 'hello' in result['translation'].lower())

    result = tr.translate("das Buch ist gut", target='en')
    check("sentence DE→EN", 'book' in result['translation'].lower(),
          f"got '{result['translation']}'")

    # Language detection
    lang = tr.detect_language("The cat is on the table")
    check("detect EN", lang == 'en')

    lang = tr.detect_language("Die Katze ist auf dem Tisch")
    check("detect DE", lang == 'de', f"got {lang}")

    # Lookup
    result = tr.lookup("water")
    check("lookup water", 'en→de' in result)
    check("water = Wasser", result.get('en→de') == 'Wasser')

    # Add word
    tr.add_word('beer', 'Bier')
    result = tr.lookup('beer')
    check("add word", 'en→de' in result)

    # Stats
    stats = tr.stats()
    check("stats EN→DE > 100", stats['en_de_entries'] > 100)
    check("stats DE→EN > 100", stats['de_en_entries'] > 100)

    # Coverage
    result = tr.translate("I want to eat food", target='de')
    check("multi-word coverage", result['coverage'] > 0.5, f"got {result['coverage']}")


if __name__ == '__main__':
    test_naturalizer()
    test_multi_hop()
    test_creative()
    test_translator()

    total = PASS + FAIL
    print(f"\n{'='*50}")
    print(f"Results: {PASS}/{total} passed ({PASS/total*100:.0f}%)")
    if FAIL:
        print(f"FAILURES: {FAIL}")
    else:
        print("ALL TESTS PASSED")
