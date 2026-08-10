"""
Benchmarks for regex_builder, differ, template, spellcheck, reviewer
=====================================================================
5 modules, 40+ tests — zero external dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.regex_builder import RegexBuilder
from core.differ import Differ
from core.template import TemplateEngine, render_template
from core.spellcheck import SpellChecker
from core.reviewer import CodeReviewer

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


def test_regex_builder():
    print("\n=== RegexBuilder ===")
    rb = RegexBuilder()

    # NL → Regex
    result = rb.from_natural_language("find all email addresses")
    check("NL→email regex", result['pattern'] is not None and result['name'] == 'email',
          f"got {result.get('name')}")
    check("NL confidence", result['confidence'] >= 0.8, f"got {result['confidence']}")

    result = rb.from_natural_language("match IPv4 addresses")
    check("NL→ipv4 regex", result['name'] == 'ipv4', f"got {result.get('name')}")

    result = rb.from_natural_language("find all URLs")
    check("NL→url regex", result['name'] == 'url', f"got {result.get('name')}")

    # Custom patterns — NL first tries library match, then custom
    result = rb.from_natural_language("starts with PREFIX")
    check("NL→custom starts-with", result['pattern'] is not None,
          f"got {result.get('pattern')}")

    result = rb.from_natural_language("contains foobar")
    check("NL→custom contains", result['pattern'] is not None and 'foobar' in result['pattern'],
          f"got {result.get('pattern')}")

    # Test regex
    test_result = rb.test(r'\d+', "I have 3 cats and 42 dogs")
    check("test finds matches", test_result['count'] == 2, f"got {test_result['count']}")
    check("test match values", test_result['matches'][0]['match'] == '3')
    check("test match values 2", test_result['matches'][1]['match'] == '42')

    # Test invalid regex
    test_result = rb.test(r'[invalid', "text")
    check("test invalid regex", test_result.get('valid') is False)

    # Explain
    explanation = rb.explain(r'\b\d{4}-\d{2}-\d{2}\b')
    check("explain has output", len(explanation) > 20, f"len={len(explanation)}")
    check("explain mentions digit", 'digit' in explanation.lower() or '\\d' in explanation)

    # Pattern library
    patterns = rb.list_patterns()
    check("pattern library size", len(patterns) >= 20, f"got {len(patterns)}")
    check("email in library", 'email' in patterns)

    # Get specific pattern
    email = rb.get_pattern('email')
    check("get_pattern returns dict", email is not None and 'pattern' in email)

    # Add custom pattern
    rb.add_pattern('custom_test', r'TEST-\d+', 'Test ID', ['TEST-123'])
    check("add_pattern", rb.get_pattern('custom_test') is not None)

    # Test real matching
    import re
    email_pattern = rb.get_pattern('email')['pattern']
    matches = re.findall(email_pattern, "contact user@test.com or admin@foo.org")
    check("email regex matches", len(matches) == 2, f"got {len(matches)}")


def test_differ():
    print("\n=== Differ ===")
    d = Differ()

    old = "line1\nline2\nline3"
    new = "line1\nmodified\nline3\nline4"

    # Basic diff
    result = d.diff(old, new)
    check("diff returns lines", len(result) > 0)
    types = [r.line_type for r in result]
    check("diff has removals", 'remove' in types)
    check("diff has additions", 'add' in types)
    check("diff has equal", 'equal' in types)

    # Unified diff
    unified = d.unified_diff(old, new)
    check("unified has ---", '---' in unified)
    check("unified has +++", '+++' in unified)
    check("unified has @@", '@@' in unified)
    check("unified has -line2", '-line2' in unified)
    check("unified has +modified", '+modified' in unified)

    # Word diff
    word_ops = d.word_diff("the quick brown fox", "the slow brown cat")
    op_types = [op for op, _ in word_ops]
    check("word diff has changes", 'remove' in op_types or 'add' in op_types)

    # Word diff HTML
    html = d.word_diff_html("hello world", "hello earth")
    check("word diff html has del", '<del>' in html)
    check("word diff html has ins", '<ins>' in html)

    # Similarity
    sim1 = d.similarity("abc\ndef\nghi", "abc\ndef\nghi")
    check("similarity identical=1.0", sim1 == 1.0, f"got {sim1}")

    sim2 = d.similarity("abc\ndef", "xyz\nqrs")
    check("similarity different<0.5", sim2 < 0.5, f"got {sim2}")

    # Stats
    stats = d.stats(old, new)
    check("stats has added", stats['added'] > 0)
    check("stats has removed", stats['removed'] > 0)
    check("stats total_changes", stats['total_changes'] == stats['added'] + stats['removed'])

    # Structural diff
    old_code = "def foo():\n    pass\n\ndef bar():\n    return 1\n"
    new_code = "def foo():\n    return 42\n\ndef baz():\n    return 2\n"
    struct = d.structural_diff(old_code, new_code)
    change_types = [c['type'] for c in struct]
    check("structural finds modified", 'modified' in change_types)
    check("structural finds added", 'added' in change_types)
    check("structural finds removed", 'removed' in change_types)

    # Patch generation and application
    patch = d.generate_patch(old, new, 'test.py')
    check("patch has filename", 'test.py' in patch)
    applied = d.apply_patch(old, patch)
    check("patch applies correctly", applied.strip() == new.strip(),
          f"got '{applied[:50]}...'")

    # Empty diff
    empty = d.diff("same", "same")
    check("identical gives equal only",
          all(r.line_type == 'equal' for r in empty))


def test_template():
    print("\n=== TemplateEngine ===")
    engine = TemplateEngine()

    # Simple variable
    result = engine.render("Hello, {{name}}!", {'name': 'World'})
    check("simple variable", result == "Hello, World!", f"got '{result}'")

    # HTML escaping
    result = engine.render("{{content}}", {'content': '<script>alert("xss")</script>'})
    check("html escaping", '&lt;' in result and '&gt;' in result, f"got '{result}'")

    # Raw (unescaped)
    result = engine.render("{{{content}}}", {'content': '<b>bold</b>'})
    check("raw unescaped", '<b>bold</b>' in result, f"got '{result}'")

    # Truthy section
    result = engine.render("{{#show}}visible{{/show}}", {'show': True})
    check("truthy section shown", result == "visible", f"got '{result}'")

    # Falsy section hidden
    result = engine.render("{{#show}}visible{{/show}}", {'show': False})
    check("falsy section hidden", result == "", f"got '{result}'")

    # Inverted section
    result = engine.render("{{^logged_in}}Please log in{{/logged_in}}", {'logged_in': False})
    check("inverted section", "Please log in" in result, f"got '{result}'")

    # Loop
    result = engine.render("{{#items}}[{{name}}]{{/items}}",
                          {'items': [{'name': 'a'}, {'name': 'b'}, {'name': 'c'}]})
    check("loop rendering", result == "[a][b][c]", f"got '{result}'")

    # Dot notation
    result = engine.render("{{user.name}}", {'user': {'name': 'David'}})
    check("dot notation", result == "David", f"got '{result}'")

    # Partials
    engine.register_partial('greeting', 'Hello, {{name}}!')
    result = engine.render("{{>greeting}}", {'name': 'Test'})
    check("partials", result == "Hello, Test!", f"got '{result}'")

    # Helpers
    result = engine.render("{{upper name}}", {'name': 'hello'})
    check("helper upper", result == "HELLO", f"got '{result}'")

    # Predefined templates
    result = render_template('python_function', {
        'name': 'add',
        'params': 'a, b',
        'docstring': 'Add two numbers.',
        'body': ['result = a + b'],
        'return_value': 'result',
    })
    check("predefined template", 'def add(a, b):' in result, f"got '{result[:60]}'")

    # Unknown template
    result = render_template('nonexistent', {})
    check("unknown template", 'Unknown' in result)

    # Empty list section
    result = engine.render("{{#items}}x{{/items}}", {'items': []})
    check("empty list = no output", result == "", f"got '{result}'")

    # Nested context
    result = engine.render("{{#user}}{{name}} ({{age}}){{/user}}",
                          {'user': {'name': 'Alice', 'age': 30}})
    check("nested context", 'Alice' in result and '30' in result, f"got '{result}'")


def test_spellcheck():
    print("\n=== SpellChecker ===")
    sc = SpellChecker()

    # Correct word stays correct
    result = sc.correct("the")
    check("correct word unchanged", result[0][0].lower() == "the", f"got {result}")

    # Misspelled word — "helo" → "help" (closest in small dictionary)
    result = sc.correct("helo")
    check("helo gets correction", len(result) > 0 and result[0][0].lower() != "helo",
          f"got {result}")

    # Tech terms not flagged
    result = sc.correct("nginx")
    check("tech term nginx", result[0][1] == 1.0, f"got {result}")

    result = sc.correct("kubernetes")
    check("tech term kubernetes", result[0][1] == 1.0)

    # All caps skip
    result = sc.correct("JSON")
    check("all caps skipped", result[0][1] == 1.0)

    # Digits skip
    result = sc.correct("test123")
    check("digits skipped", result[0][1] == 1.0)

    # Known word probability
    prob = sc.probability("the")
    check("the has high probability", prob > 0, f"got {prob}")

    # Unknown word probability
    prob = sc.probability("xyzzyplugh")
    check("unknown word prob=0", prob == 0)

    # check_text
    errors = sc.check_text("Ths is a tset of speling")
    check("check_text finds errors", len(errors) > 0, f"found {len(errors)}")
    misspelled = [e['word'] for e in errors]
    check("Ths flagged", 'Ths' in misspelled, f"got {misspelled}")

    # auto_correct
    corrected = sc.auto_correct("the quik brown fox")
    check("auto_correct changes text", corrected != "the quik brown fox",
          f"got '{corrected}'")

    # Add custom word
    sc.add_word("foobarqux", 500)
    result = sc.correct("foobarqux")
    check("custom word recognized", result[0][1] == 1.0)

    # Add multiple words
    sc.add_words(["bleep", "bloop"])
    check("add_words works", sc.probability("bleep") > 0)

    # Known set
    known = sc.known({"the", "xyzzy123notaword"})
    check("known filters correctly", "the" in known and "xyzzy123notaword" not in known)

    # Case preservation
    result = sc.correct("Helo")
    if result and result[0][0] != "Helo":
        check("case preserved", result[0][0][0].isupper(), f"got '{result[0][0]}'")
    else:
        check("case preserved", True)  # word might not have correction


def test_reviewer():
    print("\n=== CodeReviewer ===")
    reviewer = CodeReviewer()

    # Clean code
    clean = '''
def greet(name):
    """Say hello."""
    return f"Hello, {name}!"
'''
    issues = reviewer.review(clean)
    check("clean code few issues", len(issues) <= 1, f"got {len(issues)} issues")

    # Unused import
    code_unused_import = '''
import os
import sys

def main():
    """Entry point."""
    print(sys.argv)
'''
    issues = reviewer.review(code_unused_import)
    codes = [i.code for i in issues]
    check("detects unused import", 'R001' in codes, f"got {codes}")

    # Function too long
    long_func = 'def long():\n    """Long function."""\n' + '    x = 1\n' * 60
    issues = reviewer.review(long_func)
    codes = [i.code for i in issues]
    check("detects long function", 'R002' in codes, f"got {codes}")

    # Too many arguments
    many_args = '''
def foo(a, b, c, d, e, f, g):
    """Many args."""
    return a
'''
    issues = reviewer.review(many_args)
    codes = [i.code for i in issues]
    check("detects too many args", 'R003' in codes, f"got {codes}")

    # Missing docstring
    no_doc = '''
def public_func(x):
    return x * 2
'''
    issues = reviewer.review(no_doc)
    codes = [i.code for i in issues]
    check("detects missing docstring", 'R004' in codes, f"got {codes}")

    # Bare except
    bare_except = '''
def risky():
    """Risky."""
    try:
        x = 1
    except:
        pass
'''
    issues = reviewer.review(bare_except)
    codes = [i.code for i in issues]
    check("detects bare except", 'R006' in codes, f"got {codes}")

    # Mutable default
    mutable_default = '''
def bad(items=[]):
    """Bad default."""
    items.append(1)
    return items
'''
    issues = reviewer.review(mutable_default)
    codes = [i.code for i in issues]
    check("detects mutable default", 'R007' in codes, f"got {codes}")

    # Global usage
    global_code = '''
x = 0
def increment():
    """Increment."""
    global x
    x += 1
'''
    issues = reviewer.review(global_code)
    codes = [i.code for i in issues]
    check("detects global", 'R008' in codes, f"got {codes}")

    # Dead code
    dead_code = '''
def dead():
    """Has dead code."""
    return 42
    x = 1
'''
    issues = reviewer.review(dead_code)
    codes = [i.code for i in issues]
    check("detects dead code", 'R009' in codes, f"got {codes}")

    # Naming convention
    bad_naming = '''
def BadName():
    """Bad."""
    pass
'''
    issues = reviewer.review(bad_naming)
    codes = [i.code for i in issues]
    check("detects bad naming", 'R011' in codes, f"got {codes}")

    # Syntax error
    issues = reviewer.review("def foo(:")
    check("handles syntax error", issues[0].code == 'E001', f"got {issues[0].code}")

    # Report
    report = reviewer.report(code_unused_import)
    check("report has content", 'Code Review' in report, f"got '{report[:40]}'")

    # Score
    score = reviewer.score(clean)
    check("score returns dict", 'score' in score and 'grade' in score)
    check("clean code high score", score['score'] >= 70, f"got {score['score']}")

    # Clean report
    really_clean = '''
def add(a, b):
    """Add two numbers."""
    return a + b
'''
    report = reviewer.report(really_clean)
    check("clean report says clean", 'clean' in report.lower() or 'No issues' in report,
          f"got '{report[:60]}'")


if __name__ == '__main__':
    test_regex_builder()
    test_differ()
    test_template()
    test_spellcheck()
    test_reviewer()

    total = PASS + FAIL
    print(f"\n{'='*50}")
    print(f"Results: {PASS}/{total} passed ({PASS/total*100:.0f}%)")
    if FAIL:
        print(f"FAILURES: {FAIL}")
    else:
        print("ALL TESTS PASSED")
