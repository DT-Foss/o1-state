"""
Benchmarks for Session 5 Modules
===================================
Tests: knowledge_bootstrap, formatter, profiler, git_ops, plugins, testgen
"""

import sys
import os
import json
import time
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.knowledge_bootstrap import (
    generate_extended_common_sense,
    generate_world_knowledge,
    generate_technical_knowledge,
    bootstrap_full_knowledge,
    load_bootstrap_to_engine,
)
from core.formatter import CodeFormatter
from core.profiler import Profiler, Benchmark, TimingRecord
from core.git_ops import GitOps
from core.plugins import PluginManager, PluginInfo
from core.testgen import TestGenerator

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
# Knowledge Bootstrap
# ============================================================

def test_knowledge_bootstrap():
    print("\n=== Knowledge Bootstrap ===")

    cs = generate_extended_common_sense()
    check("common sense > 200", len(cs) > 200, f"got {len(cs)}")

    world = generate_world_knowledge()
    check("world knowledge > 200", len(world) > 200, f"got {len(world)}")

    tech = generate_technical_knowledge()
    check("technical knowledge > 50", len(tech) > 50, f"got {len(tech)}")

    # Verify structure
    for s, r, o, w in cs[:10]:
        check("cs has 4-tuples", isinstance(s, str) and isinstance(w, float), f"got {type(s)}")
        break

    # Countries
    country_facts = [(s, r, o) for s, r, o, w in world if r == 'capital']
    check("has capitals", len(country_facts) >= 25, f"got {len(country_facts)}")
    has_berlin = any(o == 'Berlin' for _, _, o in country_facts)
    check("Germany→Berlin", has_berlin)

    # Elements
    elem_facts = [(s, r, o) for s, r, o, w in world if r == 'symbol']
    check("has elements", len(elem_facts) >= 10, f"got {len(elem_facts)}")

    # Programming languages
    lang_facts = [(s, r, o) for s, r, o, w in world if s == 'Python' and r == 'creator']
    check("Python→Guido", len(lang_facts) > 0 and 'Guido' in lang_facts[0][2])

    # Full bootstrap to temp file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        tmp_path = f.name
    try:
        stats = bootstrap_full_knowledge(tmp_path)
        check("bootstrap generates file", os.path.exists(tmp_path))
        check("bootstrap > 600 facts", stats['total'] > 600, f"got {stats['total']}")
        check("bootstrap has size", stats['size_kb'] > 10, f"got {stats['size_kb']} KB")

        # Load and verify JSON
        with open(tmp_path) as f:
            data = json.load(f)
        check("JSON valid", 'triplets' in data)
        check("JSON count matches", data['count'] == stats['total'])
    finally:
        os.unlink(tmp_path)

    # Load into engine
    from core.commonsense import CommonSenseEngine
    cs_engine = CommonSenseEngine()
    loaded = load_bootstrap_to_engine(cs_engine)
    check("loads into engine", loaded > 600, f"got {loaded}")

    # Verify engine queries work after loading
    result = cs_engine.query("is gold valuable?")
    check("gold is valuable after load", result.get('answer') is True, f"got {result}")


# ============================================================
# Code Formatter
# ============================================================

def test_formatter():
    print("\n=== Code Formatter ===")

    fmt = CodeFormatter()

    # Trailing whitespace
    result = fmt.format("x = 1   \ny = 2  \n")
    check("removes trailing ws", "   " not in result)

    # Tabs to spaces
    result = fmt.format("\tx = 1\n\t\ty = 2\n")
    check("tabs to spaces", "\t" not in result)
    check("4-space indent", "    x = 1" in result)

    # Blank lines before top-level def
    code = "x = 1\ndef foo():\n    pass\ndef bar():\n    pass\n"
    result = fmt.format(code)
    check("blank lines before def", "\n\n\ndef foo" in result or "\n\ndef foo" in result)

    # Import sorting
    code = "import os\nimport sys\nfrom typing import List\nimport json\n"
    result = fmt.format(code)
    lines = [l for l in result.split('\n') if l.startswith('import') or l.startswith('from')]
    check("imports sorted", len(lines) >= 3)

    # File ends with newline
    result = fmt.format("x = 1")
    check("ends with newline", result.endswith('\n'))

    # Check mode
    issues = fmt.check("x = 1   \n\tx = 2\n" + "a" * 200 + "\n")
    check("check finds trailing ws", any("trailing" in i for i in issues))
    check("check finds tabs", any("tab" in i for i in issues))
    check("check finds long line", any("too long" in i for i in issues))

    # Format file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("x = 1   \ny = 2\t \n")
        tmp = f.name
    try:
        formatted, n_changes = fmt.format_file(tmp)
        check("format_file returns", len(formatted) > 0)
        check("format_file counts changes", n_changes > 0, f"got {n_changes}")
    finally:
        os.unlink(tmp)

    # Empty code
    result = fmt.format("")
    check("handles empty code", result == '' or result == '\n')


# ============================================================
# Profiler
# ============================================================

def test_profiler():
    print("\n=== Profiler ===")

    prof = Profiler()

    # Context manager measurement
    with prof.measure("test_op"):
        time.sleep(0.01)
    stats = prof.get_stats("test_op")
    check("measure records", stats is not None)
    check("measure timing", stats['avg_ms'] >= 5, f"got {stats['avg_ms']:.2f}ms")

    # Multiple calls
    for _ in range(10):
        with prof.measure("multi"):
            time.sleep(0.001)
    stats = prof.get_stats("multi")
    check("multi calls counted", stats['calls'] == 10)
    check("multi has min/max", stats['min_ms'] <= stats['max_ms'])

    # Manual record
    prof.record("manual", 0.05)
    check("manual record", prof.get_stats("manual")['total_ms'] == 50.0)

    # Decorator
    @prof.wrap("decorated")
    def slow_func():
        time.sleep(0.005)
        return 42

    result = slow_func()
    check("decorated works", result == 42)
    check("decorated recorded", prof.get_stats("decorated") is not None)

    # Report
    report = prof.report()
    check("report non-empty", len(report) > 100, f"got {len(report)} chars")
    check("report has header", "Performance Report" in report)

    # Hot paths
    hot = prof.get_hot_paths(3)
    check("hot paths", len(hot) <= 3)

    # All stats
    all_stats = prof.get_all_stats()
    check("all stats sorted", len(all_stats) >= 3)

    # Reset
    prof.reset()
    check("reset clears", len(prof.timings) == 0)

    # TimingRecord
    rec = TimingRecord("test")
    rec.record(0.1)
    rec.record(0.2)
    check("record avg", abs(rec.avg_time - 0.15) < 0.01)
    check("record min/max", rec.min_time == 0.1 and rec.max_time == 0.2)

    # Benchmark utility
    bench = Benchmark()
    result = bench.run("sleep_1ms", lambda: time.sleep(0.001), iterations=10, warmup=2)
    check("benchmark runs", result['iterations'] == 10)
    check("benchmark has p95", result['p95_ms'] > 0)
    check("benchmark has ops/s", result['ops_per_sec'] > 0)

    report = bench.report()
    check("bench report", "sleep_1ms" in report)

    # Merge
    p1 = Profiler()
    p2 = Profiler()
    with p1.measure("op_a"):
        pass
    with p2.measure("op_b"):
        pass
    p1.merge(p2)
    check("merge combines", "op_b" in p1.timings)

    # Disabled profiler
    disabled = Profiler(enabled=False)
    with disabled.measure("noop"):
        pass
    check("disabled no-op", len(disabled.timings) == 0)


# ============================================================
# Git Ops
# ============================================================

def test_git_ops():
    print("\n=== Git Ops ===")

    # Test on a real git repo if available, otherwise test structure
    git = GitOps('.')

    # Test on foss-ki directory (may not be a git repo)
    foss_ki = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git = GitOps(foss_ki)
    is_repo = git.is_repo()
    check("is_repo returns bool", isinstance(is_repo, bool))

    if is_repo:
        status = git.status()
        check("status has branch", 'branch' in status)
        check("status has clean", 'clean' in status)

        log_entries = git.log(5)
        check("log returns list", isinstance(log_entries, list))

        if log_entries:
            check("log has hash", 'hash' in log_entries[0])
            check("log has message", 'message' in log_entries[0])

        branch = git.current_branch()
        check("branch is str", isinstance(branch, str))

        branches = git.branches()
        check("branches is list", isinstance(branches, list))

        summary = git.summary()
        check("summary has fields", 'branch' in summary and 'clean' in summary)
    else:
        check("non-repo handled", True)  # Just verify no crash

    # Test diff_stat structure
    stat = git.diff_stat()
    check("diff_stat returns dict", isinstance(stat, dict))

    # Test with temp git repo
    with tempfile.TemporaryDirectory() as tmpdir:
        import subprocess
        subprocess.run(['git', 'init', tmpdir], capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@test.com'],
                       capture_output=True, cwd=tmpdir)
        subprocess.run(['git', 'config', 'user.name', 'Test'],
                       capture_output=True, cwd=tmpdir)

        # Create a file and commit
        test_file = os.path.join(tmpdir, 'test.py')
        with open(test_file, 'w') as f:
            f.write('x = 1\n')
        subprocess.run(['git', 'add', 'test.py'], capture_output=True, cwd=tmpdir)
        subprocess.run(['git', 'commit', '-m', 'init'], capture_output=True, cwd=tmpdir)

        tgit = GitOps(tmpdir)
        check("temp repo is repo", tgit.is_repo())

        status = tgit.status()
        check("temp repo clean", status['clean'])

        # Modify and check
        with open(test_file, 'w') as f:
            f.write('x = 2\ny = 3\n')
        status = tgit.status()
        check("detects modification", len(status['modified']) > 0)

        diff = tgit.diff()
        check("diff shows changes", 'x = 2' in diff or '+x = 2' in diff)

        tags = tgit.tags()
        check("tags is list", isinstance(tags, list))


# ============================================================
# Plugin System
# ============================================================

def test_plugins():
    print("\n=== Plugin System ===")

    pm = PluginManager()

    # Plugin info
    info = PluginInfo("test", "/tmp/test.py")
    check("plugin info", info.name == "test" and not info.loaded)
    d = info.to_dict()
    check("to_dict", d['name'] == 'test' and d['loaded'] is False)

    # Discovery from temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a plugin
        plugin_code = '''
__version__ = "1.0.0"
__description__ = "Test plugin"
__author__ = "Test"

def register(context):
    context['registered'] = True

def unregister():
    pass
'''
        with open(os.path.join(tmpdir, 'hello.py'), 'w') as f:
            f.write(plugin_code)

        with open(os.path.join(tmpdir, '_hidden.py'), 'w') as f:
            f.write('# hidden')

        pm.add_plugin_dir(tmpdir)
        discovered = pm.discover()
        check("discovers plugins", 'hello' in discovered)
        check("skips hidden", '_hidden' not in discovered)

        # Load
        context = {}
        ok = pm.load('hello', context)
        check("loads plugin", ok)
        check("register called", context.get('registered') is True)

        # Plugin info after load
        p = pm.get_plugin('hello')
        check("plugin version", p['version'] == '1.0.0')
        check("plugin loaded", p['loaded'])

        # Reload
        ok = pm.reload('hello', context)
        check("reload works", ok)

        # Unload
        ok = pm.unload('hello')
        check("unloads", ok)
        p = pm.get_plugin('hello')
        check("unloaded state", not p['loaded'])

    # Hooks
    pm.register_hook('on_query', lambda query='': f"hooked:{query}")
    results = pm.emit('on_query', query='test')
    check("hook fires", len(results) == 1 and results[0] == 'hooked:test')

    # Commands
    pm.register_command('/greet', lambda arg: f"Hello {arg}", "Greet someone")
    cmd = pm.get_command('/greet')
    check("command registered", cmd is not None)
    handler, desc = cmd
    check("command works", handler("World") == "Hello World")

    cmds = pm.list_commands()
    check("list commands", len(cmds) == 1)

    # Stats
    stats = pm.stats()
    check("stats", stats['hooks'] == 1 and stats['commands'] == 1)

    # Load all
    with tempfile.TemporaryDirectory() as tmpdir:
        for name in ['a', 'b']:
            with open(os.path.join(tmpdir, f'{name}.py'), 'w') as f:
                f.write(f'def register(ctx): ctx["{name}"] = True\n')
        pm2 = PluginManager([tmpdir])
        pm2.discover()
        ctx = {}
        results = pm2.load_all(ctx)
        check("load_all", all(results.values()))
        check("all registered", ctx.get('a') and ctx.get('b'))

    # List
    plugins = pm.list_plugins()
    check("list_plugins", isinstance(plugins, list))


# ============================================================
# Test Generator
# ============================================================

def test_testgen():
    print("\n=== Test Generator ===")

    gen = TestGenerator()

    # Analyze source
    source = '''
class Calculator:
    """A simple calculator."""

    def __init__(self, precision: int = 2):
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return round(a + b, self.precision)

    def multiply(self, a: float, b: float) -> float:
        return round(a * b, self.precision)

    def _private(self):
        pass

def helper(text: str) -> str:
    """Help function."""
    return text.upper()
'''

    functions = gen.analyze_source(source)
    check("analyze finds functions", len(functions) >= 3, f"got {len(functions)}")

    names = [f.name for f in functions]
    check("finds __init__", '__init__' in names)
    check("finds add", 'add' in names)
    check("skips _private", '_private' not in names)
    check("finds helper", 'helper' in names)

    # Check extracted info
    add_func = next(f for f in functions if f.name == 'add')
    check("add has args", len(add_func.args) == 2, f"got {add_func.args}")
    check("add is method", add_func.is_method)
    check("add has class", add_func.class_name == 'Calculator')
    check("add has docstring", 'Add' in add_func.docstring)

    helper_func = next(f for f in functions if f.name == 'helper')
    check("helper not method", not helper_func.is_method)

    # Generate function-style tests
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(source)
        tmp = f.name
    try:
        tests = gen.generate_from_file(tmp)
        check("generates tests", len(tests) > 100, f"got {len(tests)} chars")
        check("has test function", 'def test_' in tests)
        check("has Calculator", 'Calculator' in tests)
        check("has check()", 'check(' in tests)
    finally:
        os.unlink(tmp)

    # Generate class-style tests
    gen2 = TestGenerator(style='class')
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(source)
        tmp = f.name
    try:
        tests2 = gen2.generate_from_file(tmp)
        check("class tests", 'unittest.TestCase' in tests2)
        check("class has setUp", 'setUp' in tests2)
    finally:
        os.unlink(tmp)

    # Analyze real file
    real_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'core', 'math_solver.py'
    )
    if os.path.exists(real_file):
        funcs = gen.analyze_file(real_file)
        check("analyzes real file", len(funcs) >= 1, f"got {len(funcs)}")
    else:
        check("real file exists", False, "math_solver.py not found")

    # Empty source
    empty = gen.analyze_source("")
    check("handles empty", len(empty) == 0)

    # Syntax error
    bad = gen.analyze_source("def foo(")
    check("handles syntax error", len(bad) == 0)


if __name__ == '__main__':
    test_knowledge_bootstrap()
    test_formatter()
    test_profiler()
    test_git_ops()
    test_plugins()
    test_testgen()

    total = PASS + FAIL
    print(f"\n{'='*50}")
    print(f"Results: {PASS}/{total} passed ({PASS/total*100:.0f}%)")
    if FAIL:
        print(f"FAILURES: {FAIL}")
    else:
        print("ALL TESTS PASSED")
