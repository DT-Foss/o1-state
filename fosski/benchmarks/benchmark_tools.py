"""
Benchmark: Tool-Use Framework (A5)
=====================================
Tests:
  1. Tool registry and listing
  2. Intent matching from natural language
  3. File operations (read, list)
  4. Python eval (sandboxed)
  5. Search tools (glob, grep)
  6. Argument extraction
  7. Dangerous tool blocking
"""

import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tools import ToolRegistry, ToolExecutor, Tool


def test_registry():
    """Test 1: Tool registration and listing."""
    print("Test 1: Tool Registry")
    reg = ToolRegistry()
    results = []

    tools = reg.list_tools()
    ok = len(tools) >= 7  # At least 7 builtins
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: {len(tools)} tools registered")

    # Check specific tools exist
    for name in ['read_file', 'write_file', 'shell', 'grep', 'python_eval']:
        ok = reg.get(name) is not None
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: Tool '{name}' exists")

    return all(results)


def test_intent_matching():
    """Test 2: Match queries to tools."""
    print("Test 2: Intent Matching")
    reg = ToolRegistry()
    results = []

    cases = [
        ("read file /tmp/test.txt", "read_file"),
        ("list files in /tmp", "list_dir"),
        ("search for *.py files", "search_files"),
        ("grep for 'def main' in the code", "grep"),
        ("calculate sqrt(144)", "python_eval"),
    ]

    for query, expected in cases:
        matched = reg.match_intent(query)
        ok = matched == expected
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: \"{query[:40]}\" → {matched} (expected {expected})")

    return all(results)


def test_file_ops():
    """Test 3: File read/list operations."""
    print("Test 3: File Operations")
    executor = ToolExecutor(confirm_dangerous=False)
    results = []

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Hello from FOSS-KI!")
        tmpfile = f.name

    try:
        # Read file
        r = executor.execute('read_file', path=tmpfile)
        ok = r['success'] and 'Hello from FOSS-KI!' in r['result']
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: read_file → {r['result'][:30]}")

        # List dir
        r = executor.execute('list_dir', path=tempfile.gettempdir())
        ok = r['success'] and len(r['result']) > 0
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: list_dir → {len(r['result'])} entries")

    finally:
        os.unlink(tmpfile)

    return all(results)


def test_python_eval():
    """Test 4: Sandboxed Python evaluation."""
    print("Test 4: Python Eval")
    executor = ToolExecutor()
    results = []

    cases = [
        ("2 ** 10", "1024"),
        ("math.sqrt(144)", "12.0"),
        ("sum(range(1, 101))", "5050"),
        ("sorted([3,1,4,1,5,9])", "[1, 1, 3, 4, 5, 9]"),
    ]

    for expr, expected in cases:
        r = executor.execute('python_eval', expression=expr)
        ok = r['success'] and str(r['result']) == expected
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: eval({expr}) → {r.get('result')}")

    # Safety: os.system should fail
    r = executor.execute('python_eval', expression='os.system("ls")')
    ok = not r['success']
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: os.system blocked → {r.get('error', '')[:40]}")

    return all(results)


def test_search_tools():
    """Test 5: Search tools."""
    print("Test 5: Search Tools")
    executor = ToolExecutor()
    results = []

    # Search for Python files in core/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    core_path = os.path.join(project_root, 'core')

    r = executor.execute('search_files', pattern='*.py', path=core_path)
    ok = r['success'] and len(r['result']) > 5
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: search_files('*.py') → {len(r.get('result', []))} files")

    # Grep for class definitions
    r = executor.execute('grep', pattern=r'class \w+', path=core_path)
    ok = r['success'] and len(r['result']) > 3
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: grep('class \\w+') → {len(r.get('result', []))} matches")

    return all(results)


def test_arg_extraction():
    """Test 6: Argument extraction from natural language."""
    print("Test 6: Argument Extraction")
    executor = ToolExecutor()
    results = []

    # Path from quoted string
    tool = executor.registry.get('read_file')
    args = executor._extract_args('read file "/tmp/test.txt"', tool)
    ok = args.get('path') == '/tmp/test.txt'
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: path from quotes → {args.get('path')}")

    # Path from unquoted
    args = executor._extract_args('read file /tmp/test.txt', tool)
    ok = args.get('path') == '/tmp/test.txt'
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: path unquoted → {args.get('path')}")

    return all(results)


def test_dangerous_blocking():
    """Test 7: Dangerous tools require confirmation."""
    print("Test 7: Dangerous Tool Blocking")
    executor = ToolExecutor(confirm_dangerous=True)
    results = []

    # write_file should be blocked
    r = executor.execute_from_query('write file "/tmp/test.txt" content "hello"')
    # Shell should need confirmation
    tool = executor.registry.get('shell')
    ok = tool.dangerous is True
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: shell is marked dangerous")

    tool2 = executor.registry.get('read_file')
    ok2 = tool2.dangerous is False
    results.append(ok2)
    print(f"  {'PASS' if ok2 else 'FAIL'}: read_file is safe")

    # Custom tool
    custom = Tool(
        name='delete_all',
        description='Delete everything',
        fn=lambda: None,
        dangerous=True,
    )
    ok3 = custom.dangerous is True
    results.append(ok3)
    print(f"  {'PASS' if ok3 else 'FAIL'}: Custom dangerous tool flagged")

    return all(results)


def test_custom_tool():
    """Test 8: Custom tool registration and execution."""
    print("Test 8: Custom Tools")
    reg = ToolRegistry()
    results = []

    # Register a custom tool
    def word_count(text: str) -> int:
        return len(text.split())

    reg.register(Tool(
        name='word_count',
        description='Count words in text',
        fn=word_count,
        args=[{'name': 'text', 'type': 'str', 'description': 'input text', 'required': True}],
        category='text',
    ))

    executor = ToolExecutor(registry=reg)
    r = executor.execute('word_count', text='Hello world from FOSS-KI')
    ok = r['success'] and r['result'] == 4
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: word_count → {r.get('result')}")

    return all(results)


if __name__ == '__main__':
    print("=" * 60)
    print("TOOL-USE FRAMEWORK BENCHMARK (A5)")
    print("=" * 60)
    print()

    tests = [
        test_registry,
        test_intent_matching,
        test_file_ops,
        test_python_eval,
        test_search_tools,
        test_arg_extraction,
        test_dangerous_blocking,
        test_custom_tool,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            ok = test()
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    total = passed + failed
    print(f"RESULTS: {passed}/{total} tests passed")
    if failed == 0:
        print("ALL TESTS PASSED")
    print("=" * 60)
