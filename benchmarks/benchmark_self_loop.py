"""
Benchmarks for Self-Improvement Modules
=========================================
Tests: abstract_interpreter, causal_tracer, epistemic, planner, self_loop
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.abstract_interpreter import AbstractInterpreter, Prediction, SymValue, SymType
from core.causal_tracer import CausalTracer, TraceResult
from core.epistemic import EpistemicState
from core.planner import GoalPlanner, Plan
from core.self_loop import SelfLoop

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
# Abstract Interpreter
# ============================================================

def test_abstract_interpreter():
    print("\n=== Abstract Interpreter ===")
    ai = AbstractInterpreter()

    # Simple arithmetic
    pred = ai.interpret("x = 2 + 3\nprint(x)")
    check("simple add", pred.predicted_output == '5', f"got '{pred.predicted_output}'")
    check("var x = 5", pred.variables.get('x') and pred.variables['x'].value == 5)
    check("confidence VALUE", pred.confidence == 'VALUE')

    # String operations
    ai2 = AbstractInterpreter()
    pred = ai2.interpret("s = 'hello'\nprint(s.upper())")
    check("string upper", pred.predicted_output == 'HELLO', f"got '{pred.predicted_output}'")

    # List operations
    ai3 = AbstractInterpreter()
    pred = ai3.interpret("a = [3, 1, 2]\na.sort()\nprint(a)")
    check("list sort", pred.predicted_output == '[1, 2, 3]', f"got '{pred.predicted_output}'")

    # For loop
    ai4 = AbstractInterpreter()
    pred = ai4.interpret("total = 0\nfor i in range(5):\n    total += i\nprint(total)")
    check("for loop sum", pred.predicted_output == '10', f"got '{pred.predicted_output}'")

    # List comprehension
    ai5 = AbstractInterpreter()
    pred = ai5.interpret("result = [x * 2 for x in [1, 2, 3]]\nprint(result)")
    check("list comp", pred.predicted_output == '[2, 4, 6]', f"got '{pred.predicted_output}'")

    # While loop
    ai6 = AbstractInterpreter()
    pred = ai6.interpret("n = 10\nwhile n > 1:\n    n = n // 2\nprint(n)")
    check("while loop", pred.predicted_output == '1', f"got '{pred.predicted_output}'")

    # If/else
    ai7 = AbstractInterpreter()
    pred = ai7.interpret("x = 10\nif x > 5:\n    y = 'big'\nelse:\n    y = 'small'\nprint(y)")
    check("if/else", pred.predicted_output == 'big', f"got '{pred.predicted_output}'")

    # Function call
    ai8 = AbstractInterpreter()
    pred = ai8.interpret("def double(n):\n    return n * 2\nresult = double(7)\nprint(result)")
    check("function call", pred.predicted_output == '14', f"got '{pred.predicted_output}'")

    # Dictionary
    ai9 = AbstractInterpreter()
    pred = ai9.interpret("d = {'a': 1, 'b': 2}\nprint(d['a'])")
    check("dict access", pred.predicted_output == '1', f"got '{pred.predicted_output}'")

    # Syntax error detection
    ai10 = AbstractInterpreter()
    pred = ai10.interpret("def foo(\n    pass")
    check("syntax error", len(pred.errors) > 0)
    check("confidence NONE", pred.confidence == 'NONE')

    # Type inference
    ai11 = AbstractInterpreter()
    pred = ai11.interpret("import os\nx = os.getcwd()")
    check("symbolic import", pred.variables.get('x') is not None)
    check("import side effect", 'import os' in pred.side_effects)

    # Built-in functions
    ai12 = AbstractInterpreter()
    pred = ai12.interpret("print(len([1, 2, 3]))")
    check("len()", pred.predicted_output == '3', f"got '{pred.predicted_output}'")

    ai13 = AbstractInterpreter()
    pred = ai13.interpret("print(sum([1, 2, 3, 4]))")
    check("sum()", pred.predicted_output == '10', f"got '{pred.predicted_output}'")

    ai14 = AbstractInterpreter()
    pred = ai14.interpret("print(sorted([3, 1, 2]))")
    check("sorted()", pred.predicted_output == '[1, 2, 3]', f"got '{pred.predicted_output}'")

    # Nested function
    ai15 = AbstractInterpreter()
    pred = ai15.interpret("def add(a, b):\n    return a + b\ndef mul(a, b):\n    return a * b\nprint(add(mul(2, 3), 4))")
    check("nested calls", pred.predicted_output == '10', f"got '{pred.predicted_output}'")

    # String methods
    ai16 = AbstractInterpreter()
    pred = ai16.interpret("s = 'hello world'\nprint(s.split())")
    check("str.split", pred.predicted_output == "['hello', 'world']", f"got '{pred.predicted_output}'")

    # Multiple prints
    ai17 = AbstractInterpreter()
    pred = ai17.interpret("for i in range(3):\n    print(i)")
    check("multi print", pred.predicted_output == '0\n1\n2', f"got '{pred.predicted_output}'")

    # Empty program
    ai18 = AbstractInterpreter()
    pred = ai18.interpret("")
    check("empty program", pred.predicted_output == '')
    check("no errors", len(pred.errors) == 0)


# ============================================================
# Causal Tracer
# ============================================================

def test_causal_tracer():
    print("\n=== Causal Tracer ===")
    ct = CausalTracer()

    # Crash analysis: IndexError
    code = "a = [1, 2, 3]\nprint(a[5])"
    result = ct.trace(code, actual_error="IndexError: list index out of range")
    check("crash detected", result.has_error)
    check("root cause exists", result.root_cause is not None)
    check("fix suggestion", len(result.fix_suggestion) > 0, f"got '{result.fix_suggestion}'")

    # Output mismatch
    code = "x = 2 + 2\nprint(x)"
    result = ct.trace(code, expected_output="5", actual_output="4")
    check("mismatch detected", result.has_error)

    # NameError
    result = ct.trace("print(xyz)", actual_error="NameError: name 'xyz' is not defined")
    check("name error", result.root_cause is not None)
    check("name fix", 'xyz' in result.fix_suggestion or 'not defined' in result.fix_suggestion,
          f"got '{result.fix_suggestion}'")

    # ZeroDivision
    result = ct.trace("x = 1 / 0", actual_error="ZeroDivisionError: division by zero")
    check("zero div", 'zero' in result.fix_suggestion.lower())

    # TypeError
    result = ct.trace("x = 'a' + 1", actual_error="TypeError: can only concatenate str to str")
    check("type error fix", 'str' in result.fix_suggestion.lower() or 'type' in result.fix_suggestion.lower())

    # Clean code (no errors)
    result = ct.trace("x = 5\nprint(x)", expected_output="5", actual_output="5")
    check("clean code ok", not result.has_error)

    # Prediction vs actual comparison
    result = ct.compare_predict_vs_actual("x = 2 + 3\nprint(x)", "5")
    check("predict matches", not result.has_error or result.prediction.confidence == 'VALUE')


# ============================================================
# Epistemic Awareness
# ============================================================

def test_epistemic():
    print("\n=== Epistemic Awareness ===")
    ep = EpistemicState()

    # Record queries
    ep.record_query("What is 2+2?", "math", "HIGH", True, True, "math_solver")
    ep.record_query("What is pi?", "math", "HIGH", True, True, "math_solver")
    ep.record_query("What is sqrt(-1)?", "math", "MEDIUM", True, False, "math_solver")
    ep.record_query("Capital of France?", "factual", "HIGH", True, True, "bootstrap")
    ep.record_query("Capital of Narnia?", "factual", "LOW", False, None, "")

    # Competence map
    comp = ep.competence_map()
    check("math tracked", 'math' in comp)
    check("math accuracy", comp['math']['accuracy'] > 0.5, f"got {comp['math']['accuracy']}")
    check("factual tracked", 'factual' in comp)

    # Should answer
    should = ep.should_answer("What is 3+3?", "math")
    check("should answer math", should['should_answer'])
    check("math confidence > 0", should['confidence'] > 0)

    # Frontier
    check("frontier has narnia", any('narnia' in k for k in ep.frontier))

    # I don't know
    idk = ep.i_dont_know("What is the meaning of life?")
    check("idk response", len(idk) > 10, f"got '{idk}'")

    # Calibration
    cal = ep.calibration_report()
    check("calibration report", 'HIGH' in cal)
    check("HIGH total > 0", cal['HIGH']['total'] > 0)

    # Learning priorities
    priorities = ep.learning_priorities()
    check("priorities exist", len(priorities) >= 0)

    # Summary
    summary = ep.summary()
    check("summary non-empty", len(summary) > 50)

    # Knowledge confidence
    conf = ep.knowledge_confidence("france", "capital", "bootstrap")
    check("knowledge conf > 0", conf > 0)

    # Record many failures in a domain
    for i in range(15):
        ep.record_query(f"code {i}", "code", "HIGH", True, False, "code_gen")
    should_code = ep.should_answer("write hello world", "code")
    check("poor domain → don't answer", not should_code['should_answer'],
          f"got should={should_code['should_answer']}, acc={should_code['domain_accuracy']:.2f}")

    # Save/load
    import tempfile
    path = os.path.join(tempfile.gettempdir(), 'test_epistemic.json')
    ep.save(path)
    ep2 = EpistemicState()
    ep2.load(path)
    check("save/load", 'math' in ep2.competence)
    os.unlink(path)


# ============================================================
# Goal Planner
# ============================================================

def test_planner():
    print("\n=== Goal Planner ===")
    gp = GoalPlanner()

    # Code generation plan
    plan = gp.plan("build a web scraper tool")
    check("code plan", len(plan.steps) > 3, f"got {len(plan.steps)} steps")
    check("plan summary", len(plan.summary()) > 50)
    check("progress 0%", plan.progress == 0.0)

    # Debug plan
    plan = gp.plan("fix the bug in parser.py")
    check("debug plan", len(plan.steps) > 3)
    check("has verify step", any(s.step_type == 'check' for s in plan.steps))

    # Generic plan
    plan = gp.plan("do something unusual")
    check("generic plan", len(plan.steps) >= 3)

    # Plan execution flow
    plan = gp.plan("test the login system")
    next_steps = plan.next_steps()
    check("next steps", len(next_steps) > 0)

    # Execute first step
    idx = next_steps[0]
    plan.mark_running(idx)
    plan.mark_done(idx, "step completed")
    check("step done", plan.steps[idx].status.value == 'done')
    check("progress > 0", plan.progress > 0)

    # Self-improvement plan
    plan = gp.plan_self_improve("handle fractions in math solver")
    check("self-improve plan", len(plan.steps) >= 6)
    check("has epistemic step", any('epistemic' in s.description.lower() for s in plan.steps))
    check("has predict step", any('predict' in s.description.lower() for s in plan.steps))
    check("has compare step", any('compare' in s.description.lower() for s in plan.steps))
    check("has attribute step", any('attribute' in s.description.lower() for s in plan.steps))

    # Code task plan
    plan = gp.plan_code_task("add error handling to parser",
                             context={'file': 'parser.py', 'error': 'ValueError'})
    check("code task plan", len(plan.steps) > 5)
    check("reads file", any('parser.py' in s.description for s in plan.steps))

    # Atomic decomposition
    atoms = gp.decompose_atomic('read_file')
    check("atomic steps", len(atoms) > 2)
    check("has open", any('open' in a.lower() for a in atoms))


# ============================================================
# Self Loop
# ============================================================

def test_self_loop():
    print("\n=== Self Loop ===")
    sl = SelfLoop()

    # Verify correct code
    result = sl.verify_code("x = 2 + 3\nprint(x)", expected_output="5")
    check("correct code", result['success'])
    check("prediction match", result.get('match') is True or result.get('match') is None)

    # Verify wrong expected
    result = sl.verify_code("print(4)", expected_output="5")
    check("wrong output detected", not result['success'])
    check("has root cause", 'root_cause' in result or 'trace_result' in result)

    # Predict and check
    result = sl.predict_and_check("x = 10\nfor i in range(5):\n    x += i\nprint(x)")
    check("predict check", result['actual'] == '20', f"got actual='{result['actual']}'")
    check("understanding", result['understanding'] in ('FULL', 'PARTIAL', 'NONE'))

    # Diagnostic
    diag = sl.run_diagnostic()
    check("diagnostic", 'cycles_run' in diag)
    check("prediction accuracy", 'prediction_accuracy' in diag)
    check("epistemic summary", len(diag['epistemic_summary']) > 0)

    # Stats
    stats = sl.stats()
    check("stats string", 'Cycles' in stats)


if __name__ == '__main__':
    test_abstract_interpreter()
    test_causal_tracer()
    test_epistemic()
    test_planner()
    test_self_loop()

    total = PASS + FAIL
    print(f"\n{'='*50}")
    print(f"Results: {PASS}/{total} passed ({PASS/total*100:.0f}%)")
    if FAIL:
        print(f"FAILURES: {FAIL}")
    else:
        print("ALL TESTS PASSED")
