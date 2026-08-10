#!/usr/bin/env python3
"""
Benchmark: New Modules — Summarizer, Translator, Embeddings, Scheduler, Watcher
=================================================================================
"""

import sys
import os
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return 1 if passed else 0


def run():
    print("=" * 60)
    print("BENCHMARK: New Modules (Session 3 Alpha)")
    print("=" * 60)
    total = 0
    passed = 0

    # ============================================================
    # SUMMARIZER
    # ============================================================
    print("\n=== SUMMARIZER ===")
    from core.summarizer import Summarizer, TopicExtractor, split_sentences

    text = """
    Albert Einstein was a German-born theoretical physicist who is widely held
    to be one of the greatest and most influential scientists of all time.
    Best known for developing the theory of relativity, Einstein also made
    important contributions to quantum mechanics. His mass-energy equivalence
    formula E = mc², which arises from relativity theory, has been called
    "the world's most famous equation". He received the 1921 Nobel Prize in
    Physics for his discovery of the law of the photoelectric effect, a pivotal
    step in the development of quantum theory. His work is also known for its
    influence on the philosophy of science. In a 1999 poll of 130 leading
    physicists worldwide by the British journal Physics World, Einstein was
    ranked the greatest physicist of all time. His intellectual achievements
    and originality have made the word Einstein broadly synonymous with genius.
    """

    # 1. Sentence splitting
    total += 1
    sents = split_sentences(text)
    passed += test("Sentence splitting", len(sents) >= 5, f"{len(sents)} sentences")

    # 2. Basic summarization
    total += 1
    s = Summarizer()
    result = s.summarize(text, n_sentences=2)
    ok = len(result['sentences']) == 2 and len(result['summary']) < len(text)
    passed += test("Summarize to 2 sentences", ok,
                   f"compression={result['stats']['compression']}")

    # 3. Headline generation
    total += 1
    headline = s.headline(text)
    ok = len(headline) > 10 and len(headline) < 150
    passed += test("Headline generation", ok, f"'{headline}'")

    # 4. Bullet points
    total += 1
    bullets = s.bullet_points(text, n_points=3)
    ok = len(bullets) == 3
    passed += test("Bullet points", ok, f"{len(bullets)} points")

    # 5. Query-focused summary
    total += 1
    result = s.summarize(text, query="Nobel Prize", n_sentences=2)
    ok = 'Nobel' in result['summary'] or 'prize' in result['summary'].lower()
    passed += test("Query-focused (Nobel Prize)", ok, f"'{result['summary'][:80]}...'")

    # 6. Sentence compression
    total += 1
    from core.summarizer import Summarizer as S2
    s2 = S2()
    compressed = s2._compress_sentence(
        "It is worth noting that basically Einstein was very extremely famous."
    )
    ok = len(compressed) < len("It is worth noting that basically Einstein was very extremely famous.")
    passed += test("Sentence compression", ok, f"'{compressed}'")

    # 7. Topic extraction
    total += 1
    te = TopicExtractor()
    topics = te.extract_topics(text, n=5)
    ok = len(topics) >= 3
    passed += test("Topic extraction", ok, f"topics={topics}")

    # ============================================================
    # TRANSLATOR
    # ============================================================
    print("\n=== TRANSLATOR ===")
    from core.translator import Translator

    t = Translator(use_api_fallback=False)

    # 8. DE→EN simple
    total += 1
    r = t.translate("Das ist gut", source='de', target='en')
    ok = 'good' in r['translation'].lower()
    passed += test("DE→EN 'Das ist gut'", ok, f"'{r['translation']}'")

    # 9. EN→DE simple
    total += 1
    r = t.translate("The house is big", source='en', target='de')
    ok = r['translation'] != "The house is big"  # Something was translated
    passed += test("EN→DE 'The house is big'", ok, f"'{r['translation']}'")

    # 10. Auto-detect German
    total += 1
    r = t.translate("Ich bin müde und möchte schlafen")
    ok = r['source_lang'] == 'de'
    passed += test("Auto-detect German", ok, f"detected={r['source_lang']}")

    # 11. Auto-detect English
    total += 1
    r = t.translate("The weather is beautiful today")
    ok = r['source_lang'] == 'en'
    passed += test("Auto-detect English", ok, f"detected={r['source_lang']}")

    # 12. Phrase translation
    total += 1
    r = t.translate("Zum Beispiel ist es auf jeden Fall wichtig", source='de', target='en')
    ok = 'for example' in r['translation'].lower() or 'definitely' in r['translation'].lower()
    passed += test("Phrase translation", ok, f"'{r['translation']}'")

    # 13. Compound splitting
    total += 1
    result = t._split_compound('Haustür', t.de_en)
    ok = result is not None and 'door' in result.lower()
    passed += test("Compound splitting (Haustür)", ok, f"'{result}'")

    # 14. Custom word
    total += 1
    t.add_word('Tensorfeld', 'tensor field', 'de')
    r = t.translate("Das Tensorfeld ist wichtig", source='de', target='en')
    ok = 'tensor field' in r['translation'].lower()
    passed += test("Custom word addition", ok, f"'{r['translation']}'")

    # ============================================================
    # EMBEDDINGS
    # ============================================================
    print("\n=== EMBEDDINGS ===")
    from core.embeddings import EmbeddingEngine, cosine_similarity

    corpus = [
        "Python is a programming language used for web development and AI",
        "Java is an object-oriented programming language for enterprise software",
        "The cat sat on the mat and looked at the window",
        "Dogs are loyal pets that enjoy walks and playing fetch",
        "Machine learning uses algorithms to learn from data patterns",
        "Deep learning is a subset of machine learning using neural networks",
        "The weather today is sunny with clear blue skies",
        "Rain is expected tomorrow with thunderstorms in the afternoon",
        "Einstein developed the theory of relativity in physics",
        "Newton discovered the laws of gravity and motion",
    ]

    # 15. Fit embeddings
    total += 1
    engine = EmbeddingEngine(dim=8)
    engine.fit(corpus, labels=[f"doc_{i}" for i in range(len(corpus))])
    ok = len(engine.embeddings) == len(corpus)
    passed += test("Fit embeddings", ok, f"{len(engine.embeddings)} docs, dim={len(engine.embeddings[0])}")

    # 16. Similarity: programming languages should be similar
    total += 1
    sim_prog = cosine_similarity(engine.embeddings[0], engine.embeddings[1])
    sim_prog_cat = cosine_similarity(engine.embeddings[0], engine.embeddings[2])
    ok = sim_prog > sim_prog_cat
    passed += test("Semantic similarity (programming > random)", ok,
                   f"prog={sim_prog:.3f}, cross={sim_prog_cat:.3f}")

    # 17. Most similar search
    total += 1
    similar = engine.most_similar("programming code software", top_k=3)
    ok = len(similar) >= 2
    labels = [s[0] for s in similar]
    passed += test("Most similar search", ok, f"top={labels}")

    # 18. Clustering
    total += 1
    clusters = engine.cluster(k=3)
    ok = clusters['k'] == 3 and len(clusters['assignments']) == len(corpus)
    passed += test("K-means clustering", ok, f"k={clusters['k']}, clusters={clusters['clusters']}")

    # ============================================================
    # SCHEDULER
    # ============================================================
    print("\n=== SCHEDULER ===")
    from core.scheduler import TaskScheduler, Pipeline, Priority, EventTrigger

    # 19. Task scheduling with dependencies
    total += 1
    sched = TaskScheduler()
    sched.add('fetch', lambda: 'data_fetched', priority=Priority.HIGH)
    t2 = sched.add('process', lambda _dep_task_0001=None, **kw: f'processed:{_dep_task_0001}',
                    depends_on=['task_0001'])
    sched.add('report', lambda _dep_task_0002=None, **kw: f'report:{_dep_task_0002}',
              depends_on=['task_0002'])
    results = sched.run_all()
    ok = results.get('task_0001') == 'data_fetched'
    passed += test("Task scheduling", ok, f"results={results}")

    # 20. Execution order (topological sort)
    total += 1
    sched2 = TaskScheduler()
    sched2.add('A', lambda: 'a')
    sched2.add('B', lambda: 'b', depends_on=['task_0001'])
    sched2.add('C', lambda: 'c', depends_on=['task_0001'])
    sched2.add('D', lambda: 'd', depends_on=['task_0002', 'task_0003'])
    order = sched2.get_execution_order()
    ok = order.index('task_0001') < order.index('task_0002')
    ok = ok and order.index('task_0001') < order.index('task_0003')
    ok = ok and order.index('task_0002') < order.index('task_0004')
    passed += test("Topological sort", ok, f"order={order}")

    # 21. Retry on failure
    total += 1
    call_count = [0]
    def flaky():
        call_count[0] += 1
        if call_count[0] < 3:
            raise ValueError("not yet")
        return "success"

    sched3 = TaskScheduler()
    sched3.add('flaky', flaky, max_retries=3, retry_delay=0.01)
    results = sched3.run_all()
    ok = results.get('task_0001') == 'success' and call_count[0] == 3
    passed += test("Retry logic", ok, f"attempts={call_count[0]}")

    # 22. Pipeline
    total += 1
    pipe = Pipeline("test")
    pipe.add_step("double", lambda _input=None, **kw: (_input or 0) * 2)
    pipe.add_step("add_ten", lambda _input=None, **kw: (_input or 0) + 10)
    pipe.add_step("to_string", lambda _input=None, **kw: f"result={_input}")
    result = pipe.run(initial_input=5)
    ok = result['result'] == "result=20" and result['completed']
    passed += test("Pipeline execution", ok, f"result={result['result']}")

    # 23. Event trigger
    total += 1
    fired = [False]
    trigger = EventTrigger(
        "test",
        condition=lambda ctx: ctx.get('value', 0) > 10,
        action=lambda ctx: fired.__setitem__(0, True),
        cooldown=0,
    )
    trigger.check({'value': 5})   # Should not fire
    ok1 = not fired[0]
    trigger.check({'value': 15})  # Should fire
    ok = ok1 and fired[0]
    passed += test("Event trigger", ok, f"fired={fired[0]}")

    # ============================================================
    # FILE WATCHER
    # ============================================================
    print("\n=== FILE WATCHER ===")
    from core.watcher import FileWatcher, ChangeType

    # 24. Detect file creation
    total += 1
    with tempfile.TemporaryDirectory() as tmpdir:
        watcher = FileWatcher(tmpdir, poll_interval=0.1, debounce=0, recursive=False)
        watcher._take_snapshot()

        # Create a file
        test_file = os.path.join(tmpdir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('hello')

        events = watcher.scan()
        ok = any(e.change_type == ChangeType.CREATED for e in events)
        passed += test("Detect file creation", ok, f"{len(events)} events")

    # 25. Detect file modification
    total += 1
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('hello')

        watcher = FileWatcher(tmpdir, poll_interval=0.1, debounce=0, recursive=False)
        watcher._take_snapshot()

        time.sleep(0.05)
        with open(test_file, 'w') as f:
            f.write('hello world')

        events = watcher.scan()
        ok = any(e.change_type == ChangeType.MODIFIED for e in events)
        passed += test("Detect file modification", ok, f"{len(events)} events")

    # 26. Detect file deletion
    total += 1
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('hello')

        watcher = FileWatcher(tmpdir, poll_interval=0.1, debounce=0, recursive=False)
        watcher._take_snapshot()

        os.unlink(test_file)
        events = watcher.scan()
        ok = any(e.change_type == ChangeType.DELETED for e in events)
        passed += test("Detect file deletion", ok, f"{len(events)} events")

    # 27. Pattern matching callback
    total += 1
    py_events = []
    with tempfile.TemporaryDirectory() as tmpdir:
        watcher = FileWatcher(tmpdir, poll_interval=0.1, debounce=0, recursive=False)
        watcher.add_pattern('*.py', lambda e: py_events.append(e))
        watcher._take_snapshot()

        # Create .py and .txt files
        with open(os.path.join(tmpdir, 'test.py'), 'w') as f:
            f.write('print("hello")')
        with open(os.path.join(tmpdir, 'test.txt'), 'w') as f:
            f.write('hello')

        watcher.scan()
        ok = len(py_events) == 1 and py_events[0].path.endswith('.py')
        passed += test("Pattern matching callback (*.py only)", ok,
                       f"py_events={len(py_events)}")

    # 28. Ignore patterns
    total += 1
    with tempfile.TemporaryDirectory() as tmpdir:
        watcher = FileWatcher(tmpdir, poll_interval=0.1, debounce=0, recursive=False)
        watcher._take_snapshot()

        with open(os.path.join(tmpdir, 'test.pyc'), 'w') as f:
            f.write('bytecode')
        with open(os.path.join(tmpdir, 'real.py'), 'w') as f:
            f.write('code')

        events = watcher.scan()
        paths = [e.path for e in events]
        ok = not any('.pyc' in p for p in paths) and any('.py' in p for p in paths)
        passed += test("Ignore patterns (.pyc ignored)", ok,
                       f"events={[os.path.basename(p) for p in paths]}")

    # Summary
    print("\n" + "=" * 60)
    pct = (passed / total * 100) if total else 0
    print(f"RESULT: {passed}/{total} ({pct:.0f}%)")
    print("=" * 60)
    return passed, total


if __name__ == '__main__':
    run()
