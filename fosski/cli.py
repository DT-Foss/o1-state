#!/usr/bin/env python3
"""
FOSS-KI CLI — Command-line interface for the Markov Chain AI
==============================================================
Usage:
    foss-ki ask "What is the capital of France?"
    foss-ki feed "France's capital is Paris. Germany's capital is Berlin."
    foss-ki feed --file wikipedia.txt
    foss-ki train --text data/alice.txt
    foss-ki chat
    foss-ki selftest [--facts model.json]
    foss-ki gaps
    foss-ki patterns
    foss-ki status
    foss-ki save model.json
    foss-ki load model.json
    foss-ki info
"""

import sys
import os
import argparse
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.dialog import DialogSystem  # noqa: E402
from core.metacognition import MetacognitionEngine  # noqa: E402


class FossKICLI:
    """CLI wrapper around FOSS-KI DialogSystem."""

    def __init__(self):
        self.dialog = DialogSystem(knowledge_dim=128)
        self.meta = MetacognitionEngine(self.dialog.knowledge)
        self.model_path = None

    def cmd_ask(self, question):
        """Ask a single question."""
        result = self.dialog.turn(question)
        print(f"  {result['response']}")
        print(f"  [{result['confidence']}] subject={result['subject']}, "
              f"relation={result['relation']}")

    def cmd_feed(self, text=None, file_path=None):
        """Feed text to extract facts via metacognition pipeline."""
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.pdf':
                try:
                    import fitz
                    doc = fitz.open(file_path)
                    pages = []
                    for page in doc:
                        pages.append(page.get_text())
                    text = '\n'.join(pages)
                    doc.close()
                    print(f"  Read {len(pages)} pages from PDF")
                except ImportError:
                    print("  Error: PyMuPDF (fitz) not installed. Run: pip install pymupdf")
                    return
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

        if not text:
            print("Error: no text provided")
            return

        n_before = self.dialog.knowledge.n_facts
        n_new = self.meta.ingest(text)
        n_after = self.dialog.knowledge.n_facts

        print(f"  Extracted {n_new} new facts ({n_before} → {n_after} total)")
        # Show last N facts added
        if n_new > 0:
            for s, r, o in self.dialog.knowledge.facts[-min(n_new, 10):]:
                print(f"    ({s}, {r}, {o})")
            if n_new > 10:
                print(f"    ... and {n_new - 10} more")
        print(f"  Corpus indexed for gap-filling: "
              f"{sum(len(v) for v in self.meta.filler._corpus.values())} entries")

    def cmd_train(self, text_path):
        """Train language model on text file."""
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"  Training FLM on {len(text)} characters...")
        t0 = time.time()
        # Access the underlying engine's language model
        try:
            from core.engine import FossKI
            fki = FossKI()
            fki.train_language(text[:100000])  # Cap at 100K for speed
            t_train = time.time() - t0
            bpc = fki.text_bpc(text[100000:101000]) if len(text) > 101000 else None
            print(f"  Trained in {t_train:.1f}s")
            if bpc:
                print(f"  BPC on held-out: {bpc:.3f}")
        except Exception as e:
            print(f"  Error: {e}")

    def cmd_chat(self):
        """Interactive chat mode with full Agent pipeline."""
        from core.agent import Agent

        # Create Agent (handles all pipeline setup internally)
        agent = Agent(workspace_dir=os.path.dirname(os.path.abspath(__file__)))

        # Auto-bootstrap if no facts loaded
        if agent.dialog.knowledge.n_facts == 0:
            print("  Auto-loading world knowledge...")
            n = agent.bootstrap()
            print(f"  Loaded {n} facts.")
            print()

        print("FOSS-KI Agent — Transformer-Free AI Assistant")
        print(f"  Knowledge: {agent.dialog.knowledge.n_facts} facts | CPU-only | No API calls")
        print("  Capabilities: Q&A, Code, Math, Files, Reasoning, Text Generation")
        print("  Commands: 'quit', 'load <file>', 'feed <text>', 'bootstrap',")
        print("            'status', 'export <path>', 'help'")
        print("-" * 60)

        # Track conversation for export
        chat_history = []

        while True:
            try:
                user_input = input("\n  You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye.")
                break

            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("  Goodbye.")
                break
            if user_input.lower().startswith('load '):
                path = user_input[5:].strip()
                self.cmd_feed(file_path=path)
                continue
            if user_input.lower().startswith('feed '):
                self.cmd_feed(text=user_input[5:].strip())
                continue
            if user_input.lower() == 'status':
                self.cmd_status()
                continue
            if user_input.lower() == 'gaps':
                self.cmd_gaps()
                continue
            if user_input.lower() in ('explore', 'frontiers'):
                self.cmd_explore()
                continue
            if user_input.lower() == 'bootstrap':
                self.cmd_bootstrap()
                continue
            if user_input.lower() == 'selftest':
                self.cmd_selftest()
                continue
            if user_input.lower().startswith('export'):
                parts = user_input.split(maxsplit=1)
                path = parts[1].strip() if len(parts) > 1 else None
                self._export_chat(chat_history, path)
                continue

            t0 = time.time()
            result = agent.process(user_input)
            t_ms = (time.time() - t0) * 1000

            chat_history.append({
                'user': user_input,
                'response': result['response'],
                'intent': result.get('intent', ''),
                'source': result.get('source', ''),
                'latency_ms': round(t_ms),
            })

            print(f"  FOSS-KI: {result['response']}")
            print(f"  [{result['intent']}, {result['source']}, {t_ms:.0f}ms]")

    def _export_chat(self, history, path=None):
        """Export conversation history as Markdown or JSON."""
        if not history:
            print("  No conversation to export.")
            return

        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        if path is None:
            path = f"conversation_{ts}.md"

        if path.endswith('.json'):
            export = {
                'engine': 'FOSS-KI',
                'exported': datetime.now().isoformat(),
                'turns': len(history),
                'conversation': history,
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(export, f, indent=2, ensure_ascii=False)
        else:
            if not path.endswith('.md'):
                path += '.md'
            lines = [
                f"# FOSS-KI Conversation",
                f"*Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
                f"*Turns: {len(history)}*\n",
                "---\n",
            ]
            for i, turn in enumerate(history, 1):
                lines.append(f"**You:** {turn['user']}\n")
                lines.append(f"**FOSS-KI:** {turn['response']}\n")
                lines.append(f"*[{turn['intent']}, {turn['source']}, {turn['latency_ms']}ms]*\n")
                lines.append("---\n")
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

        print(f"  Exported {len(history)} turns to {path}")

    def cmd_save(self, path):
        """Save knowledge store to file."""
        self.dialog.knowledge.save(path)
        print(f"  Saved {self.dialog.knowledge.n_facts} facts to {path}")

    def cmd_load(self, path):
        """Load knowledge store from file."""
        from core.knowledge import KnowledgeStore
        self.dialog.knowledge = KnowledgeStore.load(path)
        # Re-wire metacognition to the new knowledge store
        self.meta = MetacognitionEngine(self.dialog.knowledge)
        print(f"  Loaded {self.dialog.knowledge.n_facts} facts from {path}")

    def cmd_bootstrap(self, save_path=None, wikidata=False):
        """Load core world knowledge. Brain-first: loads .brain if available."""
        from core.agent import Agent
        t0 = time.time()
        agent = Agent(workspace_dir=os.path.dirname(os.path.abspath(__file__)))
        n = agent.bootstrap(use_brain=True, fetch_wikidata=wikidata)
        self.dialog = agent.dialog
        self.meta = MetacognitionEngine(self.dialog.knowledge)
        elapsed = time.time() - t0
        print(f"  Bootstrapped {n} facts in {elapsed:.1f}s")
        if save_path:
            self.cmd_save(save_path)

    def cmd_wikidata(self, categories=None, verbose=True):
        """Pull structured knowledge from Wikidata SPARQL endpoint."""
        from core.agent import Agent
        from core.wikidata import wikidata_live_import

        # Bootstrap first (from brain if available)
        agent = Agent(workspace_dir=os.path.dirname(os.path.abspath(__file__)))
        agent.bootstrap(use_brain=True, fetch_wikidata=False)
        self.dialog = agent.dialog

        print("Wikidata Live Import")
        print("=" * 40)
        t0 = time.time()

        stats = wikidata_live_import(
            self.dialog.knowledge,
            categories=categories,
            verbose=verbose,
        )

        elapsed = time.time() - t0
        print(f"\nTotal: {stats['total']} new facts in {elapsed:.1f}s")
        print(f"KB now: {self.dialog.knowledge.n_facts} facts")

        if stats.get('errors'):
            for err in stats['errors']:
                print(f"  ERROR: {err}")

        # Save updated brain
        from core.brain import BrainSnapshot
        brain_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data', 'foss-ki.brain'
        )
        snap = BrainSnapshot.capture_from_parts(
            knowledge_store=self.dialog.knowledge,
        )
        snap.save(brain_path, compressed=True)
        size_kb = os.path.getsize(brain_path) / 1024
        print(f"Brain saved: {brain_path} ({size_kb:.0f} KB)")

        self.meta = MetacognitionEngine(self.dialog.knowledge)
        return stats

    def cmd_brain(self, action='info'):
        """Brain file management."""
        from core.brain import BrainSnapshot
        brain_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data', 'foss-ki.brain'
        )

        if action == 'info':
            if not os.path.exists(brain_path):
                print("  No brain file found. Run 'bootstrap' or 'wikidata' first.")
                return
            snap = BrainSnapshot.load(brain_path)
            print(snap.summary())
            size_kb = os.path.getsize(brain_path) / 1024
            print(f"  Size:     {size_kb:.0f} KB")
            print(f"  Path:     {brain_path}")

        elif action == 'rebuild':
            print("  Rebuilding brain from scratch...")
            from core.agent import Agent
            agent = Agent(workspace_dir=os.path.dirname(os.path.abspath(__file__)))
            n = agent.bootstrap(use_brain=False, fetch_wikidata=False)
            print(f"  Done: {n} facts")

        elif action == 'delete':
            if os.path.exists(brain_path):
                os.remove(brain_path)
                print(f"  Brain deleted: {brain_path}")
            else:
                print("  No brain file to delete.")

    def cmd_distill(self, model='qwen3:4b', max_entities=50,
                    overnight=False, hours=8, no_verify=False):
        """Distill knowledge from local LLM into brain file."""
        # Bootstrap first
        self.cmd_bootstrap(wikidata=False)

        brain_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data', 'foss-ki.brain'
        )

        from core.distillation import DistillationEngine
        engine = DistillationEngine(
            self.dialog.knowledge,
            model=model,
            verify=not no_verify,
            verbose=True,
        )

        if not engine.check_available():
            print(f"\n  Model '{model}' not available in Ollama.")
            print(f"  Run: ollama pull {model}")
            return

        print(f"\n  FOSS-KI Distillation Engine")
        print(f"  Model: {model}")
        print(f"  Starting KB: {self.dialog.knowledge.n_facts} facts")
        print()

        if overnight:
            engine.run_overnight(hours=hours, brain_path=brain_path)
        else:
            engine.run(max_entities=max_entities, brain_path=brain_path)

    def cmd_grow(self, model='qwen3:4b', max_entities=20, hours=0,
                 no_harvest=False, no_infer=False, no_distill=False):
        """
        Full growth pipeline: harvest → distill → infer → save.
        The path to 1M facts.
        """
        from core.agent import Agent

        agent = Agent(workspace_dir=os.path.dirname(os.path.abspath(__file__)))
        n = agent.bootstrap(use_brain=True)
        self.dialog = agent.dialog
        print(f"  Phase 0: Bootstrap — {n} facts")

        brain_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data', 'foss-ki.brain'
        )

        # Phase 1: Harvest stdlib
        if not no_harvest:
            from core.harvester import KnowledgeHarvester
            harvester = KnowledgeHarvester(self.dialog.knowledge)
            new = harvester.harvest_all(verbose=False)
            print(f"  Phase 1: Harvest — +{new} facts ({self.dialog.knowledge.n_facts} total)")

        # Phase 2: Distill from LLM (gap-filling + all layers)
        if not no_distill:
            from core.distillation import DistillationEngine
            engine = DistillationEngine(
                self.dialog.knowledge, model=model,
                verify=False, verbose=False)
            if engine.check_available():
                if hours > 0:
                    engine.run_overnight(hours=hours, brain_path=brain_path)
                else:
                    engine.run(max_entities=max_entities, brain_path=brain_path,
                              save_brain=False)
                    # Also run common sense + word relations on sample entities
                    entities = self.dialog.knowledge.all_entities()[:50]
                    engine.distill_common_sense(entities, max_questions=100)
                    engine.distill_word_relations(entities, max_questions=100)
                print(f"  Phase 2: Distill — {engine.reasoning.stats['triplets_stored']} "
                      f"new ({self.dialog.knowledge.n_facts} total)")
            else:
                print(f"  Phase 2: Distill — SKIPPED (Ollama not available)")

        # Phase 3: 3-pass inference
        if not no_infer:
            from core.inference import InferenceEngine
            inf = InferenceEngine(self.dialog.knowledge)
            inf.run(store_inferred=True)
            total_inferred = sum(inf.stats.values())
            print(f"  Phase 3: Infer — +{total_inferred} facts ({self.dialog.knowledge.n_facts} total)")

        # Save brain
        from core.brain import BrainSnapshot
        snap = BrainSnapshot.capture_from_parts(
            knowledge_store=self.dialog.knowledge)
        snap.save(brain_path, compressed=True)
        size_kb = os.path.getsize(brain_path) / 1024
        print(f"\n  Done. Brain: {size_kb:.0f} KB, {self.dialog.knowledge.n_facts} facts")

    def cmd_harvest(self, stdlib_only=False):
        """Harvest facts from Python modules via introspection."""
        from core.agent import Agent
        from core.harvester import KnowledgeHarvester

        agent = Agent(workspace_dir=os.path.dirname(os.path.abspath(__file__)))
        n = agent.bootstrap(use_brain=True)
        self.dialog = agent.dialog
        print(f"  Loaded {n} facts")

        harvester = KnowledgeHarvester(self.dialog.knowledge)
        if stdlib_only:
            new = harvester.harvest_stdlib(verbose=True)
        else:
            new = harvester.harvest_all(verbose=True)

        print(f"\n  Harvested {new} new facts")
        print(f"  KB total: {self.dialog.knowledge.n_facts}")

        # Save brain
        from core.brain import BrainSnapshot
        brain_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data', 'foss-ki.brain'
        )
        snap = BrainSnapshot.capture_from_parts(
            knowledge_store=self.dialog.knowledge)
        snap.save(brain_path, compressed=True)
        size_kb = os.path.getsize(brain_path) / 1024
        print(f"  Brain saved: {size_kb:.0f} KB")

    def cmd_infer(self):
        """Run 3-pass inference to amplify knowledge base."""
        from core.agent import Agent
        from core.inference import InferenceEngine
        import time as _time

        agent = Agent(workspace_dir=os.path.dirname(os.path.abspath(__file__)))
        n = agent.bootstrap(use_brain=True)
        self.dialog = agent.dialog
        print(f"  Loaded {n} facts")

        engine = InferenceEngine(self.dialog.knowledge)
        t0 = _time.time()
        inferred = engine.run(store_inferred=True)
        elapsed = _time.time() - t0

        print(f"  {engine.summary()}")
        print(f"  Time: {elapsed:.1f}s")

        # Save updated brain
        from core.brain import BrainSnapshot
        brain_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data', 'foss-ki.brain'
        )
        snap = BrainSnapshot.capture_from_parts(
            knowledge_store=self.dialog.knowledge)
        snap.save(brain_path, compressed=True)
        size_kb = os.path.getsize(brain_path) / 1024
        print(f"  Brain saved: {size_kb:.0f} KB ({self.dialog.knowledge.n_facts} facts)")

    def cmd_selftest(self, facts_path=None, test_path=None):
        """Run self-test benchmark."""
        if facts_path:
            self.cmd_load(facts_path)

        # Default test queries if no test file provided
        if test_path:
            with open(test_path, 'r', encoding='utf-8') as f:
                test_queries = json.load(f)
        else:
            # Auto-generate test queries from stored facts
            test_queries = []
            for s, r, o in self.dialog.knowledge.facts:
                if r in ('capital', 'language', 'population', 'location',
                         'inventor', 'discoverer', 'creator', 'author',
                         'formula', 'symbol', 'founder', 'birthplace'):
                    test_queries.append({
                        'subject': s,
                        'relation': r,
                        'expected': o,
                    })

        if not test_queries:
            print("  No test queries available. Feed facts first or provide --tests file.")
            return

        print(f"  Running self-test with {len(test_queries)} queries...")
        t0 = time.time()
        result = self.meta.selftest(test_queries)
        elapsed = time.time() - t0

        print(f"  Accuracy: {result['accuracy']:.0%} ({result['correct']}/{result['total']})")
        print(f"  Time: {elapsed:.1f}s")

        # Show failures
        failures = [r for r in result['results'] if not r['correct']]
        if failures:
            print(f"\n  Failures ({len(failures)}):")
            for f in failures[:20]:
                print(f"    ✗ ({f['subject']}, {f['relation']}) "
                      f"→ {f['got']} (expected: {f['expected']}, {f['confidence']})")

        # Show regressions/improvements
        if result['regressions']:
            print(f"\n  Regressions ({len(result['regressions'])}):")
            for r in result['regressions']:
                print(f"    ↓ ({r['subject']}, {r['relation']}): "
                      f"{r['was']} → {r['now']} (expected: {r['expected']})")

        if result['improvements']:
            print(f"\n  Improvements ({len(result['improvements'])}):")
            for r in result['improvements']:
                print(f"    ↑ ({r['subject']}, {r['relation']}): {r['was']} → {r['now']}")

        print(f"\n  Checkpoint: {result['checkpoint']}")

    def cmd_regression(self):
        """Run natural-language regression tests through full dialog pipeline."""
        import json as _json

        # Auto-bootstrap via Agent (handles both dialog AND math/code/reasoning)
        from core.agent import Agent
        agent = Agent()
        n = agent.bootstrap()
        self.dialog = agent.dialog
        self.meta = agent.meta
        self._agent = agent
        print(f"  Loaded {n} facts")

        regression_path = os.path.join(os.path.dirname(__file__),
                                       'data', 'regression_tests.json')
        if not os.path.exists(regression_path):
            print("  No regression tests found at data/regression_tests.json")
            return

        with open(regression_path, 'r', encoding='utf-8') as f:
            tests = _json.load(f)

        print(f"  Running {len(tests)} regression tests...")
        t0 = time.time()
        passed = 0
        failed = 0
        errors = 0
        failures = []

        for test in tests:
            # Multi-turn sequence test
            if 'sequence' in test:
                try:
                    for step in test['sequence']:
                        if hasattr(self, '_agent'):
                            result = self._agent.process(step['query'])
                        else:
                            result = self.dialog.turn(step['query'])
                    # Only check the last step
                    response = result.get('response', '') or ''
                    last = test['sequence'][-1]
                    expected = last.get('expected_contains', '')
                    if expected and expected.lower() in response.lower():
                        passed += 1
                    elif last.get('expected_not_error'):
                        passed += 1
                    else:
                        failed += 1
                        failures.append({
                            'query': ' → '.join(s['query'] for s in test['sequence']),
                            'expected': expected,
                            'got': response[:120],
                            'category': test.get('category', ''),
                            'bug_ref': test.get('bug_ref', ''),
                        })
                except Exception as e:
                    errors += 1
                    failures.append({
                        'query': ' → '.join(s['query'] for s in test['sequence']),
                        'expected': 'no crash',
                        'got': f'EXCEPTION: {e}',
                        'category': test.get('category', ''),
                        'bug_ref': 'crash',
                    })
                continue

            query = test['query']
            try:
                # Route through Agent (handles math, code, reasoning + dialog)
                if hasattr(self, '_agent'):
                    result = self._agent.process(query)
                else:
                    result = self.dialog.turn(query)
                response = result.get('response', '') or ''

                if test.get('expected_not_error'):
                    # Just check it didn't crash
                    passed += 1
                    continue

                expected = test.get('expected_contains', '')
                expected_any = test.get('expected_contains_any', [])
                not_expected = test.get('expected_not_contains', '')
                if not_expected and not_expected.lower() in response.lower():
                    failed += 1
                    failures.append({
                        'query': query,
                        'expected': f'NOT {not_expected}',
                        'got': response[:120],
                        'category': test.get('category', ''),
                        'bug_ref': test.get('bug_ref', ''),
                    })
                    continue
                # Check expected_contains_any (any match = pass)
                if expected_any:
                    if any(e.lower() in response.lower() for e in expected_any):
                        passed += 1
                        continue
                    else:
                        failed += 1
                        failures.append({
                            'query': query,
                            'expected': f'any of {expected_any}',
                            'got': response[:120],
                            'category': test.get('category', ''),
                            'bug_ref': test.get('bug_ref', ''),
                        })
                        continue
                if expected.lower() in response.lower():
                    passed += 1
                else:
                    failed += 1
                    failures.append({
                        'query': query,
                        'expected': expected,
                        'got': response[:120],
                        'category': test.get('category', ''),
                        'bug_ref': test.get('bug_ref', ''),
                    })
            except Exception as e:
                errors += 1
                failures.append({
                    'query': query,
                    'expected': test.get('expected_contains', 'no crash'),
                    'got': f'EXCEPTION: {e}',
                    'category': test.get('category', ''),
                    'bug_ref': 'crash',
                })

        elapsed = time.time() - t0
        total = passed + failed + errors
        avg_ms = elapsed / max(total, 1) * 1000
        max_budget = 500  # ms per query
        print(f"  Result: {passed}/{total} passed ({passed/total:.0%})")
        print(f"  Time: {elapsed:.1f}s ({avg_ms:.0f}ms avg, budget: {max_budget}ms)")
        if avg_ms > max_budget:
            print(f"  WARNING: avg latency {avg_ms:.0f}ms exceeds {max_budget}ms budget!")

        if failures:
            print(f"\n  Failures ({len(failures)}):")
            for f in failures:
                print(f"    ✗ \"{f['query']}\"")
                print(f"      expected: {f['expected']}")
                print(f"      got: {f['got']}")
                if f['bug_ref']:
                    print(f"      ref: {f['bug_ref']}")

        if errors:
            print(f"\n  Errors: {errors} queries crashed (graceful failure broken!)")

        return {'passed': passed, 'failed': failed, 'errors': errors, 'total': total,
                'avg_ms': avg_ms, 'elapsed': elapsed}

    def cmd_gaps(self):
        """Show knowledge gap report."""
        print(self.meta.gap_report())

    def cmd_explore(self):
        """Show frontier exploration report (VizDoom-ported knowledge explorer)."""
        print(self.meta.frontier_report())

    def cmd_patterns(self):
        """Discover new extraction patterns from failures."""
        print(self.meta.learn_patterns())

    def cmd_status(self):
        """Show metacognition status."""
        print(self.meta.status())

    def cmd_nexus(self, watch=False, status=False, tasks=None):
        """Start Gamma — FOSS-KI as Nexus agent."""
        from core.agent import Agent
        from core.nexus import NexusAdapter

        agent = Agent()
        n = agent.bootstrap()
        print(f"  Gamma initialized with {n} facts")

        nx = NexusAdapter(agent)

        # Process existing messages first
        results = nx.process_new()
        extracted = sum(r['facts_extracted'] for r in results)
        tracked = sum(r['tasks_found'] for r in results)
        print(f"  Processed {len(results)} messages, "
              f"extracted {extracted} facts, tracked {tracked} task mentions")

        if status:
            print()
            print(nx.status_report())
            return

        if tasks is not None:
            task_results = nx.query_tasks(tasks)
            if not task_results:
                print("  No matching tasks found.")
            else:
                for tid, info in sorted(task_results):
                    print(f"  {tid}: [{info['status']}] {info['description'][:60]}")
            return

        if watch:
            print(f"\n  Gamma watching Nexus... (Ctrl+C to stop)")
            try:
                nx.watch()
            except KeyboardInterrupt:
                print("\n  Gamma stopped.")
                print(nx.status_report())
        else:
            print()
            print(nx.status_report())

    def cmd_shovel(self, output_path=None):
        """Export knowledge with dummy substitutions (Shovel Mode)."""
        from core.shovel import ShovelMode

        if self.dialog.knowledge.n_facts == 0:
            from core.wikidata import WikidataImporter
            imp = WikidataImporter(self.dialog.knowledge)
            n = imp.import_countries_simple()
            print(f"  Bootstrapped {n} facts")

        shovel = ShovelMode(self.dialog.knowledge)
        export = shovel.export_dummy()
        stats = shovel.get_stats()

        print(f"  Shovel Mode: {export['n_facts']} facts, "
              f"{export['n_entities_mapped']} entities mapped")
        print(f"  Types: {stats['types']}")

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export, f, indent=2, ensure_ascii=False)
            print(f"  Exported to {output_path}")
            # Save mapping separately (NEVER export this)
            map_path = output_path.replace('.json', '_MAPPING_SECRET.json')
            with open(map_path, 'w', encoding='utf-8') as f:
                json.dump(shovel.export_mapping(), f, indent=2, ensure_ascii=False)
            print(f"  Mapping saved to {map_path} (KEEP SECRET)")
        else:
            print("  Use --output to export. Sample:")
            for f in export['facts'][:5]:
                print(f"    {f}")

    def cmd_info(self):
        """Show system info."""
        print(f"  FOSS-KI — Markov Chain AI")
        print(f"  Knowledge facts: {self.dialog.knowledge.n_facts}")
        print(f"  Conversation turns: {self.dialog.n_turns}")
        print(f"  Entities seen: {len(self.dialog.context_entities)}")
        print(f"  Knowledge dim: {self.dialog.knowledge.dim}")


def main():
    parser = argparse.ArgumentParser(
        description="FOSS-KI — Markov Chain AI (no Transformer, no GPU)")
    sub = parser.add_subparsers(dest='command')

    # ask
    p_ask = sub.add_parser('ask', help='Ask a question')
    p_ask.add_argument('question', type=str)
    p_ask.add_argument('--facts', type=str, help='Load facts file first')

    # feed
    p_feed = sub.add_parser('feed', help='Feed text to extract facts')
    p_feed.add_argument('text', nargs='?', type=str, help='Text to process')
    p_feed.add_argument('--file', '-f', type=str, help='File to read')

    # train
    p_train = sub.add_parser('train', help='Train language model on text')
    p_train.add_argument('--text', type=str, required=True)

    # chat
    sub.add_parser('chat', help='Interactive chat mode')

    # save/load
    p_save = sub.add_parser('save', help='Save knowledge to file')
    p_save.add_argument('path', type=str)

    p_load = sub.add_parser('load', help='Load knowledge from file')
    p_load.add_argument('path', type=str)

    # bootstrap
    p_boot = sub.add_parser('bootstrap', help='Load core knowledge (countries, science, people)')
    p_boot.add_argument('--save', type=str, help='Save facts after bootstrap')
    p_boot.add_argument('--wikidata', action='store_true', help='Also fetch from Wikidata SPARQL')

    # wikidata
    p_wiki = sub.add_parser('wikidata', help='Pull structured knowledge from Wikidata')
    p_wiki.add_argument('--categories', type=str, nargs='*',
                        help='Categories to import (default: all)')

    # brain
    p_brain = sub.add_parser('brain', help='Brain file management')
    p_brain.add_argument('action', nargs='?', default='info',
                         choices=['info', 'rebuild', 'delete'],
                         help='Action: info (default), rebuild, delete')

    # distill
    p_distill = sub.add_parser('distill', help='Distill knowledge from local LLM')
    p_distill.add_argument('--model', type=str, default='qwen3:4b',
                           help='Ollama model (default: qwen3:4b)')
    p_distill.add_argument('--max-entities', type=int, default=50,
                           help='Max entities per run (default: 50)')
    p_distill.add_argument('--overnight', action='store_true',
                           help='Run overnight (8h default)')
    p_distill.add_argument('--hours', type=int, default=8,
                           help='Hours for overnight run (default: 8)')
    p_distill.add_argument('--no-verify', action='store_true',
                           help='Skip consistency check (faster, less safe)')

    # grow
    p_grow = sub.add_parser('grow', help='Full growth: harvest → distill → infer')
    p_grow.add_argument('--model', type=str, default='qwen3:4b')
    p_grow.add_argument('--max-entities', type=int, default=20)
    p_grow.add_argument('--hours', type=int, default=0,
                        help='Hours for overnight distillation (0=single batch)')
    p_grow.add_argument('--no-harvest', action='store_true')
    p_grow.add_argument('--no-infer', action='store_true')
    p_grow.add_argument('--no-distill', action='store_true')

    # harvest
    p_harvest = sub.add_parser('harvest', help='Harvest facts from Python stdlib')
    p_harvest.add_argument('--stdlib-only', action='store_true',
                           help='Only stdlib (skip optional packages)')

    # infer
    sub.add_parser('infer', help='Run 3-pass inference to amplify knowledge')

    # selftest
    p_selftest = sub.add_parser('selftest', help='Run self-test benchmark')
    p_selftest.add_argument('--facts', type=str, help='Load facts file first')
    p_selftest.add_argument('--tests', type=str, help='Test queries JSON file')

    # regression
    sub.add_parser('regression', help='Run NL regression tests through dialog pipeline')

    # gaps
    sub.add_parser('gaps', help='Show knowledge gap report')

    # explore
    sub.add_parser('explore', help='Frontier detection — find unexplored knowledge domains')

    # patterns
    sub.add_parser('patterns', help='Discover extraction patterns from failures')

    # status
    sub.add_parser('status', help='Show metacognition status')

    # shovel
    p_shovel = sub.add_parser('shovel', help='Export KB with dummy names (Shovel Mode)')
    p_shovel.add_argument('--output', '-o', type=str, help='Output JSON path')

    # info
    sub.add_parser('info', help='Show system info')

    # nexus
    p_nexus = sub.add_parser('nexus', help='Start Gamma (Nexus agent)')
    p_nexus.add_argument('--watch', action='store_true', help='Watch mode (continuous)')
    p_nexus.add_argument('--status', action='store_true', help='Show Gamma status report')
    p_nexus.add_argument('--tasks', type=str, nargs='?', const='', help='Query tracked tasks')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    cli = FossKICLI()

    if args.command == 'ask':
        if args.facts:
            cli.cmd_load(args.facts)
        cli.cmd_ask(args.question)
    elif args.command == 'feed':
        cli.cmd_feed(text=args.text, file_path=args.file)
    elif args.command == 'train':
        cli.cmd_train(args.text)
    elif args.command == 'chat':
        cli.cmd_chat()
    elif args.command == 'save':
        cli.cmd_save(args.path)
    elif args.command == 'load':
        cli.cmd_load(args.path)
    elif args.command == 'bootstrap':
        cli.cmd_bootstrap(save_path=args.save,
                          wikidata=getattr(args, 'wikidata', False))
    elif args.command == 'wikidata':
        cli.cmd_wikidata(categories=getattr(args, 'categories', None))
    elif args.command == 'distill':
        cli.cmd_distill(
            model=getattr(args, 'model', 'qwen3:4b'),
            max_entities=getattr(args, 'max_entities', 50),
            overnight=getattr(args, 'overnight', False),
            hours=getattr(args, 'hours', 8),
            no_verify=getattr(args, 'no_verify', False),
        )
    elif args.command == 'grow':
        cli.cmd_grow(
            model=getattr(args, 'model', 'qwen3:4b'),
            max_entities=getattr(args, 'max_entities', 20),
            hours=getattr(args, 'hours', 0),
            no_harvest=getattr(args, 'no_harvest', False),
            no_infer=getattr(args, 'no_infer', False),
            no_distill=getattr(args, 'no_distill', False),
        )
    elif args.command == 'harvest':
        cli.cmd_harvest(stdlib_only=getattr(args, 'stdlib_only', False))
    elif args.command == 'infer':
        cli.cmd_infer()
    elif args.command == 'brain':
        cli.cmd_brain(action=getattr(args, 'action', 'info'))
    elif args.command == 'selftest':
        cli.cmd_selftest(facts_path=args.facts, test_path=args.tests)
    elif args.command == 'regression':
        cli.cmd_regression()
    elif args.command == 'gaps':
        cli.cmd_gaps()
    elif args.command == 'explore':
        cli.cmd_explore()
    elif args.command == 'patterns':
        cli.cmd_patterns()
    elif args.command == 'status':
        cli.cmd_status()
    elif args.command == 'info':
        cli.cmd_info()
    elif args.command == 'shovel':
        cli.cmd_shovel(output_path=args.output)
    elif args.command == 'nexus':
        cli.cmd_nexus(watch=args.watch, status=args.status,
                      tasks=args.tasks)


if __name__ == '__main__':
    main()
