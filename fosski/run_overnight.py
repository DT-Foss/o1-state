#!/usr/bin/env python3
"""
FOSS-KI Complete Capability Distillation Daemon
=================================================
24/7 autonomous extraction from local LLM.
NOT just facts — everything: language, patterns, code, reasoning, grammar.

10 Layers per cycle:
  1. Facts (gap-filling)        → KB triplets
  2. FLM Corpus (text gen)      → language model training
  3. Reformulations             → formulierer templates
  4. Reasoning Patterns (CoT)   → constraint solver cases
  5. Common Sense (yes/no)      → physical/social rules
  6. Word Relations             → antonyms, synonyms, hypernyms
  7. Code Templates             → task→method→code patterns
  8. Dialog Patterns            → conversation transitions
  9. Grammar Rules              → error detection
  10. Taxonomies + Causal       → knowledge graph structure

Usage:
    python3 run_overnight.py                    # 8h default
    python3 run_overnight.py --hours 0          # infinite
    python3 run_overnight.py --model qwen3:4b   # specific model
    nohup python3 run_overnight.py --hours 0 > data/overnight_run.log 2>&1 &
"""

import sys
import os
import time
import argparse
import signal
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(DATA_DIR, 'overnight.log')),
    ]
)
log = logging.getLogger('overnight')

# Graceful shutdown
_shutdown = False

def _signal_handler(sig, frame):
    global _shutdown
    log.info(f"Shutdown signal received ({sig}). Finishing current layer...")
    _shutdown = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ============================================================================
# SEED TOPICS — domains an AI engine actually needs
# ============================================================================
SEED_TOPICS = {
    'programming': [
        'python', 'javascript', 'typescript', 'rust', 'go', 'java', 'c++',
        'haskell', 'ruby', 'swift', 'kotlin', 'scala', 'elixir', 'lua',
        'linked list', 'binary tree', 'hash table', 'red-black tree', 'trie',
        'graph', 'stack', 'queue', 'heap', 'bloom filter', 'skip list',
        'quicksort', 'mergesort', 'binary search', 'depth-first search',
        'breadth-first search', 'dijkstra algorithm', 'dynamic programming',
        'gradient descent', 'k-means clustering', 'neural network',
        'singleton pattern', 'factory pattern', 'observer pattern',
        'strategy pattern', 'decorator pattern', 'mvc pattern',
        'http', 'tcp', 'udp', 'websocket', 'grpc', 'mqtt', 'ssh', 'dns',
        'recursion', 'closure', 'polymorphism', 'inheritance',
        'encapsulation', 'concurrency', 'mutex', 'deadlock',
        'garbage collection', 'type inference', 'memoization',
    ],
    'language': [
        'noun', 'verb', 'adjective', 'adverb', 'preposition',
        'conjunction', 'pronoun', 'article', 'interjection',
        'subject', 'predicate', 'clause', 'phrase',
        'dependent clause', 'relative clause',
        'past tense', 'present tense', 'future tense',
        'subjunctive mood', 'passive voice', 'active voice',
        'gerund', 'infinitive', 'participle', 'modal verb',
        'morpheme', 'phoneme', 'syntax', 'semantics',
        'idiom', 'metaphor', 'analogy', 'rhetoric',
    ],
    'math_cs': [
        'derivative', 'integral', 'matrix', 'vector', 'eigenvalue',
        'fourier transform', 'probability', 'bayes theorem',
        'markov chain', 'graph theory', 'set theory', 'boolean algebra',
        'turing machine', 'regular expression', 'context-free grammar',
        'halting problem', 'lambda calculus', 'information theory',
        'entropy', 'kolmogorov complexity',
    ],
}


def _get_or_create_flm(agent):
    """Get or create the FLM (Foss Language Model) for training."""
    if hasattr(agent, '_flm') and agent._flm is not None:
        return agent._flm
    try:
        from core.foss_lm import FossLanguageModel
        flm = FossLanguageModel()
        # Bootstrap from existing corpus if available
        corpus_path = os.path.join(DATA_DIR, 'flm_corpus.txt')
        if os.path.exists(corpus_path):
            with open(corpus_path, 'r', encoding='utf-8') as f:
                existing = f.read()
            if len(existing) > 100:
                flm.train(existing)
                log.info(f"  FLM bootstrapped from existing corpus ({len(existing)} chars)")
        agent._flm = flm
        return flm
    except Exception as e:
        log.warning(f"FLM creation failed: {e}")
        return None


def _inject_seeds(kb, cycle_num):
    """Inject seed entities that don't exist yet."""
    domains = list(SEED_TOPICS.keys())
    domain = domains[cycle_num % len(domains)]
    seeds = SEED_TOPICS[domain]
    injected = 0
    for entity in seeds:
        if entity.lower() not in kb._entity_subject_index:
            kb.store_fact(entity, 'type', f'(seed:{domain})')
            injected += 1
    return domain, injected


# ============================================================================
# FLM CORPUS TOPICS — diverse text generation prompts
# ============================================================================
FLM_PROMPTS = [
    # Technical writing
    "Write a paragraph explaining how a hash table works.",
    "Write a technical description of how TCP ensures reliable data transfer.",
    "Explain the difference between a stack and a queue.",
    "Describe how garbage collection works in programming languages.",
    "Explain what recursion is and give an analogy.",
    "Write a paragraph about the difference between compiled and interpreted languages.",
    "Explain how public-key cryptography works in simple terms.",
    "Describe the concept of Big O notation for algorithm analysis.",
    # Formal/professional
    "Write a formal email declining a meeting invitation.",
    "Write a professional summary of a software project status.",
    "Write a bug report for a login page that doesn't validate email format.",
    "Write a code review comment suggesting improvements to error handling.",
    # Explanatory
    "Explain how neural networks learn, in 3 sentences.",
    "Explain the difference between HTTP and HTTPS.",
    "Write a paragraph about why testing is important in software development.",
    "Explain what a database index is and why it speeds up queries.",
    # Narrative / varied style
    "Describe a day in the life of a software developer.",
    "Write a short comparison of Python and JavaScript for beginners.",
    "Explain the concept of technical debt using a metaphor.",
    "Write instructions for setting up a Python virtual environment.",
    # Science / general knowledge
    "Write a paragraph about how photosynthesis works.",
    "Explain Newton's three laws of motion briefly.",
    "Describe how the internet routes packets from source to destination.",
    "Explain the water cycle in 4 sentences.",
    "Write about the difference between weather and climate.",
]


def run_cycle(agent, model, batch_size, cycle_num, brain_path):
    """
    One COMPLETE distillation cycle — all 10 layers.
    Each layer gets a time budget, not just one layer per cycle.
    """
    kb = agent.dialog.knowledge
    start_facts = kb.n_facts
    stats = {}

    # Phase 0: Seed topics
    domain, n_seeds = _inject_seeds(kb, cycle_num)
    if n_seeds > 0:
        log.info(f"  Seeded {n_seeds} {domain} entities")

    # Phase 1: Harvest stdlib (only first cycle — deduped after that)
    if cycle_num <= 1:
        from core.harvester import KnowledgeHarvester
        h = KnowledgeHarvester(kb)
        stats['harvest'] = h.harvest_all(verbose=False)
    else:
        stats['harvest'] = 0

    if _shutdown:
        return _finish_cycle(agent, kb, start_facts, stats, brain_path, cycle_num)

    # Phase 2: LLM distillation — ALL layers
    from core.distillation import DistillationEngine
    engine = DistillationEngine(kb, model=model, verify=False, verbose=False)

    if not engine.check_available():
        log.warning(f"Ollama model '{model}' not available — skipping LLM layers")
        return _finish_cycle(agent, kb, start_facts, stats, brain_path, cycle_num)

    # --- Layer 1: Facts (gap-filling with entity-type-aware questions) ---
    min_rels = max(2, 5 - cycle_num // 5)
    engine.run(max_entities=batch_size, min_relations=min_rels, save_brain=False)
    stats['L1_facts'] = engine.reasoning.stats['triplets_stored']
    log.info(f"  L1 Facts: {stats['L1_facts']} stored")
    if _shutdown:
        return _finish_cycle(agent, kb, start_facts, stats, brain_path, cycle_num)

    # --- Layer 2: FLM Corpus — generate text AND train language model ---
    import re as _re
    corpus_path = os.path.join(DATA_DIR, 'flm_corpus.txt')
    start_idx = (cycle_num * 5) % len(FLM_PROMPTS)
    prompts = FLM_PROMPTS[start_idx:start_idx + 5]
    n_texts = 0
    corpus_texts = []
    for prompt in prompts:
        if _shutdown:
            break
        answer = engine.backend.ask(prompt)
        if answer:
            clean = _re.sub(r'<think>.*?</think>', '', answer,
                            flags=_re.DOTALL).strip()
            if clean and len(clean) > 30:
                with open(corpus_path, 'a', encoding='utf-8') as f:
                    f.write(clean + '\n\n')
                corpus_texts.append(clean)
                n_texts += 1
    # TRAIN the FLM on collected corpus — this is what makes FOSS-KI write better
    if corpus_texts:
        flm = _get_or_create_flm(agent)
        if flm is not None:
            for text in corpus_texts:
                flm.train(text)
            log.info(f"  L2 FLM trained on {n_texts} texts ({sum(len(t) for t in corpus_texts)} chars)")
    stats['L2_corpus'] = n_texts
    log.info(f"  L2 Corpus: {n_texts} texts")
    if _shutdown:
        return _finish_cycle(agent, kb, start_facts, stats, brain_path, cycle_num)

    # --- Layer 3: Reformulations (how to say things differently) ---
    templates = engine.generate_reformulations(max_templates=10)
    stats['L3_reformulations'] = len(templates)
    # Store templates in KB AND train FLM on them
    if templates:
        for rel, tmpl in templates:
            kb.store_fact(rel, 'reformulation_template', tmpl)
        # Also train FLM on the template text
        flm = _get_or_create_flm(agent)
        if flm:
            for _, tmpl in templates:
                flm.train(tmpl)
    log.info(f"  L3 Reformulations: {len(templates)}")
    if _shutdown:
        return _finish_cycle(agent, kb, start_facts, stats, brain_path, cycle_num)

    # --- Layer 4: Reasoning Patterns (CoT extraction) ---
    cot_path = os.path.join(DATA_DIR, 'reasoning_solutions.txt')
    n_cot = engine.distill_reasoning_patterns(max_problems=5, output_path=cot_path)
    stats['L4_reasoning'] = n_cot
    # Also train FLM on reasoning solutions
    if n_cot > 0 and os.path.exists(cot_path):
        flm = _get_or_create_flm(agent)
        if flm:
            with open(cot_path, 'r', encoding='utf-8') as f:
                flm.train(f.read())
    log.info(f"  L4 Reasoning: {n_cot} patterns")
    if _shutdown:
        return _finish_cycle(agent, kb, start_facts, stats, brain_path, cycle_num)

    # --- Layer 5: Common Sense (yes/no rules) ---
    entities = kb.all_entities()[:50]
    n_cs = engine.distill_common_sense(entities, max_questions=30)
    stats['L5_common_sense'] = n_cs
    log.info(f"  L5 Common Sense: {n_cs} rules")
    if _shutdown:
        return _finish_cycle(agent, kb, start_facts, stats, brain_path, cycle_num)

    # --- Layer 6: Word Relations ---
    n_wr = engine.distill_word_relations(entities, max_questions=30)
    stats['L6_word_relations'] = n_wr
    log.info(f"  L6 Word Relations: {n_wr}")
    if _shutdown:
        return _finish_cycle(agent, kb, start_facts, stats, brain_path, cycle_num)

    # --- Layer 7: Code Templates ---
    n_code = engine.distill_code_templates(max_tasks=10)
    stats['L7_code'] = n_code
    log.info(f"  L7 Code: {n_code} templates")
    if _shutdown:
        return _finish_cycle(agent, kb, start_facts, stats, brain_path, cycle_num)

    # --- Layer 8: Dialog Patterns ---
    n_dialog = engine.distill_dialog_patterns(max_scenarios=4)
    stats['L8_dialog'] = n_dialog
    log.info(f"  L8 Dialog: {n_dialog} patterns")
    if _shutdown:
        return _finish_cycle(agent, kb, start_facts, stats, brain_path, cycle_num)

    # --- Layer 9: Grammar Rules ---
    n_grammar = engine.distill_grammar_rules(max_checks=6)
    stats['L9_grammar'] = n_grammar
    log.info(f"  L9 Grammar: {n_grammar} rules")
    if _shutdown:
        return _finish_cycle(agent, kb, start_facts, stats, brain_path, cycle_num)

    # --- Layer 10: Taxonomies + Causal Chains ---
    n_tax = engine.distill_taxonomies(entities[:30], max_questions=20)
    n_causal = engine.distill_causal_chains(entities[:30], max_questions=20)
    stats['L10_taxonomy'] = n_tax
    stats['L10_causal'] = n_causal
    log.info(f"  L10 Taxonomy: {n_tax}, Causal: {n_causal}")

    # Self-optimize
    engine.self_optimize()

    return _finish_cycle(agent, kb, start_facts, stats, brain_path, cycle_num)


def _finish_cycle(agent, kb, start_facts, stats, brain_path, cycle_num):
    """Inference pass, save brain (with FLM if trained), log summary."""
    # 3-pass inference (amplify)
    from core.inference import InferenceEngine
    inf = InferenceEngine(kb)
    inf.run(store_inferred=True)
    stats['infer'] = sum(inf.stats.values())

    # Save brain — include FLM if it was trained
    from core.brain import BrainSnapshot
    flm = getattr(agent, '_flm', None)
    snap = BrainSnapshot.capture_from_parts(knowledge_store=kb, lm_model=flm)
    snap.save(brain_path, compressed=True)

    new_total = kb.n_facts
    delta = new_total - start_facts
    size_kb = os.path.getsize(brain_path) / 1024

    # Build layer summary
    layer_str = ' | '.join(f'{k}={v}' for k, v in stats.items()
                           if k.startswith('L') and v > 0)

    log.info(
        f"Cycle {cycle_num}: {start_facts} → {new_total} (+{delta}) | "
        f"brain={size_kb:.0f}KB | {layer_str}"
    )

    return delta


def main():
    parser = argparse.ArgumentParser(
        description='FOSS-KI Complete Capability Distillation Daemon')
    parser.add_argument('--hours', type=float, default=8,
                        help='Duration in hours (0=infinite, default=8)')
    parser.add_argument('--model', type=str, default='qwen3:4b',
                        help='Ollama model for distillation')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Entities per gap-fill cycle')
    parser.add_argument('--cooldown', type=int, default=15,
                        help='Seconds between cycles (default=15)')
    args = parser.parse_args()

    brain_path = os.path.join(DATA_DIR, 'foss-ki.brain')

    # Bootstrap
    from core.agent import Agent
    agent = Agent(workspace_dir=os.path.dirname(os.path.abspath(__file__)))
    n = agent.bootstrap(use_brain=True)
    log.info(f"Bootstrapped: {n} facts | Model: {args.model}")
    log.info(f"Duration: {'infinite' if args.hours == 0 else f'{args.hours}h'}")
    log.info(f"10-Layer Complete Capability Distillation")

    if args.hours > 0:
        end_time = time.time() + args.hours * 3600
    else:
        end_time = float('inf')

    cycle_num = 0
    total_new = 0
    stale_cycles = 0

    while time.time() < end_time and not _shutdown:
        cycle_num += 1
        try:
            delta = run_cycle(agent, args.model, args.batch_size,
                              cycle_num, brain_path)
            total_new += delta

            if delta == 0:
                stale_cycles += 1
                if stale_cycles >= 10:
                    log.info("10 stale cycles — all layers saturated. Stopping.")
                    break
            else:
                stale_cycles = 0

        except Exception as e:
            log.error(f"Cycle {cycle_num} failed: {e}", exc_info=True)
            time.sleep(10)
            continue

        if time.time() < end_time and not _shutdown:
            time.sleep(args.cooldown)

    log.info(f"=== Distillation Complete ===")
    log.info(f"Cycles: {cycle_num} | New facts: {total_new} | "
             f"Final KB: {agent.dialog.knowledge.n_facts}")

    # Corpus stats
    corpus_path = os.path.join(DATA_DIR, 'flm_corpus.txt')
    if os.path.exists(corpus_path):
        size_mb = os.path.getsize(corpus_path) / (1024 * 1024)
        log.info(f"FLM Corpus: {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
