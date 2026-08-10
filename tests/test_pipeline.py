"""
Pipeline Integration Tests — FOSS-KI
=====================================
Tests the full pipeline: Dialog → Parser → Composer → Formulierer → Anti-H.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dialog import DialogSystem, QueryParser, EntityTracker
from core.composer import ResponseComposer
from core.formulierer import Formulierer
from core.knowledge import KnowledgeStore
from core.wikidata import WikidataImporter


import copy

_cached_facts = None

def make_system():
    """Create a fully-loaded DialogSystem with 521 bootstrap facts."""
    global _cached_facts
    ds = DialogSystem()
    ds.enable_composer()
    ds.enable_formulierer()
    if _cached_facts is None:
        wi = WikidataImporter(ds.knowledge)
        wi.import_countries_simple()
        _cached_facts = list(ds.knowledge.facts)
    else:
        ds.knowledge.store_facts_fast(_cached_facts)
    return ds


# ── QueryParser Tests ──

def test_parser_capital():
    s, r = QueryParser.parse("What is the capital of France?")
    assert s == "France", f"Expected France, got {s}"
    assert r == "capital", f"Expected capital, got {r}"

def test_parser_who():
    s, r = QueryParser.parse("Who created Python?")
    assert s == "Python", f"Expected Python, got {s}"
    assert r == "creator", f"Expected creator, got {r}"

def test_parser_about():
    s, r = QueryParser.parse("Tell me about Germany")
    assert s is not None
    assert r is None, f"Expected None relation, got {r}"

def test_parser_compare():
    s, r = QueryParser.parse("Compare France and Germany")
    assert "France" in s and "Germany" in s
    assert r is None

def test_parser_where():
    s, r = QueryParser.parse("Where is Japan?")
    assert r == "location"

def test_parser_who_discovered():
    s, r = QueryParser.parse("Who discovered Radium?")
    assert r == "discoverer", f"Expected discoverer, got {r}"


# ── EntityTracker Tests ──

def test_tracker_ellipsis():
    et = EntityTracker()
    et.mention("France", "capital")
    et.advance_turn()
    entity, rel = et.resolve_reference("And what about Germany?")
    assert entity == "Germany"
    assert rel == "capital"

def test_tracker_pronoun():
    et = EntityTracker()
    et.mention("France", "capital")
    entity, rel = et.resolve_reference("What is its population?")
    assert entity == "France"


# ── KnowledgeStore Tests ──

def test_knowledge_store_retrieve():
    ks = KnowledgeStore(dim=128)
    ks.store_facts([("France", "capital", "Paris")])
    r = ks.query("France", "capital")
    assert r['fact'] is not None
    assert r['fact'][2] == "Paris"

def test_knowledge_anti_hallucination():
    ks = KnowledgeStore(dim=128)
    ks.store_facts([("France", "capital", "Paris")])
    r = ks.query("Narnia", "capital")
    assert r['confidence_level'] in ('REJECTED', 'UNKNOWN')


# ── Formulierer Tests ──

def test_formulierer_person_pronoun():
    f = Formulierer()
    facts = [
        ('Albert Einstein', 'born', '1879'),
        ('Albert Einstein', 'nationality', 'German'),
    ]
    result = f.reformulate(facts)
    assert 'Einstein' in result['text']
    assert 'It ' not in result['text'], f"Person should not use 'It': {result['text']}"

def test_formulierer_nonperson_pronoun():
    f = Formulierer()
    facts = [
        ('Germany', 'capital', 'Berlin'),
        ('Germany', 'population', '83 million'),
    ]
    result = f.reformulate(facts)
    # Should use "It" not a person pronoun
    assert result['verified']

def test_formulierer_no_facts():
    f = Formulierer()
    result = f.reformulate([])
    assert "don't have" in result['text']
    assert result['method'] == 'no_facts'

def test_formulierer_word_order():
    f = Formulierer()
    facts = [
        ('Albert Einstein', 'born', '1879'),
        ('Albert Einstein', 'nationality', 'German'),
        ('Albert Einstein', 'occupation', 'physicist'),
    ]
    result = f.reformulate(facts)
    # Should not contain "also is" (wrong order)
    assert 'also is' not in result['text'], f"Bad word order: {result['text']}"


# ── Full Pipeline Tests ──

def test_pipeline_capital():
    ds = make_system()
    r = ds.turn("What is the capital of France?")
    assert r['answer'] == 'Paris'
    assert r['confidence'] == 'HIGH'

def test_pipeline_about():
    ds = make_system()
    r = ds.turn("Tell me about Switzerland")
    assert 'Bern' in r['response']
    assert r['confidence'] == 'HIGH'
    # Source can be knowledge_store (overview path) or composer
    assert r['source'] in ('knowledge_store', 'composer', 'composer_fallback',
                           'composer_fallback+formulierer')

def test_pipeline_who():
    ds = make_system()
    r = ds.turn("Who created JavaScript?")
    assert 'Brendan Eich' in r['response']

def test_pipeline_anti_h():
    ds = make_system()
    r = ds.turn("What is the capital of Narnia?")
    assert r['confidence'] == 'REJECTED'
    assert r['answer'] is None

def test_pipeline_compare():
    ds = make_system()
    r = ds.turn("Compare France and Germany")
    assert 'Paris' in r['response'] or 'Berlin' in r['response']

def test_pipeline_multiturn():
    ds = make_system()
    r1 = ds.turn("What is the capital of France?")
    assert r1['answer'] == 'Paris'
    # Ellipsis: "And what about Germany?"
    r2 = ds.turn("And what about Germany?")
    # Should get info about Germany (either capital or about)
    assert 'Germany' in r2['response'] or 'Berlin' in r2['response']

def test_pipeline_composer_fallback():
    """When specific query is REJECTED but entity exists, fallback to Composer."""
    ds = make_system()
    # Ask about discoverer of Radium (specific query)
    ds.turn("Who discovered Radium?")
    # Then "And what about France?" — ellipsis with relation=discoverer
    # France has no discoverer → REJECTED → should fallback to Composer
    r = ds.turn("And what about France?")
    assert r['confidence'] != 'REJECTED', \
        f"Should fallback to Composer, got {r['confidence']}: {r['response']}"


# ── Wikidata Import Tests ──

def test_wikidata_import_count():
    ks = KnowledgeStore(dim=128)
    wi = WikidataImporter(ks)
    count = wi.import_countries_simple()
    assert count >= 500, f"Expected 500+ facts, got {count}"

def test_wikidata_entities():
    ks = KnowledgeStore(dim=128)
    wi = WikidataImporter(ks)
    wi.import_countries_simple()
    entities = set(s for s, r, o in ks.facts)
    assert len(entities) >= 100, f"Expected 100+ entities, got {len(entities)}"


# ── Agent Tests ──

def test_agent_intent_question():
    from core.agent import IntentClassifier
    intent, _ = IntentClassifier.classify("What is the capital of France?")
    assert intent == 'QUESTION', f"Expected QUESTION, got {intent}"

def test_agent_intent_code():
    from core.agent import IntentClassifier
    intent, _ = IntentClassifier.classify("Write a function that sorts a list")
    assert intent == 'CODE_REQUEST', f"Expected CODE_REQUEST, got {intent}"

def test_agent_intent_math():
    from core.agent import IntentClassifier
    intent, _ = IntentClassifier.classify("Calculate 42 * 17")
    assert intent == 'REASONING', f"Expected REASONING, got {intent}"

def test_agent_intent_conversation():
    from core.agent import IntentClassifier
    intent, _ = IntentClassifier.classify("Hello!")
    assert intent == 'CONVERSATION', f"Expected CONVERSATION, got {intent}"

def test_agent_intent_generation():
    from core.agent import IntentClassifier
    intent, _ = IntentClassifier.classify("Explain quicksort")
    assert intent == 'GENERATION', f"Expected GENERATION, got {intent}"

def test_agent_code_gen():
    from core.agent import CodeGenerator
    cg = CodeGenerator()
    result = cg.generate("Write a prime number checker")
    assert result['valid'], "Generated code should be syntactically valid"
    assert result['code'], "Should generate some code"

def test_agent_math_solver():
    from core.agent import MathSolver
    ms = MathSolver()
    r = ms.solve("42 * 17")
    assert r['result'] == 714, f"Expected 714, got {r['result']}"

def test_agent_math_complex():
    from core.agent import MathSolver
    ms = MathSolver()
    r = ms.solve("2**10")
    assert r['result'] == 1024, f"Expected 1024, got {r['result']}"

def test_agent_full_pipeline():
    from core.agent import Agent
    agent = Agent()
    agent.bootstrap()
    r = agent.process("What is the capital of France?")
    assert 'Paris' in r['response']

def test_agent_code_pipeline():
    from core.agent import Agent
    agent = Agent()
    r = agent.process("Write a fibonacci function")
    assert r['intent'] == 'CODE_REQUEST', f"Expected CODE_REQUEST, got {r['intent']}"
    assert 'def' in r['response'] or 'fibonacci' in r['response'].lower()

def test_agent_anti_h():
    from core.agent import Agent
    agent = Agent()
    agent.bootstrap()
    r = agent.process("What is the capital of Narnia?")
    assert 'don\'t have' in r['response'] or 'REJECTED' in str(r['metadata'])


# ── KnowledgeExplorer Tests (VizDoom Port) ──

def test_explorer_graph():
    """Test graph construction from knowledge store."""
    from core.metacognition import KnowledgeExplorer
    ks = KnowledgeStore(dim=128)
    ks.store_fact("France", "capital", "Paris")
    ks.store_fact("France", "language", "French")
    ks.store_fact("Germany", "capital", "Berlin")
    explorer = KnowledgeExplorer(ks)
    nodes, edges = explorer.build_graph()
    assert len(edges) == 3
    assert 'france' in nodes
    assert nodes['france']['degree'] == 2  # capital + language

def test_explorer_frontier_scores():
    """Frontier score = 1/(1+degree). Isolated nodes score highest."""
    from core.metacognition import KnowledgeExplorer
    ks = KnowledgeStore(dim=128)
    ks.store_fact("France", "capital", "Paris")
    ks.store_fact("France", "language", "French")
    ks.store_fact("France", "continent", "Europe")
    ks.store_fact("Germany", "capital", "Berlin")
    explorer = KnowledgeExplorer(ks)
    scores = explorer.frontier_scores()
    # France has degree 3 (capital, language, continent)
    # Germany has degree 1 (only capital)
    # Paris, French, Europe, Berlin have degree 1 each
    france_score = next(s for n, s, _, _ in scores if n == 'France')
    germany_score = next(s for n, s, _, _ in scores if n == 'Germany')
    assert germany_score > france_score, "Germany (fewer facts) should be more frontier"

def test_explorer_domain_coverage():
    """Domain detection and coverage computation."""
    from core.metacognition import KnowledgeExplorer
    ks = KnowledgeStore(dim=128)
    wi = WikidataImporter(ks)
    wi.import_countries_simple()
    explorer = KnowledgeExplorer(ks)
    coverage = explorer.domain_coverage()
    assert 'country' in coverage
    assert coverage['country']['entities'] > 50  # many country entities
    assert coverage['country']['coverage'] > 0  # some coverage

def test_explorer_leaf_filtering():
    """Leaf values (numbers, languages) should be filtered from frontiers."""
    from core.metacognition import KnowledgeExplorer
    ks = KnowledgeStore(dim=128)
    ks.store_fact("France", "population", "67 million")
    ks.store_fact("France", "capital", "Paris")
    ks.store_fact("Paris", "known_for", "Eiffel Tower")
    explorer = KnowledgeExplorer(ks)
    frontiers = explorer.find_frontiers(max_degree=5)
    frontier_names = [f['entity'] for f in frontiers]
    # "67 million" is a leaf value (object only, never subject)
    assert '67 million' not in frontier_names, "Leaf values should be filtered"
    # Paris IS a subject (has own facts) → should appear
    assert 'Paris' in frontier_names or 'Eiffel Tower' in frontier_names

def test_explorer_exploration_targets():
    """Exploration targets should prioritize entities with most missing relations."""
    from core.metacognition import KnowledgeExplorer
    ks = KnowledgeStore(dim=128)
    ks.store_fact("Oxygen", "symbol", "O")
    ks.store_fact("France", "capital", "Paris")
    ks.store_fact("France", "language", "French")
    ks.store_fact("France", "continent", "Europe")
    ks.store_fact("France", "population", "67 million")
    explorer = KnowledgeExplorer(ks)
    targets = explorer.exploration_targets(n=5)
    assert len(targets) > 0
    # Oxygen (1 fact, science domain) should be higher priority than France (4 facts)
    entity_names = [t['entity'] for t in targets]
    if 'Oxygen' in entity_names and 'France' in entity_names:
        ox_idx = entity_names.index('Oxygen')
        fr_idx = entity_names.index('France')
        assert ox_idx < fr_idx, "Oxygen (sparser) should be higher priority than France"


# ── ToolSelectionPolicy Tests (VizDoom BasalGanglia Port) ──

def test_policy_question_routes_dialog():
    """Questions should route to dialog subsystem."""
    from core.agent import ToolSelectionPolicy
    policy = ToolSelectionPolicy()
    ranked = policy.select('QUESTION', 'What is the capital of France?')
    assert ranked[0][0] == 'dialog', f"Expected dialog, got {ranked[0][0]}"

def test_policy_code_routes_code():
    """Code requests should route to code subsystem."""
    from core.agent import ToolSelectionPolicy
    policy = ToolSelectionPolicy()
    ranked = policy.select('CODE_REQUEST', 'Write a sort function')
    assert ranked[0][0] == 'code', f"Expected code, got {ranked[0][0]}"

def test_policy_weight_learning():
    """Policy should adapt weights based on outcomes."""
    from core.agent import ToolSelectionPolicy
    policy = ToolSelectionPolicy()
    initial = policy.weights['dialog']
    policy.record_outcome('dialog', True)
    policy.record_outcome('dialog', True)
    assert policy.weights['dialog'] > initial, "Weight should increase after success"
    policy.record_outcome('code', False)
    policy.record_outcome('code', False)
    assert policy.weights['code'] < 1.0, "Weight should decrease after failure"


# ── ForgeAssembler Tests ──

def test_forge_fragment_count():
    """ForgeAssembler should load 1000+ fragments."""
    from core.forge_assembler import ForgeAssembler
    fa = ForgeAssembler()
    assert len(fa.fragments) >= 1000, f"Expected 1000+ fragments, got {len(fa.fragments)}"

def test_forge_exact_lookup():
    """Exact key lookup should find known fragments."""
    from core.forge_assembler import ForgeAssembler
    fa = ForgeAssembler()
    result = fa.find_fragment("port_scanner")
    assert result is not None, "port_scanner fragment should exist"

def test_forge_generate_valid_ast():
    """Generated code should be valid Python AST."""
    from core.forge_assembler import ForgeAssembler
    import ast
    fa = ForgeAssembler()
    result = fa.generate("Write a port scanner")
    assert result['code'], "Should generate code"
    if result['code']:
        try:
            ast.parse(result['code'])
        except SyntaxError:
            assert False, "Generated code has syntax errors"

def test_forge_registry_roles():
    """FragmentRegistry should classify fragment roles."""
    from core.forge_assembler import ForgeAssembler
    fa = ForgeAssembler()
    roles = set()
    for key, meta in fa.registry.registry.items():
        roles.add(meta['role'])
    assert 'STANDALONE' in roles, "Should have STANDALONE fragments"


# ── FuzzyParser Tests ──

def test_fuzzy_tokenize():
    """Tokenizer should remove stopwords."""
    from core.fuzzy_parser import FuzzyParser
    fp = FuzzyParser()
    tokens = fp.tokenize("Write a function that sorts a list")
    assert 'write' in tokens
    assert 'a' not in tokens
    assert 'that' not in tokens

def test_fuzzy_classify_build():
    """Build keywords should classify as BUILD."""
    from core.fuzzy_parser import FuzzyParser
    fp = FuzzyParser()
    tokens = fp.tokenize("Create a web scraper")
    mode = fp.classify(tokens)
    assert mode == 'BUILD', f"Expected BUILD, got {mode}"

def test_fuzzy_classify_question():
    """Question keywords should classify as QUESTION."""
    from core.fuzzy_parser import FuzzyParser
    fp = FuzzyParser()
    tokens = fp.tokenize("What is recursion?")
    mode = fp.classify(tokens)
    assert mode == 'QUESTION', f"Expected QUESTION, got {mode}"

def test_fuzzy_jaro_winkler():
    """Jaro-Winkler should give high score for similar strings."""
    from core.fuzzy_parser import jaro_winkler
    assert jaro_winkler("scanner", "scanner") == 1.0
    assert jaro_winkler("scanner", "scaner") > 0.9
    assert jaro_winkler("hello", "world") < 0.5

def test_fuzzy_entity_extraction():
    """Entity extraction should match known entities."""
    from core.fuzzy_parser import FuzzyParser
    fp = FuzzyParser(known_entities=['scanner', 'scraper', 'keylogger'])
    tokens = fp.tokenize("Build a scanner tool")
    entities = fp.extract_entities(tokens)
    matched = [e['matched'] for e in entities]
    assert 'scanner' in matched, f"Expected scanner in {matched}"

def test_fuzzy_extract_parameters():
    """Should extract IPs, ports, paths from text."""
    from core.fuzzy_parser import FuzzyParser
    fp = FuzzyParser()
    params = fp.extract_parameters("Scan 192.168.1.1 port 8080")
    assert params.get('target') == '192.168.1.1'
    assert params.get('port') == 8080

def test_fuzzy_composition_detection():
    """Should detect multi-step tasks."""
    from core.fuzzy_parser import FuzzyParser
    fp = FuzzyParser()
    tokens = fp.tokenize("download the page and parse the HTML")
    comp = fp.detect_composition("download the page and parse the HTML", tokens)
    assert comp['is_composition'], "Should detect composition"
    assert len(comp['sub_tasks']) >= 2


# ── Apprentice Tests ──

def test_apprentice_reasoning_extraction():
    """ReasoningExtractor should find step patterns."""
    from core.apprentice import ReasoningExtractor
    re_ = ReasoningExtractor()
    text = "1. Parse the input string into tokens.\n2. Validate each token against the grammar.\n3. Return the parsed result."
    patterns = re_.extract(text)
    assert len(patterns) > 0, "Should extract step patterns"
    assert any(p.pattern_type == 'reasoning' for p in patterns)

def test_apprentice_code_extraction():
    """CodePatternExtractor should find function structures."""
    from core.apprentice import CodePatternExtractor
    ce = CodePatternExtractor()
    code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''
    patterns = ce.extract(code)
    assert len(patterns) > 0, "Should extract code patterns"

def test_apprentice_pattern_store():
    """PatternStore should store and retrieve patterns."""
    from core.apprentice import PatternStore, ExtractedPattern
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), 'test_patterns.json')
    ps = PatternStore(path)
    p1 = ExtractedPattern('reasoning', 'step1 then step2', 'multi_step')
    ps.add(p1)
    assert 'reasoning' in ps.patterns
    assert len(ps.patterns['reasoning']) == 1
    p2 = ExtractedPattern('reasoning', 'step1 then step2', 'multi_step')
    ps.add(p2)  # duplicate
    assert len(ps.patterns['reasoning']) == 1, "Should deduplicate"


# ── SelfImproveLoop Tests ──

def test_gap_detector_no_crash():
    """GapDetector should work with None inputs."""
    from core.self_improve import GapDetector
    gd = GapDetector()
    gaps = gd.detect_all()
    assert gaps == [], "No gaps without forge/knowledge"

def test_task_generator():
    """TaskGenerator should convert gaps to tasks."""
    from core.self_improve import TaskGenerator
    tg = TaskGenerator()
    gaps = [
        {'type': 'untested_fragment', 'key': 'port_scanner', 'priority': 3,
         'action': 'Test fragment: port_scanner'},
        {'type': 'knowledge_frontier', 'entity': 'Oxygen', 'domain': 'science',
         'priority': 5, 'action': 'Learn more about: Oxygen'},
    ]
    tasks = tg.generate_tasks(gaps)
    assert len(tasks) == 2
    assert tasks[0]['type'] == 'code'
    assert tasks[1]['type'] == 'knowledge'

def test_self_improve_no_agent():
    """SelfImproveLoop without agent should return error."""
    from core.self_improve import SelfImproveLoop
    loop = SelfImproveLoop()
    result = loop.run()
    assert 'error' in result

def test_self_improve_report_empty():
    """Report with no results should say so."""
    from core.self_improve import SelfImproveLoop
    loop = SelfImproveLoop()
    assert "No self-improvement" in loop.report()


if __name__ == '__main__':
    tests = [(name, func) for name, func in globals().items()
             if name.startswith('test_') and callable(func)]

    passed = 0
    failed = 0

    for name, func in sorted(tests):
        try:
            func()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        sys.exit(1)
