"""
Metacognition — Self-Improving Feedback Loops
===============================================
The system that makes FOSS-KI get smarter WITHOUT human intervention.

Three cycles:
  Cycle 1 (Gap-Fill): Query rejected → find raw text → re-extract → answer
  Cycle 2 (Self-Test): Run benchmarks → diff with last run → classify regressions
  Cycle 3 (Self-Extend): Analyze gap logs → find systematic patterns → propose new regex

Every component used here ALREADY EXISTS in the system.
This module just WIRES THEM TOGETHER into feedback loops.
"""

import json
import os
import re
import time
from collections import Counter


class GapLog:
    """
    Persistent log of knowledge gaps discovered during operation.

    Every REJECTED query is a gap. The log classifies each gap:
    - EXTRACTION_FAILURE: entity exists in corpus, extractor missed it
    - KNOWLEDGE_GAP: entity not in corpus at all
    - ENCODING_FAILURE: fact exists but similarity too low (threshold issue)

    Persists to JSON for cross-session learning.
    """

    def __init__(self, log_path=None):
        if log_path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_path = os.path.join(base, 'data', 'gap_log.json')
        self.log_path = log_path
        self.gaps = self._load()

    def _load(self):
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def save(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self.gaps, f, ensure_ascii=False, indent=2)

    def log_gap(self, query_subject, query_relation, gap_type,
                raw_text=None, details=None):
        """Record a knowledge gap."""
        entry = {
            'subject': query_subject,
            'relation': query_relation,
            'type': gap_type,  # EXTRACTION_FAILURE, KNOWLEDGE_GAP, ENCODING_FAILURE
            'timestamp': time.time(),
            'resolved': False,
            'details': details or '',
        }
        if raw_text:
            # Store first 500 chars of relevant text for re-extraction
            entry['raw_text'] = raw_text[:500]
        self.gaps.append(entry)

    def mark_resolved(self, subject, relation=None):
        """Mark gaps as resolved when they get filled."""
        for gap in self.gaps:
            if gap['subject'].lower() == subject.lower():
                if relation is None or gap.get('relation') == relation:
                    gap['resolved'] = True

    def unresolved(self):
        """Get all unresolved gaps."""
        return [g for g in self.gaps if not g['resolved']]

    def by_frequency(self):
        """Rank gaps by how often they were queried."""
        counts = Counter()
        for gap in self.gaps:
            if not gap['resolved']:
                key = (gap['subject'], gap.get('relation', ''))
                counts[key] += 1
        return counts.most_common()


class GapFiller:
    """
    Cycle 1: Automatic gap filling.

    When a query is REJECTED:
    1. Check if the entity exists somewhere in stored corpus text
    2. If yes → extraction failure → try alternative extraction strategies
    3. If no → knowledge gap → log for future text ingestion

    Uses existing components:
    - KnowledgeStore.query() for detection (REJECTED = gap)
    - TripletExtractor for re-extraction
    - Corpus store for raw text lookup
    """

    def __init__(self, knowledge_store, extractor=None, corpus=None):
        self.knowledge = knowledge_store
        self.gap_log = GapLog()
        self._corpus = corpus or {}  # entity → raw text paragraphs

        if extractor is None:
            from .extractor import TripletExtractor
            extractor = TripletExtractor()
        self.extractor = extractor

    def add_corpus(self, entity, text):
        """Store raw text associated with an entity for future re-extraction."""
        entity_lower = entity.lower()
        if entity_lower not in self._corpus:
            self._corpus[entity_lower] = []
        self._corpus[entity_lower].append(text)

    def add_corpus_bulk(self, text):
        """
        Store raw text and auto-index by detected entities.
        Splits text into paragraphs and indexes each by its main entity.
        """
        paragraphs = re.split(r'\n\s*\n|\n#\s+', text)
        for para in paragraphs:
            para = para.strip()
            if len(para) < 20:
                continue
            # Find first named entity as index key
            m = re.match(r'^#?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', para)
            if m:
                entity = m.group(1)
                self.add_corpus(entity, para)

    def check_and_fill(self, subject, relation=None):
        """
        Attempt to fill a knowledge gap automatically.

        Returns:
            dict with:
                filled: bool
                gap_type: str
                new_facts: list of (S, R, O) added
                details: str
        """
        result = {
            'filled': False,
            'gap_type': None,
            'new_facts': [],
            'details': '',
        }

        # Step 1: Check if we already know this
        query_result = self.knowledge.query(subject, relation)
        if query_result['confidence_level'] in ('HIGH', 'MEDIUM'):
            result['details'] = 'Already known'
            return result

        # Step 2: Check corpus for raw text about this entity
        subject_lower = subject.lower()
        raw_texts = self._corpus.get(subject_lower, [])

        if not raw_texts:
            # Also try title case and partial matches
            for key in self._corpus:
                if subject_lower in key or key in subject_lower:
                    raw_texts.extend(self._corpus[key])

        if not raw_texts:
            # Knowledge gap: entity not in corpus
            result['gap_type'] = 'KNOWLEDGE_GAP'
            result['details'] = f"No corpus text found for '{subject}'"
            self.gap_log.log_gap(subject, relation, 'KNOWLEDGE_GAP')
            self.gap_log.save()
            return result

        # Step 3: Extraction failure — try re-extraction with different strategies
        result['gap_type'] = 'EXTRACTION_FAILURE'
        new_facts = []

        for text in raw_texts:
            # Strategy 1: Standard extraction
            triplets = self.extractor.extract_from_text(text)

            # Strategy 2: Extraction with explicit context subject
            triplets_ctx = self.extractor.extract_from_text(
                f"# {subject}\n{text}"
            )

            # Strategy 3: Sentence-level with forced subject
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for sent in sentences:
                triplets_sent = self.extractor.extract_from_sentence(
                    sent, context_subject=subject
                )
                triplets.extend(triplets_sent)

            triplets.extend(triplets_ctx)

            # Deduplicate and filter relevant
            seen = set()
            existing = {(s.lower(), r, o.lower()) for s, r, o in self.knowledge.facts}
            for s, r, o in triplets:
                key = (s.lower(), r, o.lower())
                if key not in seen and key not in existing:
                    # Only add facts relevant to the queried entity
                    if s.lower() == subject_lower or o.lower() == subject_lower:
                        self.knowledge.store_fact(s, r, o)
                        new_facts.append((s, r, o))
                        seen.add(key)

        if new_facts:
            result['filled'] = True
            result['new_facts'] = new_facts
            result['details'] = f"Extracted {len(new_facts)} new facts via re-extraction"
            self.gap_log.mark_resolved(subject, relation)
        else:
            result['details'] = f"Corpus text found but extraction yielded no new facts"
            self.gap_log.log_gap(subject, relation, 'EXTRACTION_FAILURE',
                                raw_text=raw_texts[0] if raw_texts else None)

        self.gap_log.save()
        return result

    def fill_query(self, subject, relation=None):
        """
        Query with automatic gap-filling.

        If the first query fails, try to fill the gap and re-query.
        This is the main entry point for self-improving queries.

        Returns same format as KnowledgeStore.query() but with extra fields:
            gap_filled: bool (whether gap-filling was attempted)
            new_facts: list of facts added during gap-filling
        """
        # First attempt
        result = self.knowledge.query(subject, relation)
        result['gap_filled'] = False
        result['new_facts'] = []

        if result['confidence_level'] in ('HIGH', 'MEDIUM'):
            return result

        # Attempt gap-fill
        fill_result = self.check_and_fill(subject, relation)

        if fill_result['filled']:
            # Re-query after filling
            result = self.knowledge.query(subject, relation)
            result['gap_filled'] = True
            result['new_facts'] = fill_result['new_facts']

            # Also try query_smart for synonym/reverse coverage
            if result['confidence_level'] not in ('HIGH', 'MEDIUM'):
                smart = self.knowledge.query_smart(subject, relation)
                smart['gap_filled'] = True
                smart['new_facts'] = fill_result['new_facts']
                return smart
        else:
            result['gap_filled'] = False
            result['gap_type'] = fill_result['gap_type']

        return result

    def gap_report(self):
        """Generate a report of current knowledge gaps."""
        unresolved = self.gap_log.unresolved()
        freq = self.gap_log.by_frequency()

        extraction_fails = [g for g in unresolved if g['type'] == 'EXTRACTION_FAILURE']
        knowledge_gaps = [g for g in unresolved if g['type'] == 'KNOWLEDGE_GAP']

        lines = [
            f"Gap Report: {len(unresolved)} unresolved gaps",
            f"  Extraction failures: {len(extraction_fails)}",
            f"  Knowledge gaps: {len(knowledge_gaps)}",
            "",
            "Top gaps by frequency:",
        ]
        for (subj, rel), count in freq[:10]:
            lines.append(f"  {count}× ({subj}, {rel or '*'})")

        return "\n".join(lines)


class SelfTester:
    """
    Cycle 2: Automated self-testing.

    Runs benchmark queries, compares with last run, classifies changes.
    Stores results as JSON checkpoints for regression tracking.

    Uses existing components:
    - KnowledgeStore.query() / query_smart()
    - Benchmark query sets
    """

    def __init__(self, knowledge_store, checkpoint_dir=None):
        self.knowledge = knowledge_store
        if checkpoint_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            checkpoint_dir = os.path.join(base, 'data', 'selftest_checkpoints')
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def run(self, test_queries):
        """
        Run a set of test queries and record results.

        Args:
            test_queries: list of dicts with keys:
                subject: str
                relation: str (optional)
                expected: str (expected answer)

        Returns:
            dict with:
                total: int
                correct: int
                accuracy: float
                results: list of per-query results
                regressions: list of queries that got worse since last run
                improvements: list of queries that got better
        """
        results = []
        correct = 0

        for tq in test_queries:
            subject = tq['subject']
            relation = tq.get('relation')
            expected = tq.get('expected', '').lower()

            # Query with smart fallback
            qr = self.knowledge.query_smart(subject, relation)

            got_answer = False
            answer = None
            if qr['fact']:
                answer = qr['fact'][2]
                if answer.lower() == expected:
                    got_answer = True
                    correct += 1

            results.append({
                'subject': subject,
                'relation': relation,
                'expected': expected,
                'got': answer,
                'correct': got_answer,
                'confidence': qr['confidence_level'],
            })

        # Save checkpoint
        checkpoint = {
            'timestamp': time.time(),
            'total': len(test_queries),
            'correct': correct,
            'accuracy': correct / max(len(test_queries), 1),
            'results': results,
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"selftest_{int(time.time())}.json"
        )
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

        # Compare with last checkpoint
        regressions, improvements = self._diff_with_last(results)

        return {
            'total': len(test_queries),
            'correct': correct,
            'accuracy': correct / max(len(test_queries), 1),
            'results': results,
            'regressions': regressions,
            'improvements': improvements,
            'checkpoint': checkpoint_path,
        }

    def _diff_with_last(self, current_results):
        """Compare current results with the most recent checkpoint."""
        checkpoints = sorted([
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith('selftest_') and f.endswith('.json')
        ])

        # Need at least 2 checkpoints (current one was just saved)
        if len(checkpoints) < 2:
            return [], []

        last_path = os.path.join(self.checkpoint_dir, checkpoints[-2])
        with open(last_path, 'r', encoding='utf-8') as f:
            last = json.load(f)

        # Build lookup from last results
        last_by_key = {}
        for r in last.get('results', []):
            key = (r['subject'], r.get('relation', ''))
            last_by_key[key] = r

        regressions = []
        improvements = []

        for r in current_results:
            key = (r['subject'], r.get('relation', ''))
            if key in last_by_key:
                prev = last_by_key[key]
                if prev['correct'] and not r['correct']:
                    regressions.append({
                        'subject': r['subject'],
                        'relation': r.get('relation'),
                        'was': prev['got'],
                        'now': r['got'],
                        'expected': r['expected'],
                    })
                elif not prev['correct'] and r['correct']:
                    improvements.append({
                        'subject': r['subject'],
                        'relation': r.get('relation'),
                        'was': prev['got'],
                        'now': r['got'],
                    })

        return regressions, improvements

    def history(self):
        """Get accuracy history across all checkpoints."""
        checkpoints = sorted([
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith('selftest_') and f.endswith('.json')
        ])

        history = []
        for cp_file in checkpoints:
            cp_path = os.path.join(self.checkpoint_dir, cp_file)
            with open(cp_path, 'r', encoding='utf-8') as f:
                cp = json.load(f)
            history.append({
                'timestamp': cp['timestamp'],
                'accuracy': cp['accuracy'],
                'total': cp['total'],
                'correct': cp['correct'],
            })

        return history


class GapPrioritizer:
    """
    Cycle 3 (partial): Rank knowledge gaps by impact.

    Analyzes gap logs to find systematic patterns:
    - Which relations are most commonly missing?
    - Which entities are most queried but unknown?
    - Which extraction patterns fail most often?

    Produces actionable reports for self-extension or human review.
    """

    def __init__(self, gap_log=None):
        if gap_log is None:
            gap_log = GapLog()
        self.gap_log = gap_log

    def prioritize(self):
        """
        Analyze gaps and return prioritized action items.

        Returns:
            list of dicts sorted by priority:
                category: str (e.g., 'population', 'capital')
                count: int (number of queries)
                subjects: list of str
                action: str (suggested fix)
        """
        # Group by relation
        rel_groups = {}
        entity_groups = {}

        for gap in self.gap_log.unresolved():
            rel = gap.get('relation') or 'unknown'
            subj = gap['subject']

            if rel not in rel_groups:
                rel_groups[rel] = []
            rel_groups[rel].append(subj)

            if subj not in entity_groups:
                entity_groups[subj] = []
            entity_groups[subj].append(rel)

        # Build priority list
        priorities = []

        # Relations with most gaps
        for rel, subjects in sorted(rel_groups.items(),
                                     key=lambda x: len(x[1]), reverse=True):
            unique_subjects = list(set(subjects))
            if rel == 'unknown':
                action = "Need structured text about these entities"
            else:
                action = f"Need '{rel}' facts — feed text with {rel} information"

            priorities.append({
                'category': rel,
                'count': len(subjects),
                'subjects': unique_subjects[:10],
                'type': 'relation_gap',
                'action': action,
            })

        return priorities

    def suggest_text_sources(self):
        """
        Based on gap patterns, suggest what kind of text to ingest.

        Returns human-readable suggestions.
        """
        priorities = self.prioritize()
        if not priorities:
            return "No gaps detected. System knowledge appears complete for current queries."

        lines = ["Knowledge Gap Report — Suggested Actions:", ""]

        for i, p in enumerate(priorities[:5], 1):
            lines.append(f"{i}. {p['count']} queries about '{p['category']}'")
            lines.append(f"   Entities: {', '.join(p['subjects'][:5])}")
            lines.append(f"   Action: {p['action']}")
            lines.append("")

        total = sum(p['count'] for p in priorities)
        lines.append(f"Total unresolved: {total} gaps across {len(priorities)} categories")

        return "\n".join(lines)


class PatternLearner:
    """
    Cycle 3: Learn new extraction patterns from failures.

    Analyzes sentences where extraction failed despite containing
    the target entity. Finds common sentence structures and
    proposes new regex patterns.

    Pipeline:
    1. Collect failed sentences (entity in text, no triplet extracted)
    2. Normalize: replace entity with {ENTITY}, capitalize to {CAP}
    3. Cluster by structure (bag of function words + position)
    4. Generate regex from cluster prototype
    5. Validate: pattern must match >N known cases, 0 false positives
    """

    # Sentence templates we know about (to avoid rediscovering them)
    KNOWN_STRUCTURES = {
        'X is the R of Y',
        'Y is the R of X',
        'X is a/an Y',
        'X was born in Y',
        'X was founded in Y',
        'Its R is Y',
    }

    # Function words to keep in structural templates
    FUNCTION_WORDS = {
        'is', 'are', 'was', 'were', 'has', 'have', 'had',
        'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for',
        'by', 'with', 'from', 'and', 'or', 'but', 'its', 'their',
        'as', 'also', 'known', 'called', 'named', 'located',
        'which', 'that', 'who', 'whose', 'where', 'when',
    }

    def __init__(self, extractor=None):
        if extractor is None:
            from .extractor import TripletExtractor
            extractor = TripletExtractor()
        self.extractor = extractor
        self.failed_sentences = []  # (sentence, entity, relation)
        self.learned_patterns = []  # validated regex patterns

    def collect_failure(self, sentence, entity, relation=None):
        """Record a sentence where extraction failed for a known entity."""
        self.failed_sentences.append({
            'sentence': sentence,
            'entity': entity,
            'relation': relation,
        })

    def analyze_failures(self, corpus_texts, knowledge_store):
        """
        Scan corpus for sentences containing known entities where
        extraction produces no relevant triplets.

        This is the main discovery method — finds what the extractor misses.
        """
        # Get all known entities
        entities = set()
        for s, r, o in knowledge_store.facts:
            entities.add(s.lower())
            entities.add(o.lower())

        for text in corpus_texts:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 15:
                    continue

                # Check if sentence contains a known entity
                sent_lower = sent.lower()
                for entity in entities:
                    if entity in sent_lower and len(entity) > 2:
                        # Try extraction
                        triplets = self.extractor.extract_from_sentence(sent)
                        # Check if any triplet involves this entity
                        relevant = [t for t in triplets
                                    if entity in t[0].lower() or entity in t[2].lower()]
                        if not relevant:
                            # Failed extraction — record it
                            self.collect_failure(sent, entity)

    def _normalize_sentence(self, sentence, entity):
        """
        Convert sentence to structural template.

        "Berlin is the capital of Germany" with entity="germany"
        becomes "{CAP} is the {WORD} of {ENTITY}"
        """
        sent = sentence
        # Replace the entity (case-insensitive) with placeholder
        pattern = re.compile(re.escape(entity), re.IGNORECASE)
        sent = pattern.sub('{ENTITY}', sent)

        # Replace other capitalized words with {CAP}
        sent = re.sub(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', '{CAP}', sent)

        # Replace numbers with {NUM}
        sent = re.sub(r'\b\d[\d,.\s]*\b', '{NUM}', sent)

        # Simplify: collapse multiple spaces
        sent = re.sub(r'\s+', ' ', sent).strip()

        return sent

    def find_patterns(self, min_count=3):
        """
        Cluster failed sentences by structure and propose new patterns.

        Returns list of dicts:
            template: str (normalized sentence structure)
            count: int (how many sentences match)
            examples: list of original sentences
            proposed_regex: str (suggested extraction regex)
        """
        if not self.failed_sentences:
            return []

        # Normalize all failed sentences
        templates = Counter()
        template_examples = {}

        for fail in self.failed_sentences:
            tmpl = self._normalize_sentence(fail['sentence'], fail['entity'])
            templates[tmpl] += 1
            if tmpl not in template_examples:
                template_examples[tmpl] = []
            if len(template_examples[tmpl]) < 5:
                template_examples[tmpl].append(fail['sentence'])

        # Filter to templates with enough instances
        results = []
        for tmpl, count in templates.most_common():
            if count < min_count:
                continue

            proposed = self._template_to_regex(tmpl)
            results.append({
                'template': tmpl,
                'count': count,
                'examples': template_examples[tmpl],
                'proposed_regex': proposed,
            })

        return results

    def _template_to_regex(self, template):
        """
        Convert a structural template to a proposed regex pattern.

        "{CAP} is the {WORD} of {ENTITY}"
        becomes the corresponding regex with named groups
        """
        regex = template

        # Entity and CAP both become named entity captures
        # First {CAP} → subject, {ENTITY} → also a capture
        cap_count = 0

        def replace_cap(m):
            nonlocal cap_count
            cap_count += 1
            group_name = 'o' if cap_count == 1 else f'cap{cap_count}'
            return f"(?P<{group_name}>[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)"

        regex = re.sub(r'\{CAP\}', replace_cap, regex)
        regex = regex.replace('{ENTITY}',
                              "(?P<s>[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)")
        regex = regex.replace('{NUM}', r'[\d,.\s]+')

        # Replace remaining words with \s+ separated
        # Escape special regex chars in the connecting words
        parts = regex.split()
        escaped_parts = []
        for part in parts:
            if '(?P<' in part or '[' in part:
                escaped_parts.append(part)
            else:
                escaped_parts.append(re.escape(part))
        regex = r'\s+'.join(escaped_parts)

        return regex

    def validate_pattern(self, regex_str, test_sentences, expected_extractions):
        """
        Validate a proposed pattern against known-good test cases.

        Returns:
            dict with:
                valid: bool
                matches: int
                false_positives: int
                precision: float
        """
        try:
            pattern = re.compile(regex_str)
        except re.error:
            return {'valid': False, 'matches': 0, 'false_positives': 0,
                    'precision': 0.0, 'error': 'Invalid regex'}

        matches = 0
        false_positives = 0

        for sent in test_sentences:
            m = pattern.search(sent)
            if m:
                matches += 1
                # Check if it's a true positive
                groups = m.groupdict()
                is_expected = False
                for exp_s, exp_r, exp_o in expected_extractions:
                    if (groups.get('s', '').lower() == exp_s.lower() and
                            groups.get('o', '').lower() == exp_o.lower()):
                        is_expected = True
                        break
                if not is_expected:
                    false_positives += 1

        precision = (matches - false_positives) / max(matches, 1)

        return {
            'valid': matches > 0 and false_positives == 0,
            'matches': matches,
            'false_positives': false_positives,
            'precision': precision,
        }

    def report(self, min_count=2):
        """Human-readable report of discovered patterns."""
        patterns = self.find_patterns(min_count=min_count)

        if not patterns:
            return f"No systematic patterns found (analyzed {len(self.failed_sentences)} failures)"

        lines = [
            f"Pattern Discovery Report ({len(self.failed_sentences)} failures analyzed)",
            "=" * 60,
            "",
        ]

        for i, p in enumerate(patterns, 1):
            lines.append(f"Pattern {i}: {p['template']}")
            lines.append(f"  Frequency: {p['count']} sentences")
            lines.append(f"  Proposed regex: {p['proposed_regex']}")
            lines.append(f"  Examples:")
            for ex in p['examples'][:3]:
                lines.append(f"    - {ex[:80]}...")
            lines.append("")

        return "\n".join(lines)


class KnowledgeExplorer:
    """
    Cycle 4: Frontier Detection for Knowledge Graphs.
    Ported from VizDoom spatial exploration (foss-v2).

    In VizDoom: the agent scans a visited-grid and drives toward
    unvisited tiles in expanding rings.

    Here: the knowledge graph IS the grid.
        - Entities = Nodes (tiles)
        - Relations = Edges (visited connections)
        - Frontier = Entities with few edges (low connectivity)
        - Exploration score = inverse edge density

    The explorer prioritizes knowledge domains where the system
    has the least coverage, driving metacognition toward the
    biggest gaps in understanding.

    Graph metrics:
        - degree(entity) = number of relations it participates in
        - domain_density = avg degree of entities in a domain
        - frontier_score = 1 / (1 + degree) — high = unexplored
        - isolation = entities with only 1 edge (dead-end nodes)
    """

    # Known domain categories for grouping entities
    DOMAIN_HINTS = {
        'country': {'capital', 'language', 'continent', 'population',
                    'currency', 'government', 'area', 'gdp'},
        'person': {'birthplace', 'birth_year', 'nationality', 'profession',
                   'known_for', 'death_year', 'field', 'invention'},
        'science': {'formula', 'symbol', 'discoverer', 'inventor',
                    'discovery_year', 'atomic_number', 'unit'},
        'geography': {'location', 'height', 'area', 'type', 'continent'},
        'technology': {'founder', 'founded', 'headquarters', 'product',
                       'industry', 'ceo'},
        'literature': {'author', 'year', 'genre', 'language', 'characters'},
    }

    def __init__(self, knowledge_store):
        self.knowledge = knowledge_store

    def build_graph(self):
        """
        Build adjacency from facts.

        Returns:
            nodes: dict entity → {degree, relations, connected_to}
            edges: list of (entity_a, relation, entity_b)
        """
        nodes = {}
        edges = []

        for s, r, o in self.knowledge.facts:
            s_lower = s.lower()
            o_lower = o.lower()

            # Initialize nodes
            if s_lower not in nodes:
                nodes[s_lower] = {
                    'name': s, 'degree': 0,
                    'relations': set(), 'connected_to': set()
                }
            if o_lower not in nodes:
                nodes[o_lower] = {
                    'name': o, 'degree': 0,
                    'relations': set(), 'connected_to': set()
                }

            # Add edges
            nodes[s_lower]['degree'] += 1
            nodes[s_lower]['relations'].add(r)
            nodes[s_lower]['connected_to'].add(o_lower)

            nodes[o_lower]['degree'] += 1
            nodes[o_lower]['relations'].add(r)
            nodes[o_lower]['connected_to'].add(s_lower)

            edges.append((s_lower, r, o_lower))

        return nodes, edges

    def frontier_scores(self):
        """
        Compute frontier score for every entity.

        frontier_score = 1 / (1 + degree)
        High score = barely explored = frontier tile.

        Returns:
            list of (entity_name, score, degree, relations)
            sorted by score descending (most frontier first)
        """
        nodes, _ = self.build_graph()

        scored = []
        for _, info in nodes.items():
            score = 1.0 / (1.0 + info['degree'])
            scored.append((
                info['name'],
                score,
                info['degree'],
                sorted(info['relations']),
            ))

        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored

    def _is_leaf_value(self, entity_name, info):
        """
        Detect if an entity is a leaf value (population number, language name,
        continent name) vs a real entity worth exploring.

        Leaf values are objects in facts that only appear as objects,
        never as subjects with their own relations.
        """
        name = entity_name.lower()

        # Numbers / populations / years
        if re.match(r'^[\d,.\s]+(?:\s*million|\s*billion)?$', name):
            return True

        # Only appears as object, never as subject with own facts
        is_subject = any(
            s.lower() == name for s, _, _ in self.knowledge.facts
        )
        if not is_subject and info['degree'] <= 2:
            return True

        return False

    def find_frontiers(self, max_results=20, min_degree=0, max_degree=3):
        """
        Find frontier entities — low connectivity nodes.
        Direct port of VizDoom _compute_frontier_direction.

        In VizDoom: scan expanding rings from current tile.
        Here: scan the graph outward from well-known entities,
        find neighbors with few connections.

        Filters out leaf values (numbers, languages, continents)
        that are just object endpoints, not real explorable entities.

        Args:
            max_results: how many frontier entities to return
            min_degree: minimum degree to count (0 = isolates)
            max_degree: maximum degree to be a frontier

        Returns:
            list of dicts:
                entity: str
                degree: int
                frontier_score: float
                relations: list of str (what we know)
                missing: list of str (what we DON'T know)
                domain: str (detected domain)
        """
        nodes, _ = self.build_graph()
        frontiers = []

        for _, info in nodes.items():
            if info['degree'] < min_degree or info['degree'] > max_degree:
                continue

            # Skip leaf values — they're not explorable entities
            if self._is_leaf_value(info['name'], info):
                continue

            # Detect domain
            domain = self._detect_domain(info['relations'])

            # Find missing relations for this domain
            expected = self.DOMAIN_HINTS.get(domain, set())
            existing = info['relations']
            missing = sorted(expected - existing)

            frontiers.append({
                'entity': info['name'],
                'degree': info['degree'],
                'frontier_score': 1.0 / (1.0 + info['degree']),
                'relations': sorted(existing),
                'missing': missing,
                'domain': domain,
            })

        # Sort: highest frontier score, then most missing relations
        frontiers.sort(key=lambda x: (-x['frontier_score'],
                                       -len(x['missing']),
                                       x['entity']))

        return frontiers[:max_results]

    def domain_coverage(self):
        """
        Compute coverage per domain.

        Returns:
            dict domain → {
                entities: int,
                avg_degree: float,
                coverage: float (0-1, fraction of expected relations filled),
                frontier_count: int (entities with degree ≤ 2),
                densest: str (most connected entity),
                sparsest: str (least connected entity),
            }
        """
        nodes, _ = self.build_graph()
        domains = {}

        for _, info in nodes.items():
            domain = self._detect_domain(info['relations'])
            if domain not in domains:
                domains[domain] = {
                    'entities': [],
                    'total_degree': 0,
                    'total_coverage': 0.0,
                    'frontier_count': 0,
                }

            d = domains[domain]
            d['entities'].append((info['name'], info['degree']))
            d['total_degree'] += info['degree']

            if info['degree'] <= 2:
                d['frontier_count'] += 1

            # Coverage: what fraction of expected relations are present?
            expected = self.DOMAIN_HINTS.get(domain, set())
            if expected:
                filled = len(info['relations'] & expected)
                d['total_coverage'] += filled / len(expected)

        result = {}
        for domain, d in domains.items():
            n = len(d['entities'])
            entities_sorted = sorted(d['entities'], key=lambda x: x[1])
            expected = self.DOMAIN_HINTS.get(domain, set())

            result[domain] = {
                'entities': n,
                'avg_degree': d['total_degree'] / n if n else 0,
                'coverage': d['total_coverage'] / n if n and expected else 0,
                'frontier_count': d['frontier_count'],
                'densest': entities_sorted[-1][0] if entities_sorted else '',
                'sparsest': entities_sorted[0][0] if entities_sorted else '',
            }

        return result

    def exploration_targets(self, n=10):
        """
        Get the top-N entities that need the most exploration.

        Combines frontier score with domain gap analysis.
        This is the equivalent of VizDoom's "drive toward nearest
        unvisited tile" — but for knowledge.

        Returns:
            list of dicts:
                entity: str
                priority: float (higher = more urgent)
                reason: str (why this needs exploration)
                suggested_queries: list of str
        """
        frontiers = self.find_frontiers(max_results=50, max_degree=5)
        targets = []

        for f in frontiers:
            # Priority = frontier_score * (1 + missing_count)
            missing_count = len(f['missing'])
            priority = f['frontier_score'] * (1 + missing_count)

            # Generate suggested queries
            queries = []
            for rel in f['missing'][:5]:
                queries.append(f"What is the {rel} of {f['entity']}?")
            if not queries:
                queries.append(f"Tell me more about {f['entity']}")

            reason_parts = []
            if f['degree'] <= 1:
                reason_parts.append("nearly isolated node")
            elif f['degree'] <= 3:
                reason_parts.append("sparse connections")
            if missing_count > 0:
                reason_parts.append(
                    f"{missing_count} expected {f['domain']} relations missing"
                )

            targets.append({
                'entity': f['entity'],
                'priority': priority,
                'reason': '; '.join(reason_parts) if reason_parts else 'low connectivity',
                'suggested_queries': queries,
                'domain': f['domain'],
                'degree': f['degree'],
            })

        targets.sort(key=lambda x: -x['priority'])
        return targets[:n]

    def _detect_domain(self, relations):
        """Detect which domain an entity belongs to based on its relations."""
        if not relations:
            return 'unknown'

        best_domain = 'unknown'
        best_overlap = 0

        for domain, expected_rels in self.DOMAIN_HINTS.items():
            overlap = len(relations & expected_rels)
            if overlap > best_overlap:
                best_overlap = overlap
                best_domain = domain

        return best_domain

    def report(self, top_n=10):
        """
        Human-readable frontier exploration report.
        """
        nodes, edges = self.build_graph()
        coverage = self.domain_coverage()
        targets = self.exploration_targets(n=top_n)

        lines = [
            "Knowledge Frontier Report",
            "=" * 50,
            f"  Total entities: {len(nodes)}",
            f"  Total relations: {len(edges)}",
            f"  Avg degree: {sum(n['degree'] for n in nodes.values()) / max(len(nodes), 1):.1f}",
            "",
            "Domain Coverage:",
        ]

        for domain, info in sorted(coverage.items(),
                                     key=lambda x: x[1]['coverage']):
            bar_len = int(info['coverage'] * 20)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            lines.append(
                f"  {domain:12s} [{bar}] {info['coverage']:.0%} "
                f"({info['entities']} entities, "
                f"{info['frontier_count']} frontiers)"
            )

        lines.append("")
        lines.append(f"Top {len(targets)} Exploration Targets:")

        for i, t in enumerate(targets, 1):
            lines.append(
                f"  {i}. {t['entity']} "
                f"[{t['domain']}, degree={t['degree']}, "
                f"priority={t['priority']:.2f}]"
            )
            lines.append(f"     Reason: {t['reason']}")
            if t['suggested_queries']:
                lines.append(f"     → {t['suggested_queries'][0]}")

        return "\n".join(lines)


class MetacognitionEngine:
    """
    Master controller for all metacognition cycles.

    Wires together: GapFiller + SelfTester + GapPrioritizer
                  + KnowledgeExplorer (frontier detection).
    Provides a single interface for self-improving operation.
    """

    def __init__(self, knowledge_store, extractor=None):
        self.knowledge = knowledge_store
        self.filler = GapFiller(knowledge_store, extractor=extractor)
        self.tester = SelfTester(knowledge_store)
        self.prioritizer = GapPrioritizer(self.filler.gap_log)
        self.pattern_learner = PatternLearner(extractor=self.filler.extractor)
        self.explorer = KnowledgeExplorer(knowledge_store)

    def query(self, subject, relation=None):
        """
        Self-improving query: try → fill gap → re-try.
        Main entry point replacing direct KnowledgeStore.query().
        """
        return self.filler.fill_query(subject, relation)

    def ingest(self, text):
        """
        Ingest text into both knowledge store AND corpus for future gap-filling.
        """
        # Store raw text for future re-extraction
        self.filler.add_corpus_bulk(text)

        # Extract and store facts now
        n_new = self.filler.extractor.extract_and_store(text, self.knowledge)

        return n_new

    def selftest(self, test_queries):
        """Run self-test and return report."""
        return self.tester.run(test_queries)

    def learn_patterns(self, min_count=2):
        """
        Analyze extraction failures and discover new patterns.
        Cycle 3: self-extension.
        """
        # Feed corpus texts into pattern learner
        corpus_texts = []
        for texts in self.filler._corpus.values():
            corpus_texts.extend(texts)

        if corpus_texts:
            self.pattern_learner.analyze_failures(corpus_texts, self.knowledge)

        return self.pattern_learner.report(min_count=min_count)

    def explore(self, n=10):
        """
        Cycle 4: Frontier exploration — find what the system doesn't know.
        Ported from VizDoom frontier detection.
        """
        return self.explorer.exploration_targets(n=n)

    def frontier_report(self):
        """Full frontier exploration report."""
        return self.explorer.report()

    def gap_report(self):
        """Get prioritized gap report."""
        return self.prioritizer.suggest_text_sources()

    def status(self):
        """Overall metacognition status."""
        unresolved = len(self.filler.gap_log.unresolved())
        total_gaps = len(self.filler.gap_log.gaps)
        resolved = total_gaps - unresolved
        history = self.tester.history()

        lines = [
            "Metacognition Status",
            "=" * 40,
            f"Knowledge facts: {self.knowledge.n_facts}",
            f"Corpus entries: {sum(len(v) for v in self.filler._corpus.values())}",
            f"Total gaps logged: {total_gaps}",
            f"  Resolved: {resolved}",
            f"  Unresolved: {unresolved}",
            f"Self-test runs: {len(history)}",
        ]

        if history:
            latest = history[-1]
            lines.append(f"  Latest accuracy: {latest['accuracy']:.1%} "
                         f"({latest['correct']}/{latest['total']})")

            if len(history) >= 2:
                prev = history[-2]
                delta = latest['accuracy'] - prev['accuracy']
                direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
                lines.append(f"  Trend: {direction} {abs(delta):.1%}")

        return "\n".join(lines)
