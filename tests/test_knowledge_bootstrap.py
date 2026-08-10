"""
Regression test for core/knowledge_bootstrap.py::load_bootstrap_to_knowledge_store.

Bug: the loader used 4-tuple unpacking (s, r, o, _w) but data/knowledge_full.json
(schema v2, count=4855) stores flat 3-tuples [s, r, o], causing a ValueError on
the real data file. Covers both triplet shapes so a regression to either one
is caught.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.knowledge_bootstrap import load_bootstrap_to_knowledge_store
from core.knowledge import KnowledgeStore


class FakeStore:
    """Minimal stand-in for KnowledgeStore.store_fact to isolate the loader."""

    def __init__(self):
        self.facts = []

    def store_fact(self, subject, relation, obj):
        self.facts.append((subject, relation, obj))


def _write_triplets(triplets):
    fd, path = tempfile.mkstemp(suffix='.json')
    with os.fdopen(fd, 'w') as f:
        json.dump({'version': '2.0', 'source': 'test', 'count': len(triplets), 'triplets': triplets}, f)
    return path


class TestLoadBootstrapToKnowledgeStore(unittest.TestCase):

    def test_flat_3tuples_v2_schema(self):
        """The real knowledge_full.json v2 shape: [s, r, o], no weight."""
        path = _write_triplets([
            ['France', 'capital', 'Paris'],
            ['Germany', 'capital', 'Berlin'],
        ])
        try:
            store = FakeStore()
            loaded = load_bootstrap_to_knowledge_store(store, path)
            self.assertEqual(loaded, 2)
            self.assertIn(('France', 'capital', 'Paris'), store.facts)
            self.assertIn(('Germany', 'capital', 'Berlin'), store.facts)
        finally:
            os.remove(path)

    def test_4tuples_with_weight_legacy_schema(self):
        """Older/legacy shape: [s, r, o, weight] must still work."""
        path = _write_triplets([
            ['France', 'capital', 'Paris', 4.0],
            ['Germany', 'capital', 'Berlin', 4.0],
        ])
        try:
            store = FakeStore()
            loaded = load_bootstrap_to_knowledge_store(store, path)
            self.assertEqual(loaded, 2)
            self.assertIn(('France', 'capital', 'Paris'), store.facts)
        finally:
            os.remove(path)

    def test_real_knowledge_full_json_end_to_end(self):
        """The actual on-disk data/knowledge_full.json against a real KnowledgeStore."""
        real_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'knowledge_full.json'
        )
        if not os.path.exists(real_path):
            self.skipTest("data/knowledge_full.json not present in this environment")
        store = KnowledgeStore()
        loaded = load_bootstrap_to_knowledge_store(store, real_path)
        self.assertGreater(loaded, 4000)
        result = store.query(subject='France', relation='capital')
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
