"""
Regression tests for Task 16's "receipts" additions -- LiveCausalAdapter's
citations/contested query() fields, and repl.py's _append_receipt trace
integration.

Slow (full FossKIRepl boot) -- run explicitly, not part of a fast suite.
"""
import os
import shutil
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser("~/fosski-venv/adapter"))
sys.path.insert(0, os.path.expanduser("~"))

from livecausal_bridge import LiveStore  # noqa: E402
from live_causal_adapter import LiveCausalAdapter  # noqa: E402


class TestReceiptFields(unittest.TestCase):
    """Adapter-level: citations/contested fields on query()'s result."""

    def setUp(self):
        self.store_dir = os.path.expanduser("~/fosski-venv/test_receipts_store")
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)
        store = LiveStore(self.store_dir)
        # Base fact, single citing segment.
        store.append_segment([
            {"trigger": "Water", "mechanism": "boils_at", "outcome": "100C",
             "trigger_key": "water", "outcome_key": "100c",
             "doc_coord": "fact:1", "evidence_count": 1, "use_count": 0, "meta": {}},
        ])
        # A genuine SS2 conflict: same edge_key, two different mechanisms,
        # each in its own segment (segment-provenance fallback needs one
        # segment per independent source, see build_contested_store.py).
        for i in range(3):
            store.append_segment([
                {"trigger": "Coffee", "mechanism": "causes", "outcome": "alertness",
                 "trigger_key": "coffee", "outcome_key": "alertness",
                 "doc_coord": f"study:{i}", "evidence_count": 1, "use_count": 0, "meta": {}},
            ])
        store.append_segment([
            {"trigger": "Coffee", "mechanism": "correlates_with", "outcome": "alertness",
             "trigger_key": "coffee", "outcome_key": "alertness",
             "doc_coord": "study:x", "evidence_count": 1, "use_count": 0, "meta": {}},
        ])

    def tearDown(self):
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

    def test_citations_present_and_correct_count(self):
        a = LiveCausalAdapter(self.store_dir)
        r = a.query(subject="Water", relation="boils_at")
        self.assertIn("citations", r)
        self.assertEqual(len(r["citations"]), 1)
        self.assertEqual(r["citations"][0][1], 0)  # idx
        self.assertEqual(len(r["citations"][0][0]), 12)  # short sha

    def test_uncontested_fact_has_no_contested_field(self):
        a = LiveCausalAdapter(self.store_dir)
        r = a.query(subject="Water", relation="boils_at")
        self.assertIsNone(r["contested"])

    def test_genuine_conflict_detected_as_contested(self):
        a = LiveCausalAdapter(self.store_dir)
        r = a.query(subject="Coffee", relation="causes")
        self.assertIsNotNone(r["contested"])
        self.assertEqual(r["contested"]["winner_mechanism"], "causes")
        self.assertEqual(r["contested"]["ratio"], "3:1")
        self.assertEqual(r["contested"]["counts_by_mechanism"]["causes"], 3)
        self.assertEqual(r["contested"]["counts_by_mechanism"]["correlates_with"], 1)

    def test_contested_ratio_updates_live_after_cut(self):
        a = LiveCausalAdapter(self.store_dir)
        r1 = a.query(subject="Coffee", relation="causes")
        self.assertEqual(r1["contested"]["ratio"], "3:1")

        causes_shas = []
        for sha in a.segments():
            for _s, _i, rec in a.graph.store.iter_records(sha):
                if rec["mechanism"] == "causes":
                    causes_shas.append(sha)
        a.drop_segments(causes_shas[:2])

        r2 = a.query(subject="Coffee", relation="causes")
        self.assertEqual(r2["contested"]["ratio"], "1:1")
        self.assertEqual(r2["contested"]["winner_mechanism"], "correlates_with")


if __name__ == "__main__":
    unittest.main()
