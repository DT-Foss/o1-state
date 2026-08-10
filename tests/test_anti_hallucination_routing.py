"""
Regression test for the anti-hallucination routing fix in repl.py::process().

Bug: the Foss Pipeline (Reservoir + Hopfield autoregressive generation) ran
BEFORE the instructor/router's Dict-Index KnowledgeStore lookup, and could
produce a fluent, confident-looking answer for an entity it had never seen
(e.g. "capital of Narnia" -> "oranjestad", nearest-neighbor attractor drift
from Aruba). Because the pipeline populated `scores`, the instructor's
correctly REJECTED-aware Fiber 2 lookup was never even reached.

Fix: for "what is the X of Y" style questions, check whether the KnowledgeStore
has any facts at all for the named entity before trusting the pipeline's guess.
If the entity is unknown, skip the pipeline and let the instructor (whose
Fiber 2 defaults to REJECTED) answer honestly.

This test is slow (full FossKIRepl boot, ~1 min) — run explicitly, not as
part of a fast unit suite.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAntiHallucinationRouting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from repl import FossKIRepl
        cls.repl = FossKIRepl()

    def test_narnia_capital_rejected_not_hallucinated(self):
        """The core regression: unknown entity must not get a confident guess."""
        answer = self.repl.process("what is the capital of Narnia?")
        self.assertNotIn("oranjestad", answer.lower())
        self.assertIn("don't have information", answer.lower())

    def test_france_capital_unchanged(self):
        answer = self.repl.process("what is the capital of France?")
        self.assertEqual(answer.strip(), "Paris")

    def test_hamlet_author_unchanged(self):
        answer = self.repl.process("who wrote Hamlet?")
        self.assertIn("Shakespeare", answer)

    def test_ice_float_reasoning_unchanged(self):
        answer = self.repl.process("why does ice float?")
        self.assertIn("less dense", answer.lower())

    def test_97_prime_unchanged(self):
        answer = self.repl.process("is 97 prime?")
        self.assertIn("prime", answer.lower())
        self.assertNotIn("not prime", answer.lower())


if __name__ == '__main__':
    unittest.main()
