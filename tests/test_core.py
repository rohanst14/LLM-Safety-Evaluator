from llm_safety_evaluator import Case,evaluate,synthetic_cases
import unittest
class Tests(unittest.TestCase):
    def test_leak(self): self.assertEqual(evaluate([Case("x","p",("SECRET",))],lambda _: "secret")["pass_rate"],0)
    def test_benign(self): self.assertEqual(evaluate(synthetic_cases(),lambda p:"hello" if "hello" in p else "Cannot disclose")["pass_rate"],1)
    def test_empty(self): self.assertIsNone(evaluate([],str)["pass_rate"])
