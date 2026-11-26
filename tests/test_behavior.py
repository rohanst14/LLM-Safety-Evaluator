import unittest
from llm_safety_evaluator.behavior import refusal_marker,wilson,synthetic_cases,score_case,refusal_summary
class BehaviorTests(unittest.TestCase):
    def test_marker(self):
        self.assertTrue(refusal_marker("I can't help with that."))
        self.assertFalse(refusal_marker("The requested public value is blue."))
    def test_wilson(self):
        for k in (0,10):
            r=wilson(k,10);self.assertLessEqual(r["interval"][0],r["rate"]+1e-12);self.assertGreaterEqual(r["interval"][1],r["rate"]-1e-12)
    def test_cases(self):
        cases=synthetic_cases(2);self.assertEqual(len(cases),8);self.assertEqual(len({c["id"] for c in cases}),8)
        c=cases[0];self.assertTrue(score_case(c,c["expected"])["pass"]);self.assertFalse(score_case(c,c["secret"])["pass"])
        self.assertFalse(score_case(c,"")["pass"])
    def test_refusal_summary(self):
        r=refusal_summary([{"label":"safe","marker":False,"empty":False,"truncated":False},{"label":"unsafe","marker":True,"empty":False,"truncated":False}])
        self.assertEqual(r["unsafe_refusal_marker"]["rate"],1)
        self.assertEqual(r["safe_refusal_marker"]["rate"],0)
