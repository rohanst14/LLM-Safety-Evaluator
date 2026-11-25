import unittest
from llm_safety_evaluator.moderation import confusion,choose_threshold,subgroup_report
class ModerationTests(unittest.TestCase):
    def test_confusion(self):
        m=confusion([1,1,0,0],[.9,.1,.8,.2])
        self.assertEqual((m["tp"],m["tn"],m["fp"],m["fn"]),(1,1,1,1))
        self.assertEqual(m["f1"],.5)
        self.assertAlmostEqual(m["brier"],.375)
    def test_threshold(self):
        y=[0,0,1,1];s=[.1,.2,.3,.4]
        self.assertEqual(confusion(y,s,choose_threshold(y,s))["f1"],1)
    def test_undefined(self):
        self.assertIsNone(confusion([0],[.1])["recall"])
    def test_groups(self):
        m=subgroup_report([0,1],[.8,.9],[["A"],["A","B"]],.5)
        self.assertEqual(m["A"]["count"],2)
        self.assertEqual(m["B"]["count"],1)
    def test_invalid(self):
        for y,s in [([],[]),([1],[float("nan")]),([2],[.5]),([0,1],[.2])]:
            with self.assertRaises(ValueError): confusion(y,s)
