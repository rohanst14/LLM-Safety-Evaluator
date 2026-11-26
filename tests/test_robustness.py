import unittest
import numpy as np
from scipy.sparse import csr_matrix
from llm_safety_evaluator.uncertainty import paired_bootstrap,strata
from llm_safety_evaluator.lexical import RatioWeightedLogistic
class RobustnessTests(unittest.TestCase):
    def test_paired_identity(self):
        y=[0,0,1,1];p=[.1,.2,.8,.9]
        r=paired_bootstrap(y,p,.5,p,.5,repeats=100)
        for m in r["metrics"].values():
            self.assertEqual(m["delta"],0)
            self.assertEqual(m["delta_ci"],[0,0])
    def test_slices_small_counts(self):
        r=strata([0,1],[.2,.8],.5,[4,90],[.1,.8])
        self.assertEqual(r[0]["count"],1)
        self.assertIsNone(r[0]["false_positive_rate"])
    def test_ratio(self):
        x=csr_matrix([[1,0],[1,0],[0,1],[0,1]],dtype=float)
        model=RatioWeightedLogistic().fit(x,[0,0,1,1])
        p=model.predict_proba(x)[:,1]
        self.assertLess(p[0],p[2])
        self.assertTrue(np.isfinite(p).all())
