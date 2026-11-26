import unittest
import numpy as np
from llm_safety_evaluator.evaluation import metrics,policy_threshold,ranking_curves,paired_correctness,calibrate,calibrated_predict
class EvaluationTests(unittest.TestCase):
    def test_perfect(self):
        m=metrics([0,0,1,1],[0,0,1,1])
        for k in ("accuracy","precision","recall","specificity","balanced_accuracy","f1","f2","mcc","kappa","roc_auc","average_precision"): self.assertAlmostEqual(m[k],1.)
        for k in ("brier","ece","log_loss"): self.assertAlmostEqual(m[k],0.)
    def test_majority(self):
        m=metrics([0,0,0,1],[.25]*4)
        self.assertEqual(m["accuracy"],.75)
        self.assertEqual(m["balanced_accuracy"],.5)
        self.assertEqual(m["f1"],0.)
        self.assertEqual(m["roc_auc"],.5)
        self.assertEqual(m["average_precision"],.25)
    def test_policy(self):
        y=[0,0,1,1];p=[.1,.6,.7,.9]
        t=policy_threshold(y,p)
        self.assertGreaterEqual(metrics(y,p,t)["recall"],.8)
        self.assertEqual(t,.7)
    def test_calibration(self):
        y=[0,0,1,1];p=[.1,.2,.6,.8]
        model=calibrate(y,[p])
        result=calibrated_predict(model,[p])
        self.assertTrue(np.all((result>=0)&(result<=1)))
        self.assertGreater(result[-1],result[0])
    def test_curves(self):
        c=ranking_curves([0,1],[.1,.9])
        self.assertEqual(c["roc"][0],{"x":0.,"y":0.})
        self.assertEqual(c["roc"][-1],{"x":1.,"y":1.})
    def test_paired(self):
        m=paired_correctness([0,1],[.2,.2],.5,[.2,.8],.5)
        self.assertEqual(m["candidate_only_correct"],1)
        self.assertEqual(m["accuracy_delta"],.5)
