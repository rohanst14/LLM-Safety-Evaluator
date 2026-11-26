import unittest
import numpy as np
from scipy.sparse import csr_matrix
from llm_safety_evaluator.encoder import fuse
class EncoderTests(unittest.TestCase):
    def test_fusion(self):
        x=csr_matrix([[1.,0.],[0.,1.]])
        z=np.array([[.2,.4],[.5,.6]])
        np.testing.assert_allclose(fuse(x,z).toarray(),np.hstack([x.toarray(),3*z]),rtol=1e-6)
    def test_misalignment(self):
        with self.assertRaises(ValueError):fuse(csr_matrix([[1.]]),np.ones((2,2)))
