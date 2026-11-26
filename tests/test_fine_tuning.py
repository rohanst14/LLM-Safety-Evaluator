import unittest,importlib.util
@unittest.skipUnless(importlib.util.find_spec("torch"),"PyTorch required for training loss tests")
class FineTuningTests(unittest.TestCase):
    def test_losses_and_gradients(self):
        import torch
        from llm_safety_evaluator.fine_tuning import binary_loss
        for mode in ("bce","weighted_bce","focal"):
            x=torch.tensor([-1.,0.,1.],requires_grad=True);y=torch.tensor([0.,1.,1.])
            v=binary_loss(x,y,mode,3.);v.backward()
            self.assertTrue(torch.isfinite(v));self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(x.grad.abs().sum().item(),0)
    def test_focal_gamma_zero(self):
        import torch
        from llm_safety_evaluator.fine_tuning import binary_loss
        x=torch.tensor([-2.,2.]);y=torch.tensor([0.,1.])
        self.assertTrue(torch.allclose(binary_loss(x,y,"bce"),binary_loss(x,y,"focal",gamma=0)))
