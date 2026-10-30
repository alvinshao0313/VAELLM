"""CPU mathematical checks only; these do not validate an 8B model or quality."""
import json
import unittest

import torch
from torch.nn import functional as F

import numerics as n


class MathTests(unittest.TestCase):
    def test_confident_hard_target(self):
        logits = torch.tensor([[9., -9., -9.]], dtype=torch.float32)
        target = torch.tensor([0])
        old_value, old_gradient = n.value_gradient(lambda z: F.cross_entropy(z, target), logits)
        new_value, new_gradient = n.value_gradient(lambda z: n.fixed_hard_cross_entropy(z, target).sum(), logits)
        reference, gradient64 = n.value_gradient(lambda z: n.fixed_hard_cross_entropy(z, target).sum(), logits.double())
        self.assertEqual(float(old_value), 0.)
        self.assertEqual(float(old_gradient[0, 0]), 0.)
        self.assertGreater(float(new_value), 0.)
        # One FP32 log-sum-exp rounding near -17 can amplify exp by about 2e-6.
        torch.testing.assert_close(new_value.double(), reference, rtol=2e-6, atol=0.)
        torch.testing.assert_close(new_gradient.double(), gradient64, rtol=2e-6, atol=0.)
        self.assertLess(float(new_gradient.double().sum(-1).abs().max()), 1e-14)

    def test_identity_raw_teacher_has_zero_gradient(self):
        generator = torch.Generator().manual_seed(137)
        for classes in (4, 151936):
            logits = torch.randn((2, classes), generator=generator)
            _, gradient = n.value_gradient(lambda z: n.fixed_raw_soft_cross_entropy(z, logits).sum(), logits)
            self.assertEqual(torch.count_nonzero(gradient).item(), 0)

    def test_eq9_near_teacher_matches_fp64(self):
        teacher = torch.tensor([[2., 1., 0.]], dtype=torch.float32)
        student = teacher + torch.tensor([[-1e-7, 1e-7, 0.]], dtype=torch.float32)
        reference = F.softmax(student.double(), -1) - F.softmax(teacher.double(), -1)
        candidate = n.raw_teacher_gradient(student, teacher)
        torch.testing.assert_close(candidate.double(), reference, rtol=2e-6, atol=1e-14)

    def test_unnormalized_cache_uses_mass(self):
        logits = torch.tensor([[.2, -.4, 1.3]], dtype=torch.float64)
        cached = F.log_softmax(torch.tensor([[.4, -.1, 1.]], dtype=torch.float64), -1) + .01
        _, gradient = n.value_gradient(lambda z: n.cached_kl_reference(z, cached), logits)
        p = cached.exp()
        expected = p.sum(-1, keepdim=True) * F.softmax(logits, -1) - p
        torch.testing.assert_close(gradient, expected, rtol=1e-13, atol=1e-15)
        self.assertGreater(float((gradient - (F.softmax(logits, -1) - p)).norm()), .001)

    def test_eq5_eq6_float64_autograd(self):
        logits = torch.tensor([[.4, 1.1, -.2], [2., -.3, .1]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([0, 2])
        self.assertTrue(torch.autograd.gradcheck(n.fixed_log_softmax, (logits,)))
        self.assertTrue(torch.autograd.gradcheck(lambda z: n.fixed_hard_cross_entropy(z, target), (logits,)))
        torch.testing.assert_close(n.fixed_log_softmax(logits), F.log_softmax(logits, -1), rtol=1e-13, atol=1e-15)


if __name__ == '__main__':
    torch.set_num_threads(4)
    result = unittest.main(exit=False, verbosity=2).result
    print(json.dumps(dict(status='passed' if result.wasSuccessful() else 'failed',
        tests=result.testsRun, scope='CPU mathematics only, no model or training validation')), flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
