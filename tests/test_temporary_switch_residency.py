import unittest

import torch

from e2e_common.temporary_switch_linear import TemporarySwitchLinear


class TemporarySwitchLinearTests(unittest.TestCase):
    def test_set_temporary_switches_weights(self):
        student = torch.ones(2, 3)
        teacher = torch.full((2, 3), 2.0)
        module = TemporarySwitchLinear(3, 2, student, teacher)
        x = torch.ones(4, 3)
        module.set_temporary(True)
        out_student = module(x)
        module.set_temporary(False)
        out_teacher = module(x)
        self.assertTrue(torch.allclose(out_student, x @ student.T))
        self.assertTrue(torch.allclose(out_teacher, x @ teacher.T))


if __name__ == "__main__":
    unittest.main()
