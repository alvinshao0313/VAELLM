import unittest

from torch import nn

from train_utils.utils import collect_linears


_CATEGORIES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        for category in _CATEGORIES:
            setattr(self, category, nn.Linear(2, 2))


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyLayer()])


class CatInlineRemainingLoraTests(unittest.TestCase):
    def test_seven_category_remaining_target_state_machine(self):
        model = _TinyModel()
        expected = {
            "q_proj": _CATEGORIES[1:],
            "k_proj": _CATEGORIES[2:],
            "v_proj": _CATEGORIES[3:],
            "o_proj": _CATEGORIES[4:],
            "gate_proj": _CATEGORIES[5:],
            "up_proj": _CATEGORIES[6:],
            "down_proj": [],
        }
        for category in _CATEGORIES:
            setattr(model.model.layers[0], category, nn.Identity())
            remaining = collect_linears(
                model,
                transpose_modules=["q_proj", "v_proj", "o_proj", "down_proj"],
                only_decoder_projections=True,
                target_categories=_CATEGORIES,
            )
            self.assertEqual(list(dict.fromkeys(ref.category for ref in remaining)), expected[category])

