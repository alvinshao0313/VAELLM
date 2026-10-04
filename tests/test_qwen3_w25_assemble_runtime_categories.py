from copy import deepcopy

import pytest

from experiments.qwen3_w25.assemble_runtime_categories import _validate_runtime_configs


def _configs():
    base = {"model_type": "qwen3", "hidden_size": 4096, "use_cache": True,
            "pad_token_id": None, "rope_theta": 1000000.0}
    donor = deepcopy(base)
    donor.update(use_cache=False, pad_token_id=151643)
    return base, donor


def test_only_valid_runtime_differences_are_recorded_without_mutating_sources():
    base, donor = _configs()
    before = deepcopy((base, donor))
    result = _validate_runtime_configs(base, donor, 151643)
    assert result["differences"] == {
        "use_cache": {"base": True, "donor": False},
        "pad_token_id": {"base": None, "donor": 151643},
    }
    assert (base, donor) == before


@pytest.mark.parametrize("field,value", [("hidden_size", 2048), ("rope_theta", 10000.0),
                                         ("eos_token_id", 151645)])
def test_architecture_or_other_config_difference_is_rejected(field, value):
    base, donor = _configs()
    donor[field] = value
    with pytest.raises(ValueError, match="outside the permitted runtime fields"):
        _validate_runtime_configs(base, donor, 151643)


@pytest.mark.parametrize("pad", [0, 151644, True, "151643"])
def test_invalid_padding_id_is_rejected(pad):
    base, donor = _configs()
    donor["pad_token_id"] = pad
    with pytest.raises(ValueError, match="original tokenizer pad id"):
        _validate_runtime_configs(base, donor, 151643)


@pytest.mark.parametrize("cache", [0, 1, None, "false"])
def test_non_boolean_cache_value_is_rejected(cache):
    base, donor = _configs()
    donor["use_cache"] = cache
    with pytest.raises(ValueError, match="use_cache must be bool"):
        _validate_runtime_configs(base, donor, 151643)
