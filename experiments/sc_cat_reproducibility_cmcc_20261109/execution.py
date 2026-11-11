"""Bind FA2 deterministic backward before any Transformers consumer is imported."""
import os
import sys


def configure_deterministic_backward():
    name = 'transformers.modeling_flash_attention_utils'
    loaded = sys.modules.get(name)
    if loaded is not None and not loaded.deterministic_g:
        raise RuntimeError('Configure FA2 determinism before importing the Transformers attention consumer.')
    os.environ['FLASH_ATTENTION_DETERMINISTIC'] = '1'
    from transformers import modeling_flash_attention_utils as flash
    if not flash.flash_241 or not flash.deterministic_g:
        raise RuntimeError('The installed FA2 consumer did not enable deterministic backward.')
    return dict(environment='FLASH_ATTENTION_DETERMINISTIC=1',
        effective=True, consumer=flash.__file__, algorithm_and_seeds_unchanged=True)
