"""Use the reviewed native fit with only its private CPU loader replaced."""
from copy import deepcopy
from pathlib import Path
from types import FunctionType

from experiments.cat_auto_protocol_cmcc_20261104 import fit_adapter as native
from experiments.cat_auto_protocol_cmcc_20261104.adapter import file_sha
from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from .data import build_block_data_loaders


def fit_code_identity():
    result = deepcopy(native.fit_code_identity())
    result['data_path'] = 'native_CPU_sampler_batched_gather_v1'
    for path in (Path(__file__), Path(__file__).with_name('data.py')):
        result['source_sha256'][str(path.resolve())] = file_sha(path)
    return result


def _validate_identity(identity):
    native._validate_actual_identity(identity)
    require(identity['code_identity'] == fit_code_identity(),
        'The batched fit requires its actual wrapper and CPU gather source identity.')


def _private_pipeline(hooks):
    require('_build_block_data_loaders' not in hooks, 'Do not stack another loader override.')
    return native._private_pipeline({**hooks, '_build_block_data_loaders': build_block_data_loaders})


def fit_category(model, *, inventory, recipe, run_identity, calibration_bundle,
        activation_bundle=None, device, output_dir, technical_steps=None):
    namespace = dict(native.fit_category.__globals__)
    namespace.update(_validate_actual_identity=_validate_identity,
        _private_pipeline=_private_pipeline, fit_code_identity=fit_code_identity)
    entry = FunctionType(native.fit_category.__code__, namespace, native.fit_category.__name__,
        native.fit_category.__defaults__, native.fit_category.__closure__)
    entry.__kwdefaults__ = native.fit_category.__kwdefaults__
    return entry(model, inventory=inventory, recipe=recipe, run_identity=run_identity,
        calibration_bundle=calibration_bundle, activation_bundle=activation_bundle,
        device=device, output_dir=output_dir, technical_steps=technical_steps)
