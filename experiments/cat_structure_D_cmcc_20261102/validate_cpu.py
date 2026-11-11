"""Validate the real D recipe, frozen namespace and native evaluation dispatch."""
import importlib
import os
from pathlib import Path
import sys

from . import bindings as b


def main():
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Use caller-activated CPU bitvae.')
    for path in b.HERE.glob('*.py'):
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    argv, parsed, identity, normalized = b.recipe(b.OUTPUT / 'formal/native')
    cat, hf, training, vae, cfgs = parsed
    b.require(cat.seed == 31 and cat.linear_group_size == 36 and cat.batch_size == 8192
        and cfgs['gate_proj'].base_ch == 128 and cfgs['gate_proj'].decoder_base_ch == 256
        and cfgs['gate_proj'].decoder_type == 'asymmetric'
        and all(cfgs[name].base_ch == 128 and cfgs[name].decoder_type == 'symmetric'
            for name in ('up_proj', 'down_proj')), 'Incorrect actual D recipe.')
    from . import evaluate, source
    from experiments.adaptive_class_evaluation_20261030 import bindings as common
    before = (common.source, common.command, common.make_config, common.check_config)
    try:
        source.inspect(b.OUTPUT / 'formal/missing', b.OUTPUT / 'formal/missing/report.json')
    except FileNotFoundError:
        rejected = True
    else:
        raise ValueError('An absent formal endpoint was accepted.')
    b.require(before == (common.source, common.command, common.make_config, common.check_config),
        'The pure native source reader changed evaluation dispatch.')
    checks = []
    for name in ('evaluate', 'full8', 'predict', 'summarize'):
        try:
            getattr(evaluate, name)(['--help'])
        except SystemExit as error:
            b.require(error.code == 0, 'Actual component CLI failed: ' + name)
            checks.append(dict(component=name, exit_code=0))
        else:
            raise ValueError('Expected actual parser help: ' + name)
    modules = {}
    for name in ('train_utils.cat_train_pipeline', 'train_utils.cat_after_category_common',
            'train_utils.checkpoint_v6', 'litebsq.vae_linear', 'rotation.model_utils'):
        path = Path(importlib.import_module(name).__file__).resolve()
        b.require(path.is_relative_to(b.SNAPSHOT), 'Execution escaped the frozen native N snapshot: ' + name)
        modules[name] = str(path)
    b.require(not b.torch.cuda.is_initialized(), 'CPU validation initialized CUDA.')
    b.write(b.RESULT / 'cpu_validation.json', dict(status='verified_real_D_recipe_native_dispatch',
        exit_code=0, source_binding=b.source_binding(), native_argv=argv,
        runtime_identity=identity, normalized=normalized, actual_component_CLI=checks,
        missing_source_rejected=rejected, source_reader_preserves_dispatch=True,
        actual_native_modules=modules, CUDA_initialized=False, source_sha256=b.pins()))


if __name__ == '__main__':
    main()
