"""One fixed full-H96 BR36 candidate on a genuinely completed fresh C0."""
from pathlib import Path

from experiments.block_reconstruction_cmcc_20261030 import bindings as core
from experiments.block_reconstruction_cmcc_20261030 import data, export as deployment, fit, rolling, runtime
from experiments.fresh_fixed2_cmcc_20261030 import evaluate as c0
from experiments.cmcc_runtime_20261030 import adaptive_evaluation as shared, sequence
from experiments.cmcc_runtime_20261030.paths import ARTIFACTS, PYTHON, REPO, TOPIC

torch, checkpoint = core.torch, core.checkpoint
require, read, write, sha, digest, file_binding = core.require, core.read, core.write, core.sha, core.digest, core.file_binding
HERE = Path(__file__).resolve().parent
OUTPUT = ARTIFACTS/'block_reconstruction_formal_cmcc_20261031'
SHORT = ARTIFACTS/'block_reconstruction_cmcc_20261030/adjacent_short_v1'
SUPPORT = ARTIFACTS/'adaptive_class_cmcc_20261030/evaluation_support'
_READERS_CONFIGURATION = None
LAYERS, GROUPS = tuple(range(36)), tuple(range(32))
CONTRACT = dict(layers=list(LAYERS), groups=list(GROUPS), updates_per_layer=32, total_updates=1152,
    objective='sum FP32 valid-token hidden SSE of all nine entries / (group valid tokens * 4096)',
    teacher='Independent original BF16 prefix', student='Updated own prefix, advanced only after exact BF16 commit',
    trainable='Current complete Transformer block seven existing native/compact decoders only',
    frozen='All codes, routes, protection, rotation, nondecoder and other-block fields; no LoRA',
    optimizer='AdamW', base_lr=1e-6, betas=[.9, .95], eps=1e-8, weight_decay=0., foreach=False,
    joint_clip=1.5, schedule_steps=500, warmup_steps=0, fresh_optimizer_per_layer=True, seed=137,
    precision='FP32 masters, actual BF16 decoder/block compute, FP32 loss reduction; frozen BF16 commits',
    native_prewarm='Complete same-device216 native in original order, canonical group8; retain current block only')


def configure_readers(support=SUPPORT):
    """Initialize the fixed cmcc reader once, before installing any dispatch."""
    global _READERS_CONFIGURATION
    from experiments.adaptive_class_evaluation_20261030 import bindings as official
    support = Path(support).resolve(strict=True)
    require(support == SUPPORT.resolve(strict=True), 'Use this candidate\'s actual fixed cmcc evaluation support.')
    prepared = support/'prepared.json'
    if _READERS_CONFIGURATION is None:
        shared.setup(support)
        _READERS_CONFIGURATION = file_binding(prepared)
    require(_READERS_CONFIGURATION == file_binding(prepared)
        and official.BASE.resolve(strict=True) == core.HF.resolve(strict=True)
        and official.REFERENCE == support/'config.json' and official.TEMPLATE == support/'template.json',
        'The initialized cmcc support binding, local HF base or reader paths changed.')
    # In particular, do not call shared.setup again here: its generic command
    # would replace a BR or BR->E dispatch already installed by evaluate.setup.
    return _READERS_CONFIGURATION.copy()


def code_bindings():
    paths = set(HERE.glob('*.py'))
    paths.update(Path(module.__file__).resolve() for module in (data, rolling, fit, runtime, deployment,
        core, c0, shared, sequence, data.legacy, data.original, data.legacy.core,
        data.legacy.mmlu, data.legacy.base))
    return {**core.runtime_bindings(), **{str(path.resolve()): sha(path) for path in sorted(paths)}}


def verified_short():
    report_path, audit_path = SHORT/'report.json', SHORT/'fresh_audit.json'
    report, audit = read(report_path), read(audit_path)
    exits = (SHORT/'run.exit', SHORT/'fresh.exit', SHORT.with_suffix('.exit'))
    require(all(path.read_text().strip() == '0' for path in exits)
        and report['schema'] == 'BR_adjacent_blocks_short2_v1' and report['validation_only'] is True
        and report['completed_layers'] == [0, 1] and report['updates_per_layer'] == 2
        and audit['schema'] == 'BR_adjacent_blocks_short2_fresh_audit_v1' and audit['status'] == 'verified'
        and audit['training_report'] == file_binding(report_path) and audit['full_state_exact']
        and audit['frozen_complement_exact'] and audit['same_complete_input_BF16_logits_exact'],
        'Reuse only the actual completed adjacent-block mixed GPU mechanism evidence.')
    math_paths = [core.HERE/name for name in
        ('bindings.py', 'data.py', 'rolling.py', 'runtime.py', 'fit.py', 'export.py', 'run_short.py', 'fresh.py')]
    math_pins = {str(path.resolve()): report['source_sha256'][str(path.resolve())] for path in math_paths}
    sequence.check_pins(math_pins)
    return dict(report=file_binding(report_path), fresh=file_binding(audit_path),
        exits=[file_binding(path) for path in exits], numerical_core_sha256=math_pins,
        source_checkpoint_id=report['source_binding']['checkpoint_id'], checkpoint_id=audit['checkpoint_id'],
        short_weights_loaded=False, scope='Reuse tested math, mixed lifecycle and peak shapes; not a formal parent or quality result.')


def bind_C0(directory, report):
    configure_readers()
    directory, report = Path(directory).resolve(), Path(report).resolve()
    require(directory.is_dir() and (directory/checkpoint.META_FILE).is_file() and report.is_file(),
            'Require the actual completed mixed C0 and its pipeline report; native B or a missing future parent is not eligible.')
    value = read(report)
    require(value.get('schema') == 'fresh_C0_fixed2_pipeline_v1'
        and value.get('status') == 'completed_fresh_C0_fixed2_pipeline' and value.get('exit_code') == 0,
        'Require the genuine complete C0; no short, intermediate category, prototype or renamed BR endpoint.')
    return c0.source(directory, report)


@torch.no_grad()
def load_student(parent, parent_meta):
    model, meta, loaded, budget = checkpoint.load(parent['checkpoint_dir'])
    require(meta == parent_meta and not loaded.missing_keys and not loaded.unexpected_keys,
            'Load the exact verified C0 full persistent source strictly.')
    deployment.deployment_contract(model)
    require(budget == parent['bit_budget'], 'The actual C0 storage ledger changed before BR.')
    for name, module in runtime.modules(model).items():
        core.original.validate_supported(module, name)
    core.install_precision_runtime(model, torch.bfloat16)
    from experiments.muon_compact_cmcc_20261030 import inputs
    teacher = inputs.teacher_identity(model.state_dict(),
        [name for group in checkpoint.inventories(model).values() for name in group])
    flags = runtime.remember_deployment_flags(model)
    return model, dict(teacher_identity=teacher, deployment_flags=flags, post_conversion_budget=budget,
                       source_state_digest=digest(model.state_dict()), extra_conversion_or_fit_updates=0)
