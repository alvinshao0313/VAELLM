"""Fixed D CE10/CE50 recovery through same-source05, using native K50 geometry."""
from __future__ import annotations

import ast
import gc
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOPIC = ROOT / 'result/qwen3_8b_mmlu65_20261004'
PREVIOUS = TOPIC / 'qwen_recompression_followup_20261026'
OUTPUT = PREVIOUS / 'qwen_recompression_d_recovery_20261027'
DATA = Path('/data/star/VAELLM/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/qwen_recompression_d_recovery_20261027')
SOURCE = Path('/data/star/VAELLM/qwen_recompression_20261026/D/native/home_star_models_Qwen_Qwen3-8B_20261026_225932/final_model')
SOURCE_ID = 'e4b4e88f-071d-4efc-8692-9635b717ff66'
SOURCE_META_SHA = 'af1601189fb831a24de6c1738fcd85b5d7df6076442ee36be5d2c82a008a40a7'
NATIVE_REPORT = TOPIC / 'qwen_recompression_20261026/D_resume/D_down.json'
PREPARED = OUTPUT / 'prepared.json'
GPUS = {'ce10': 5, 'ce50': 4}
PREDECESSORS = {
    'ce10': dict(pid=4060124, start_ticks=234744733, report=str(TOPIC / 'qwen_recompression_20261026/D_resume/chain.json')),
    'ce50': dict(pid=4098943, start_ticks=235073176, report=str(PREVIOUS / 'queue/lane_4/queue.json')),
}
TAIL = ROOT / 'experiments/recompression_recovery_tail_20261027'
NIGHT = ROOT / 'experiments/night_sequence_20261026'
sys.dont_write_bytecode = True
sys.path.insert(0, str(TAIL))
sys.path.insert(0, str(NIGHT))
import common as native
import run as stages
from recovery_core import audit_final
from recovery_eval import evaluate as evaluate_full8

require, read, write, sha, stat = native.require, native.read, native.write, native.sha, native.stat
processes = native.module(NIGHT / 'queue_wait.py', '_D_recovery_pidfd_wait')


def training_argv(recipe):
    from experiments.qwen3_w25.reproduction_recipe import training_command
    args = training_command(recipe, source_dir=SOURCE, hf_model=native.HF,
        output_dir=DATA / recipe / 'training', export_gpu=GPUS[recipe])[3:]
    index = args.index('--export_cuda_visible_devices')
    del args[index:index + 2]
    args[args.index('--save_total_limit') + 1] = '5'
    return args


def check():
    record = read(PREPARED)
    require(record['status'] == 'cpu_verified' and record['schema'] == 'fixed_D_CE10_CE50_2000_then05_v1', 'Use the actual completed fixed D preparation.')
    for path, digest in record['source_sha256'].items():
        require(sha(path) == digest, 'Pinned D implementation/input changed: ' + path)
    for item in record['input_stats']:
        require(stat(item['path']) == item, 'Bound D model/data stat changed: ' + item['path'])
    require(native.native_context() == record['native_context'], 'Actual native training/loading context changed.')
    return record


def prepare():
    import torch
    from compressed_e2e_fintuning.args import parse_args
    from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state
    from experiments.qwen3_w25.audit_e2e_decoder import persistent_inventory
    from experiments.qwen3_w25.reproduction_recipe import DATA_FILES, MIXED_DATA, TASK_MIX
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'D preparation is CPU-only.')
    require(not PREPARED.exists() and not DATA.exists(), 'Use one fresh D recovery preparation/output.')
    torch.set_num_threads(4)
    started = time.monotonic()
    context = native.native_context()
    report = read(NATIVE_REPORT)
    require(report['status'] == 'completed' and report['exit_code'] == 0 and report['profile'] == 'D', 'Actual D native source must have completed successfully.')
    entries = [row for row in report['saved_boundaries'] if row['kind'] == 'final_model']
    require(len(entries) == 1 and Path(entries[0]['path']).resolve() == SOURCE
        and entries[0]['checkpoint_id'] == SOURCE_ID and sha(SOURCE / 'checkpoint_meta.json') == SOURCE_META_SHA,
        'Actual D native source identity differs.')
    meta, budget = native.checkpoint(SOURCE)
    model, loaded_meta, loaded = native.load_cpu(SOURCE)
    require(meta == loaded_meta and meta['checkpoint_id'] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys, 'Actual D native fresh strict load failed.')
    count = _assert_state_equal(model.state_dict(), _load_state(SOURCE), 'D original native fresh strict3423')
    inventory, banks = persistent_inventory(model)
    require(count == 3423 and sum(map(len, inventory.values())) == 1512 and sum(map(len, banks.values())) == 504,
        'Actual complete D persistent/decoder/packed-bank inventory changed.')
    target_name = 'model.layers.0.mlp.down_proj'
    target = model.get_submodule(target_name)
    geometry = dict(spec=next(row for row in meta['converted_modules'] if row['name'] == target_name),
        decoder_shapes={key: list(model.state_dict()[key].shape) for key in inventory[target_name]},
        packed_shapes=[list(target.get_stage_part_vq_storage(stage, 0).shape) for stage in range(2)],
        logical_shapes=[list(target.get_stage_part_vq_spec(stage, 0)['logical_shape']) for stage in range(2)])
    require(all(shape[-1] == 50 for shape in geometry['logical_shapes'])
        and all(shape[-1] == 7 for shape in geometry['packed_shapes']), 'Require the actual D K50/seven-byte bank geometry; do not use K48 constants.')
    for key in inventory[target_name]:
        require(model.state_dict()[key].dtype == torch.bfloat16, 'D native deployment decoder storage must be BF16.')
    del model
    gc.collect()
    repair_path = PREVIOUS / 'recovery/data_path_repair.json'
    repair = read(repair_path)
    require(repair['status'] == 'verified' and repair['all_five_actual_preset_loaders_passed'] and len(repair['presets']) == 5,
        'Reuse the actual verified five-preset train-only loader evidence.')
    task_manifest = ROOT / 'result/qwen3_8b_w25_20260930/data/task_manifest.json'
    manifest = read(task_manifest)
    require(manifest['samples'] == 176597 and manifest['sha256'] == '894a04aea907efebd8b0f52f040c5635750bb30dfb64d9c34e30a97421a155a1'
        and sha(manifest['path']) == manifest['sha256'] and 'auxiliary_train/train' in manifest['split_policy'], 'Use the actual audited176597-row task train file.')
    for relative in DATA_FILES:
        require((native.SNAPSHOT / relative).resolve() == (ROOT / relative).resolve(), 'Mature snapshot preset resolves to another dataset: ' + relative)
    commands = {}
    for recipe in GPUS:
        argv = training_argv(recipe)
        cfg, _hf, args = parse_args(argv)
        require(cfg.train_mode == 'decoder' and tuple(cfg.target_layers) == tuple(range(36)) and cfg.target_modules == 'all'
            and tuple(cfg.opt.distill_fp32_components) == ('decoder',) and args.max_steps == 2000
            and args.per_device_train_batch_size == 4 and args.gradient_accumulation_steps == 1
            and cfg.data.model_max_length == 1024 and args.seed == args.data_seed == 0
            and args.save_steps == 400 and args.save_total_limit == 5 and cfg.opt.decoder_lr == 1e-5
            and args.get_warmup_steps(2000) == 20 and args.bf16 and not args.fp16
            and str(getattr(args.lr_scheduler_type, 'value', args.lr_scheduler_type)) == 'cosine'
            and cfg.loss.loss_type == 'kd_top_mass' and cfg.loss.top_k == 100 and cfg.loss.temperature == 1
            and cfg.loss.alpha == (.9 if recipe == 'ce10' else .5)
            and cfg.loss.prompt_loss_weight == (.3 if recipe == 'ce10' else 0.)
            and cfg.aux.norm_train_mode == cfg.aux.lm_head_train_mode == cfg.aux.residual_lora_mode == 'none',
            'Original CE10/task50 or CE50/task100 decoder2000 recipe changed.')
        commands[recipe] = argv
    for recipe, row in PREDECESSORS.items():
        current = processes.process_start(row['pid'])
        require(current in (None, row['start_ticks']), 'Bound predecessor PID identity changed: ' + recipe)
        actual = read(row['report'])
        require(actual['pid'] == row['pid'] and int(actual['start_ticks']) == row['start_ticks'], 'Actual predecessor report identity differs.')
    paths = {NATIVE_REPORT, SOURCE / 'checkpoint_meta.json', SOURCE / 'config.json', repair_path, task_manifest,
        ROOT / 'result/qwen3_8b_w25_20260930/data/sources.json', NIGHT / 'queue_wait.py', NIGHT / 'recovery_core.py', NIGHT / 'recovery_eval.py',
        TAIL / 'common.py', TAIL / 'endpoints.py', TAIL / 'run.py', native.TOPIC / 'qwen_c1_dual_down_pv32_20261019/pv_cpu_loader.py',
        ROOT / 'experiments/cat_recompression/longbench.py', *HERE.glob('*.py')}
    paths.update(Path(path) for path in context.values())
    for name in ('experiments/qwen3_w25/reproduction_recipe.py', 'experiments/qwen3_w25/audit_e2e_decoder.py',
        'experiments/qwen3_w25/export_e2e_selected.py', 'experiments/qwen3_w25/assemble_categories.py',
        'experiments/qwen3_w25/assemble_runtime_categories.py', 'train_utils/checkpoint_bpw.py', 'train_utils/checkpoint_v6.py',
        'train_utils/distill_precision.py'):
        paths.update({ROOT / name, native.SNAPSHOT / name})
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
    record = dict(schema='fixed_D_CE10_CE50_2000_then05_v1', status='cpu_verified', created_unix=time.time(),
        source_dir=str(SOURCE), source_checkpoint_id=SOURCE_ID, source_bit_budget=budget,
        source_fresh_strict_tensors=3423, decoder_tensors=1512, code_banks=504, frozen_tensors=1911,
        actual_D_L0_down_geometry=geometry, native_context=context, predecessor_bindings=PREDECESSORS, fixed_gpus=GPUS,
        training_argv=commands, protocol=dict(train_mode='decoder', FP32_master=True, BF16_compute_and_export=True,
            CE10_mix=MIXED_DATA, CE50_mix=TASK_MIX, steps=2000, batch_size=4, max_length=1024,
            seed=0, decoder_lr=1e-5, scheduler='cosine2000/warm20', saved_training_steps=[400, 800, 1200, 1600, 2000],
            save_total_limit=5, donor_evaluation='terminal full8 only', assembly='CE50 attention plus same-source CE10 MLP',
            assembly_full8_samples=24742, assembly_longbench_samples=4750, longbench_timeout_seconds=86400,
            max_logical_bpw=2.6, max_lora_rank=0, no_K50_local_joint_tail=True, no_intermediate_quality_selection=True),
        reused_evidence=dict(CE2000_production='existing night C/G decoder2000 native finalization/save/fresh/full8',
            D_native_K50_source='actual completed500 CAT/native export and in-progress full9; quality does not gate this recovery',
            task_train_provenance=manifest, no_new_GPU_smoke=True),
        source_sha256={str(path): sha(path) for path in sorted(paths)},
        input_stats=[stat(SOURCE / meta['state_dict_file']), *(stat(ROOT / relative) for relative in DATA_FILES)],
        CUDA_initialized=False, elapsed_seconds=time.monotonic() - started)
    write(PREPARED, record)
    require(not torch.cuda.is_initialized(), 'CPU D preparation initialized CUDA.')
    return record


def train(recipe):
    import torch
    from compressed_e2e_fintuning.args import parse_args
    from compressed_e2e_fintuning.runtime_v6 import run as native_run
    from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
    from train_utils.checkpoint_v6 import load_v6_training_step_meta
    from transformers import TrainerCallback
    check()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(GPUS[recipe]) and torch.cuda.device_count() == 1, 'Use the one assigned D recipe GPU.')
    cfg, hf, args = parse_args(training_argv(recipe))
    require(not Path(cfg.run_root_dir).exists(), 'Every D recipe starts fresh from the same native source.')
    saved = []
    class SavedSteps(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            path = Path(args.output_dir) / ('checkpoint-' + str(state.global_step))
            meta = load_v6_training_step_meta(str(path))
            require(meta['train_mode'] == 'decoder' and meta['round_base_checkpoint_id'] == SOURCE_ID
                and (path / 'training_model_state.pt').is_file(), 'Saved real D decoder step provenance differs.')
            saved.append(dict(step=int(state.global_step), path=str(path.resolve()), metadata_sha256=sha(path / 'checkpoint_meta.json')))
            return control
    configure_e2e_determinism(bool(args.full_determinism))
    set_e2e_seed(args.seed)
    result = native_run(cfg, hf, args, extra_callbacks=(SavedSteps(),))
    require(result['global_step'] == 2000 and [row['step'] for row in saved] == [400, 800, 1200, 1600, 2000]
        and all((Path(row['path']) / 'training_model_state.pt').is_file() for row in saved), 'All five real D saved checkpoints and terminal2000 are required.')
    write(OUTPUT / recipe / 'training_result.json', dict(status='completed', recipe=recipe, source_checkpoint_id=SOURCE_ID,
        native_result=result, saved_training_checkpoints=saved, intermediate_quality_evaluated=False))


def assemble():
    from endpoints import save_native
    from experiments.qwen3_w25.assemble_categories import EXPECTED, _expected_state, _load_state, _replace_modules
    from experiments.qwen3_w25.assemble_runtime_categories import _validate_sources
    records = {recipe: read(OUTPUT / recipe / 'pipeline.json') for recipe in GPUS}
    require(all(row['status'] == 'complete_donor' and row['source_checkpoint_id'] == SOURCE_ID for row in records.values()), 'Both actual D terminal donors must complete native audit and full8.')
    base, donor = (Path(records[key]['final_model']) for key in ('ce50', 'ce10'))
    meta, donor_meta, names, _original, tokenizer, runtime = _validate_sources(base, donor, ['gate_proj', 'up_proj', 'down_proj'])
    require(len(names) == 108, 'Fixed05 requires all108 same-source CE10 MLP projections.')
    for recipe, directory in (('ce50', base), ('ce10', donor)):
        audit = read(OUTPUT / recipe / 'native_audit.json')
        require(audit['status'] == 'verified' and audit['source_checkpoint_id'] == SOURCE_ID
            and audit['checkpoint_id'] == read(directory / 'checkpoint_meta.json')['checkpoint_id'], 'Actual D donor provenance differs.')
    expected = _expected_state(_load_state(base), _load_state(donor), names, EXPECTED)
    model, _, _ = native.load_cpu(base)
    donor_model, _, _ = native.load_cpu(donor)
    _replace_modules(model, donor_model, names)
    del donor_model
    gc.collect()
    provenance = dict(source_checkpoint_id=SOURCE_ID, base_checkpoint_dir=str(base), base_checkpoint_id=meta['checkpoint_id'],
        donor_checkpoint_dir=str(donor), donor_checkpoint_id=donor_meta['checkpoint_id'], categories=['gate_proj', 'up_proj', 'down_proj'],
        replaced_targets=names, source_config_runtime_fields=runtime, purpose='Fixed D05 inference assembly; no exact-resume claim.')
    audit = save_native(model, DATA / '05_ce50_attention_ce10_mlp/final_model', meta, expected, provenance, tokenizer)
    require(audit['checkpoint_id'] not in (meta['checkpoint_id'], donor_meta['checkpoint_id']), 'D05 must have an independent native identity.')
    write(OUTPUT / '05_ce50_attention_ce10_mlp/native_audit.json', audit)
    return audit
