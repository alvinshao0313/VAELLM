"""Fixed fresh C/G CE10/CE50 decoder2000, saved steps, and same-source assembly."""
from __future__ import annotations

import argparse
import ast
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from recovery_core import (CORE_FILES, DATA, GPUS, HERE, HF, OUTPUT, PREPARED, ROOT, SCHEMA,
    SNAPSHOT, SOURCES, SOURCE_IDS, TOPIC, assemble, audit_final, check_gpu, check_prepared,
    cpu_source, environment, own_sources, read, require, sha, stat, write)


def training_argv(profile, recipe):
    from experiments.qwen3_w25.reproduction_recipe import training_command
    command = training_command(recipe, source_dir=SOURCES[profile], hf_model=HF,
                               output_dir=DATA / profile / recipe / 'training', export_gpu=GPUS[(profile, recipe)])
    args = command[3:]
    index = args.index('--export_cuda_visible_devices')
    del args[index:index + 2]
    args[args.index('--save_total_limit') + 1] = '5'
    return args


def prepare(path):
    import torch
    from compressed_e2e_fintuning.args import parse_args
    from experiments.qwen3_w25.reproduction_recipe import DATA_FILES, MIXED_DATA, TASK_MIX
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU preparation only.')
    require(not path.exists(), 'Do not overwrite the completed recovery preparation.')
    start = time.monotonic()
    task_manifest = ROOT / 'result/qwen3_8b_w25_20260930/data/task_manifest.json'
    manifest = read(task_manifest)
    task = Path(manifest['path'])
    require(manifest['samples'] == 176597 and task.stat().st_size == manifest['size_bytes'] == 191870906
            and sha(task) == manifest['sha256'] == '894a04aea907efebd8b0f52f040c5635750bb30dfb64d9c34e30a97421a155a1'
            and manifest['split_policy'] == 'train for seven task datasets; cais/mmlu auxiliary_train/train; evaluation validation/test excluded',
            'Use the existing audited176597-row train-only task file.')
    commands = {}
    for profile in SOURCES:
        for recipe in ('ce10', 'ce50'):
            args = training_argv(profile, recipe)
            cfg, _hf, training = parse_args(args)
            require(cfg.train_mode == 'decoder' and tuple(cfg.target_layers) == tuple(range(36))
                    and cfg.target_modules == 'all' and tuple(cfg.opt.distill_fp32_components) == ('decoder',)
                    and cfg.aux.residual_lora_mode == cfg.aux.norm_train_mode == cfg.aux.lm_head_train_mode == 'none'
                    and training.max_steps == 2000 and training.per_device_train_batch_size == 4
                    and training.gradient_accumulation_steps == 1 and training.seed == training.data_seed == 0
                    and training.save_steps == 400 and training.save_total_limit == 5
                    and str(getattr(training.lr_scheduler_type, 'value', training.lr_scheduler_type)) == 'cosine'
                    and training.get_warmup_steps(2000) == 20 and training.bf16,
                    'The fixed original all-decoder recipe changed.')
            commands[profile + '_' + recipe] = args
    sources = {profile: cpu_source(profile) for profile in SOURCES}
    files = [SNAPSHOT / name for name in CORE_FILES if (SNAPSHOT / name).is_file()] + own_sources() + [task_manifest,
        ROOT / 'result/qwen3_8b_w25_20260930/data/sources.json',
        TOPIC / 'qwen_c1_dual_down_pv32_20261019/pv_cpu_loader.py',
        SNAPSHOT / 'rotation/hadamard_utils.py', SNAPSHOT / 'rotation/hadamard_extended.py',
        ROOT / 'experiments/cat_recompression/longbench.py']
    files.extend(source / 'checkpoint_meta.json' for source in SOURCES.values())
    for filename in own_sources():
        ast.parse(filename.read_text(encoding='utf-8'))
    same = {name: (ROOT / name).read_bytes() == (SNAPSHOT / name).read_bytes()
            for name in CORE_FILES[:8]}
    require(all(same.values()), 'Core runtime differs from the mature full-RHT context; inspect before launch.')
    protocol = dict(train_mode='decoder', layers=list(range(36)), modules='all', decoder_tensors=1512,
        fp32_master=True, compute_dtype='bfloat16', export_dtype='bfloat16', external_lora_rank=0,
        batch_size=4, gradient_accumulation_steps=1, max_length=1024, steps=2000, seed=0, data_seed=0,
        decoder_lr=1e-5, scheduler='cosine2000', warmup_steps=20, saved_training_steps=[400,800,1200,1600,2000],
        save_total_limit=5, CE10=dict(dataset_mix=MIXED_DATA, loss='kd_top_mass', alpha_KL=.9, prompt_loss_weight=.3),
        CE50=dict(dataset_mix=TASK_MIX, loss='kd_top_mass', alpha_KL=.5, prompt_loss_weight=0.),
        task_manifest=manifest, middle_checkpoint_quality_evaluated=False,
        final_rule='fixed CE50 attention plus same-source CE10 MLP; both complete full9 regardless donorfull8',
        max_logical_bpw=2.6, no_automatic_retry=True, no_automatic_budget_extension=True)
    record = dict(schema=SCHEMA, status='cpu_verified', sources=sources, training_argv=commands,
        protocol=protocol, source_sha256={str(p): sha(p) for p in files},
        file_stats=[stat(ROOT / name) for name in DATA_FILES] + [stat(source / 'pytorch_model.bin') for source in SOURCES.values()],
        mature_snapshot_core_exact=same, cpu_only=True, GPU_training_or_quality_verified=False,
        elapsed_seconds=time.monotonic() - start)
    write(path, record)
    return record


def train(profile, recipe, gpu, prepared):
    from compressed_e2e_fintuning.args import parse_args
    from compressed_e2e_fintuning.runtime_v6 import run as native_run
    from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
    from train_utils.checkpoint_v6 import load_v6_training_step_meta
    from transformers import TrainerCallback
    import torch
    check_prepared(prepared)
    require(gpu == GPUS[(profile, recipe)] and os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu)
            and torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Use this arm single assigned physical GPU.')
    cfg, hf, training = parse_args(training_argv(profile, recipe))
    require(not Path(cfg.run_root_dir).exists(), 'Every recovery uses a fresh independent training directory.')
    saved_steps = []

    class SavedSteps(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            path = Path(args.output_dir) / ('checkpoint-' + str(state.global_step))
            meta = load_v6_training_step_meta(str(path))
            require(meta['train_mode'] == 'decoder' and (path / 'training_model_state.pt').is_file()
                    and meta['round_base_checkpoint_id'] == SOURCE_IDS[profile], 'Saved real decoder step has invalid provenance.')
            saved_steps.append(dict(step=state.global_step, path=str(path.resolve()),
                                    metadata_sha256=sha(path / 'checkpoint_meta.json')))
            return control

    configure_e2e_determinism(bool(training.full_determinism))
    set_e2e_seed(training.seed)
    result = native_run(cfg, hf, training, extra_callbacks=(SavedSteps(),))
    require(result['global_step'] == 2000 and [row['step'] for row in saved_steps] == [400,800,1200,1600,2000],
            'All five real training checkpoints must be saved.')
    for row in saved_steps:
        require((Path(row['path']) / 'training_model_state.pt').is_file(), 'An intermediate training endpoint was deleted.')
    write(OUTPUT / profile / recipe / 'training_result.json', dict(status='completed', profile=profile, recipe=recipe,
        source_checkpoint_id=SOURCE_IDS[profile], native_result=result, saved_training_checkpoints=saved_steps,
        intermediate_quality_evaluated=False))


def run_donor(profile, recipe, gpu, prepared):
    from recovery_eval import evaluate, run_stage
    check_prepared(prepared)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and gpu == GPUS[(profile, recipe)], 'Fixed arm CPU driver/GPU assignment.')
    output = OUTPUT / profile / recipe
    require(not output.exists(), 'Never overwrite a donor pipeline.')
    output.mkdir(parents=True)
    record = dict(status='running', profile=profile, recipe=recipe, gpu=gpu, source_checkpoint_id=SOURCE_IDS[profile],
                  prepared_sha256=sha(prepared), started_unix=time.time(), stages={})
    path = output / 'pipeline.json'
    write(path, record)
    try:
        check_gpu(gpu)
        argv = [sys.executable, '-B', '-u', str(HERE / 'recovery.py'), 'train', '--profile', profile,
                '--recipe', recipe, '--gpu', str(gpu), '--prepared', str(prepared)]
        record['stages']['train'] = dict(command=argv, started_unix=time.time())
        write(path, record)
        run_stage(argv, name='train', output=output, gpu=str(gpu), seconds=50400)
        record['stages']['train'].update(exit_code=0, finished_unix=time.time())
        result = read(output / 'training_result.json')
        final = Path(result['native_result']['saved_model_dir'])
        selected = Path(result['saved_training_checkpoints'][-1]['path'])
        audit_final(SOURCES[profile], final, selected, output / 'native_audit.json')
        record.update(final_model=str(final), checkpoint_id=read(final / 'checkpoint_meta.json')['checkpoint_id'])
        record['stages']['native_audit'] = dict(exit_code=0, evidence=str(output / 'native_audit.json'))
        write(path, record)
        evaluation = evaluate(final, output / 'evaluation_full8', gpu, evidence=output / 'native_audit.json')
        record['stages']['full8'] = dict(exit_code=0, evidence=str(output / 'evaluation_full8/evaluation.json'))
        record.update(status='complete_donor', full8_mean_percent=evaluation['full8_mean_percent'])
    except BaseException as exc:
        record.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        write(path, record)
    return record


def run_assembly(profile, gpu, prepared):
    from recovery_eval import evaluate
    check_prepared(prepared)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and gpu == {'C': 0, 'G': 2}[profile], 'Use the fixed profile leader GPU.')
    output = OUTPUT / profile / 'assembly'
    require(not output.exists(), 'Only one fixed same-source assembly/full9 is scheduled.')
    output.mkdir(parents=True)
    record = dict(status='running', profile=profile, gpu=gpu, started_unix=time.time())
    path = output / 'pipeline.json'
    write(path, record)
    try:
        final = DATA / profile / 'assembly/final_model'
        audit = assemble(profile, final)
        write(output / 'native_audit.json', audit)
        record.update(final_model=str(final), checkpoint_id=audit['checkpoint_id'])
        write(path, record)
        check_gpu(gpu)
        result = evaluate(final, output / 'evaluation_full9', gpu, full9=True,
                          evidence=output / 'native_audit.json', foreground=True)
        record.update(status='complete_assembly_full9', nine_task_score=result['nine_task_score'], passed=result['passed'])
    except BaseException as exc:
        record.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        write(path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--prepared', type=Path, default=PREPARED)
    for action in ('run', 'train'):
        p = sub.add_parser(action)
        p.add_argument('--profile', choices=('C','G'), required=True)
        p.add_argument('--recipe', choices=('ce10','ce50'), required=True)
        p.add_argument('--gpu', type=int, required=True)
        p.add_argument('--prepared', type=Path, default=PREPARED)
    p = sub.add_parser('assemble'); p.add_argument('--profile', choices=('C','G'), required=True)
    p.add_argument('--gpus', choices=('0','2'), required=True); p.add_argument('--prepared', type=Path, default=PREPARED)
    p = sub.add_parser('evaluate'); p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--full9', action='store_true'); p.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.prepared)
    elif args.action == 'train':
        train(args.profile, args.recipe, args.gpu, args.prepared)
        return
    elif args.action == 'run':
        result = run_donor(args.profile, args.recipe, args.gpu, args.prepared)
    elif args.action == 'assemble':
        result = run_assembly(args.profile, int(args.gpus), args.prepared)
    else:
        from recovery_eval import evaluate
        result = evaluate(args.checkpoint, args.output, args.gpu, full9=args.full9, evidence=args.evidence)
    print(json.dumps({key: result[key] for key in ('status','checkpoint_id','elapsed_seconds') if key in result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
