"""Immutable native fitter/targets and one explicitly fixed complete candidate."""
from pathlib import Path
import os
import sys

from experiments.progressive_residual_bsq_20261030.recovery_compare_20261101 import bindings as old
from experiments.mlp_m4096_cmcc_20261101 import data

parent=old.parent
torch,require,read,write,sha=old.torch,old.require,old.read,old.write,old.sha
HERE=Path(__file__).resolve().parent
ROOT,HF,PYTHON=old.REPO,old.HF,old.PYTHON
OUTPUT=parent.OUTPUT_ROOT/'s4_from_bf16_20261102'
CATEGORIES=tuple(old.CATEGORIES)
CONTRACT=dict(origin='Original HF BF16 factory; no old compressed checkpoint inputs',
    fit=dict(categories=list(CATEGORIES),stages=4,updates_per_stage=10000,M=36,
        vector_batch=8192,gate_vector_batch=4096,seed=31,category_seed_stride=100000,
        protection='Original BF16 Alpaca train512x256 seed31, fixed statistics; no refresh'),
    recovery=dict(kind='custom cumulative-prefix native packed decoder static TF',
        budget_rule='Full14042 pooled-correct-count adaptive64/128/256/384; no best-point selection or rollback',
        initial_budget=dict(drop_below36=64,drop36_to84=128,drop85_or_more=256,down_minimum=128),
        continuation=dict(nodes=[64,128,256,384],complete_initial_budget=True,
            remaining_questions_at_least=15,recent_adjacent_node_gain_at_least=15,stage_cap=384),
        groups=list(range(128)),seed=137,lr=1e-6,schedule_steps=500,warmup=0,
        optimizer='fresh AdamW per category',betas=[.9,.95],eps=1e-8,weight_decay=0.,foreach=False,joint_clip=1.5,
        trainable='All36*j already compressed projection decoders, never only current36',
        target='M4096 once .5F+.5gold, gold-prefix full-vocabulary Q32/H96 teacher KL',
        commit='BF16 decoder commit at each category end; future dense/codes/protection/rotation/backbone frozen',
        fitting_RNG='Save/restore Python NumPy CPU and both CUDA RNG around the entire recovery/readout hook'),
    readouts=dict(MMLU='Before/after each category and all visited64/128/256/384 budget nodes, full0shot B64',
        final='Same selected final down UUID: seven B32 and LongBench21/4750; reuse post-down MMLU',
        panels=[2,10,34,42],quality_early_stop=False),physical_GPUs=[6,7],
    final_lowbit_acceptance_evaluated=False)
binding=old.binding


def native_recipe(output):
    # The frozen experiment uses file-local imports; keep its actual context.
    directory=str(parent.HERE)
    if directory not in sys.path:sys.path.insert(0,directory)
    return parent.module(parent.HERE/'recipe.py','_s4_fresh_original_recipe').parse(output)


def pins():
    return {**old.pins(),**data.b.code_bindings(),
        **{str(p):sha(p) for p in sorted(HERE.glob('*.py'))}}


def environment(gpus):
    env=parent.environment(gpus)
    env['PYTHONPATH']=':'.join(map(str,(parent.SAMPLE,ROOT,parent.s.SNAPSHOT)))
    return env


def resources():
    return old.resources()


def restore_rope_alias(model):
    rope=model.model.rotary_emb
    require(rope.inv_freq.dtype==rope.original_inv_freq.dtype==torch.float32
        and torch.equal(rope.inv_freq.cpu(),rope.original_inv_freq.cpu()),'Preserve the actual original FP32 RoPE values.')
    rope.original_inv_freq=rope.inv_freq


def initialize():
    require(Path(sys.executable).resolve()==PYTHON.resolve()
        and os.environ.get('CONDA_DEFAULT_ENV')=='bitvae','Use caller-activated bitvae.')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')


def event(directory):
    directory=Path(directory).resolve(strict=True)
    meta=read(directory/'checkpoint_meta.json')
    return dict(path=str(directory),checkpoint_id=meta['checkpoint_id'],
        metadata_sha256=sha(directory/'checkpoint_meta.json'),
        weight_stat=parent.s.stat(directory/meta['state_dict_file']))


def cleanup_model(directory,reason):
    directory=Path(directory).resolve(strict=True)
    require(directory.is_relative_to(OUTPUT.resolve()),'Only completed independent new-run model payloads may be cleaned.')
    meta=read(directory/'checkpoint_meta.json')
    names=[meta['state_dict_file'],'cat_runtime_state.pt','special_tokens_map.json','merges.txt',
        'added_tokens.json','tokenizer.json','vocab.json','tokenizer_config.json']
    removed=[]
    for name in names:
        path=directory/name
        if not path.exists():continue
        require(not path.is_symlink() and path.resolve().parent==directory and path.is_file(),
            'Refuse cleanup outside the exact completed model directory.')
        info=path.stat();removed.append(dict(path=str(path),bytes=info.st_size,allocated_bytes=info.st_blocks*512))
    record=dict(reason=reason,files=removed,logical_bytes=sum(row['bytes'] for row in removed),
        allocated_bytes=sum(row['allocated_bytes'] for row in removed),
        retained='Config, native metadata, per-tensor expected state, source pins, raw metrics/samples, core logs')
    # Record the exact paths/reason/size before removing these completed payloads.
    print(__import__('json').dumps(dict(event='completed_model_cleanup',**record)),flush=True)
    for row in removed:Path(row['path']).unlink()
    require(all(not Path(row['path']).exists() for row in removed),'Completed payload cleanup incomplete.')
    return record
