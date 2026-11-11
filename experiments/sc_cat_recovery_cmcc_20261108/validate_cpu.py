"""CPU rule counterexamples and actual tiny Qwen native KD/save/export/resume."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import functools
import json
import os
from pathlib import Path
import tempfile
import time

import torch
from transformers import BertTokenizerFast

from experiments.cat_auto_protocol_cmcc_20261104 import data as d
from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from experiments.cat_auto_protocol_cmcc_20261104.live import architecture, fresh_native
from experiments.cat_auto_protocol_cmcc_20261104.nodes import NodeScore
from experiments.cat_auto_protocol_cmcc_20261104.validate_recovery_cpu import score, rejects
from experiments.cat_auto_protocol_cmcc_20261104.validate_native_builder_cpu import fixture_sources, tiny_root
from experiments.sc_cat_finish_cmcc_20261108.resume import equal_tree
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from train_utils import checkpoint_v6 as v6
from .boundary import export_selected_node, export_zero_kd_boundary
from .builder import make_native_recovery_trainer, RecoveryExecutionPolicy
from .handoff import choose_top2
from .identity import source_pins
from .mutable import capture_mutable
from .selection import TrajectoryResult, SelectedCategory, compare, calibration, select_category, long_check
from .settings import RecoverySettings, normalize, round_candidates, ROUNDS


def rule_checks():
    import ast
    import sys
    from rotation import hadamard_utils
    assert Path(hadamard_utils.__file__).resolve()==Path('rotation/hadamard_utils.py').resolve()
    from . import muon_math
    class LocalRequire(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id=='b' and node.attr=='require':
                return ast.copy_location(ast.Name(id='require', ctx=node.ctx), node)
            return self.generic_visit(node)
    original=ast.parse(Path('experiments/muon_compact_cmcc_20261030/optimizer.py').read_text())
    isolated=ast.parse(Path(muon_math.__file__).read_text())
    for name in ('stages','zeropower','direction'):
        a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in isolated.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(LocalRequire().visit(a))==ast.dump(b)
    assert not any('qwen_native_category_rotation/source_snapshot' in str(p) for p in sys.path)
    inv = dict(prefix=('a',), current=('a',), protected=('a',), future=('b',))
    settings = RecoverySettings()
    assert len(round_candidates('combination', settings, **inv)) == 7
    assert len(round_candidates('scope', replace(settings, groups='DPL'), **inv)) == 1
    assert len(round_candidates('scope', replace(settings, groups='DPL'), **dict(inv, prefix=('x', 'a'), protected=('x', 'a')))) == 4
    seen = set()
    for name in ROUNDS:
        candidates = round_candidates(name, replace(settings, groups='DPL') if name != 'combination' else settings, **inv)
        seen.update(candidates)
    # Two defaults, with the winner's DL already included in the seven-way round.
    assert len(seen)+1 == 23
    assert len(round_candidates('combination', settings, **dict(inv, future=()))) == 3
    assert round_candidates('rank', replace(settings, groups='P'), **inv) == ()
    rows = [dict(candidate_id=n, recipe=dict(transpose=False, group_signature=['x'], value=n),
        run_identity={'cpu_fixture': True}, full8_mean=s, logical_bits=b, packed_bits=b) for n,s,b in [('anchor', .5, 5), ('a', .7, 6), ('b', .6, 4)]]
    assert [r['candidate_id'] for r in choose_top2(rows, 'a')[0]] == ['a', 'b']
    assert [r['candidate_id'] for r in choose_top2(rows[:1], 'anchor')[0]] == ['anchor']
    assert len(choose_top2([*rows, deepcopy(rows[1])], 'a')[1]) == 3
    def tr(identifier, values, raw=.4, status='complete', bits=100):
        return TrajectoryResult('k', 'own_parent', 'raw_'+identifier, 'recipe_'+identifier, identifier, 'tr_'+identifier,
            bits, bits, settings, score(raw, 'raw_'+identifier), tuple(NodeScore(i, score(s, identifier+str(i)), identifier+str(i))
            for i,s in zip((125,250,375,500), values)), status)
    a, b, c = tr('a', (.7,.7,.7,.7)), tr('b', (.6,.6,.9,.8)), tr('c', (.8,.8,.8,.8))
    assert calibration([a,b,c], [])['status'] == 'failed'
    assert calibration([a,b,c], ['tr_b'])['status'] == 'passed'
    assert compare([a])[0][1].step == 500
    assert select_category([tr('a', (.5,)*4, raw=.5)])[1]['selected_step'] == 500
    assert select_category([tr('a', (.5,)*4, raw=.6)])[1]['selected_step'] == 0
    assert select_category([tr('a', (.5,)*4, raw=.6)])[1]['selected_trajectory_id'] == 'raw0:raw_a'
    assert long_check([tr('a', (.5,.6,.7,.8))], 'recipe_a', []) is not None
    assert long_check([tr('a', (.5,.6,.7,.8))], 'recipe_a', ['x','y']) is None
    short = tr('a', (.95,.9), status='screened250')
    assert select_category([short,b])[0] == short
    rejects(select_category, [replace(short, status='pending250')])
    assert compare([a, tr('b', (.7,)*4, bits=90)])[0][0].candidate_id == 'b'
    return dict(top2_all_rounds_no_anchor_priority=True, duplicates_and_single_candidate=True,
        Muon_math_AST_unchanged_without_N_runtime=True,
        first_class_max500_trajectories=23, full_grid_and_screened250=True,
        calibration_reversal_and_preexisting_control=True, raw0_strict_and_later_tie=True,
        persistent_long_quota=True, inactive_owner_dedup=True)


def protected_root(root, tokenizer):
    inventory, old_dir, _ = tiny_root(root, tokenizer)
    model, meta, _ = v6.load_v6_full_checkpoint_into_model(architecture(old_dir, attn_implementation='sdpa'), str(old_dir), strict=True)
    for name in meta['compressed_targets']:
        decoders = [Decoder(in_dim=8, out_dim=8, hidden_dim=8, num_res_blocks=0,
            norm_type='layer', decoder_type='asymmetric', use_checkpoint=False, num_models=1).to(dtype=torch.bfloat16).extract_single(0) for _ in range(2)]
        codes = [torch.randint(2, (16,1,8), dtype=torch.bool) for _ in range(2)]
        layer = VAELinear(in_features=16, out_features=16, compressed_in_features=8,
            bias=None, original_weight=None, vq_weight=codes[0], decoder=decoders[0],
            codebook_dim=8, transpose=False, stage_vq_weights=codes,
            stage_decoders=decoders, stage_codebook_dims=[8,8],
            protected_input_indices=torch.arange(8), protected_input_qvalues=torch.randint(1,100,(8,16),dtype=torch.uint8),
            protected_input_scales=torch.full((8,), .01, dtype=torch.bfloat16), protected_channel_quant_format='int8')
        layer.pack_parallel_stage_decoder_(trainable=False)
        owner, field = name.rsplit('.',1)
        setattr(model.get_submodule(owner), field, layer)
    path = root/'protected_raw'
    v6.save_v6_full_checkpoint(model, str(path), checkpoint_kind='round_base',
        compressed_targets=meta['compressed_targets'], pending_dense_targets=meta['pending_dense_targets'],
        completed_categories=meta['completed_categories'], compression_categories=meta['compression_categories'],
        target_layers=meta['target_layers'], target_modules=meta['target_modules'],
        cat_runtime_state=v6.load_v6_cat_runtime_state(str(old_dir), required=True),
        base_model_path=meta['base_model_path'], extra_meta=meta['extra_meta'])
    return inventory, path


def actual_checks(root, *, resume):
    vocab = root/'vocab.txt'
    vocab.write_text('\n'.join(['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','hello','world','tiny','train','answer']
        + [f'unused{i}' for i in range(10,128)])+'\n')
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
    tokenizer.eos_token = tokenizer.sep_token
    inventory, raw_dir = protected_root(root, tokenizer)
    sources = fixture_sources(root)
    prepared = d.freeze_data_plan(sources, sources[0], tokenizer,
        model_identity=inventory['model_identity'], category_ids=['k_proj'], search_seed=31)
    settings = RecoverySettings(groups='DPL', lora_rank=24, lora_alpha=96, lora_dropout=0.,
        bound_optimizer='muon', lora_optimizer='muon')
    policy = RecoveryExecutionPolicy('cpu', 'sdpa', 'none', 'none', teacher_output_pin_memory=False)
    common = dict(round_base_dir=raw_dir, inventory=inventory, category_id='k_proj', settings=settings,
        prepared_data=prepared, sources=sources, tokenizer=tokenizer, execution_policy=policy, phase='cpu_fixture')
    run = make_native_recovery_trainer(**common, output_dir=root/'train',
        technical_stop_after=None if resume else 2, stop_at=250 if resume else None)
    try:
        initial, _, _ = capture_mutable(run.selection)
        result = run.train()
        final, _, manifest = capture_mutable(run.selection)
        assert all(torch.isfinite(v).all() for v in final.values())
        assert any(not torch.equal(v, initial[n]) for n,v in final.items() if n.endswith('protected_code_coordinate'))
        assert any(not torch.equal(v, initial[n]) for n,v in final.items() if n.endswith('protected_log_scale_delta'))
        optimizer = run.trainer.optimizer.state_dict()
        groups = optimizer['param_groups']
        assert {g['group_name'] for g in groups if g['muon_kappa'] is not None} == {'decoder','lora','q','scale'}
        assert all(v>0 for v in run.selection.protection.calls.values())
        step = result['actual_steps']
        node_dir = Path(run.trainer.args.output_dir)/f'checkpoint-{step}'
        node = v6.load_v6_training_step_meta(str(node_dir))
        selected = SelectedCategory('k_proj','cpu_recipe','cpu_candidate',settings,step,'cpu_trajectory',node['checkpoint_id'],
            run.provenance['round_base_id'],run.provenance['semantic_seeds'],'cpu_fixture')
    finally:
        run.close()
    current = [r['actual_name'] for r in inventory['projections'] if r['category_id']=='k_proj']
    exported = export_selected_node(node_directory=node_dir, destination=root/'export', selected=selected,
        current_targets=current, phase='cpu_fixture', model_factory=functools.partial(architecture,attn_implementation='sdpa'))
    fresh, proof = fresh_native(root/'export', model_factory=functools.partial(architecture,attn_implementation='sdpa'))
    with torch.no_grad():
        logits = fresh(input_ids=prepared['calibration'][0], use_cache=False).logits
    assert torch.isfinite(logits).all() and proof['state_exact']
    assert not any('protected_code_coordinate' in n or 'protected_log_scale_delta' in n for n in fresh.state_dict())
    zero = replace(selected, selected_step=0, source_node_id=selected.source_round_base_id)
    z = export_zero_kd_boundary(round_base_dir=raw_dir, destination=root/'zero', selected=zero,
        current_targets=current, phase='cpu_fixture', model_factory=functools.partial(architecture,attn_implementation='sdpa'))
    resume_verified = False
    ownership = {}
    for group in ('D','P','L','DP','DL','PL'):
        only = replace(settings, groups=group, bound_optimizer='adamw', lora_optimizer='muon')
        check = make_native_recovery_trainer(**dict(common, settings=only),
            output_dir=root/('ownership_'+group), technical_stop_after=2)
        try:
            check.train()
            contract = check.optimizer.effective_contract()
            ownership[group] = [g['group_name'] for g in contract['groups']]
            expected = ({'decoder','decoder_aux'} if 'D' in group else set()) | ({'q','scale'} if 'P' in group else set()) | ({'lora'} if 'L' in group else set())
            assert set(ownership[group]) == expected
        finally:
            check.close()
    fixed = {g['group_name']:g['calibration'] for g in groups if g['kind']=='muon'}
    pair = make_native_recovery_trainer(**dict(common, settings=replace(settings, decoder_lr=2e-5,
        lora_lr=5e-5, protection_multiplier=2.)), output_dir=root/'fixed_kappa',
        fixed_calibrations=fixed, technical_stop_after=2)
    try:
        pair.train()
        assert {g['group_name']:g['calibration'] for g in pair.optimizer.param_groups if g['kind']=='muon'} == fixed
    finally:
        pair.close()
    if resume:
        resumed = make_native_recovery_trainer(**common, output_dir=root/'resume', resume_from=str(node_dir), stop_at=500)
        try:
            resumed.train()
            resume_verified = resumed.proof.checked and len(resumed.observer.observed)==250
            reloaded = resumed.trainer.optimizer.state_dict()['param_groups']
            assert [g['calibration'] for g in reloaded] == [g['calibration'] for g in groups]
        finally:
            resumed.close()
    return dict(actual_KD_updates=step+(250 if resume else 0)+14, first_segment_updates=step,
        schedule_max=500, actual_native_B4_workers16=True,
        joint_q_scale_changed=True, four_independent_Muon_kappas={g['group_name']:g['muon_kappa'] for g in groups},
        variable_LoRA_export=dict(rank=24,alpha=96,dropout=0.), mutable_count=len(manifest),
        actual_combination_ownership=ownership, fixed_kappa_across_learning_rate_pairs=True,
        complete_native_save_and_export=True, fresh_state_and_finite_forward=True, zero_raw_export=True,
        native250_resume_to500=resume_verified, GPU_verified=False, full8_or_quality_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU check must hide CUDA.')
    path = Path(args.output)
    require(not path.exists(), 'Preserve earlier validation evidence.')
    torch.set_num_threads(2)
    began = time.monotonic()
    rules = rule_checks()
    print('rule_checks passed', flush=True)
    with tempfile.TemporaryDirectory(prefix='new_SC_CAT_CPU_', dir=path.parent) as tmp:
        actual = actual_checks(Path(tmp), resume=args.resume)
    value = dict(status='CPU_verified', rules=rules, actual=actual, source_pins=source_pins(),
        elapsed_seconds=time.monotonic()-began, CUDA_initialized=torch.cuda.is_initialized(), formal_eligible=False)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(actual), flush=True)


if __name__ == '__main__':
    main()
