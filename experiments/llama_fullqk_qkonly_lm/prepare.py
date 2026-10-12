"""CPU preparation from original Llama full-QK; existing method, new source/scope."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.llama_ce_reproduction.callback import stage_commands
from experiments.llama_fullqk_qkonly_lm.train import (
    CONTROL, OUTPUT, PUBLIC, R2, ROOT, SOURCE, SOURCE_ID, check_inputs, configuration, read, training_arguments)
from experiments.longbench_v1.common import check_hashes, file_stat, sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.qwen_fullqk_lm_recovery.train import completed_proof
from experiments.w25_eval.acceptance import collect_results, validate_checkpoint
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import selected_names
from experiments.w25_eval.validate_decoder_subset_training import expected_inventory
from train_utils.runtime_snapshot import to_runtime_jsonable


def prepare():
    require(Path.cwd() == ROOT and Path(__file__).resolve().is_relative_to(ROOT)
            and os.environ.get('PYTHONPATH') == str(ROOT) and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized() and not OUTPUT.exists(), 'Use new CPU-only bitvae/S preparation.')
    control = read(CONTROL)
    qprepared_path = R2 / 'general_lm_qwen_fullqk_qkonly_s2000/launch.prepared.json'
    qprepared = read(qprepared_path)
    qproof = completed_proof()
    require(qproof == qprepared['resource_validation'], 'Previously measured H252 backward proof changed.')
    qreview_path = R2 / 'general_lm_qwen_fullqk_qkonly_s2000/step_0400/independent_review.json'
    qreview = read(qreview_path)
    require(qreview['status'] == 'verified_independent_cpu_review'
            and qreview['actual_scope']['decoder_tensors'] == 432
            and qreview['actual_scope']['frozen_cached_projections'] == 180
            and qreview['actual_scope']['optimizer_exact']
            and qreview['native_audit_reused']['fresh_strict_reload_state_exact']
            and qreview['native_audit_reused']['frozen_tensors_compared'] == 2991
            and qreview['training_return']['observed_step_after_callback'] > 400,
            'Require completed full-rotation Q/K subset/cache/export/callback evidence.')
    llreview_path = CONTROL.parent / 'step_0400/independent_review.json'
    llreview = read(llreview_path); llaudit = read(CONTROL.parent / 'step_0400/audit.json')
    require(llreview['status'] == 'verified' and llaudit['status'] == 'verified'
            and llaudit['serialized_tensors_compared'] == llaudit['reloaded_tensors_compared'] == 2979
            and llaudit['selected_decoder_tensor_count'] == 1344 and llaudit['code_bank_count'] == 448
            and llaudit['fresh_strict_reload_state_exact'] and llaudit['frozen_bytes_unchanged'],
            'Reuse actual Llama mixed-D16/native callback and independent export evidence.')
    pair_path = SOURCE.parent.parent / 'independent_review.json'
    pair, launch, audit = (read(p) for p in (pair_path, SOURCE.parent/'launch.json', SOURCE.parent/'audit.json'))
    require(pair['status'] == 'independent_cpu_verified' and pair['arms']['full']['checkpoint_id'] == SOURCE_ID
            and launch['status'] == 'completed' and launch['exit_code'] == 0
            and launch['checkpoint_id'] == audit['checkpoint_id'] == SOURCE_ID
            and audit['status'] == 'verified' and audit['source_union_save_fresh_tensors'] == 2979
            and audit['replaced_projection_count'] == 64 and audit['preserved_projection_count'] == 160,
            'Use the actual completed Llama Q/K-full assembled source, not a donor or Qwen source.')
    exits = [SOURCE.parent/f'{n}.exit' for n in ('train','assembly','mmlu','seven','dispatcher','outer')]
    require(all(p.read_text().strip() == '0' for p in exits), 'The fresh source has incomplete stages.')
    architecture = model_inventory(read(SOURCE/'checkpoint_meta.json')['base_model_path'])
    source, meta, budget = validate_checkpoint(SOURCE, architecture)
    require(architecture == control['architecture'] and architecture['model_type'] == 'llama'
            and meta['checkpoint_id'] == SOURCE_ID and budget['budget_passed']
            and budget['user_linear_bits'] == budget['packed_linear_bits'] == 17095180288
            and budget['original_linear_weights'] == 6979321856 and budget['code_padding_bits'] == 0,
            'Actual source architecture/logical/physical budget changed.')
    for field in ('user_linear_bits','packed_linear_bits','original_linear_weights','components'):
        require(budget[field] == audit['bit_budget'][field], 'Source budget differs from native assembly audit.')
    names = selected_names('all','q_proj,k_proj',architecture)
    inv = expected_inventory(source,architecture,names)
    require(inv['selected_projections'] == 2*architecture['num_hidden_layers'] == 64
            and inv['selected_decoder_tensors'] == 384 and inv['selected_decoder_elements'] == 1101824
            and inv['frozen_tensors'] == 2595 and inv['serialized_tensors'] == 2979
            and inv['all_code_banks'] == 448 and inv['selected_projection_code_banks'] == 128,
            'Actual Llama Q/K inventory or shape changed.')
    specs = {x['name']:x for x in meta['converted_modules']}
    qmeta = read(Path(qprepared['source_checkpoint'])/'checkpoint_meta.json')
    qspecs = {x['name']:x for x in qmeta['converted_modules']}
    geometry_keys = ('in_features','out_features','compressed_in_features','compressed_out_features',
        'codebook_dim','transpose','residual_stages','stage_codebook_dims','stage_decoders','stage_vq_weights',
        'protected_input_indices','protected_input_qvalues','protected_input_scales','protected_channel_quant_format')
    for name in names:
        require(specs[name]['weight_rotation']['block_size'] == 0
                and all(specs[name][k] == qspecs[name][k] for k in geometry_keys),
                'Q/K geometry is not covered by the real H252 FP32 backward proof: '+name)
    protocol_paths = [SOURCE.parent/f'{n}.json' for n in ('mmlu','seven')]
    original = collect_results(protocol_paths,architecture,checkpoint=(source,meta))
    require(original['mean_percent'] == pair['arms']['full']['mean_percent'], 'Source raw score differs.')
    for a,b in zip(protocol_paths,control['control_result_paths'],strict=True):
        aa,bb = (read(p)['evals']['lm_eval'] for p in (a,b))
        require(all(aa[k] == bb[k] for k in ('configs','versions','n_shot','n_samples')), 'Original full8 protocols differ.')
    inputs = [CONTROL,qprepared_path,qreview_path,llreview_path,CONTROL.parent/'step_0400/audit.json',
        pair_path,SOURCE.parent/'launch.json',SOURCE.parent/'audit.json',SOURCE/'checkpoint_meta.json',
        SOURCE/'config.json',SOURCE/'assembly_report.json',Path(qprepared['source_checkpoint'])/'checkpoint_meta.json',
        Path(control['data']['manifest']),Path(control['data']['canonical_statistics']),*protocol_paths,*exits]
    evidence = {str(p):sha256(p) for p in inputs}; evidence.update(qproof['evidence_sha256'])
    # Retain only small proof inputs; historical donor payload paths inside audits are provenance, not dependencies.
    source_pins = deepcopy(qprepared['source_sha256'])
    for p in sorted(Path(__file__).parent.glob('*.py')):
        ast.parse(p.read_text());require(p.read_bytes() == (PUBLIC/p.relative_to(ROOT)).read_bytes(),'Public/S source differs.')
        source_pins[str(p)] = sha256(p)
    cache = sum(out*inp*2 for name,(out,inp) in architecture['projection_shapes'].items() if name not in names)
    require(cache == 12616466432, 'Actual frozen160 BF16 dense-cache estimate differs.')
    record = {k:deepcopy(control[k]) for k in ('hf_source','input_data_stat','data','schedule','protocol')}
    record.update(schema='llama_fullqk_qkonly_fineweb_2000_v1',kind='formal_s2000',status='prepared_unbound',
        gpu=None,export_gpu=None,queued=False,automatic_retry=False,cwd=str(ROOT),python_executable=sys.executable,
        output_dir=str(OUTPUT),source_checkpoint=str(source),source_checkpoint_id=SOURCE_ID,
        source_weight_stat=file_stat(source/'pytorch_model.bin'),source_bit_budget=budget,
        architecture=architecture,expected_inventory=inv,selected_targets=list(names),
        rotation_blocks={n:s['weight_rotation']['block_size'] for n,s in specs.items()},
        source_sha256=source_pins,input_evidence_sha256=evidence,environment=deepcopy(qprepared['environment']),
        control_result_paths=list(map(str,protocol_paths)),control_mean_percent=original['mean_percent'],
        training_argv=training_arguments(OUTPUT),argv=[sys.executable,'-u','-m','experiments.llama_fullqk_qkonly_lm.train',
            '--launch_record',str(OUTPUT/'launch.json')],resource_budget_seconds=64800,termination_grace_seconds=20,
        historical_e767_context={'checkpoint_id':control['source_checkpoint_id'],'mean_percent':control['control_mean_percent'],
            'boundary':'History only; different source and trainable scope, not a matched all-decoder arm.'},
        validation_reuse={'h252_backward':qproof,'qk_cached_subset_callback':str(qreview_path),
            'llama_native_callback':str(llreview_path),'source_pair':str(pair_path),
            'boundary':'Identical selected QTrue/KFalse4032 H252 geometry; Llama32 instead of Qwen36 layers and frozen160 rather than180. Reuse native Llama loading/training/callback and mixedD16 evidence, no new GPU diagnostic.'},
        readiness={'cpu':'pending','gpu':'not_scheduled','reused_evidence':'verified'},
        comparison_boundary='Primary recovery gain versus original4e1 fullQK. e767 is history, not same-source allD control; Qwen scope results are method evidence, not a Llama quality prediction.',
        scientific_boundary='Fresh original fullQK; FineWeb2000/save400/B4/max1024/LR3e-6/warm20. Only Q/K decoders train; packed bits, signs, protection, other160 decoders, model norms/head/embedding freeze. No LoRA or prior diagnostic weights.',
        estimate={'storage_bytes_per_branch':35000000000,'disk_free_at_prepare':shutil.disk_usage(OUTPUT.parent).free,
            'frozen_BF16_dense_cache_bytes':cache,'qwen_allD_measured_peak_bytes':35621856256,
            'simple_peak_plus_frozen_cache_bytes':35621856256+cache,'training_peak_gib_estimate':[45,60],
            'training_hours_estimate':[2,10],'hard_seconds':64800,
            'boundary':'No measured Llama fullQK subset throughput/peak yet; geometry is covered but different model/tokenizer attention costs remain. Root must recheck shared disk and GPU window before binding.'},
        experience={'source_pair_mean_gain_pp':pair['full_vs_block32']['mean_delta_pp'],
            'source_still_below_e767_pp':original['mean_percent']-control['control_mean_percent'],
            'reason':'Full Q/K improved all8 versus freshblock but removed prior Q/K recovery. Try existing restricted decoder recovery; do not infer final quality from rotation or Qwen outcomes.'},cuda_initialized=False)
    cfg,hf,args,actual = configuration(record)
    require(tuple(actual) == names and record['estimate']['disk_free_at_prepare']>=35000000000,
            'Actual parser scope or minimum output space differs.')
    record['cpu_configuration']={'canonical_config':to_runtime_jsonable(cfg),'hf_args':to_runtime_jsonable(hf),
        'training_args':to_runtime_jsonable(args),'canonical_difference_from_original_llama_recipe':
        ['student_checkpoint_dir','target_modules','run_root_dir']}
    record['stage_templates']=[{'name':n,'argv':a,'allowed_exit_codes':e} for n,a,e in stage_commands(
        checkpoint=OUTPUT/'NATIVE_RUN/trainer_state/checkpoint-400',source_dir=SOURCE,output_dir=OUTPUT/'step_0400')]
    check_inputs(record); require(not torch.cuda.is_initialized(),'CPU preparation initialized CUDA.')
    record['readiness']['cpu']='passed'; OUTPUT.mkdir(); write_json(OUTPUT/'launch.prepared.json',record)
    print({'status':'prepared_unbound','path':str(OUTPUT/'launch.prepared.json'),
        'sha256':sha256(OUTPUT/'launch.prepared.json'),'sources':len(source_pins),'inputs':len(evidence),
        'inventory':inv,'cache_bytes':cache,'cuda_initialized':False},flush=True)


if __name__ == '__main__':
    prepare()
