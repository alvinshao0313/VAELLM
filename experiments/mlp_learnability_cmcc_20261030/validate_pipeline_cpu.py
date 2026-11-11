"""Actual full108 installation and ordinary-BF16 export interfaces, CPU only."""
import argparse
import ast
import gc
import os
from pathlib import Path
import time

from . import bindings as b
from . import export,lifecycle as life
from .free_dense import replace_all_mlp,verify_trainable_contract

torch = b.torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'Use CPU-only caller-activated bitvae.')
    b.require(not args.output.exists(), 'Choose a new full-interface CPU evidence directory.')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    started = time.monotonic()
    for path in b.HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    binding = b.source(b.SOURCE_B,prototype_only=True)
    try:
        b.source(b.SOURCE_B,prototype_only=False)
    except ValueError:
        native_quality_rejected = True
    else:
        native_quality_rejected = False
    b.require(native_quality_rejected, 'The native prototype must never pass as formal C0.')
    model,_meta,_budget = b.load_source(binding)
    source_non_mlp = life.non_mlp_proof(model)
    b.install_precision_runtime(model,torch.bfloat16)
    print('strict actual B; full252 CPU native group8 prewarm starts',flush=True)
    warm = b.prewarm_deployment(model)
    references,protection = life.capture_source(model)
    print('complete native deployment caches and108 actual W0 captured',flush=True)
    active,inventory = replace_all_mlp(model,binding)
    verify_trainable_contract(model)
    b.require(life.non_mlp_proof(model) == source_non_mlp
        and all(model.get_submodule(name).source['W0_sha256'] == record['W0_sha256'] for name,record in protection.items()),
        'All108 install changed source W0 or non-MLP frozen state.')
    extra_proof = life.frozen_proof(model,set(active))
    b.require(extra_proof['extra_entries'] == 108, 'Separate Tensor/source-JSON proof must cover all108 custom extra_state entries.')
    print('all108 free-dense FP32 masters and original W0/attention/backbone verified',flush=True)
    deployment,audit = export.build(model,'dense',binding,active,source_non_mlp,protection)
    del model,active,references
    gc.collect()
    directory = args.output/'dense_zero_update_interface'
    meta,budget,state = b.checkpoint.save(deployment,directory,provenance=dict(
        schema='all_MLP_CPU_zero_update_export_interface_v1',prototype_only=True,validation_only=True,
        diagnostic_overbudget=True,actual_updates=0,source_binding=binding,
        scope='CPU installation/serialization only; no short2 GPU or learned endpoint claim.'))
    expected = {name:life.tensor_record(value) for name,value in state.items()}
    del state,deployment
    gc.collect()
    loaded,saved,strict,measured = b.checkpoint.load(directory)
    actual = life.state_records(loaded)
    b.require(not strict.missing_keys and not strict.unexpected_keys and actual['tensors']==expected
        and not actual['extra'] and life.non_mlp_proof(loaded)==source_non_mlp and measured==budget,
        'Canonical ordinary-BF16 dense checkpoint did not reload strictly with all state/source fields exact.')
    for name,record in protection.items():
        module = loaded.get_submodule(name)
        ids = torch.tensor(record['protected_indices'],dtype=torch.int64)
        b.require(type(module) is torch.nn.Linear and module.weight.dtype == torch.bfloat16
            and b.digest({'columns':module.weight.index_select(1,ids)}) == record['protected_BF16_sha256'],
            'Fresh ordinary Linear lost actual protected BF16 columns: '+name)
    b.require(len(saved['native_modules'])==144 and len(saved['pending_dense_modules'])==108
        and not saved['compact_modules'] and not measured['budget_passed'] and not measured['final_budget_applicable'],
        'CPU dense diagnostic must retain the honest144native/108dense overbudget contract.')
    dense_fields = len(expected)
    del loaded,actual,expected
    gc.collect()
    print('canonical dense144native/108ordinary-BF16 strict full state and protection verified',flush=True)
    d_evidence_path = b.OUTPUT/'export_cpu_v1/report.json'
    d_evidence = b.read(d_evidence_path)
    b.require(d_evidence['status']=='actual_B_CPU_decoder_export_build_verified'
        and d_evidence['source_binding']==binding, 'Reuse only the separately completed actual-source D CPU export evidence.')
    removed = []
    for path in sorted(directory.iterdir()):
        b.require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(args.output.resolve()), 'CPU cleanup path escaped this own generated checkpoint.')
        removed.append(dict(path=str(path.resolve()),bytes=path.stat().st_size,
                            reason='Consumed zero-update CPU interface payload; exact-state/budget evidence retained in report.'))
    report = dict(schema='all_MLP_full_interface_CPU_v1',status='verified_actual_B_full_CPU_interfaces_GPU_pending',exit_code=0,
        source_binding=binding,complete_source_strict=True,prototype_B_rejected_as_formal_C0=native_quality_rejected,
        prewarm=warm,all108_original_W0_exact=True,all108_nonprotected_FP32_only=True,
        source_non_mlp_exact=True,custom_extra_state_separately_bound=extra_proof,
        active_elements=sum(row['active_elements'] for row in inventory),dense_export=audit,dense_budget=budget,
        ordinary_dense_fresh_strict_fields=dense_fields,ordinary_dense_protection_exact=True,
        reused_D_export_evidence=b.file_binding(d_evidence_path),
        source_sha256={str(path):b.sha(path) for path in b.HERE.glob('*.py')},cleanup=removed,
        elapsed_seconds=time.monotonic()-started,CUDA_initialized=torch.cuda.is_initialized(),
        scope='CPU full-family materialization/ownership/export/strict state only. CUDA group8 batched numerics, full-model logits/backward, two-GPU short2 and learned quality remain unverified.')
    b.require(not report['CUDA_initialized'], 'CPU interface validation initialized CUDA.')
    b.write(args.output/'report.json',report)
    for row in removed: Path(row['path']).unlink()
    directory.rmdir()
    print('CPU full interfaces verified; removed consumed payload bytes '+str(sum(row['bytes'] for row in removed)),flush=True)


if __name__ == '__main__':
    main()
