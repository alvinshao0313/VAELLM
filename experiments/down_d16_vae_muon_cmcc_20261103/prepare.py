"""CPU actual D16 owners/flat initializations, explicit batch and original suffix binding."""
import ast
from dataclasses import asdict
import difflib
import inspect
import json
from pathlib import Path
import sys
import time

from . import bindings as b
from . import optimizer,evaluate,native_fit
from litebsq.llm_vae import MultiLayerVAE
from experiments.adaptive_native_model_20261029.training_checkpoint import MixedCatTrainer


def main():
    b.cpu_environment();b.torch.set_num_threads(4)
    b.require(not b.PREPARED.exists() and not b.OUTPUT.exists(),'Do not overwrite preparation or an earlier candidate.')
    started=time.monotonic();source=b.sources();controls=b.controls()
    parsed,identity,baseline=b.recipe(b.OUTPUT/'formal')
    _activation,_plan,restored=b.base.restore_cat_runtime_state(parsed[-1],access_token=parsed[1].access_token)
    b.require(restored==baseline,'Restore the genuine original calibration runtime unchanged.')
    old=b.read(b.base.RESULT/'prepared.json')
    b.require(old['source_binding']==source,'Reuse frozen-prefix/live target evidence only for the identical actual after-up.')
    args=b.stage_args(parsed);b.recovery.native.set_seed(31)
    initializations=[];layout=None
    # The genuine interstage encoding/extraction can construct decoder objects.
    # A second bare CPU constructor is not the real stage2 RNG path; bind its
    # original digest here and compare both actual factories in the GPU short.
    for stage in range(1):
        model=MultiLayerVAE(args)
        named,layout,affine=optimizer.derive_layout(model)
        digest=b.c.digest({f'group0.parameter{i}':value.detach() for i,value in enumerate(model.parameters())})
        b.require(digest==controls['D16_initializations'][stage]['initial_parameter_digest'],
            'CPU canonical D16 flat initialization differs from its actual original D16 Adam stage.')
        b.require(sum(value.numel() for value in named.values())==397728
            and sum(named[name].numel() for name in layout)==368640
            and sum(named[name].numel() for name in affine)==29088
            and [spec['matrix_shape'] for spec in layout.values()]==[[36,128,16],[36,24,128],[36,128,24],[36,16,128]],
            'Use the real D16 397728 owners and144 independent matrices, not D32 constants.')
        initializations.append(dict(stage=stage,digest=digest,D16_control_exact=True,parameter_elements=397728,
            matrix_elements=368640,auxiliary_elements=29088))
    original=b.ORIGINAL_MUV_OPTIMIZER.read_text(encoding='utf-8');actual=Path(optimizer.__file__).read_text(encoding='utf-8')
    expected=original.replace("SCHEMA='cat_down_vae_muon_bf16_v1'","SCHEMA='down_D16_vae_muon_bf16_v1'").replace('==767808','==397728')
    b.require(actual==expected,'Mature Muon math/aux/state/precision/calibration code must remain exact; only schema and D16 count change.')
    source_text=Path(native_fit.__file__).read_text(encoding='utf-8')
    b.require(source_text==(b.REPO/'experiments/down_geometry_baseline_cmcc_20261103/fit.py').read_text(encoding='utf-8')
        and 'batch_size=settings[\'batch\']' in source_text and b.SETTINGS['batch']==16384,
        'The genuine D16 fit body must use explicit16384, not original fit_next hard8192.')
    parsed2,_identity,_baseline,stage=b.recovery_recipe(b.OUTPUT/'recovery')
    reference=b.torch.load(b.REFERENCE_ARGS,map_location='cpu',weights_only=False)
    cls,kwargs=MixedCatTrainer.get_optimizer_cls_and_kwargs(reference)
    kwargs={key:value for key,value in kwargs.items() if key not in ('params','model','optimizer_dict')}
    b.require(cls is b.torch.optim.AdamW and json.loads(json.dumps(kwargs))==dict(lr=1e-4,betas=[.9,.999],eps=1e-8)
        and stage.config.opt.resolved_decoder_lr()==1e-5 and stage.stage_seed==37 and stage.config.data.data_seed==31,
        'The downstream actual original AdamW/seed/data contract changed.')
    from experiments.cat_down_vae_muon_closure_cmcc_20261103 import evaluate as previous_evaluate
    b.require(inspect.getsource(evaluate.bound_longbench_plan)==inspect.getsource(previous_evaluate.bound_longbench_plan),
        'Reuse the proven absolute deadline exactly, including detached native/outer bounds and original grace.')
    b.require(not any(name.startswith('experiments.down_d8_neighbor_cmcc_20261103') for name in sys.modules),
        'D16 must use its original native consumer, without importing the D8 tile extension.')
    for path in b.HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    original_cpu=b.TOPIC/'cat_down_vae_muon_cmcc_20261103/prepared.json'
    record=dict(schema='D16_Muon_prepared_v1',status='CPU_verified_D16_Muon_shape_batch_initialization_and_suffix',
        command=[sys.executable,*sys.argv],created_unix=time.time(),source_binding=source,controls=controls,contract=b.CONTRACT,
        live_down_source_digest=old['live_down_source_digest'],frozen_prefix_digest=old['frozen_prefix_digest'],
        reused_frozen_prefix_evidence=str(b.base.RESULT/'prepared.json'),reused_frozen_prefix_sha256=b.sha(b.base.RESULT/'prepared.json'),
        actual_D16_CPU_initializations=initializations,stage2_initialization_bound_for_actual_short=controls['D16_initializations'][1],
        layout=layout,actual_explicit_vector_batch=16384,actual_independent_matrices=144,
        mature_Muon_math_aux_state_reused=dict(CPU_evidence=str(original_cpu),CPU_evidence_sha256=b.sha(original_cpu),
            source_sha256=b.sha(b.ORIGINAL_MUV_OPTIMIZER),exact_math_aux_state_source_equal=True,
            only_diff='Own state schema and total owner count767808->397728; no repeated generic NS/aux/state audit.'),
        optimizer_unified_diff=''.join(difflib.unified_diff(original.splitlines(True),actual.splitlines(True))),
        runtime_identity=identity,original_runtime_identity=baseline,category_configs={name:asdict(cfg) for name,cfg in parsed[-2].items()},
        original_recovery_stage=asdict(stage),original_recovery_optimizer=dict(cls=cls.__module__+'.'+cls.__name__,
            constructor_kwargs=json.loads(json.dumps(kwargs)),decoder_lr=1e-5,weight_decay=0.,FP32_decoder_leaves=1728,
            FP32_decoder_elements=controls['D16_raw_budget']['components']['decoder']['budget_bits']//16,fresh_state=True),
        original_recovery_all_fields_JSON_normalized_exact_except_output_dir=True,
        absolute_deadline_implementation_exact_to_proven_MuV_closure=True,
        CUDA_initialized=b.torch.cuda.is_initialized(),source_sha256=b.pins(),elapsed_seconds=time.monotonic()-started,
        boundary='CPU real D16 owner/batch/initialization/unchanged mature math and suffix proof. Actual new-shape short and GPU kappa/peak/fresh still required; no control weights or recovery state used.')
    b.require(not record['CUDA_initialized'],'CPU evidence initialized CUDA.')
    b.OUTPUT.mkdir(parents=True);b.write(b.PREPARED,record)
    print(json.dumps(dict(status=record['status'],seconds=record['elapsed_seconds'],initializations=initializations,
        total=397728,matrices=368640,auxiliary=29088,CUDA_initialized=False)),flush=True)


if __name__=='__main__':main()
