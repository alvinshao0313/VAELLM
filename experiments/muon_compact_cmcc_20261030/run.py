"""Exactly two complete original OPD updates on the real mixed B interface parent."""
import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback

from transformers import AutoModelForCausalLM

from . import bindings as b
from . import fit, inputs, lifecycle, rollout


@b.torch.no_grad()
def load_pair(record, observe):
    teacher = AutoModelForCausalLM.from_pretrained(str(b.HF),local_files_only=True,trust_remote_code=False,
        torch_dtype=b.torch.bfloat16,attn_implementation='flash_attention_2').eval().requires_grad_(False)
    teacher.seqlen = 2048
    policy = b.core.native_policy(teacher)
    b.require(not any(isinstance(module,b.core.VAELinear) for module in teacher.modules()), 'Teacher must be full HF.')
    model,meta,result,budget = b.checkpoint.load(b.OUTPUT/'fixed2_parent')
    b.require(meta['checkpoint_id'] == record['source_checkpoint_id'] and not result.missing_keys
        and not result.unexpected_keys and b.core.native_policy(model) == policy, 'Mixed student/teacher identity or policy changed.')
    model.to('cuda:0')
    modules = b.selected_modules(model)
    # A real native vs compact consumer comparison records expected cross-kernel BF16 differences.
    native_meta = b.read(b.SOURCE/'checkpoint_meta.json')
    source = b.torch.load(b.SOURCE/native_meta['state_dict_file'],map_location='cpu',weights_only=True,mmap=True)
    spec = next(item for item in native_meta['converted_modules'] if item['name'].endswith('.gate_proj'))
    from experiments.adaptive_native_model_20261029.native_prefix import load_native
    relative = {key[len(spec['name'])+1:]:value for key,value in source.items() if key.startswith(spec['name']+'.')}
    original = load_native(spec,relative).to('cuda:0')
    original._decoder_compute_dtype = b.torch.bfloat16
    original.disable_trainable_decode()
    from litebsq.vae_linear import prime_named_vae_linear_cache
    prime_named_vae_linear_cache([(spec['name'],original)],dtype=b.torch.bfloat16,clear_existing=True,group_size=8)
    sample = b.torch.randn((2,3,original.in_features),device='cuda:0',dtype=b.torch.bfloat16)
    native_output = original(sample)
    compact_output = modules[spec['name']](sample)
    delta = (native_output.float()-compact_output.float()).double()
    consumer = dict(name=spec['name'],scope='Real full original-coordinate gate Linear; native packed vs compact independent-stage BF16',
        relative_L2=float(delta.norm()/native_output.double().norm()),max_abs=float(delta.abs().max()),
        finite=bool(b.torch.isfinite(native_output).all() and b.torch.isfinite(compact_output).all()),
        no_numerical_compensation=True,claim='Cross-kernel conversion difference only; not quality benefit')
    b.require(consumer['finite'] and consumer['relative_L2'] < .03,
        'Material native/compact consumer mismatch beyond the known sub-percent BF16 conversion range.')
    del original,source,relative,sample,native_output,compact_output,delta
    lifecycle.clear(modules)
    gc.collect()
    b.torch.cuda.empty_cache()
    observe('mixed_student_and_full_HF_teacher_loaded',dict(native=216,compact=36,consumer=consumer))
    return model,teacher,dict(native_policy=policy,teacher='Current independently loaded full-HF BF16/FA2',
        source_teacher_nonprojection=record['teacher_identity']['nonprojection_tensors_compared'],
        strict_persistent_tensors=len(model.state_dict()),budget=budget,conversion_consumer=consumer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--timeout-seconds',type=int,default=7200)
    args = parser.parse_args()
    b.require(args.gpu == 6 and os.environ.get('CUDA_VISIBLE_DEVICES') == '6'
        and b.torch.cuda.device_count() == 1 and sys.executable == '/home/cmcc/miniconda3/envs/bitvae/bin/python',
        'Use activated bitvae and the exclusively assigned physical GPU6.')
    prepared = b.read(b.OUTPUT/'prepared.json')
    for binding in prepared['bindings'].values():
        b.require(b.binding(binding['path']) == binding, 'Changed prepared dependency: '+binding['path'])
    for binding in prepared['record']['teacher_identity']['hf_shards']:
        path = b.HF / b.Path(binding['path']).name
        b.require(path.stat().st_size == binding['bytes'], 'HF shard size changed after CPU identity verification.')
    run = b.OUTPUT/'short2'
    b.require(not run.exists(), 'Never overwrite or automatically retry the short2 execution.')
    run.mkdir()
    started = time.monotonic()
    identity = dict(schema='mixed_compact_Muon_short2_v1',source_checkpoint_id=prepared['record']['source_checkpoint_id'],
        original_B=prepared['conversion']['source_checkpoint_id'],seed=137,phase='short2',group_indices=[0,1],
        physical_GPU=6,pid=os.getpid(),argv=sys.argv,python=sys.executable,cwd=os.getcwd(),
        prepared=b.binding(b.OUTPUT/'prepared.json'),contract=b.CONTRACT,
        implementation={str(path.resolve()):b.binding(path) for path in b.HERE.glob('*.py')})
    b.core.write(run/'execution.json',identity)
    observe = lambda event,values: print(json.dumps(dict(event=event,elapsed_seconds=time.monotonic()-started,**values)),flush=True)
    b.torch.set_num_threads(4)
    b.torch.manual_seed(137)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    try:
        dataset,recovery,cache,evidence = inputs.load_inputs()
        record = prepared['record']
        model,teacher,load_audit = load_pair(record,observe)
        tokenizer = b.core.build_tokenizer(str(b.HF))
        training,initial = fit.fit(model,teacher,tokenizer,recovery,record,'short2',observe,started+args.timeout_seconds)
        del teacher,recovery
        gc.collect()
        b.torch.cuda.empty_cache()
        b.torch._C._cuda_clearCublasWorkspaces()
        b.torch.cuda.empty_cache()
        with b.torch.no_grad():
            modules = b.selected_modules(model)
            expected_decoder = {name:p.detach().cpu().bfloat16() for name,p in lifecycle.decoder_parameters(modules).items()}
            training['deployment'] = lifecycle.deployment_mode(model,modules)
            endpoint = lifecycle.copy_parameters(modules)
            from .prepare import exact
            exact(endpoint,expected_decoder,'final FP32 cast vs BF16 endpoint')
            b.require(lifecycle.frozen_proof(model,set(endpoint)) == record['validated_frozen_state'], 'Frozen complement changed at deployment.')
            model.config.use_cache = True
            fixture = dict(input_ids=dataset['training_cases'][0]['input_ids'].unsqueeze(0))
            output = model(input_ids=fixture['input_ids'].to('cuda:0'),use_cache=False,logits_to_keep=1)
            b.require(output.logits.dtype == b.torch.bfloat16 and b.torch.isfinite(output.logits).all().item(), 'Final actual BF16 output invalid.')
            fixture['logits'] = output.logits.cpu()
            fixture['source_case_id'] = dataset['training_cases'][0]['case_id']
            b.torch.save(fixture,run/'same_input_output.pt')
            patch = run/'decoder_endpoint.pt'
            b.torch.save(dict(schema='mixed_compact_Muon_decoder_endpoint_v1',state=endpoint,execution=identity,steps=2),patch)
            output_dir = run/'final_model'
            meta,budget,state = b.checkpoint.save(model,output_dir,provenance=dict(original_parent=b.binding(b.SOURCE/'checkpoint_meta.json'),
                interface_parent=record['source_checkpoint_id'],short2_execution=identity,actual_updates=2,planned_schedule=500,
                validation_only=True,not_eligible_as_quality_or_formal_parent=True))
            expected = {key:b.core.tensor_record(value) for key,value in state.items()}
            b.core.write(run/'expected_state.json',expected)
            exact(endpoint,{key:state[key] for key in endpoint},'serialized endpoint')
            # Restore the actual interface source masters and frozen complement without replacing Parameters.
            parameters = dict(model.named_parameters())
            for key,value in initial.items():
                parameters[key].data.copy_(value.to(parameters[key].device))
            lifecycle.clear(modules)
            exact(lifecycle.copy_parameters(modules),initial,'interface source restoration')
            b.require(lifecycle.frozen_proof(model,set(initial)) == record['validated_frozen_state'], 'Source restoration changed frozen state.')
        report = dict(**identity,status='training_and_BF16_export_complete_fresh_pending',training=training,load_audit=load_audit,
            target_cache=cache,unchanged_M_target_evidence=evidence,source_restored_exact=True,
            saved_decoder_matches_final_FP32_cast_exact=True,endpoint=b.binding(patch),
            export=dict(directory=str(output_dir),checkpoint_id=meta['checkpoint_id'],bit_budget=budget,
                tensors=len(state),expected_state=b.binding(run/'expected_state.json'),fixture=b.binding(run/'same_input_output.pt')),
            elapsed_seconds=time.monotonic()-started)
        b.core.write(run/'report.json',report)
        (run/'run.exit').write_text('0\n',encoding='utf-8')
        observe('mixed_Muon_short2_training_export_complete',dict(report=str(run/'report.json')))
    except BaseException as error:
        b.core.write(run/'failure.json',dict(**identity,error=str(error),traceback=traceback.format_exc()))
        (run/'run.exit').write_text('1\n',encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
