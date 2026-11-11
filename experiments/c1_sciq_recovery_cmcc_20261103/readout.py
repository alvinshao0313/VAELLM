"""Fixed first32 SciQ and first32 retained-aux readout on raw/control/new; 192 B1 FW."""
from collections import defaultdict
import gc
import json
import os
import time

from . import bindings as b
from experiments.muon_compact_cmcc_20261030 import rollout


def main():
    b.configure()
    b.require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '4' and b.torch.cuda.device_count() == 1,
        'Use only assigned GPU4 for the fixed readout.')
    b.executor.check_resources(['4'])
    prepared = b.read_prepared(b.PREPARED)
    b.require(not (b.EVIDENCE/'readout.json').exists(), 'Never replace a readout.')
    b.torch.set_num_threads(4)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    dataset = b.inputs.read_dataset()
    by_id = {case['case_id']:case for case in dataset['training_cases']}
    selected = dataset['evidence']['readout_case_ids']
    teacher = b.torch.load(b.OUTPUT/'teacher/targets.pt',map_location='cpu',weights_only=True)['targets']
    composed = b.read(b.original.inputs.HOT/'teacher_targets.json')
    component = composed['components'][0]
    pure_aux = b.torch.load(b.original.inputs.checked(component['payload_binding']),
        map_location='cpu',weights_only=True,mmap=True)['targets']
    models = dict(raw=b.original.SOURCE/'final_model',
        control=b.original.OUTPUT/'C1_H96_Muon32/formal32/final_model',candidate=b.OUTPUT/'formal32/final_model')
    control_acceptance = b.read(b.original.EVIDENCE/'evaluation/longbench/merged/nine_task_acceptance.json')
    b.require(control_acceptance['checkpoint_id'] == b.CONTROL_ID
        and control_acceptance['compressed_nine_task_score'] == 64.89598790075989, 'Use designated completed original Muon32 control.')
    started,results = time.monotonic(),{}
    for name,directory in models.items():
        model,meta,strict,budget = b.checkpoint.load(directory)
        expected = dict(raw=b.PARENT_ID,control=b.CONTROL_ID,
            candidate=b.read(b.OUTPUT/'formal32/report.json')['export']['checkpoint_id'])[name]
        b.require(meta['checkpoint_id'] == expected and not strict.missing_keys and not strict.unexpected_keys
            and all(budget[key] == prepared['source_budget'][key] for key in
                ('user_linear_bits','packed_linear_bits','original_linear_weights','full_model_tensor_bytes')),
            'Readout must use genuine same-topology raw/control/new endpoint.')
        frozen_before = b.lifecycle.frozen_proof(model,set())
        model.to('cuda:0').eval().requires_grad_(False)
        b.lifecycle.prewarm(b.mature.selected_modules(model))
        rows = []
        with b.torch.no_grad():
            for domain,keys in selected.items():
                for key in keys:
                    case = by_id[key]
                    logits = b.mature.base.answer_logits(model,case)
                    logp = b.mature.base.log_distribution(logits,case)
                    target = teacher[key]['pure_teacher_log_probs'] if domain == 'sciq' else pure_aux[key]['target_log_probs']
                    kl = b.torch.nn.functional.kl_div(logp,target.to(logp.device),log_target=True,reduction='sum')
                    b.require(b.torch.isfinite(logp).all() and b.torch.isfinite(kl), 'Nonfinite fixed readout.')
                    rows.append(dict(domain=domain,case_id=key,input_signature=case['input_signature'],
                        conditional_teacher_KL=float(kl),conditional_gold_CE=float(-logp[0,case['gold_index']]),
                        correct=int(logp.argmax(-1).item() == case['gold_index']),input_tokens=len(case['input_ids'])))
                    del logits,logp,kl
        b.require(b.lifecycle.frozen_proof(model,set()) == frozen_before, 'Readout changed actual source state.')
        summary = {domain:dict(cases=32,conditional_teacher_KL=sum(row['conditional_teacher_KL'] for row in rows if row['domain']==domain)/32,
            conditional_gold_CE=sum(row['conditional_gold_CE'] for row in rows if row['domain']==domain)/32,
            accuracy_percent=sum(row['correct'] for row in rows if row['domain']==domain)*100/32)
            for domain in selected}
        results[name] = dict(checkpoint_id=expected,directory=str(directory),rows=rows,summary=summary,state_unchanged=True)
        print(json.dumps(dict(event='fixed_readout_complete',model=name,checkpoint_id=expected,summary=summary)),flush=True)
        del model
        gc.collect()
        b.torch.cuda.empty_cache()
        b.torch._C._cuda_clearCublasWorkspaces()
        b.torch.cuda.empty_cache()
    b.core.write(b.EVIDENCE/'readout.json',dict(schema='C1_SciQ_fixed_train_readout_v1',status='complete',
        selected_case_ids=selected,selection=dataset['evidence']['readout_selection'],B1_forward_calls=192,
        physical_GPU=4,results=results,elapsed_seconds=time.monotonic()-started,
        boundary='Train fitting diagnostic, not held-out transfer evidence; adoption uses same-UUID complete-nine and formal MMLU.'))


if __name__ == '__main__':
    main()
