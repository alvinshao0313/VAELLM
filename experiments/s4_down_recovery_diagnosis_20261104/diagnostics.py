"""Fixed early/late train panels and disjoint auxiliary-train gold observations."""
import hashlib
from pathlib import Path
from . import bindings as b
from experiments.mlp_m4096_cmcc_20261101 import pool_build
from experiments.s4_from_bf16_20261102.recovery import panel


def prepare(root,dataset):
    from datasets import Dataset
    from transformers import AutoTokenizer
    from lm_eval.evaluator_utils import get_task_list
    from lm_eval.tasks import TaskManager,get_task_dict
    captured=b.read(b.data.b.DATA/'dataset_prepared.json')['source_capture']
    b.data.checked(captured)
    capture=b.read(captured['path']);source=Path(captured['path']).parent
    b.require(capture['revision']=='c30699e8356da336a370243923dbaf21066bb9fe'
        and capture['repository']=='cais/mmlu' and capture['config']=='auxiliary_train' and capture['split']=='train',
        'Require the exact official retained train capture.')
    for record in capture['files'].values():b.data.checked(record)
    auxiliary=Dataset.from_parquet([str(source/name) for name in capture['selected_parquet']],cache_dir=str(root/'aux_cache'))
    b.require(len(auxiliary)==99842 and auxiliary.column_names==['train'],'Original official row count/order changed.')
    for case in dataset['training_cases'][:4096]:
        doc=dict(auxiliary[case['source']['source_row_0based']]['train'])
        b.require(pool_build.data.digest(doc)==case['source']['doc_sha256'],'Rebuilt source differs from an actual training4096 document.')
    generated=list((root/'aux_cache').rglob('*.arrow'));b.require(len(generated)==1,'Require one actual rebuilt auxiliary Arrow.')
    arrow=b.binding(generated[0])
    excluded=set(dataset['train_ids'][:4096])
    ranked=sorted((pool_build.fit_data.rank(f'cais_mmlu/auxiliary_train/train/{i}'),i)
        for i in range(len(auxiliary)) if f'cais_mmlu/auxiliary_train/train/{i}' not in excluded)[:128]
    task=get_task_list(get_task_dict(['mmlu_abstract_algebra'],TaskManager()))[0].task
    tokenizer=AutoTokenizer.from_pretrained(str(b.HF),use_fast=True,local_files_only=True)
    codec=pool_build.native.PairTokenizer(tokenizer);cases=[];selection=[]
    for rank,(hash_value,index) in enumerate(ranked):
        doc=dict(auxiliary[index]['train']);case_id=f'cais_mmlu/auxiliary_train/train/{index}'
        b.require(doc['subject']=='' and len(doc['choices'])==4 and case_id not in excluded,'Official held-out source/selection changed.')
        context=task.doc_to_text(doc)
        native=pool_build.native.native_variant(task,doc,context,codec,'cais_mmlu_auxiliary_train_native_no_subject',index)
        pairs=native['pairs'];ids=pool_build.data.cpu_ids(pairs[0]['context_ids']);gold=doc['answer']
        b.require([p['continuation_ids'] for p in pairs]==[[362],[425],[356],[422]],'Held-out native option token boundary changed.')
        case=pool_build.data.finish_case(dict(domain='mmlu',case_id=case_id,split='auxiliary_train_untrained_diagnostic',
            input_ids=ids,prompt_input_ids=ids,native_continuation_ids=pool_build.data.cpu_ids([p['continuation_ids'] for p in pairs]),
            native_arguments=[p['arguments'] for p in pairs],gold_index=gold,target_ids=pool_build.data.cpu_ids(pairs[gold]['continuation_ids']),
            answer_query_positions=pool_build.data.cpu_ids([len(ids)-1]),answer_target_positions=pool_build.data.cpu_ids([len(ids)]),
            source=dict(repository='cais/mmlu',config='auxiliary_train',official_split='train',source_split='train',
                revision='c30699e8356da336a370243923dbaf21066bb9fe',source_row_0based=index,subject='',data_path=arrow['path'],
                source_identity=['cais/mmlu','auxiliary_train','train',index],variant='native_no_subject',
                doc_sha256=pool_build.data.digest(doc)),evidence_location=None))
        cases.append(case);selection.append(dict(case_id=case_id,hash=hash_value,rank=rank,input_signature=case['input_signature']))
    b.require(len(cases)==len({c['case_id'] for c in cases})==128,'Exactly128 unique untrained auxiliary cases required.')
    b.torch.save(cases,root/'heldout.pt')
    record=dict(cases=b.binding(root/'heldout.pt'),selection=selection,excluded_training_count=4096,
        excluded_ids_sha256=hashlib.sha256('\n'.join(sorted(excluded)).encode()).hexdigest(),
        official_source_capture=captured,actual_Arrow=arrow,actual_source_rows=99842,training4096_documents_exact=True,
        formal_test_used=False,teacher_used_for_selection=False,
        scope='Untrained same-source auxiliary_train diagnostic; not held-out57-subject test coverage')
    b.write(root/'diagnostic_data.json',record)
    return record


@b.torch.no_grad()
def gold(model,cases,baseline=None):
    values={};total_nll=0.;correct=0
    for start in range(0,len(cases),32):
        batch=cases[start:start+32];b.require(len(batch)==32,'Keep actual native M32 diagnostics.')
        prediction=b.data.mmlu_log_distribution(model,batch).cpu()
        for i,case in enumerate(batch):
            nll=float(-prediction[i,case['gold_index']]);ok=bool(prediction[i].argmax()==case['gold_index'])
            values[case['case_id']]=dict(correct=ok,gold_NLL=nll);total_nll+=nll;correct+=int(ok)
    result=dict(count=len(cases),M_correct=correct,M_accuracy=correct/len(cases),M_gold_NLL=total_nll/len(cases),M_cases=values)
    if baseline is not None:
        b.require(values.keys()==baseline['M_cases'].keys(),'Diagnostic identity changed.')
        result.update(M_repairs=sum(not baseline['M_cases'][k]['correct'] and v['correct'] for k,v in values.items()),
            M_damages=sum(baseline['M_cases'][k]['correct'] and not v['correct'] for k,v in values.items()))
    return result


@b.torch.no_grad()
def evaluate(model,dataset,targets,heldout,baseline=None,limited=False):
    mode=model.training;model.eval()
    early=panel(model,dataset,targets,None if baseline is None else baseline['early'])
    indices=b.CONTRACT['late_groups'][:1] if limited else b.CONTRACT['late_groups']
    late=[targets[k]['case'] for i in indices for k in dataset['groups'][i]['case_ids'][:32]]
    result=dict(early=early,late=gold(model,late,None if baseline is None else baseline['late']),
        heldout=gold(model,heldout[:32] if limited else heldout,None if baseline is None else baseline['heldout']),
        limited_validation=limited,scope='Early/late are training diagnostics; heldout is untrained same-source auxiliary, never an optimizer input',
        QA_diagnostic_presentations=32,QA_diagnostic_unique=16,QA_training_exposure='Separate in training.json; diagnostic forwards are not optimizer examples')
    model.train(mode)
    return result
