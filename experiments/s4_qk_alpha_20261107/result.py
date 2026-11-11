"""Complete paired eight-task outcomes against the common parent and fixed QK alpha.5."""
import json
import math
from pathlib import Path
from . import bindings as b


def paired_samples(left_paths,right_paths):
    from experiments.w25_eval.acceptance import TASK_METRICS
    tables=[]
    for paths in (left_paths,right_paths):
        tasks={}
        for path in paths:
            source=b.read(path)['evals']['lm_eval']['sample_artifacts']['tasks']
            b.require(not tasks.keys()&source.keys(),'Duplicate sample task shard.')
            tasks.update(source)
        tables.append(tasks)
    left,right=tables;b.require(left.keys()==right.keys() and len(left)==64,'Require full57 M and seven sample leaves.')
    totals={};subjects=0
    for task in sorted(left):
        rows=[]
        for record in (left[task],right[task]):
            b.require(b.sha(record['path'])==record['sha256'],'Paired raw samples changed.')
            with Path(record['path']).open(encoding='utf-8') as stream:values=[json.loads(line) for line in stream]
            mapping={row['doc_id']:row for row in values}
            b.require(len(values)==len(mapping)==record['count'],'Sample count/unique ID changed.')
            rows.append(mapping)
        a,c=rows;b.require(a.keys()==c.keys(),'Paired sample IDs differ.')
        m=task.startswith('mmlu_');key='mmlu' if m else task;metric='acc' if m else TASK_METRICS[task].split(',')[0]
        if m:subjects+=1
        item=totals.setdefault(key,dict(samples=0,left_correct=0,right_correct=0,repaired=0,damaged=0,metric=metric))
        for index,old in a.items():
            new=c[index]
            b.require(all(old[field]==new[field] for field in ('doc_hash','prompt_hash','target_hash')),
                'Paired doc/prompt/target identity differs.')
            b.require(old[metric] in (0,1) and new[metric] in (0,1),'Paired correctness must use the official binary metric.')
            before,after=bool(old[metric]),bool(new[metric])
            item['samples']+=1;item['left_correct']+=int(before);item['right_correct']+=int(after)
            item['repaired']+=int(not before and after);item['damaged']+=int(before and not after)
    b.require(subjects==57 and len(totals)==8 and totals['mmlu']['samples']==14042
        and sum(item['samples'] for item in totals.values())==24742,'Full original sample coverage changed.')
    for item in totals.values():
        item['net_correct']=item['repaired']-item['damaged']
        b.require(item['net_correct']==item['right_correct']-item['left_correct'],'Paired net count differs.')
    return dict(tasks=totals,subjects=subjects,total_samples=24742,doc_prompt_target_hashes_exact=True)


def summarize(request_path,root):
    from experiments.w25_eval.model_inventory import model_inventory
    from experiments.w25_eval.acceptance import collect_results
    from experiments.w25_eval.transfer_nine_task_acceptance import _sample_artifacts
    request=b.read(request_path);root=Path(root);checkpoint=root/'post/model';meta=b.read(checkpoint/'checkpoint_meta.json')
    paths=[root/'post/evaluation'/(name+'.json') for name in ('mmlu','seven')]
    parent_paths=list(b.PARENT_EIGHT.values());control_paths=list(b.CONTROL_EIGHT.values())
    b.require(all(b.binding(path)==request['parent_eight'][name] for name,path in b.PARENT_EIGHT.items())
        and all(b.binding(path)==request['control_eight'][name] for name,path in b.CONTROL_EIGHT.items()),'Matched reference eight artifacts changed.')
    inventory=model_inventory(b.HF)
    current=collect_results(paths,inventory,checkpoint=(checkpoint,meta))
    parent=collect_results(parent_paths,inventory,checkpoint=(b.SOURCE,b.read(b.SOURCE/'checkpoint_meta.json')))
    control_model=b.CONTROL_ROOT/'post/model'
    control=collect_results(control_paths,inventory,checkpoint=(control_model,b.read(control_model/'checkpoint_meta.json')))
    b.require(current['versions']==parent['versions']==control['versions'],'Eight-task versions differ.')
    scores={k:100*v for k,v in current['task_metrics'].items()}
    old={k:100*v for k,v in parent['task_metrics'].items()};half={k:100*v for k,v in control['task_metrics'].items()}
    b.require(set(scores)==set(old)==set(half) and len(scores)==8,'Eight-task inventory differs.')
    paired_parent=paired_samples(parent_paths,paths);paired_half=paired_samples(control_paths,paths)
    b.require(paired_parent['tasks']['mmlu']['left_correct']==10007 and paired_half['tasks']['mmlu']['left_correct']==10004,
        'Original parent or QK alpha.5 M count changed.')
    result=dict(status='complete_fixed64_QK_alpha_eight_not_final_acceptance',checkpoint_id=meta['checkpoint_id'],arm=request['arm'],
        teacher_alpha=request['teacher_alpha'],teacher_gold_mixture=request['contract']['teacher_gold_mixture'],
        scores_percent=scores,eight_mean_percent=math.fsum(scores.values())/8,parent_percent=old,
        parent_eight_mean_percent=math.fsum(old.values())/8,deltas_pp={k:scores[k]-old[k] for k in scores},
        control_half_percent=half,control_half_eight_mean_percent=math.fsum(half.values())/8,
        deltas_vs_half_pp={k:scores[k]-half[k] for k in scores},paired_parent=paired_parent,paired_half=paired_half,
        raw_samples=_sample_artifacts(paths),parent_raw_samples=_sample_artifacts(parent_paths),control_raw_samples=_sample_artifacts(control_paths),
        contract=request['contract'],data_contract=request['data_contract'],
        diagnostic_boundary='Each alpha own0-to64 mixed KL; shared gold NLL/hits and complete eight comparable',
        next='Report both fixed64 alpha endpoints vs QK.5; no automatic extension, LongBench or other target/data change')
    b.write(root/'result.json',result);return result
