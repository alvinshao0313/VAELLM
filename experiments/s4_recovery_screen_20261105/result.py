"""Same-UUID complete eight, raw sample coverage, and paired parent MMLU."""
import json
import math
from pathlib import Path
from . import bindings as b


def paired_mmlu(parent_path,current_path):
    parent=b.read(parent_path)['evals']['lm_eval']['sample_artifacts']['tasks']
    current=b.read(current_path)['evals']['lm_eval']['sample_artifacts']['tasks']
    b.require(set(parent)==set(current) and len(parent)==57,'Complete57 MMLU subjects required.')
    repaired=damaged=parent_correct=current_correct=total=0
    for task in sorted(parent):
        rows=[]
        for record in (parent[task],current[task]):
            b.require(b.sha(record['path'])==record['sha256'],'Paired raw M samples changed.')
            with Path(record['path']).open(encoding='utf-8') as stream:
                values=[json.loads(line) for line in stream]
            b.require(len(values)==record['count'],'Raw M subject count changed.')
            mapping={row['doc_id']:row for row in values}
            b.require(len(mapping)==len(values),'Duplicate M doc ID.')
            rows.append(mapping)
        a,c=rows;b.require(a.keys()==c.keys(),'Parent/current M identities differ.')
        for key,left in a.items():
            right=c[key]
            b.require(all(left[field]==right[field] for field in ('doc_hash','prompt_hash','target_hash')),
                'Parent/current M document/prompt/target differs.')
            old,new=bool(left['acc']),bool(right['acc'])
            repaired+=not old and new;damaged+=old and not new
            parent_correct+=old;current_correct+=new;total+=1
    b.require(total==14042 and parent_correct==10007,'Original parent complete M count changed.')
    return dict(subjects=57,samples=total,parent_correct=parent_correct,current_correct=current_correct,
        repaired=repaired,damaged=damaged,doc_prompt_target_hashes_exact=True)


def summarize(request_path,root):
    from experiments.w25_eval.model_inventory import model_inventory
    from experiments.w25_eval.acceptance import collect_results
    from experiments.w25_eval.transfer_nine_task_acceptance import _sample_artifacts
    request=b.read(request_path);root=Path(root);checkpoint=root/'post/model';meta=b.read(checkpoint/'checkpoint_meta.json')
    paths=[root/'post/evaluation'/(name+'.json') for name in ('mmlu','seven')]
    parent_paths=[b.PARENT_EIGHT[name] for name in ('mmlu','seven')]
    b.require(all(b.binding(path)==request['parent_eight'][name] for name,path in b.PARENT_EIGHT.items()),'Parent eight artifacts changed.')
    inventory=model_inventory(b.HF)
    current=collect_results(paths,inventory,checkpoint=(checkpoint,meta))
    old_meta=b.read(b.SOURCE/'checkpoint_meta.json')
    parent=collect_results(parent_paths,inventory,checkpoint=(b.SOURCE,old_meta))
    dense=collect_results([b.parent.BASELINE/'mmlu.json',b.parent.BASELINE/'seven.json'],inventory)
    b.require(current['versions']==parent['versions']==dense['versions'],'Eight-task versions differ.')
    scores={k:100*v for k,v in current['task_metrics'].items()}
    old={k:100*v for k,v in parent['task_metrics'].items()}
    b.require(set(scores)==set(old) and len(scores)==8,'Actual eight-task inventory differs.')
    result=dict(status='complete_fixed64_scope_screen_eight_not_final_acceptance',checkpoint_id=meta['checkpoint_id'],arm=request['arm'],
        scores_percent=scores,eight_mean_percent=math.fsum(scores.values())/8,parent_percent=old,
        parent_eight_mean_percent=math.fsum(old.values())/8,deltas_pp={k:scores[k]-old[k] for k in scores},
        paired_M=paired_mmlu(parent_paths[0],paths[0]),raw_samples=_sample_artifacts(paths),
        parent_raw_samples=_sample_artifacts(parent_paths),contract=request['contract'],
        next='Report both fixed64 endpoints; no automatic extension or LongBench')
    b.write(root/'result.json',result)
    return result
