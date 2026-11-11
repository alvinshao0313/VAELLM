"""Same-budget comparisons with original per-question identities, no quality gate."""
import fcntl
import math
from pathlib import Path
from . import bindings as b
from experiments.s4_qk_source_weights_20261107.result import paired_samples


def paths(directory):
    return [directory/'post/evaluation'/(name+'.json') for name in ('mmlu','seven')]


def refresh():
    results={};comparisons={};text=['# S4四臂64／128阶段报告','',
        '| 步数 | 臂 | MMLU正确／14042 | MMLU% | 八项均值% |','| --- | --- | --- | --- | --- |']
    for step in (64,128):
        available={}
        for arm in b.ARMS:
            directory=b.RUN/arm/f'step_{step:06d}';file=directory/'result.json'
            if not file.exists():continue
            item=b.read(file);available[arm]=item;results[f'{arm}/{step}']=b.binding(file)
            text.append(f"| {step} | {arm} | {item['paired_parent']['tasks']['mmlu']['right_correct']} | {item['scores_percent']['mmlu']:.8f} | {item['eight_mean_percent']:.8f} |")
        if 'A' in available:
            for arm,item in available.items():
                if arm=='A':continue
                base=available['A'];pair=paired_samples(paths(b.RUN/'A'/f'step_{step:06d}'),paths(b.RUN/arm/f'step_{step:06d}'))
                comparisons[f'{arm}_vs_A/{step}']=dict(paired=pair,
                    deltas_pp={k:item['scores_percent'][k]-base['scores_percent'][k] for k in item['scores_percent']},
                    eight_mean_delta_pp=item['eight_mean_percent']-base['eight_mean_percent'])
    status='complete_eight_endpoints' if len(results)==8 else 'partial_completed_endpoints'
    b.write(b.RUN/'batch_results.json',dict(status=status,results=results,comparisons=comparisons,
        boundary='Single seed; same-budget A comparisons; no LongBench/final acceptance/automatic successor'))
    text.extend(['',f'状态：{status}。完整逐题配对、逐项差值及证据见 batch_results.json。',
        '128为主比较，64为短程节点；八项不替代九项验收。'])
    (b.RUN/'stage_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def summarize(request_path,directory,step):
    from experiments.w25_eval.model_inventory import model_inventory
    from experiments.w25_eval.acceptance import collect_results
    from experiments.w25_eval.transfer_nine_task_acceptance import _sample_artifacts
    request=b.read(request_path);directory=Path(directory);checkpoint=directory/'post/model'
    meta=b.read(checkpoint/'checkpoint_meta.json');inventory=model_inventory(b.HF)
    current=collect_results(paths(directory),inventory,checkpoint=(checkpoint,meta))
    parent_paths=list(b.PARENT_EIGHT.values())
    parent=collect_results(parent_paths,inventory,checkpoint=(b.SOURCE,b.read(b.SOURCE/'checkpoint_meta.json')))
    b.require(current['versions']==parent['versions'],'Task versions changed.')
    b.require(all(b.binding(path)==request['parent_eight'][name] for name,path in b.PARENT_EIGHT.items()),
        'Parent eight binding changed.')
    scores={k:100*v for k,v in current['task_metrics'].items()};old={k:100*v for k,v in parent['task_metrics'].items()}
    b.require(len(scores)==8 and scores.keys()==old.keys(),'Complete eight required.')
    pair=paired_samples(parent_paths,paths(directory));b.require(pair['tasks']['mmlu']['left_correct']==10007,'Parent M count changed.')
    result=dict(status='complete_eight_not_final_acceptance',step=step,arm=request['arm'],checkpoint_id=meta['checkpoint_id'],
        scores_percent=scores,eight_mean_percent=math.fsum(scores.values())/8,
        parent_percent=old,parent_eight_mean_percent=math.fsum(old.values())/8,
        deltas_parent_pp={k:scores[k]-old[k] for k in scores},paired_parent=pair,raw_samples=_sample_artifacts(paths(directory)),
        training=b.binding(directory/'training/training.json'),export=b.binding(directory/'post/export.json'),
        fresh=b.binding(directory/'post/evaluation/fresh.json'),request=b.binding(request_path),contract=b.CONTRACT)
    if step==128:
        previous=directory.parent/'step_000064'
        result['paired_previous64']=paired_samples(paths(previous),paths(directory))
        earlier=b.read(previous/'result.json')
        result['deltas_previous64_pp']={k:scores[k]-earlier['scores_percent'][k] for k in scores}
        result['eight_mean_delta_previous64_pp']=result['eight_mean_percent']-earlier['eight_mean_percent']
    with (b.RUN/'.result.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);b.write(directory/'result.json',result);refresh()
    return result
