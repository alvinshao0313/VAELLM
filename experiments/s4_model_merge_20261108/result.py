"""Paired complete-eight results against both donors, with generalization boundary."""
import math
from pathlib import Path
from . import bindings as b
from experiments.s4_qk_source_weights_20261107.result import paired_samples


def summarize(request_path,root):
    from experiments.w25_eval.model_inventory import model_inventory
    from experiments.w25_eval.acceptance import collect_results
    from experiments.w25_eval.transfer_nine_task_acceptance import _sample_artifacts
    request=b.read(request_path);root=Path(root);checkpoint=root/'post/model';meta=b.read(checkpoint/'checkpoint_meta.json')
    paths=[root/'post/evaluation'/(x+'.json') for x in ('mmlu','seven')];inventory=model_inventory(b.HF)
    current=collect_results(paths,inventory,checkpoint=(checkpoint,meta))
    scores={k:100*v for k,v in current['task_metrics'].items()};mean=math.fsum(scores.values())/8
    donors={}
    for step in (64,128):
        original=b.donor_directory(step);record=request['donors'][str(step)]
        b.require(b.binding(original/'result.json')==record['result'],'Reused donor result changed.')
        donor_paths=[original/'post/evaluation'/(part+'.json') for part in ('mmlu','seven')]
        b.require(all(b.binding(p)==record['evaluation'][p.stem] for p in donor_paths),'Donor evaluation changed.')
        parent=collect_results(donor_paths,inventory,checkpoint=(original/'post/model',b.read(original/'post/model/checkpoint_meta.json')))
        b.require(parent['versions']==current['versions'],'Evaluation version differs from donors.')
        old=b.read(original/'result.json');pair=paired_samples(donor_paths,paths)
        b.require(pair['tasks']['mmlu']['left_correct']==b.DONORS[step]['correct'],'Donor M count changed.')
        donors[str(step)]=dict(checkpoint_id=old['checkpoint_id'],paired=pair,
            scores_percent=old['scores_percent'],eight_mean_percent=old['eight_mean_percent'],
            deltas_pp={k:scores[k]-old['scores_percent'][k] for k in scores},eight_mean_delta_pp=mean-old['eight_mean_percent'])
    result=dict(status='complete_merge_eight_screen_not_generalization_or_final_acceptance',arm=request['arm'],
        ratio_late=request['recipe']['ratio_late'],checkpoint_id=meta['checkpoint_id'],scores_percent=scores,
        eight_mean_percent=mean,M_correct=donors['64']['paired']['tasks']['mmlu']['right_correct'],
        donor_comparisons=donors,raw_samples=_sample_artifacts(paths),request=b.binding(request_path),
        merge=b.binding(root/'post/merge.json'),fresh=b.binding(root/'post/evaluation/fresh.json'),
        contract=b.CONTRACT,generalization_warning=b.CONTRACT['generalization'],
        next='Report all three fixed ratios; no automatic ratio refinement, training, LongBench or final algorithm adoption')
    b.write(root/'result.json',result);return result
