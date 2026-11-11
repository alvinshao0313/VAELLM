"""Same-UUID unrounded nine-task scores and the explicit S4 quality thresholds."""
import argparse
import math
from pathlib import Path
from . import bindings as b
from experiments.compare_longbench_v1 import comparison
from experiments.w25_eval.acceptance import collect_results
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.transfer_nine_task_acceptance import _sample_artifacts
from train_utils.checkpoint_bpw import audit_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();b.initialize()
    request=b.read(args.request);b.check(request);meta=b.read(b.SOURCE/'checkpoint_meta.json');inventory=model_inventory(b.HF)
    paths=[Path(request['eight'][part]['path']) for part in ('mmlu','seven')]
    current=collect_results(paths,inventory,checkpoint=(b.SOURCE,meta))
    dense=collect_results([b.parent.BASELINE/'mmlu.json',b.parent.BASELINE/'seven.json'],inventory)
    b.require(current['versions']==dense['versions'],'Eight-task versions differ from BF16.')
    lb_path=b.RUN/'merged/scores.json'
    lb=comparison(b.parent.BASELINE/'longbench/scores.json',lb_path,str(b.parent.MANIFEST))
    b.require(b.read(lb_path)['model']['checkpoint_id']==b.SOURCE_ID,'LongBench and eight UUID differ.')
    budget=audit_checkpoint(b.SOURCE,max_bpw=16,max_lora_rank=0)
    b.require(not budget['contract_violations'] and budget['compressed_target_count']==252
        and abs(budget['user_linear_bpw']-b.BPW)<1e-12
        and budget['packed_linear_bpw']==budget['user_linear_bpw'],'Selected S4 ledger changed.')
    scores={k:100*v for k,v in current['task_metrics'].items()};baseline={k:100*v for k,v in dense['task_metrics'].items()}
    scores['longbench']=lb['task_macro_mean']['compressed'];baseline['longbench']=lb['task_macro_mean']['dense']
    mean=math.fsum(scores.values())/9;BF16=math.fsum(baseline.values())/9
    correct=round(scores['mmlu']*14042/100)
    b.require(correct==10146 and abs(correct/14042*100-scores['mmlu'])<1e-10
        and abs(BF16-68.04714491790848)<1e-10,'Selected M count or BF16 nine baseline changed.')
    M_pass=correct>=10111;nine_pass=mean>b.CONTRACT['nine_mean_strict_minimum']
    result=dict(status='complete_S4_merge75_same_checkpoint_nine',checkpoint_id=b.SOURCE_ID,checkpoint=b.event(b.SOURCE),
        scores_percent=scores,BF16_percent=baseline,nine_mean_percent=mean,BF16_nine_mean_percent=BF16,
        nine_loss_pp=BF16-mean,deltas_pp={k:scores[k]-baseline[k] for k in scores},
        M_correct=correct,M_total=14042,M_loss_pp=baseline['mmlu']-scores['mmlu'],
        quality_acceptance=dict(M_passed=M_pass,nine_passed=nine_pass,S4_quality_target_passed=M_pass and nine_pass,
            M_correct_minimum=10111,nine_mean_strict_minimum=b.CONTRACT['nine_mean_strict_minimum']),
        bit_budget=budget,reused_eight=request['eight'],raw_eight_samples=_sample_artifacts(paths),
        longbench_scores=b.binding(lb_path),source_sha256=request['source_sha256'],contract=b.CONTRACT,
        full_BF16_chain_replayed=False,cross_seed_model_bitwidth_generalization_verified=False,
        overall_lowbit_algorithm_goal_complete=False,
        boundary='Numerical S4 quality acceptance is separate from full-chain reproducibility and generalization; no automatic successor')
    b.write(args.output,result);print(dict(status=result['status'],nine_mean=mean,quality=result['quality_acceptance']),flush=True)


if __name__=='__main__':main()
