"""Fixed integer-count development heuristic on complete57-subject MMLU."""
from . import bindings as b

NODES=(64,128,256,384)


def correct(payload):
    b.require(not payload['validation_only'],'Short scores cannot decide formal recovery budgets.')
    result=payload['evals']['lm_eval'];counts=result['n_samples']
    b.require(len(counts)==57 and sum(row['original'] for row in counts.values())==14042
        and all(row['effective']==row['original'] for row in counts.values()),'Require all57 subjects/14042 actual questions.')
    value=payload['verified_task_metrics']['mmlu']
    count=round(value*14042)
    b.require(abs(value-count/14042)<1e-12,'Full MMLU accuracy is not the exact pooled question count.')
    return count


def initial(previous,pre,category):
    drop=max(0,previous-pre)
    steps=64 if drop<36 else 128 if drop<85 else 256
    if category=='down_proj':steps=max(steps,128)
    return dict(previous_post_correct=previous,pre_correct=pre,added_compression_loss_questions=drop,
        initial_budget=steps,rule='drop<36:64;36<=drop<85:128;drop>=85:256;down>=128')


def decide(step,current,preceding,base,previous):
    b.require(step in NODES and base in NODES[:3],'Use the fixed complete-MMLU budget nodes.')
    remaining=previous-current;marginal=current-preceding
    extend=step<base or (step<384 and remaining>=15 and marginal>=15)
    next_step=NODES[NODES.index(step)+1] if extend else None
    return dict(step=step,correct=current,preceding_correct=preceding,remaining_questions=remaining,
        marginal_questions=marginal,initial_budget=base,continue_training=extend,next_node=next_step,
        reason='complete_initial_budget' if step<base else '384_stage_cap' if step==384 else
            'remaining_and_recent_gain_at_least15' if extend else 'remaining_or_recent_gain_below15',
        boundary='Development heuristic; not a significance test, checkpoint selection, or rollback.')


def validation_cases():
    # Arithmetic contract cases; never represented as model evaluations.
    expected=[(35,64),(36,128),(84,128),(85,256)]
    b.require(all(initial(1000,1000-drop,'q_proj')['initial_budget']==steps for drop,steps in expected),
        'Initial category-loss boundary semantics failed.')
    b.require(initial(1000,1001,'down_proj')['initial_budget']==128,'Down minimum or negative-drop handling failed.')
    b.require(decide(64,990,990,256,1000)['continue_training'],'No quality early stop within initial budget.')
    b.require(decide(128,985,970,128,1000)['next_node']==256,'Inclusive15/15 extension boundary failed.')
    b.require(not decide(128,986,970,128,1000)['continue_training'],'Remaining14 must stop.')
    b.require(not decide(128,985,971,128,1000)['continue_training'],'Marginal14 must stop.')
    b.require(not decide(384,985,970,256,1000)['continue_training'],'384 is a hard per-category cap.')
    return dict(status='verified_integer_budget_boundary_cases',model_evaluation=False,
        tested='35/36,84/85 initial drop; down minimum; initial-budget continuation; inclusive15/15;14 stop;384 cap')
