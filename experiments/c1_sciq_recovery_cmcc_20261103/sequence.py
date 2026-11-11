"""One fixed formal32 -> independent fresh -> full9, without retries or reservation."""
import json
import os
import sys
import time
import traceback

from . import bindings as b


def requests():
    def request(name,command,gpu='',devices=()):
        return dict(name=name,argv=[str(b.PYTHON),'-B','-u','-m',*command],
            environment=b.executor.environment(gpu),gpus=list(devices),
            log=str(b.EVIDENCE/(name+'.log')),exit_file=str(b.EVIDENCE/(name+'.exit')))
    package = 'experiments.c1_sciq_recovery_cmcc_20261103.'
    return [request('formal32',[package+'run','--prepared',str(b.PREPARED),'--gpu','4','--timeout-seconds','14400'],'4',('4',)),
        request('fresh',[package+'fresh','--prepared',str(b.PREPARED),'--gpu','4'],'4',('4',)),
        request('full9',[package+'evaluate','--support',str(b.SUPPORT),'--checkpoint',str(b.OUTPUT/'formal32/final_model'),
            '--pipeline-report',str(b.OUTPUT/'formal32/report.json'),'--output',str(b.EVIDENCE/'evaluation'),'--gpus','4']),
        request('readout',[package+'readout'],'4',('4',))]


def main():
    b.executor.cpu_environment()
    b.configure()
    prepared = b.read_prepared(b.PREPARED)
    b.require(not (b.EVIDENCE/'run.json').exists() and not (b.EVIDENCE/'plan.json').exists(), 'Single fixed chain; no retry/resume.')
    resources = b.executor.check_resources(['4'])
    bound = {path:item['sha256'] for path,item in b.code_bindings().items()}
    bound[str(b.PREPARED)] = b.mature.sha256(b.PREPARED)
    plan = dict(schema='C1_SciQ_single_data_arm_sequence_v1',created_unix=time.time(),source_checkpoint_id=b.PARENT_ID,
        control_checkpoint_id=b.CONTROL_ID,prepared=b.binding(b.PREPARED),source_sha256=bound,
        contract=b.CONTRACT,stages=requests(),initial_resources=resources,automatic_retry=False,automatic_resume=False,
        budget=dict(teacher_cache_seconds=1800,formal_updates=32,formal_seconds=14400,
            original_schedule=500,complete_nine=1,complete_nine_estimated_hours=6,readout_B1_FW=192),
        completion='Original Muon32 control reused. Same-UUID full9 and unchanged actual deployment budget; knowledge gain requires both nine and M above control.',
        stop='No net gain ends this512/512 configuration; no ratio/seed/step grid, reservation, retries or waiting for another branch.')
    b.core.write(b.EVIDENCE/'plan.json',plan)
    record = dict(schema=plan['schema'],status='starting',pid=os.getpid(),start_ticks=b.executor.proc_start(os.getpid()),
        started_unix=time.time(),argv=sys.argv,python=sys.executable,plan_sha256=b.mature.sha256(b.EVIDENCE/'plan.json'),stages={})
    def changed(stage):
        record['stages'][stage['name']] = stage
        record['status'] = stage['name']+'_'+stage['status']
        b.core.write(b.EVIDENCE/'run.json',record)
    try:
        b.core.write(b.EVIDENCE/'run.json',record)
        for request in plan['stages']:
            b.executor.check_pins(bound)
            if request['gpus']:
                b.executor.check_resources(request['gpus'])
            if request['name'] == 'readout':
                try:
                    b.executor.launch_child(request,changed)
                except Exception:
                    record['readout_error'] = traceback.format_exc()
                continue
            b.executor.launch_child(request,changed)
        report = b.read(b.OUTPUT/'formal32/report.json')
        quality = b.executor.validate_evaluation(b.EVIDENCE/'evaluation',report['export']['checkpoint_id'])
        control = b.read(b.original.EVIDENCE/'evaluation/longbench/merged/nine_task_acceptance.json')
        nine_delta = quality['nine_task_score']-control['compressed_nine_task_score']
        m_delta = quality['task_scores']['mmlu']-control['compressed_scores']['mmlu']
        training = report['training']
        record.update(status='completed_C1_SciQ512_aux512_formal_fresh_full9',exit_code=0,quality=quality,
            delta_vs_original_Muon32=dict(nine=nine_delta,mmlu=m_delta),
            knowledge_recovery_supported=nine_delta>0 and m_delta>0,
            decision=('further_knowledge_candidate' if nine_delta>0 and m_delta>0 else
                'combined_candidate_only' if nine_delta>0 else 'end_this512_512_configuration'),
            actual_exposure=dict(M_input_tokens=prepared['data']['M_input_tokens'],M_target_positions=1024,
                actual_OPD_supervision=training['actual_OPD_supervision']))
    except BaseException:
        record.update(status='failed',exit_code=1,error=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        b.core.write(b.EVIDENCE/'run.json',record)
        (b.EVIDENCE/'run.exit').write_text(str(record['exit_code'])+'\n',encoding='utf-8')
    print(json.dumps(dict(status=record['status'],quality=record['quality'],decision=record['decision'])),flush=True)


if __name__ == '__main__':
    main()
