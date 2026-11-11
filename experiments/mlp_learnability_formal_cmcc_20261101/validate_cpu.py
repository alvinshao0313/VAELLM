"""Actual C0/data/import/source-context checks; the GPU bridge stays separate."""
import argparse
import ast
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import bindings as b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    b.executor.cpu_environment()
    output=args.output.resolve()
    b.require(not output.exists(),'Use a fresh CPU validation directory.')
    output.mkdir(parents=True)
    began=time.monotonic()
    result=dict(status='running',python=sys.executable,argv=[sys.executable,*sys.argv])
    try:
        from . import evaluate, evaluation_bindings, fresh, fit, readout, sequence, snapshot, source
        syntax=[]
        for path in b.HERE.glob('*.py'):
            ast.parse(path.read_text(),filename=str(path))
            syntax.append(str(path))
        parent=b.source_C0()
        b.require(parent['checkpoint_id']==b.C0_ID and parent['fresh_C0']['checkpoint_id']==b.C0_ID,
            'CPU preparation must use the actual completed original C0, not an old B or short.')
        dataset,recovery,evidence=b.data.load()
        full=b.data.exposure(dataset,recovery,b.groups(False))
        b.require(full['presentations']==dict(mmlu=4096,qasper=256,hotpotqa=768)
            and full['total_input_tokens']==4212624 and full['total_answer_positions']==10944
            and Counter(b.groups(False))==Counter({index:4 for index in range(32)})
            and b.groups(True)==[2,10] and b.snapshot_steps(True)==[1,2]
            and b.snapshot_steps(False)==[64,128] and (1-127/500)==.746,
            'The real fixed128 document/token/LR budget differs.')
        evaluation_bindings.setup(b.SUPPORT)
        from experiments.adaptive_class_evaluation_20261030 import bindings as official
        callbacks=(official.source,official.source_pins,official.command)
        try:
            source.source(b.ROOT/'missing_real_snapshot/final_model',b.ROOT/'missing_real_snapshot/report.json')
        except FileNotFoundError:
            missing_rejected=True
        else:
            raise ValueError('A missing genuine snapshot was accepted.')
        b.require(callbacks==(official.source,official.source_pins,official.command),
            'The actual missing-source path replaced the installed learning dispatch.')
        from experiments.fresh_fixed2_cmcc_20261030.evaluate import source as actual_C0_source
        actual,_meta=actual_C0_source(b.C0,b.C0.parent/'report.json')
        b.require(actual==parent['fresh_C0'] and callbacks==(official.source,official.source_pins,official.command),
            'The exact actual ancestor-reader path changed parent identity or learning dispatch.')
        help_commands=[]
        for module in ('run','fresh','sequence','evaluate'):
            argv=[str(b.PYTHON),'-B','-u','-m','experiments.mlp_learnability_formal_cmcc_20261101.'+module,'--help']
            completed=subprocess.run(argv,cwd=b.REPO,env={**os.environ,**b.executor.environment('')},
                stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            b.require(completed.returncode==0,'An actual formal CLI import failed: '+module)
            help_commands.append(dict(argv=argv,exit_code=completed.returncode,stdout=completed.stdout))
        pins=b.code_bindings()
        b.executor.check_pins(pins)
        b.require(not b.torch.cuda.is_initialized(),'CPU checks initialized CUDA.')
        result.update(status='verified_real_C0_fixed_data_and_dispatch_GPU_bridge_pending',exit_code=0,
            actual_C0_binding=parent,actual_128_exposure=full,data=evidence,syntax_checked=syntax,
            installed_dispatch_preserved_on_actual_C0_and_missing_snapshot=True,missing_snapshot_rejected=missing_rejected,
            CLI=help_commands,source_sha256=pins,training_contract=b.CONTRACT,
            CUDA_initialized=False,training_or_deployment_model_constructed=False,actual_C0_state_inspected_on_CPU=True,
            boundary='Real source/metadata/data and exact dispatch contexts only. Actual two-card updates, '
                'step1 snapshot/step2 continuation/fresh and single-card consumption remain necessary.')
    except BaseException:
        result.update(status='failed',exit_code=1,error=traceback.format_exc())
        raise
    finally:
        result['elapsed_seconds']=time.monotonic()-began
        b.write(output/'validation.json',result)
    print(json.dumps(dict(status=result['status'],report=str(output/'validation.json'))),flush=True)


if __name__=='__main__':
    main()
