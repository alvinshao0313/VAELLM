"""Real prefix/target/QA128-group/shape/source/dispatch evidence; no new GPU short."""
import ast
from collections import Counter
import importlib.metadata
import os
import subprocess
import sys
import time
import traceback

from . import bindings as b
from . import data
from .fetch import SOURCE


def main():
    b.executor.cpu_environment()
    b.torch.set_num_threads(4)
    output=b.EVIDENCE/'cpu'
    b.require(not output.exists(),'Use fresh CPU evidence.')
    output.mkdir(parents=True)
    began=time.monotonic()
    record=dict(status='running')
    try:
        from . import source,evaluation_bindings,fit,snapshot,fresh,evaluate,sequence
        for path in b.HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
        parent=b.source_C0()
        dataset,recovery,evidence=data.load()
        data_contract=b.data_binding()
        exposure=data.exposure(dataset,recovery,b.groups(False))
        shapes=data.input_shapes(dataset)
        peak=max(shapes,key=lambda row:row['M_max_length'])
        b.require(exposure['presentations']==dict(mmlu=4096,qasper=256,hotpotqa=768)
            and exposure['total_answer_positions']==10944 and len(dataset['groups'])==128
            and Counter(b.groups(False))==Counter({index:1 for index in range(128)})
            and b.snapshot_steps(False)==[64,128]
            and peak['M_max_length']<=1222 and max(row['QA_max_length'] for row in shapes)==10593,
            'Changed fixed exposure/target or approved real resource-shape boundary.')
        old,old_recovery,_old_evidence=data.control.load()
        data.validate(dataset,old)
        for case in old['training_cases']:
            key=case['case_id']
            b.require(b.torch.equal(recovery[key]['target_log_probs'],old_recovery[key]['target_log_probs']),
                'Old M1024 once-mixed and QA target values must be exact.')
        evaluation_bindings.setup(b.SUPPORT)
        from experiments.adaptive_class_evaluation_20261030 import bindings as official
        callbacks=(official.source,official.source_pins,official.command)
        try:source.source(b.ROOT/'missing/final_model',b.ROOT/'missing/report.json')
        except FileNotFoundError:missing=True
        else:raise ValueError('Missing coverage snapshot accepted.')
        from experiments.fresh_fixed2_cmcc_20261030.evaluate import source as c0_source
        actual,_meta=c0_source(b.C0,b.C0.parent/'report.json')
        b.require(actual==parent['fresh_C0'] and callbacks==(official.source,official.source_pins,official.command),
            'Ancestor reader changed the installed coverage dispatch.')
        CLI=[]
        for module in ('run','fresh','evaluate','sequence'):
            argv=[str(b.PYTHON),'-B','-m','experiments.mlp_m4096_cmcc_20261101.'+module,'--help']
            result=subprocess.run(argv,cwd=b.REPO,env={**os.environ,**b.executor.environment('')},
                stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            b.require(result.returncode==0,'Actual coverage CLI failed: '+module)
            CLI.append(dict(argv=argv,exit_code=result.returncode,stdout=result.stdout))
        b.require(not b.torch.cuda.is_initialized(),'CPU checks initialized CUDA.')
        pins=b.code_bindings()
        b.executor.check_pins(pins)
        generated=list((SOURCE/'arrow_cache').rglob('*.arrow'))
        b.require(len(generated)==1,'Require the actual sole newly generated auxiliary_train Arrow capture.')
        record.update(status='verified_same_C0_M4096_real_data_targets_and_dispatch',exit_code=0,
            data_contract=data_contract,actual_C0=parent,data=evidence,shapes=shapes,actual128_exposure=exposure,
            actual_peak=peak,old_M1024_mixed_and_QA_FP32_targets_exact=True,
            original_QA_group_suffixes128_exact=True,CLI=CLI,source_sha256=pins,training_contract=b.CONTRACT,
            actual_runtime_versions=dict(python=sys.version,torch=str(b.torch.__version__),
                **{name:importlib.metadata.version(name) for name in ('transformers','flash-attn','datasets')}),
            teacher_mapping_source=b.binding(b.Path(b.life.apply_layer_device_map.__code__.co_filename)),
            new_Arrow_provenance=dict(Arrow=b.binding(generated[0]),source_capture=evidence['source_capture'],
                actual_official_rows=99842,old_Arrow_recovered=False,
                boundary='Generated from the pinned official Parquet by the recorded datasets version; not the absent old Arrow bytes'),
            CUDA_initialized=False,missing_snapshot_rejected=missing,dispatch_preserved=True,
            resource_estimate=dict(measured_control_M_maximum=958,measured_control_peak_GB=[19.835,19.583],
                conservative_new_peak_GB=[19.835*peak['M_max_length']/958,19.583*peak['M_max_length']/958],
                new_peak_measured_on_GPU=False,source='Completed same-C0 D short/formal memory evidence; full-peak linear scaling is a conservative estimate, unchanged FA2/last-head/checkpoint paths'),
            validation_reused='Completed same-C0 fixed-code D128 real training/64 and128 snapshot/fresh/single-GPU consumers; no new GPU short')
    except BaseException:
        record.update(status='failed',exit_code=1,error=traceback.format_exc())
        raise
    finally:
        record['elapsed_seconds']=time.monotonic()-began
        b.write(output/'validation.json',record)
    print(dict(status=record['status'],actual_peak=peak,exposure_tokens=exposure['total_input_tokens']),flush=True)


if __name__=='__main__':main()
