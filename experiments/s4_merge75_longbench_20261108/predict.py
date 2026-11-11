"""Original full21 runner with the proven native4bit merge loader and prewarm."""
import argparse
from pathlib import Path
from . import bindings as b
from experiments.s4_model_merge_20261108 import state
from experiments.s4_scope_objective_20261107.core import runtime
from experiments.longbench_v1 import runner
from experiments.longbench_v1.protocol import model_policy


def load_model(config,logger):
    b.require(config['model']['checkpoint_id']==b.SOURCE_ID and Path(config['model']['checkpoint']).resolve()==b.SOURCE,
        'Use only selected merge75 checkpoint.')
    model,meta,budget=state.load(config['model']['checkpoint'])
    model.to('cuda');b.restore_rope_alias(model);b.old.install_precision_runtime(model,b.torch.bfloat16)
    warm=runtime.warm_frozen(model)
    return model,dict(checkpoint_id=meta['checkpoint_id'],strict_missing=[],strict_unexpected=[],
        model_policy=model_policy(b.HF),bit_budget=budget,warmup=warm,
        loader='Existing native S4 strict merge load and BF16 group8 prewarm; parameters only changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--datasets',nargs='+',required=True);args=parser.parse_args();b.initialize()
    request=b.read(args.request);b.check(request)
    import os
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES') in b.GPUS and b.torch.cuda.device_count()==1,'One assigned GPU4..7 required.')
    b.require(b.sha(args.config)==request['config_sha256'],'Pinned LongBench config changed.')
    runner.load_model=load_model
    result=runner.run(str(args.config),str(args.output),datasets=args.datasets,max_runtime_seconds=86400)
    b.write(args.output/'predict.json',result)
    b.require(result['status']=='partition_complete' and result['exit_code']==0 and result['already_present']==0,
        'LongBench partition incomplete; no partial score/retry.')
    print(dict(status=result['status'],new_predictions=result['new_predictions']),flush=True)


if __name__=='__main__':main()
