"""Export CPU weights in a new process with the assigned GPU visible to FA2."""
import argparse
import os
from pathlib import Path
from . import bindings as b,state


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--state',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();b.initialize()
    request=b.read(args.request);b.check_request(request)
    resource=b.resources(request['recipe']['gpu'])
    saved=b.torch.load(args.state,map_location='cpu',weights_only=False)
    report=state.export(request,saved,args.output)
    report.update(training_state=b.binding(args.state),model_weights_on_CPU=True,
        assigned_GPU_visible_for_FA2_loader=resource,CUDA_initialized=b.torch.cuda.is_initialized())
    b.write(args.output/'export.json',report)
    print(dict(status='native_export_complete',checkpoint_id=report['checkpoint']['checkpoint_id']),flush=True)


if __name__=='__main__':main()
