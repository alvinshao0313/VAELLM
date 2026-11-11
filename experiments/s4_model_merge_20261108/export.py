"""Independent native merge build; no training or optimizer continuation."""
import argparse
import os
from pathlib import Path
import time
from . import bindings as b,state


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();b.initialize()
    request=b.read(args.request);b.check(request);resource=b.resources(request['recipe']['gpu']);start=time.monotonic()
    report=state.build(request,args.output)
    report.update(resource=resource,elapsed_seconds=time.monotonic()-start,**b.parent.process_event(os.getpid()))
    b.write(args.output/'merge.json',report)
    print(dict(status='native_model_merge_exported',checkpoint_id=report['checkpoint']['checkpoint_id']),flush=True)


if __name__=='__main__':main()
