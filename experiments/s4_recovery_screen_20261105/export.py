"""New process: reconstruct parent plus FP32 state and export native BF16."""
import argparse
import os
from pathlib import Path
from . import bindings as b,state,train


def execute(request_path,state_path,output):
    b.initialize();request=b.read(request_path);b.parent.s.check_pins(request['source_sha256']);resource=b.resources(request['gpu'])
    b.require(request['learning_contract']==b.LEARNING_CONTRACT,'Immutable learning contract changed.')
    model,active,optimizer,warm,budget=train.build(request)
    saved=state.restore(state_path,model,active,optimizer,request)
    b.require(saved['completed_updates']==saved['saved_execution_policy']['stop_step'],'Export only the completed explicit endpoint.')
    report=state.export(model,active,optimizer,request,saved,output)
    report.update(**b.parent.process_event(os.getpid()),training_state=b.binding(state_path),
        reconstructed_from_common_parent=True,actual_FP32_masters_Adam_RNG_cursor_exact=True,resource=resource)
    b.write(Path(output)/'export.json',report)
    print(dict(status='native_endpoint_exported_from_named_state',checkpoint_id=report['checkpoint']['checkpoint_id']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--state',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();execute(args.request,args.state,args.output)
