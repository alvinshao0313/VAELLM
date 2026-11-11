"""Launch one authorized finite chain in tmux and record its actual identity."""
import argparse
import os
import shlex
import subprocess
from . import bindings as b


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=list(b.ARMS),required=True)
    parser.add_argument('--validation',action='store_true');args=parser.parse_args();b.initialize()
    b.require(os.environ.get('CONDA_DEFAULT_ENV')=='bitvae','Activate bitvae in caller shell.')
    tag='validation_'+args.arm if args.validation else args.arm;root=b.RUN/tag
    request=b.read(root/'request.json');b.check_request(request)
    b.require(not (root/'launch.json').exists() and not (root/'run.json').exists(),'Fresh launch only; no retry/overwrite.')
    session=f'S4_SO128_{tag}_g{request["recipe"]["gpu"]}_20261108'
    argv=[str(b.PYTHON),'-B','-u','-m','experiments.s4_scope_objective_20261107.run','--arm',args.arm]
    if args.validation:argv.append('--validation')
    env=b.environment('')
    keys=('PATH','CONDA_DEFAULT_ENV','CONDA_PREFIX','PYTHONPATH','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS')
    command=shlex.join(['env',*[key+'='+env[key] for key in keys if key in env],*argv])
    command+=' > '+shlex.quote(str(root/'pipeline.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(b.ROOT),command],check=True,env=env)
    pane=int(subprocess.run(['tmux','display-message','-p','-t',session,'#{pane_pid}'],check=True,capture_output=True,text=True).stdout.strip())
    record=dict(session=session,command=command,argv=argv,request=b.binding(root/'request.json'),
        process=b.parent.process_event(pane),log=str(root/'pipeline.log'),gpu=request['recipe']['gpu'],
        validation=args.validation)
    b.write(root/'launch.json',record);print(record,flush=True)


if __name__=='__main__':main()
