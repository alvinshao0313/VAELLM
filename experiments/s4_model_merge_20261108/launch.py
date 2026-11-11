"""Record one tmux launch of the predeclared finite inference-only candidate."""
import argparse
import shlex
import subprocess
from . import bindings as b


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=list(b.ARMS),required=True)
    args=parser.parse_args();b.initialize();root=b.RUN/args.arm;request=b.read(root/'request.json');b.check(request)
    b.require(not (root/'launch.json').exists() and not (root/'run.json').exists(),'Never duplicate or overwrite a merge launch.')
    env=b.environment('');keys=('PATH','CONDA_DEFAULT_ENV','CONDA_PREFIX','PYTHONPATH','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS')
    argv=[str(b.PYTHON),'-B','-u','-m','experiments.s4_model_merge_20261108.run','--arm',args.arm]
    command=shlex.join(['env',*[k+'='+env[k] for k in keys if k in env],*argv])+' > '+shlex.quote(str(root/'pipeline.log'))+' 2>&1'
    session=f'S4_merge_{args.arm}_g{request["recipe"]["gpu"]}_20261108'
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(b.ROOT),command],env=env,check=True)
    pane=int(subprocess.run(['tmux','display-message','-p','-t',session,'#{pane_pid}'],check=True,capture_output=True,text=True).stdout)
    record=dict(session=session,command=command,argv=argv,request=b.binding(root/'request.json'),gpu=request['recipe']['gpu'],
        process=b.parent.process_event(pane),log=str(root/'pipeline.log'))
    b.write(root/'launch.json',record);print(dict(session=session,pane=record['process'],gpu=record['gpu']),flush=True)


if __name__=='__main__':main()
