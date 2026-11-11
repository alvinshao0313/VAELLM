"""Start one explicitly authorized four-card full21 supplement in tmux."""
import shlex
import subprocess
from . import bindings as b


def main():
    b.initialize();request=b.read(b.RUN/'request.json');b.check(request)
    b.require(not (b.RUN/'launch.json').exists() and not (b.RUN/'run.json').exists(),'One fresh launch only.')
    env=b.environment('');keys=('PATH','CONDA_DEFAULT_ENV','CONDA_PREFIX','PYTHONPATH','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS')
    argv=[str(b.PYTHON),'-B','-u','-m','experiments.s4_merge75_longbench_20261108.run']
    command=shlex.join(['env',*[k+'='+env[k] for k in keys if k in env],*argv])+' > '+shlex.quote(str(b.RUN/'pipeline.log'))+' 2>&1'
    session='S4_merge75_LB21_g4_7_20261108'
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(b.ROOT),command],check=True,env=env)
    pane=int(subprocess.run(['tmux','display-message','-p','-t',session,'#{pane_pid}'],check=True,capture_output=True,text=True).stdout)
    record=dict(session=session,command=command,argv=argv,request=b.binding(b.RUN/'request.json'),
        process=b.parent.process_event(pane),log=str(b.RUN/'pipeline.log'),GPUs=b.GPUS)
    b.write(b.RUN/'launch.json',record);print(dict(session=session,pane=record['process']),flush=True)


if __name__=='__main__':main()
