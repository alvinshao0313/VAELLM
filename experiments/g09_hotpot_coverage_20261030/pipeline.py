"""A single finite formal32 -> visible-CUDA native export -> complete-nine execution."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = Path('/data/star/VAELLM/g09_hotpot_coverage_20261030')
PYTHON = Path('/home/star/miniconda3/envs/bitvae/bin/python')


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def resources(gpus):
    result = subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True)
    devices = {parts[0].strip():dict(uuid=parts[1].strip(), memory_MiB=int(parts[2]))
        for line in result.stdout.splitlines() if len(parts:=line.split(',')) == 3}
    selected = {gpu:devices[gpu] for gpu in gpus}
    result = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True)
    blockers = [line for line in result.stdout.splitlines() if line.split(',')[0].strip()
        in {item['uuid'] for item in selected.values()}]
    if blockers:
        raise RuntimeError('Assigned GPUs occupied; no waiting/retry: ' + repr(blockers))
    return dict(checked_unix=time.time(), devices=selected, compute_processes=blockers)


def main():
    if Path(sys.executable).resolve() != PYTHON.resolve() or Path.cwd() != ROOT:
        raise RuntimeError('Use activated bitvae and repository cwd.')
    launch = dict(schema='G09_Hotpot192_finite_pipeline_v1', pid=os.getpid(), argv=sys.argv,
        python=sys.executable, cwd=str(Path.cwd()), started_unix=time.time(),
        formal_resources=resources(['4']), automatic_retry=False, daemon=False)
    write(OUTPUT / 'pipeline_launch.json', launch)
    with (OUTPUT / 'formal32.log').open('x', encoding='utf-8') as log:
        formal = subprocess.run(['bash',str(HERE / 'run_formal.sh')], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    (OUTPUT / 'formal32.exit').write_text(str(formal.returncode)+'\n', encoding='utf-8')
    if formal.returncode:
        raise RuntimeError('Formal training/export failed; full-nine is not started.')
    # The allocation was agreed before launch; inspect again immediately before evaluation.
    checked = resources(['5','6','7'])
    environment = {**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(ROOT),
        'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}
    command = [str(PYTHON),'-B','-u',str(HERE / 'evaluate.py'),'--gpus','5,6,7']
    write(OUTPUT / 'full9_launch.json', dict(schema='G09_Hotpot192_full9_launch_v1',
        resources=checked, started_unix=time.time(), argv=command,
        environment={key:environment[key] for key in ('CUDA_VISIBLE_DEVICES','PYTHONPATH',
            'PYTHONDONTWRITEBYTECODE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TOKENIZERS_PARALLELISM')},
        formal32_exit=0, predecessor='formal32 -> visible-CUDA native export, both completed before evaluation',
        automatic_retry=False))
    with (OUTPUT / 'full9.log').open('x', encoding='utf-8') as log:
        evaluated = subprocess.run(command, cwd=ROOT, env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    (OUTPUT / 'full9.exit').write_text(str(evaluated.returncode)+'\n', encoding='utf-8')
    write(OUTPUT / 'pipeline_terminal.json', dict(status='complete' if evaluated.returncode == 0 else 'full9_failed',
        formal32_exit=0, full9_exit=evaluated.returncode, finished_unix=time.time()))
    if evaluated.returncode:
        raise RuntimeError('Full-nine failed; no automatic retry.')


if __name__ == '__main__':
    main()
