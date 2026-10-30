"""CPU preparation of the real B mixed boundary and actual FP32 D/PEFT inventory."""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.adaptive_native_model_20261029 import checkpoint, recovery
from experiments.adaptive_vector_rvq import common as c
import torch


def fresh(output):
    original_load = torch.load
    reads = []

    def only_payload(path, *args, **kwargs):
        path = Path(path).resolve()
        c.require(path == output / 'mixed_round_base' / checkpoint.STATE_FILE, 'Fresh mixed loader attempted an old parent weight read.')
        reads.append(str(path))
        return original_load(path, *args, **kwargs)

    torch.load = only_payload
    model, meta, loaded, budget = checkpoint.load(output / 'mixed_round_base')
    torch.load = original_load
    expected = c.read(output / 'report.json')
    deployed_digest = c.digest(model.state_dict())
    c.require(deployed_digest == expected['deployment_digest'] and not loaded.missing_keys and not loaded.unexpected_keys,
              'Independent CPU fresh mixed deployment fields differ from the actual saved complete model.')
    cat, hf, training, vae, cfgs, runtime, stage = recovery.recipe(output)
    selected, native_targets, compact_targets, lr, inventory = recovery.selection(model, stage)
    c.require(inventory['actual_trainable_leaves'] == inventory['decoder_leaves'] + inventory['lora_leaves'],
              'The real PEFT/native/compact trainable inventory has an omission or overlap.')
    c.write(output / 'fresh_cpu.json', dict(status='independent_mixed_CPU_state_and_trainables_verified', pid=os.getpid(),
        persistent_tensors=len(model.state_dict()), saved_deployment_digest=deployed_digest, effective_weight_reads=reads,
        no_old_parent_weight_read=True, actual_FP32_selection=inventory, original_stage_config=asdict(stage),
        raw_runtime_identity_exact=True, budget=budget, cuda_initialized=torch.cuda.is_initialized()))
    print(json.dumps(dict(event='independent_mixed_CPU_state_and_trainables_verified', **inventory)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'Use only hidden-CUDA CPU preparation.')
    torch.set_num_threads(8)
    output = args.output.resolve()
    if args.fresh:
        fresh(output)
        return
    c.require(not output.exists(), 'Use a fresh isolated mixed-handoff preparation directory.')
    output.mkdir(parents=True)
    record = dict(status='running', pid=os.getpid(), command=[sys.executable, *sys.argv], started_unix=time.time())
    c.write(output / 'report.json', record)
    try:
        cat, hf, training, vae, cfgs, runtime, stage = recovery.recipe(output)
        model, source = recovery.build_raw_mixed()
        record.update(source=source, original_stage=asdict(stage), actual_native_fitter=str(Path(recovery.native.__file__).resolve()))
        meta, budget, payload = checkpoint.save(model, output / 'mixed_round_base', provenance=dict(**source,
            purpose='Validation-only actual raw B fixture for original500 schedule; not structure selection, formal training or quality.'))
        record.update(saved_checkpoint_id=meta['checkpoint_id'], deployment_digest=c.digest(payload),
            persistent_tensors=len(payload), actual_bit_budget=budget, mixed_round_base=str(output / 'mixed_round_base'))
        c.write(output / 'report.json', record)
        del model, payload
        gc.collect()
        subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--output', str(output), '--fresh'], cwd=ROOT, check=True)
        verified = c.read(output / 'fresh_cpu.json')
        c.require(not verified['cuda_initialized'] and verified['no_old_parent_weight_read'], 'CPU preparation violated effective payload/device isolation.')
        paths = [*Path(__file__).parent.glob('*.py'), Path(recovery.native.__file__), Path(recovery.canonical.__file__), recovery.PREPARED]
        record.update(status='CPU_verified_waiting_team_review', exit_code=0, independent_fresh=verified,
            source_sha256={str(path.resolve()): c.sha(path) for path in paths}, CUDA_used=False,
            final_budget_not_applied_to_dense_boundary=not budget['final_budget_applicable'],
            next='Root/Astra narrow review before any GPU0 execution; no third-stage/routing/long training')
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record.update(finished_unix=time.time(), elapsed_seconds=time.time() - record['started_unix'])
        c.write(output / 'report.json', record)
    print(json.dumps(dict(status=record['status'], seconds=record['elapsed_seconds'], persistent_tensors=record['persistent_tensors'])), flush=True)


if __name__ == '__main__':
    main()
