"""Freeze the single approved GPU5/.5h PUBLIC up technical/cost pilot."""
import argparse
import json
from pathlib import Path
import time

from .adapter import REPO, file_sha, public_consumer_identity
from .config import resolve_recipe
from .contracts import digest, require
from .fit_adapter import fit_code_identity
from .identity import candidate_id
from .search import validate_request


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--request', required=True); parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    report_root = REPO / 'result/qwen3_8b_mmlu65_20261004/cat_auto_protocol_cmcc_20261104'
    inventory_path = report_root / 'model_inventory_cpu.json'
    data_path = report_root / 'full_data_v0/data_plan.pt'
    inventory = json.loads(inventory_path.read_text())['inventory']
    data_report = json.loads((report_root / 'full_data_v0/report.json').read_text())
    require(data_report['exit_code'] == 0 and file_sha(data_path) == data_report['data_plan_sha256'], 'Use the actual completed frozen five-source plan.')
    consumer = public_consumer_identity()
    identity = dict(model=inventory['model_identity'], parent_uuid='BF16:' + digest(inventory['model_identity']),
        parent_weights_digest=digest(inventory['model_identity']['weight_shards_sha256']), data_identity=data_report['data_identity'],
        code_identity=fit_code_identity(), consumer_identity=consumer, search_seed=31)
    recipe = resolve_recipe(inventory, 'up_proj', '2', dim=32, stages=2, transpose=False, rotation_level='32', optimizer='adamw')
    require(recipe['D'] == recipe['K'] == 32 and recipe['S'] == 2 and recipe['B'] == 8192
        and recipe['linear_group_size'] == 36 and set(recipe['protection']['counts'].values()) == {64}, 'The approved real M36/D32K32/B8192/input64 card changed.')
    pins = {**consumer['source_sha256'], **{str(path.resolve()): file_sha(path) for path in Path(__file__).parent.glob('*.py')}}
    quantizer_source = REPO / 'litebsq/protected_channel_quant.py'
    pins[str(quantizer_source)] = file_sha(quantizer_source)
    started = time.time()
    request = dict(schema='cat_auto_explicit_candidates_v0', phase='technical_validation',
        inventory_receipt=str(inventory_path), inventory_receipt_sha256=file_sha(inventory_path),
        data_plan=str(data_path), data_plan_sha256=data_report['data_plan_sha256'], run_identity=identity,
        parent=dict(kind='BF16', path=inventory['source_evidence']['path'], uuid=identity['parent_uuid'], weights_digest=identity['parent_weights_digest']),
        candidates=[dict(candidate_id=candidate_id(identity, recipe), recipe=recipe, purpose='profile',
            complete_dependency_gpu_hours=.5, cost_basis=dict(kind='approved_global_technical_cost_pilot_cap_not_runtime_estimate',
                includes='Source verification, teacher,512x256 activation capture,fit+encode,512 proxy/local,save/actual ledger/fresh',
                no_quality_or_HPO_slot=True))], output_dir=str(Path(args.output_dir).resolve()),
        absolute_deadline_epoch=started + 1800, authorized_start_epoch=started, authorized_gpu_hours=.5, physical_GPU=5,
        authorization_reference='root/Astra2026-11-04: unique GPU5<=.5GPUh PUBLIC BF16 up technical2 cost pilot',
        shared_overhead_gpu_hours=0., shared_cost_scope='All shared overhead is inside the sole complete_dependency .5h cap, not excluded work',
        source_sha256=pins, technical_steps=2, required_model_path_evidence=None,
        slot_accounting='Technical/cost pilot only; no offline20/profile3/live2 completed fit', automatic_retry=False,
        full8=False, KD=False, formal10000=False)
    validate_request(request)
    destination = Path(args.request)
    require(not destination.exists() and not Path(request['output_dir']).exists(), 'Freeze one new request and one independent pilot output.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(request, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(request=str(destination), request_sha256=file_sha(destination), candidate_id=request['candidates'][0]['candidate_id'],
        start_epoch=started, deadline_epoch=request['absolute_deadline_epoch'], output_dir=request['output_dir'])), flush=True)


if __name__ == '__main__':
    main()
