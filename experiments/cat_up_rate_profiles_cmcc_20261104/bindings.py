"""Frozen common BF16/data/core identities and exactly three authorized cards."""
from copy import deepcopy
import json
from pathlib import Path
import time

from experiments.cat_auto_protocol_cmcc_20261104 import adapter, config, fit_adapter, search
from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.identity import candidate_id

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PILOT = REPO / 'artifacts/cat_auto_protocol_cmcc_20261104/public_up_pilot_20261104_1455'
DATA = REPO / 'result/qwen3_8b_mmlu65_20261004/cat_auto_protocol_cmcc_20261104'
CARDS = {'k32': dict(rate='2', K=32, gpu=1, technical_seconds=0, formal_seconds=2700),
    'k40': dict(rate='5/2', K=40, gpu=3, technical_seconds=900, formal_seconds=2700),
    'k48': dict(rate='3', K=48, gpu=4, technical_seconds=900, formal_seconds=2700)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def common():
    original = read(PILOT / 'request.json')
    completed = read(PILOT / 'completed.json')
    inventory_path = DATA / 'model_inventory_cpu.json'
    inventory = read(inventory_path)['inventory']
    require(adapter.file_sha(PILOT / 'request.json') == '27daf04be51d878f556150dbd8fc9da27e16efe53f21301a74a73ae153f03121'
        and adapter.file_sha(PILOT / 'completed.json') == '27de387928a29060d3e346131be03577b08d283a2ead762623e7afffb36cca74', 'Reuse only the actual complete K32 model-path receipt.')
    require(all(adapter.file_sha(path) == sha for path, sha in original['source_sha256'].items()), 'The reviewed PUBLIC core must remain frozen, including its host policy.')
    identity = deepcopy(original['run_identity'])
    require(identity['code_identity'] == fit_adapter.fit_code_identity() and identity['consumer_identity'] == adapter.public_consumer_identity(), 'Actual core/consumer family changed.')
    require(identity['model'] == inventory['model_identity'] and identity['data_identity'] == read(DATA / 'full_data_v0/report.json')['data_identity'], 'Reuse the exact original model and root DataPlan.')
    return original, completed, inventory, identity


def pins():
    original, _, _, _ = common()
    return {**original['source_sha256'], **{str(path): adapter.file_sha(path) for path in sorted(HERE.glob('*.py'))}}


def recipe_for(label):
    _, _, inventory, identity = common()
    card = CARDS[label]
    recipe = config.resolve_recipe(inventory, 'up_proj', card['rate'], dim=32, stages=2, transpose=False,
        rotation_level='32', protection_fraction='1/64', optimizer='adamw', multiplier=1)
    require((recipe['D'], recipe['K'], recipe['S'], recipe['B'], recipe['linear_group_size']) == (32, card['K'], 2, 8192, 36)
        and set(recipe['protection']['counts'].values()) == {64}, 'The approved three-rate card changed.')
    return recipe, identity


def make_request(label, phase, *, output, chain_deadline, technical_receipt=None, now=None):
    require(label in CARDS and phase in ('technical_validation', 'formal'), 'No fourth rate or other phase is authorized.')
    require(not (label == 'k32' and phase == 'technical_validation'), 'K32 reuses its existing complete technical path; do not repeat its short run.')
    original, _, _, identity = common()
    recipe, _ = recipe_for(label)
    card = CARDS[label]
    budget = card['technical_seconds'] if phase == 'technical_validation' else card['formal_seconds']
    started = time.time() if now is None else now
    receipt = None
    if phase == 'formal':
        proof_path = PILOT / 'completed.json' if label == 'k32' else Path(technical_receipt)
        require(proof_path.is_file(), 'A real completed own-K technical receipt is required before creating its formal registry.')
        receipt = dict(path=str(proof_path.resolve()), sha256=adapter.file_sha(proof_path))
    request = dict(schema=original['schema'], phase=phase, inventory_receipt=original['inventory_receipt'],
        inventory_receipt_sha256=original['inventory_receipt_sha256'], data_plan=original['data_plan'], data_plan_sha256=original['data_plan_sha256'],
        run_identity=identity, parent=original['parent'], candidates=[dict(candidate_id=candidate_id(identity, recipe), recipe=recipe,
            purpose='profile', complete_dependency_gpu_hours=budget / 3600,
            cost_basis=dict(kind='approved_finite_complete_stage_cap', includes='All teacher/capture/fit/encode/proxy/local/save/fresh/source costs',
                reference='root/user approved parallel three-rate BF16 up batch<=2.75GPUh'))],
        output_dir=str(Path(output).resolve()), absolute_deadline_epoch=min(started + budget, chain_deadline),
        authorized_start_epoch=started, authorized_gpu_hours=budget / 3600, physical_GPU=card['gpu'],
        authorization_reference='root/user2026-11-04: three rates parallel onGPU1/3/4; K40/K48 tests simultaneous; total<=2.75GPUh',
        shared_overhead_gpu_hours=0., shared_cost_scope='All shared overhead included in this sole complete stage cap',
        source_sha256=pins(), technical_steps=2 if phase == 'technical_validation' else None,
        required_model_path_evidence=receipt, full8=False, KD=False, automatic_retry=False,
        formal_initialization='Fresh original BF16 load; never resume a technical deployment bank',
        batch_label=label, fixed_rate=card['rate'], chain_absolute_deadline_epoch=chain_deadline)
    require(request['absolute_deadline_epoch'] > started, 'The original chain budget is exhausted; no extension.')
    return request
