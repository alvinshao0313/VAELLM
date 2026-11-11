"""Unapproved planning and source-bound authorization for exactly three chains."""
from copy import deepcopy
from pathlib import Path
import time

from experiments.cat_up_rate_profiles_cmcc_20261104 import bindings as old
from experiments.cat_batch_gather_cmcc_20261104 import fit, search
from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.identity import candidate_id

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CARDS = {'k32': dict(rate='2', K=32, gpu=1), 'k40': dict(rate='5/2', K=40, gpu=3), 'k48': dict(rate='3', K=48, gpu=4)}
TECHNICAL_SECONDS, FORMAL_SECONDS, CHAIN_SECONDS = 900, 3600, 4500
BATCH_GPU_HOURS = 3.75
SCHEMA = 'cat_batched_up_three_rate_proposal_v1'
CLOSURE = REPO/'artifacts/cat_up_rate_profiles_cmcc_20261104/batch_20261104_1538/closure_20261104.json'
read, write, sha = old.read, old.write, old.adapter.file_sha


def bound(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def write_new(path, value):
    require(not Path(path).exists(), 'Never overwrite an existing proposal, registry or receipt: '+str(path))
    write(path, value)


def common():
    original, _, inventory, identity = old.common()
    require(sha(original['inventory_receipt']) == original['inventory_receipt_sha256'], 'Original actual inventory receipt SHA changed.')
    identity = deepcopy(identity); identity['code_identity'] = fit.fit_code_identity()
    fit._validate_identity(identity)
    require(len(identity['code_identity']['source_sha256']) == 9, 'Use the reviewed native7 plus batched fit/data2 identity.')
    return original, inventory, identity


def pins():
    original, _, _ = common()
    paths = set(HERE.glob('*.py')) | set(old.HERE.glob('*.py'))
    paths.update((REPO/'experiments/cat_auto_protocol_cmcc_20261104').glob('*.py'))
    paths.update((REPO/'experiments/cat_batch_gather_cmcc_20261104').glob('*.py'))
    return {**original['source_sha256'], **{str(path.resolve()): sha(path) for path in sorted(paths)}}


def make_proposal(batch_root):
    original, inventory, identity = common()
    recipes = {}
    for label, card in CARDS.items():
        recipe, previous = old.recipe_for(label)
        require(recipe['S'] == 2 and recipe['K'] == card['K'] and previous['search_seed'] == identity['search_seed'] == 31,
            'Reuse the exact original fixed S2 recipes and semantic search seed.')
        recipes[label] = dict(recipe=recipe, candidate_id=candidate_id(identity, recipe))
    closure = read(CLOSURE)
    value = dict(schema=SCHEMA, status='unapproved_proposal', batch_root=str(Path(batch_root).resolve()),
        cards=CARDS, labels=list(CARDS), recipes=recipes, run_identity=identity, parent=original['parent'],
        inventory_receipt=bound(original['inventory_receipt']), data_plan=bound(original['data_plan']),
        source_sha256=pins(), technical_steps=2, original_schedule_steps=10000, formal_steps_per_stage=10000,
        technical_seconds=TECHNICAL_SECONDS, formal_seconds=FORMAL_SECONDS, chain_seconds=CHAIN_SECONDS,
        batch_GPU_hours=BATCH_GPU_HOURS, full8=False, KD=False, automatic_retry=False,
        original_batch=dict(closure=bound(CLOSURE), actual_chain_wall_GPUh=closure['cost']['chain_wall_GPUh'],
            accounting='Separate incurred historical cost; not reset, credit or transferred authorization.'),
        policy='All three own new-path technical2 receipts before own fresh BF16 S2x10000; independent chains; no retries or occupancy.',
        boundary='Proposal only. No executable request, authorization start or future deadline is prepared.')
    return dict(value, proposal_identity=digest(value))


def validate_proposal(proposal, batch_root):
    require(proposal['schema'] == SCHEMA and proposal['status'] == 'unapproved_proposal'
        and proposal['batch_root'] == str(Path(batch_root).resolve()), 'The exact unique proposed output root is required.')
    require(proposal == make_proposal(batch_root), 'Proposal source/model/data/recipes/cards/budget changed after preparation.')
    return proposal


def validate_authorization_value(proposal, proposal_reference, authorization):
    require(authorization.get('user_approved') is True, 'This proposed batch has no explicit user approval; do not start or create requests.')
    approved = authorization.get('user_authorization')
    expected = dict(proposal_identity=proposal['proposal_identity'], proposal_sha256=proposal_reference['sha256'],
        batch_root=proposal['batch_root'], labels=list(CARDS), cards=CARDS, batch_GPU_hours=BATCH_GPU_HOURS,
        technical_seconds=TECHNICAL_SECONDS, formal_seconds=FORMAL_SECONDS, chain_seconds=CHAIN_SECONDS)
    require(isinstance(approved, dict) and isinstance(approved.get('reference'), str) and bool(approved['reference'].strip())
        and all(approved.get(key) == value for key, value in expected.items()),
        'Actual user authorization must match this proposal SHA/identity/root, fixed labels/cards and complete3.75GPUh caps.')
    return approved


def authorization(proposal, proposal_path, authorization_path):
    require(read(proposal_path) == proposal, 'Execute only the actual frozen proposal file.')
    validate_proposal(proposal, Path(proposal_path).parent)
    return validate_authorization_value(proposal, bound(proposal_path), read(authorization_path))


def stage_window(started, now, phase):
    require(phase in ('technical_validation', 'formal') and now >= started, 'Use an actual current phase and nonfuture chain start.')
    seconds = TECHNICAL_SECONDS if phase == 'technical_validation' else FORMAL_SECONDS
    deadline = min(now+seconds, started+CHAIN_SECONDS)
    require(deadline > now, 'This chain absolute4500-second budget is exhausted; no extension.')
    return seconds, deadline


def technical_evidence(proposal, label, receipt_path):
    original, inventory, identity = common()
    root = Path(proposal['batch_root'])/label
    require(Path(receipt_path).resolve() == (root/'technical_validation/completed.json').resolve(),
        'Formal must consume its own new-batch new-path actual technical2 receipt.')
    registry = root/'technical_validation_request.json'; request = read(registry); completed = read(receipt_path)
    require(request['run_identity'] == identity and request['batch_label'] == label
        and request['proposal_identity'] == proposal['proposal_identity'] and request['physical_GPU'] == CARDS[label]['gpu']
        and request['phase'] == 'technical_validation' and request['technical_steps'] == 2
        and request['source_sha256'] == proposal['source_sha256'] and request['parent'] == proposal['parent']
        and completed['request_identity'] == digest(request), 'Technical receipt differs from its actual own frozen registry.')
    planned = proposal['recipes'][label]
    require(len(request['candidates']) == 1 and request['candidates'][0]['candidate_id'] == planned['candidate_id']
        and request['candidates'][0]['recipe'] == planned['recipe'] and request['candidates'][0]['purpose'] == 'profile',
        'Technical registry must contain exactly this own planned profile candidate.')
    # Keep the reviewed real GPU/two-stage/fit/resource-path checks intact.
    search.native._validate_model_path_evidence(completed, inventory, identity, request['candidates'])
    require(len(completed['candidates']) == 1 and completed['status'] == 'completed' and completed['exit_code'] == 0,
        'Require one complete successful own technical candidate.')
    row = completed['candidates'][0]
    require(row['candidate_id'] == planned['candidate_id'] and row['fit_report']['resolved_recipe'] == planned['recipe'],
        'Technical candidate must have its exact own planned K/recipe/new source identity.')
    fresh, raw, bank = row['fresh_evidence'], row['raw_checkpoint'], row['bank']
    meta_path = Path(raw['output_dir'])/'checkpoint_meta.json'; meta = read(meta_path)
    require(fresh['state_exact'] and fresh['forward_executed'] and fresh['gpu_validation']
        and fresh['same_input_same_BF16_path_logits_exact']
        and meta['checkpoint_id'] == raw['checkpoint_id'] == fresh['checkpoint_id']
        and sha(meta_path.parent/meta['state_dict_file']) == fresh['weight_digest']
        and sha(bank['state_file']) == bank['state_sha256'], 'Own actual complete raw/bank and independent fresh GPU evidence are required.')
    return bound(receipt_path)


def make_request(proposal, proposal_path, authorization_path, label, phase, *, chain_started, technical_receipt=None):
    require(label in CARDS, 'Only the three fixed labels may execute.')
    approved = authorization(proposal, proposal_path, authorization_path)
    original, _, identity = common(); now = time.time()
    seconds, deadline = stage_window(chain_started, now, phase)
    evidence = technical_evidence(proposal, label, technical_receipt) if phase == 'formal' else None
    planned = proposal['recipes'][label]
    request = dict(schema=original['schema'], phase=phase, inventory_receipt=original['inventory_receipt'],
        inventory_receipt_sha256=original['inventory_receipt_sha256'], data_plan=original['data_plan'], data_plan_sha256=original['data_plan_sha256'],
        run_identity=identity, parent=original['parent'], candidates=[dict(**planned, purpose='profile',
            complete_dependency_gpu_hours=seconds/3600, cost_basis=dict(kind='actual_user_approved_complete_stage_cap',
                includes='All source/teacher/capture/fit/encode/proxy/local/save/fresh costs', reference=approved['reference']))],
        output_dir=str(Path(proposal['batch_root'])/label/phase), absolute_deadline_epoch=deadline,
        authorized_start_epoch=now, authorized_gpu_hours=seconds/3600, physical_GPU=CARDS[label]['gpu'],
        authorization_reference=approved['reference'], actual_user_authorization=bound(authorization_path),
        shared_overhead_gpu_hours=0., shared_cost_scope='All overhead included in sole stage cap', source_sha256=proposal['source_sha256'],
        technical_steps=2 if phase == 'technical_validation' else None, required_model_path_evidence=evidence,
        full8=False, KD=False, automatic_retry=False, batch_label=label, fixed_rate=CARDS[label]['rate'],
        proposal_identity=proposal['proposal_identity'], proposal_file=bound(proposal_path),
        chain_absolute_deadline_epoch=chain_started+CHAIN_SECONDS,
        formal_initialization='Fresh original BF16; no technical weights or old timed-out state reuse')
    search.validate_request(request)
    return request
