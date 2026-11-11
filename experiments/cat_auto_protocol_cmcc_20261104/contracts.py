"""Small JSON contracts shared by planning, data and recovery implementations."""
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
import math
from typing import Any, Mapping

SCHEMA = 'cat_auto_protocol_v0_20261104'
COVERAGE_STATUSES = ('tested', 'matched_reuse', 'math_invalid', 'impl_blocked', 'budget_deferred', 'v0_fixed')
CAPABILITY_STATUSES = ('math_invalid', 'impl_blocked', 'constructible', 'verified')
FIXED_VAE_LOSS = dict(reconstruction='mse', quantizer='BSQ', gamma0=1., gamma=1., zeta=1., inv_temperature=100.,
    l1_weight=1., lfq_weight=2.5, commitment_loss_weight=.25, entropy_loss_weight=.01)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def json_value(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        require(all(isinstance(key, str) for key in value), 'Canonical object keys must be explicit strings.')
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        require(math.isfinite(value), 'Canonical identities cannot contain NaN or infinity.')
        return value
    raise ValueError('Not a canonical JSON value: ' + type(value).__name__)


def canonical_json(value: Any) -> str:
    return json.dumps(json_value(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return sha256(canonical_json(value).encode('utf-8')).hexdigest()


def _fields(value, fields, name):
    require(isinstance(value, Mapping), name + ' must be an object.')
    require(set(fields) <= set(value), name + ' missing fields: ' + ','.join(sorted(set(fields) - set(value))))
    return json_value(value)


def _text(value, name):
    require(isinstance(value, str) and bool(value.strip()), name + ' must be a nonempty string.')


def _integer(value, name, minimum=0):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        name + ' must be an integer >= ' + str(minimum))


def validate_inventory(value):
    result = _fields(value, ('model_identity', 'adapter_id', 'adapter_version', 'category_ids', 'projections', 'total_target_numel'), 'ModelInventory')
    _text(result['adapter_id'], 'adapter_id')
    require(result['category_ids'] and len(set(result['category_ids'])) == len(result['category_ids']), 'Semantic categories must be explicit and unique.')
    names, total = set(), 0
    for row in result['projections']:
        _fields(row, ('actual_name', 'category_id', 'layer_id', 'shape', 'dtype', 'numel'), 'projection')
        _text(row['actual_name'], 'actual_name')
        _integer(row['layer_id'], 'layer_id')
        require(row['actual_name'] not in names and row['category_id'] in result['category_ids'], 'Duplicate projection or unknown semantic category.')
        names.add(row['actual_name'])
        require(len(row['shape']) == 2, 'Projection geometry must be a matrix.')
        for size in row['shape']:
            _integer(size, 'shape', 1)
        require(row['numel'] == math.prod(row['shape']), 'Actual projection shape and numel disagree.')
        _text(row['dtype'], 'dtype')
        total += row['numel']
    require(names and set(row['category_id'] for row in result['projections']) == set(result['category_ids'])
        and total == result['total_target_numel'], 'Full semantic inventory or shared denominator is incomplete.')
    return result


def inventory_id(value):
    value = validate_inventory(value)
    return digest({key: value[key] for key in ('model_identity', 'adapter_id', 'adapter_version',
        'category_ids', 'projections', 'total_target_numel')})


def validate_run_identity(value):
    result = _fields(value, ('model', 'parent_uuid', 'parent_weights_digest', 'data_identity', 'code_identity', 'consumer_identity', 'search_seed'), 'RunIdentity')
    _text(result['parent_uuid'], 'parent_uuid')
    _text(result['parent_weights_digest'], 'parent_weights_digest')
    _integer(result['search_seed'], 'search_seed')
    require(all(result[key] for key in ('model', 'data_identity', 'code_identity', 'consumer_identity')), 'A reusable identity cannot omit model/data/code/consumer evidence.')
    return result


def validate_recipe(value):
    result = _fields(value, ('category_id', 'transpose', 'rotation', 'D', 'K', 'S', 'B', 'encoder', 'decoder', 'normalization',
        'protection', 'optimizer', 'loss', 'steps', 'schedule', 'warmup', 'linear_group_size', 'group_signature'), 'ResolvedRecipe')
    _text(result['category_id'], 'category_id')
    require(isinstance(result['transpose'], (bool, Mapping)), 'Explicit uniform or per-layer transpose is required.')
    for key in ('D', 'K', 'S', 'B', 'linear_group_size'):
        _integer(result[key], key, 1)
    require(result['B'] * result['D'] == 262144, 'Preserve B*D=262144; do not silently lower batch.')
    require(result['steps'] == 10000 and result['schedule'] == 'linear' and result['warmup'] == 0, 'Every VAE stage uses the complete original linear10000 schedule.')
    require(result['S'] in (2, 4), 'v0 supports coreS2 and the registered conditionalS4 only.')
    require(isinstance(result['group_signature'], list) and len(result['group_signature']) == result['linear_group_size']
        and len(set(result['group_signature'])) == result['linear_group_size'], 'The pooled group signature must name every actual category Linear once.')
    _fields(result['rotation'], ('kind',), 'rotation')
    _fields(result['protection'], ('axis', 'format', 'counts'), 'protection')
    require(result['protection']['axis'] in ('input', 'output') and result['protection']['format'] == 'int8', 'v0 protection is explicit single-axisINT8.')
    require(set(result['protection']['counts']) == set(result['group_signature']), 'Protection counts must resolve every actual projection.')
    for count in result['protection']['counts'].values():
        _integer(count, 'protection count')
    optimizer = _fields(result['optimizer'], ('kind', 'multiplier', 'base_lr', 'betas', 'weight_decay', 'kappa_contract'), 'optimizer')
    require(optimizer['kind'] in ('adamw', 'vae_muon') and optimizer['base_lr'] == .003
        and optimizer['betas'] == [.9, .95] and optimizer['weight_decay'] == 0,
        'Keep the VAE optimizer contract distinct from recovery AdamW.')
    require(optimizer['multiplier'] in (.25, .5, 1, 2, 4), 'Use only the confirmed actual update multipliers.')
    require('runtime_resolution' not in result and 'actual_kappa' not in optimizer,
        'Measured kappa belongs to result/artifact identity; never rename a queued candidate after fitting.')
    require(set(result['loss']) == set(FIXED_VAE_LOSS), 'The fixed VAE loss fields must all resolve explicitly.')
    changed = [name for name, fixed in FIXED_VAE_LOSS.items() if result['loss'][name] != fixed]
    require(len(changed) <= 1 and all(name in ('commitment_loss_weight', 'entropy_loss_weight')
        and result['loss'][name] in (FIXED_VAE_LOSS[name] * .5, FIXED_VAE_LOSS[name] * 2) for name in changed),
        'Conditional BSQ changes at most one registered contribution by .5 or2; all other loss coefficients stay fixed.')
    return result


def validate_ledger(value):
    result = _fields(value, ('N', 'logical_bits', 'packed_bits', 'tensor_bytes', 'directory_bytes', 'per_category',
        'future_dense_bits', 'component_fees', 'assumptions', 'source'), 'LedgerResult')
    for key in ('N', 'logical_bits', 'packed_bits', 'tensor_bytes', 'future_dense_bits'):
        _integer(result[key], key, 1 if key == 'N' else 0)
    require(result['logical_bits'] <= result['packed_bits'], 'Logical code count cannot exceed actual packed payload.')
    if result['directory_bytes'] is not None:
        _integer(result['directory_bytes'], 'directory_bytes')
        require(result['directory_bytes'] >= result['tensor_bytes'], 'Directory omitted actual tensor payload.')
    require(result['per_category'] and result['source'], 'Full ledger needs category ownership and a real source.')
    return result


def capability(status, reason, evidence=(), verification_scope='unverified'):
    require(status in CAPABILITY_STATUSES, 'Unknown capability status.')
    _text(reason, 'capability reason')
    require(status != 'verified' or bool(evidence), 'Verified capability needs scoped actual evidence.')
    return dict(status=status, reason=reason, evidence=json_value(list(evidence)), verification_scope=verification_scope,
        parseable=True, constructible=status in ('constructible', 'verified'))


def validate_boundary(value):
    result = _fields(value, ('uuid', 'path', 'digest', 'inventory_hash', 'completed_prefix', 'mutable_manifest',
        'export_evidence', 'fresh_evidence', 'actual_ledger', 'formal_eligible'), 'BoundaryReference')
    for key in ('uuid', 'path', 'digest', 'inventory_hash'):
        _text(result[key], key)
    require(isinstance(result['completed_prefix'], list) and len(set(result['completed_prefix'])) == len(result['completed_prefix']), 'Completed prefix must preserve actual semantic order.')
    require(isinstance(result['formal_eligible'], bool), 'Formal eligibility must be explicit.')
    if result.get('phase') == 'cpu_fixture' or result.get('verification_scope') == 'cpu_fixture':
        require(not result['formal_eligible'], 'A CPU fixture cannot acquire formal boundary identity.')
    validate_ledger(result['actual_ledger'])
    return result
