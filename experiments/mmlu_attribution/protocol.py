"""Fixed native MMLU subset and model identities for paired restoration."""
import hashlib
import json
from pathlib import Path

from experiments.longbench_v1.common import check_hashes, check_stat, sha256

ARMS = {'dense': (), 'compressed': (), 'qk': ('q_proj', 'k_proj'),
        'ov': ('o_proj', 'v_proj'), 'gate': ('gate_proj',),
        'up': ('up_proj',), 'down': ('down_proj',)}
PROTOCOL = {'task': 'mmlu', 'subjects': 57, 'limit_per_subject': 32, 'samples': 1824,
            'choice_requests': 7296, 'num_fewshot': 0, 'batch_size': 64,
            'metric': 'acc,none', 'apply_chat_template': False, 'fewshot_as_multiturn': False,
            'system_instruction': None, 'fewshot_random_seed': 1234,
            'sampling': 'Unmodified native doc_iterator(limit=32), first32 of every official test subject; no answer/prediction selection.',
            'scope': 'Paired diagnostic subset only; not full MMLU or nine-term acceptance.'}


def require(value, message):
    if not value:
        raise ValueError(message)


def value_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def case_identity(task, doc_id, doc, arguments):
    arguments = [list(pair) for pair in arguments]
    require(len(arguments) == 4 and all(len(pair) == 2 and all(isinstance(x, str) for x in pair)
                                       for pair in arguments), 'Require four native string likelihood pairs.')
    return {'task': str(task), 'doc_id': int(doc_id), 'doc': doc, 'arguments': arguments,
            'doc_sha256': value_sha(doc), 'arguments_sha256': value_sha(arguments)}


def check_preparation(path):
    path = Path(path).resolve()
    record = json.loads(path.read_text())
    require(record['schema'] == 'mmlu_category_restore_v1'
            and record['status'] == 'prepared_not_scheduled' and record['protocol'] == PROTOCOL
            and record['architecture']['base_model_path'] == record['hf_path']
            and record['model_policy']['model_type'] == record['architecture']['model_type']
            and record['architecture']['model_type'] in ('qwen3', 'llama'), 'Unexpected diagnostic source/protocol.')
    check_hashes(record['source_sha256'])
    check_hashes(record['input_evidence_sha256'])
    for item in record['hf_weight_stats']:
        check_stat(item)
    meta = json.loads((Path(record['source_checkpoint']) / 'checkpoint_meta.json').read_text())
    require(meta['checkpoint_id'] == record['source_checkpoint_id']
            and str(Path(meta['base_model_path']).resolve()) == record['hf_path'], 'Bound model identity changed.')
    require(sha256(record['cases']['path']) == record['cases']['sha256'], 'Paired input archive changed.')
    cases = [json.loads(line) for line in Path(record['cases']['path']).read_text().splitlines()]
    require(len(cases) == 1824 and len({(row['task'], row['doc_id']) for row in cases}) == 1824,
            'Missing or duplicate paired cases.')
    return record, cases
