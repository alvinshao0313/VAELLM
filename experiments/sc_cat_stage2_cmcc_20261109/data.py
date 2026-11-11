"""Freeze paired native knowledge inputs and the unchanged canonical generic stream."""
import argparse
from collections import Counter, defaultdict
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
from e2e_common.data import DATASET_MIX_SOURCE_PRESETS, record_to_mcqa_example
from train_utils.distill_data import encode_canonical_record
from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest
from .common import ROOT, read, require, binding, write, bound
from .prepare import AUX, KNOWLEDGE


def native_case(tokenizer, context, gold, source, row):
    # Verify the actual context/continuation boundary for this tokenizer.
    ids = tokenizer.encode(context, add_special_tokens=False)
    choices = []
    for letter in 'ABCD':
        pair = tokenizer.encode(context + ' ' + letter, add_special_tokens=False)
        require(pair[:len(ids)] == ids and len(pair) == len(ids) + 1,
                'Native four-choice continuation is not one token with a common prefix.')
        choices.append(pair[-1])
    require(len(set(choices)) == 4 and 0 <= gold < 4, 'Invalid option encoding or gold.')
    return dict(kind='knowledge', source=source, row=row, input_ids=torch.tensor(ids),
                choices=torch.tensor(choices), gold=int(gold), text_sha=digest(context))


def web_parser():
    directory = ROOT / 'result/qwen3_8b_mmlu65_20261004/qwen_webinstruct_train_inventory_20261015'
    old = list(sys.path)
    try:
        sys.path.insert(0, str(directory))
        spec = importlib.util.spec_from_file_location('_stage2_web_inventory', directory / 'inventory.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.inspect_row
    finally:
        sys.path[:] = old


def prepare(root):
    root = Path(root)
    audit = read(root / 'preparation.json')
    base_meta = read(bound(audit['base']['metadata']))
    teacher = str(Path(base_meta['base_model_path']).resolve())
    tokenizer = AutoTokenizer.from_pretrained(teacher, local_files_only=True, use_fast=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    old = torch.load(bound(audit['original_data']['plan']), weights_only=True, map_location='cpu')
    # Reuse a frozen, source-verified canonical sample stream, not old trained weights or losses.
    schedule = old['manifest']['updates']['k_proj']['batches'][:500]
    wanted = defaultdict(set)
    for batch in schedule:
        for row in batch['records']:
            wanted[row['alias']].add(row['raw_index'])
    records = {}
    for source in audit['original_data']['sources']:
        alias = source['alias']
        found = {}
        with open(source['path'], encoding='utf-8') as stream:
            for index, line in enumerate(stream):
                if index in wanted[alias]:
                    import json
                    found[index] = json.loads(line)
                if len(found) == len(wanted[alias]):
                    break
        require(set(found) == wanted[alias], 'Missing selected original rows: ' + alias)
        records[alias] = found
        print(dict(event='original_rows_loaded', source=alias, rows=len(found)), flush=True)
    aux = pq.read_table(AUX / 'auxiliary_train/train-00000-of-00001.parquet').to_pylist()
    batches = []
    for batch in schedule:
        encoded = []
        for reference in batch['records']:
            alias, index = reference['alias'], reference['raw_index']
            record = records[alias][index]
            require(digest(record) == reference['raw_digest'], 'Original selected record changed.')
            if alias == 'vaellm_eval_task' and index >= audit['original_data']['auxiliary_start_row']:
                aux_index = index - audit['original_data']['auxiliary_start_row']
                rendered = record_to_mcqa_example(aux[aux_index], text_format='mmlu_mcqa')
                context = str(rendered['prompt'])
                case = native_case(tokenizer, context, rendered['answer_index'], 'auxiliary', aux_index)
            else:
                preset = DATASET_MIX_SOURCE_PRESETS[alias]
                sample = encode_canonical_record(record, tokenizer, text_format=preset.text_format,
                    text_field=preset.text_field, task='sft', model_max_length=1024)
                require(sample is not None, 'An already accepted canonical record became invalid.')
                case = dict(kind='generic', source=alias, row=index, **sample)
            case['original_reference'] = reference
            encoded.append(case)
        require(len(encoded) == 4, 'The fixed update must contain four questions.')
        batches.append(encoded)

    additions = {}
    supply = {}
    parse = web_parser()
    for short, name in [('web', 'webinstruct'), ('med', 'medmcqa')]:
        spec = audit['new_sources'][name]
        rows = pq.read_table(spec['path']).to_pylist()
        eligible = []
        for index, row in enumerate(rows):
            if short == 'web':
                if row['answer_type'] != 'Multiple Choice':
                    continue
                reason, parsed, _ = parse(row)
                if reason != 'qualified':
                    continue
                context, gold = row['question'] + '\nAnswer:', 'ABCD'.index(parsed['gold_label'])
            else:
                choices = [row[k] for k in ('opa', 'opb', 'opc', 'opd')]
                require(type(row['cop']) is int and 0 <= row['cop'] < 4, 'Invalid original Med gold.')
                context = row['question'].strip() + '\n' + '\n'.join(
                    letter + '. ' + choice for letter, choice in zip('ABCD', choices)) + '\nAnswer:'
                gold = row['cop']
            eligible.append((index, context, gold))
        require(len(eligible) == (11219 if short == 'web' else 182822), 'Pinned structural supply changed.')
        # Seeded source permutation, no length, teacher, subject or test-based selection.
        order = np.random.default_rng(31).permutation(len(eligible))
        count = sum(c['kind'] == 'knowledge' for batch in batches for c in batch)
        additions[short] = [native_case(tokenizer, eligible[i][1], eligible[i][2], short, eligible[i][0])
                            for i in order[:count]]
        supply[short] = dict(official_rows=len(rows), eligible=len(eligible), selected_pool=count)

    arms = {}
    summaries = {}
    for arm, fractions in KNOWLEDGE.items():
        rng = np.random.default_rng(31)
        cursor = Counter()
        arm_batches = []
        for batch in batches:
            output = []
            for case in batch:
                if case['kind'] == 'knowledge':
                    source = str(rng.choice(['auxiliary', 'web', 'med'], p=np.array(fractions)/100))
                    if source != 'auxiliary':
                        case = additions[source][cursor[source]]
                        cursor[source] += 1
                output.append(case)
            arm_batches.append(output)
        arms[arm] = arm_batches
        flat = [c for batch in arm_batches for c in batch]
        summaries[arm] = dict(questions=len(flat), source_counts=dict(Counter(c['source'] for c in flat)),
            tokens_by_source={s: sum(len(c['input_ids']) for c in flat if c['source']==s)
                              for s in sorted({c['source'] for c in flat})},
            max_length=max(len(c['input_ids']) for c in flat),
            max_knowledge_length=max(len(c['input_ids']) for c in flat if c['kind']=='knowledge'))
    require(all(s['max_length'] <= tokenizer.model_max_length for s in summaries.values()),
            'Selected complete knowledge question exceeds actual supported context; do not truncate or skip.')
    destination = root / 'data.pt'
    require(not destination.exists(), 'Do not overwrite a frozen data plan.')
    torch.save(dict(arms=arms, teacher=teacher, pad_token_id=tokenizer.pad_token_id,
                    model_identity=old['manifest']['model_identity']), destination)
    write(root / 'data.json', dict(payload=binding(destination), teacher=teacher, summaries=summaries,
        source_audit=binding(root/'preparation.json'), supply=supply,
        original_stream='existing frozen canonical k_proj first500 B4 accepted record identities only',
        generic_length='canonical max1024, original renderer and masks',
        knowledge_length='full native prompt, no length filter or truncation',
        source_sampling='original fixed source slots; seeded categorical replacement only within knowledge slots',
        actual_steps=500, questions_per_update=4, cuda_initialized=torch.cuda.is_initialized()))
    print(dict(event='data_frozen', summaries=summaries, supply=supply), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    prepare(parser.parse_args().root)
