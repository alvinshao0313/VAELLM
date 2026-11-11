"""Freeze actual five-source streams and Alpaca calibration without model/GPU work."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from . import data as d
from .contracts import require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--sources', required=True)
    parser.add_argument('--search-seed', type=int, default=31)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('HF_DATASETS_OFFLINE') == '1'
        and os.environ.get('HF_HUB_OFFLINE') == '1', 'Prepare data on CPU from existing local bytes, without a download fallback.')
    require(not output.exists(), 'Use an independent output; never overwrite a frozen data plan.')
    output.mkdir(parents=True)
    began = time.monotonic()
    torch.set_num_threads(4)
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from e2e_common.data import DATASET_MIX_SOURCE_PRESETS, build_tokenizer
    from e2e_common.lazy_datasets import build_calibration_input_ids_lazy
    from train_utils.distill_data import build_distill_data_collator

    inventory = json.loads(Path(args.inventory).read_text(encoding='utf-8'))['inventory']
    registry = json.loads(Path(args.sources).read_text(encoding='utf-8'))
    require(tuple(row['alias'] for row in registry['sources']) == tuple(d.MIX), 'Preserve original source enumeration.')
    tokenizer = build_tokenizer(inventory['source_evidence']['path'], None)
    sources = []
    for row in registry['sources']:
        source = d.load_local_source(row['alias'], revision='converted-artifact-sha256:' + row['sha256'],
            license_record=row['license'], provenance_record=row['provenance'], expected_sha256=row['sha256'])
        require(len(source.raw) == row['rows'], 'The actual migrated source row count differs from its evidence.')
        sources.append(source)
        print(json.dumps(dict(source=source.alias, rows=len(source.raw), actual_sha256=row['sha256'])), flush=True)
    spec = registry['calibration']
    arrow = Path(spec['arrow'])
    info_path = arrow.with_name('dataset_info.json')
    info = json.loads(info_path.read_text(encoding='utf-8'))
    require(info['splits']['train']['num_examples'] == spec['rows'] and spec['revision'] in info_path.parts,
        'Bind the actual cached Alpaca revision/train partition.')
    calibration_source = d.bind_source(Dataset.from_file(str(arrow)), DATASET_MIX_SOURCE_PRESETS['alpaca'],
        revision=spec['revision'], license_record=spec['license'],
        provenance_record=dict(repository=spec['repository'], split='train', artifact_scope='actual cached train Arrow',
            dataset_info_sha256=d.file_sha256(info_path)), artifact_digest=d.file_sha256(arrow))
    require(len(calibration_source.raw) == spec['rows'], 'The actual Alpaca train rows differ.')
    prepared = d.freeze_data_plan(sources, calibration_source, tokenizer,
        model_identity=inventory['model_identity'], category_ids=inventory['category_ids'], search_seed=args.search_seed)
    original = build_calibration_input_ids_lazy('alpaca=1', tokenizer=tokenizer, nsamples=512, seqlen=256, seed=31)
    require(len(original) == 512 and all(torch.equal(a, b) for a, b in zip(original, prepared['calibration'])),
        'Root-frozen calibration changed the original actual Alpaca seed31/512x256 tokens.')
    category = inventory['category_ids'][0]
    bundle = d.build_update_bundle(prepared, sources, tokenizer, category)
    update = prepared['manifest']['updates'][category]
    loader = DataLoader(bundle.train_dataset, batch_size=4, num_workers=16, pin_memory=False,
        generator=torch.Generator().manual_seed(update['data_seed']),
        collate_fn=build_distill_data_collator(tokenizer, model_max_length=1024, dynamic_padding=True))
    iterator, observer = iter(loader), d.UpdateObserver(update)
    try:
        for index in range(32):
            observer(next(iterator), index)
    finally:
        iterator._shutdown_workers()
    require(not torch.cuda.is_initialized(), 'Source/data preparation initialized CUDA.')
    prepared['local_sources_registry'] = registry
    prepared['local_sources_registry_sha256'] = d.file_sha256(args.sources)
    torch.save(prepared, output / 'data_plan.pt')
    manifest = prepared['manifest']
    report = dict(status='actual_five_source_data_plan_CPU_verified', exit_code=0, cuda_initialized=False,
        data_identity=prepared['data_identity'], data_plan_sha256=d.file_sha256(output / 'data_plan.pt'),
        inventory_receipt_sha256=d.file_sha256(args.inventory), source_registry_sha256=d.file_sha256(args.sources),
        category_ids=inventory['category_ids'], full_potential_T1000_prefix_per_category=True,
        source_counts={source.alias: len(source.raw) for source in sources},
        calibration512x256_matches_original_Alpaca=True, calibration_source=calibration_source.identity,
        update_seeds={category: update['data_seed'] for category, update in manifest['updates'].items()},
        proposal_source_counts=manifest['proposal']['source_counts'],
        actual_training_bundle_observer=observer.report(), proposal_update_calibration_ID_exclusion=True,
        elapsed_seconds=time.monotonic() - began,
        source_sha256={str(Path(__file__).with_name(name)): d.file_sha256(Path(__file__).with_name(name))
            for name in ('prepare_data.py', 'data.py', 'identity.py', 'contracts.py')},
        boundary='Actual local train bytes/tokenizer/source sampling/row exclusion only; no model forward, quality score or GPU runtime proof.')
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('status', 'exit_code', 'elapsed_seconds', 'data_identity')}), flush=True)


if __name__ == '__main__':
    main()
