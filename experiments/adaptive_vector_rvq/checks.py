"""CPU contracts on real HF vectors; these checks make no quality claim."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback

import torch

from . import common as c
from . import codec
from .budget import counts_for_budget, select_active
from .training import fit_stage


def checks(prepared_dir, output):
    torch.set_num_threads(4)
    c.require(not torch.cuda.is_initialized() and not output.exists(), 'Use a fresh CPU-only contract output.')
    prepared = c.check_prepared(prepared_dir)
    data = torch.load(prepared['layers']['0']['path'], map_location='cpu', weights_only=True, mmap=True)
    output.mkdir(parents=True)
    report = dict(status='running', started_unix=time.time(), budget=dict(passed=False),
        freshdecode=dict(passed=False), freeze=dict(passed=False), allfinite=False,
        scope='FP32 CPU codec/active-prefix contracts using real HF vectors; no model-quality conclusion or substitute for GPU full-shape validation.')
    path = output / 'report.json'
    c.write(path, report)
    try:
        pattern = torch.tensor([1, 2, 3, 4] * 4 + [3], dtype=torch.uint8)
        packed = codec.pack_lengths(pattern)
        c.require(packed.numel() == 5 and torch.equal(codec.unpack_lengths(packed, len(pattern)), pattern), 'Actual1..4/tail two-bit packing failed.')
        target = data['vectors'][:17].clone().contiguous()
        prefix = torch.zeros_like(target)
        previous = torch.arange(len(target))
        tied, _ = select_active(torch.ones_like(target), prefix, previous, 7, 1.)
        c.require(torch.equal(tied, torch.arange(7)), 'Stable ties must use ascending original IDs.')
        stages, gates, freeze = [], [], []
        lengths = torch.ones(len(target), dtype=torch.uint8)
        for index, count in enumerate((17, 9, 5, 2)):
            active = previous if index == 0 else select_active(target, prefix, previous, count, data['source_variance'])[0]
            before = [c.digest({'codes': stage['codes'], **stage['decoder_state']}) for stage in stages]
            stage, following, trained = fit_stage(target, prefix, active, steps=1, seed=c.SEED + 1000 * index,
                device='cpu', output=output / f'cpu_stage{index + 1}', source_variance=data['source_variance'], dtype='fp32')
            after = [c.digest({'codes': old['codes'], **old['decoder_state']}) for old in stages]
            c.require(before == after and trained['allfinite'], 'Actual later-stage training changed a frozen prefix.')
            inactive = torch.ones(len(target), dtype=torch.bool)
            inactive[active] = False
            c.require(torch.equal(following[inactive], prefix[inactive]), 'Inactive vectors must skip the whole fused stage.')
            stages.append(stage)
            lengths[active] = index + 1
            gates.append(active.tolist())
            freeze.append(dict(before=before, after=after, passed=True))
            prefix, previous = following, active
        c.require(set(lengths.tolist()) == {1, 2, 3, 4}, 'Real tiny contract must exercise every possible active length.')
        tiny = dict(mode='adaptive', n_vectors=len(target), geometry=data['geometry'], fixed=data['fixed'],
            stages=stages, lengths=codec.pack_lengths(lengths))
        codec_path = output / 'real_vectors.avrvq'
        actual_ledger = codec.save(tiny, codec_path)
        fresh = codec.load(codec_path)
        decoded = codec.decode(fresh, 'cpu')
        c.require(torch.equal(decoded, prefix)
            and c.digest(codec.tensor_inventory(tiny)) == c.digest(codec.tensor_inventory(fresh)), 'Actual FP32 fresh codec reconstruction/persistent fields changed.')
        c.require(actual_ledger['code_bytes'] == 6 * int(lengths.long().sum()) and actual_ledger['count_bytes'] == 5, 'Short-vector tails or count-byte inflation were persisted.')
        # Real full-N geometry establishes feasibility and exact physical accounting.
        n = prepared['layers']['0']['vector_count']
        states = [{key: value.bfloat16().contiguous() for key, value in stage['decoder_state'].items()} for stage in stages]
        decoder_bytes = sum(c.compact(value) for value in states[0].values())
        fixed_bytes = sum(c.compact(value) for value in data['fixed'].values())
        plan = counts_for_budget(n, decoder_bytes, fixed_bytes)
        budgets = {}
        for mode in ('fixed2', 'adaptive'):
            active_counts = [n, n] if mode == 'fixed2' else plan['active_counts']
            bundle = dict(mode=mode, n_vectors=n, geometry=data['geometry'], fixed=data['fixed'], stages=[])
            if mode == 'adaptive':
                complete = torch.ones(n, dtype=torch.uint8)
                for index, count in enumerate(active_counts[1:], start=2):
                    complete[:count] = index
                bundle['lengths'] = codec.pack_lengths(complete)
            for index, count in enumerate(active_counts):
                bundle['stages'].append(dict(decoder_spec=stages[index]['decoder_spec'], decoder_state=states[index],
                    normalization=stages[index]['normalization'], codes=torch.zeros((count, 6), dtype=torch.uint8)))
            diagnostic = output / (mode + '_full_N_storage_contract.avrvq')
            measured = codec.save(bundle, diagnostic)
            loaded = codec.load(diagnostic)
            c.require(codec.ledger(loaded) == measured and c.digest(codec.tensor_inventory(loaded)) == c.digest(codec.tensor_inventory(bundle)), 'Full-N physical storage/fresh load contract failed.')
            expected = plan['control_bytes' if mode == 'fixed2' else 'adaptive_bytes']
            c.require(diagnostic.stat().st_size == measured['total_bytes'] == expected, 'Full-N actual file size disagrees with the complete preallocated budget.')
            budgets[mode] = dict(**measured, actual_file_bytes=diagnostic.stat().st_size,
                fixture_scope='Zero code-pattern serialization only; true full-N shape/fixed payload/BF16 parameter geometry, no inference or quality claim.')
            # No future consumer uses this one-shot large pattern stream; retain its measured evidence.
            resolved = diagnostic.resolve()
            c.require(resolved.parent == output.resolve() and resolved.is_relative_to(c.OUTPUT), 'Cleanup must stay inside this task output.')
            diagnostic.unlink()
            c.require(not resolved.exists(), 'One-shot storage fixture cleanup failed.')
        report.update(status='completed', budget=dict(passed=True, plan=plan, actual_storage_contracts=budgets),
            freshdecode=dict(passed=True, real_vector_count=len(target), all_lengths_covered=True,
                actual_ledger=actual_ledger, original_and_fresh_tensor_digest=c.digest(codec.tensor_inventory(fresh))),
            freeze=dict(passed=True, previous_stage_checks=freeze, active_sets=gates), allfinite=True,
            storage_fixtures_removed_bytes=sum(row['actual_file_bytes'] for row in budgets.values()))
        c.require(not torch.cuda.is_initialized(), 'CPU vector contracts initialized CUDA.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['finished_unix'] = time.time()
        c.write(path, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    checks(args.prepared_dir.resolve(), args.output_dir.resolve())
