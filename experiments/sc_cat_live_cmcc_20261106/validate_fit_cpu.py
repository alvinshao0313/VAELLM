"""Read the real frozen inputs and compile the live bridge without GPU work."""
import argparse
import os
from pathlib import Path
import time

import torch

from experiments.cat_auto_protocol_cmcc_20261104 import live as native
from experiments.cat_auto_protocol_cmcc_20261104.geometry import component_plan
from . import live_io as io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.time()
    io.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'This validation is CPU-only.')
    manifest = io.read(args.source_manifest)
    request = dict(source_manifest=io.binding(args.source_manifest), inventory_receipt=manifest['inventory'], data_plan_path=manifest['data_plan'])
    inventory, prepared, sources, tokenizer = io.load_inputs(request, with_sources=False)
    before = (native.fit_next_live_category, dict(native.fit_next_live_category.__globals__))
    entry = io.source9_live_entry()
    io.require(entry is not before[0] and native.fit_next_live_category is before[0]
        and before[1] == native.fit_next_live_category.__globals__, 'Private compilation mutated shared native code or globals.')
    pins = io.verify_fit_sources(manifest)
    selected = io.read(Path(manifest['output_root']) / 'selected_settings.json')
    rows = []
    for category in selected['categories']:
        if category['category_id'] not in io.ORDER[:2]:
            continue
        for role, reference in (('anchor', category['anchor']), ('best_different', category['best_different']['result'])):
            record = io.read(io.bound_path(reference))
            recipe = record['recipe']
            plan = component_plan(inventory, recipe)
            io.require(plan['capability']['constructible'], 'The actual selected mixed geometry is not constructible.')
            rows.append(dict(category_id=category['category_id'], role=role, candidate_id=record['candidate_id'],
                D=recipe['D'], K=recipe['K'], B=recipe['B'], source=reference, capability=plan['capability']))
    io.require(len(rows) == 4 and sources == [] and not torch.cuda.is_initialized(), 'Only the four approved real recipes were inspected.')
    io.write(args.output, dict(status='CPU_real_inputs_and_source9_live_bridge_bound', exit_code=0,
        cuda_initialized=False, data_identity=prepared['data_identity'], tokenizer_class=type(tokenizer).__name__,
        rows=rows, source_pins=pins, elapsed_seconds=time.time()-started,
        boundary='Actual input/source/geometry/bridge binding only; no model load, GPU fit, fresh forward or quality claim.'))


if __name__ == '__main__':
    main()
