"""CPU strict readback of the isolated real two-update B endpoint."""
from pathlib import Path
import argparse
import os

import common as c


def main(report_path):
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Short readback remains on CPU.')
    report_path = Path(report_path).resolve()
    report = c.read(report_path)
    c.require(report['status'] == 'short2_complete_not_formal' and report['exit_code'] == 0
        and report['short_recovery']['actual_steps'] == 2, 'Require the actual complete two-update validation.')
    event = report['saved_boundaries'][-1]
    model, meta, loaded = c.fork.load_cpu_boundary(event['path'])
    c.require(not loaded.missing_keys and not loaded.unexpected_keys
        and c.fork.state_digest(model.state_dict()) == event['state'], 'Short native full-model fresh strict readback changed.')
    c.require(meta['converted_module_count'] == 180 and len(meta['pending_dense_targets']) == 72
        and meta['lora_config'] is None, 'Invalid native merged short endpoint inventory.')
    c.fork.check_model(model, meta)
    c.require(not c.fork.torch.cuda.is_initialized(), 'Short CPU readback initialized CUDA.')
    result = dict(status='verified', actual_steps=2, schedule_max_steps=500, formal_parent_eligible=False,
        preparation_sha256=report['preparation_sha256'], source_checkpoint_id=report['source_checkpoint_id'],
        short_report=str(report_path), short_report_sha256=c.sha(report_path),
        endpoint=event, fresh_strict_state_equal=True, CUDA_initialized=False,
        rule='Formal A/B restart from their untouched real after_o forks and full original schedules; no short state is inherited.')
    path = c.OUTPUT / 'short2_validation.json'
    c.require(not path.exists(), 'Never overwrite short validation evidence.')
    c.write(path, result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=c.OUTPUT / 'B/short2.json')
    main(parser.parse_args().report)
