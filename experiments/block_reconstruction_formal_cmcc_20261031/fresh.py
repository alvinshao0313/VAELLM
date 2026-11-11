"""Independent BR36 endpoint-only strict state and actual BF16 probe readback."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from . import bindings as b
from . import source
from experiments.block_reconstruction_cmcc_20261030.fresh import comparison
from experiments.block_reconstruction_cmcc_20261030.run_short import probe


@b.torch.no_grad()
def audit(directory, gpu):
    started = time.monotonic()
    report_path = directory/'report.json'
    report = b.read(report_path)
    b.require(report.get('schema') == 'BR36_H96_32_execution_v1' and report['physical_GPUs'] == [gpu]
        and report['validation_only'] is report['prototype_only'] is False,
        'Fresh must consume this actual formal endpoint on its identical single-GPU deployment path.')
    model, meta, loaded, budget = b.checkpoint.load(directory/'final_model')
    report, expected = source.execution(directory/'final_model', report_path, meta)
    actual = b.fit.state_records(model)
    b.require(not loaded.missing_keys and not loaded.unexpected_keys and actual == expected,
              'Every independently reloaded persistent field must be strict and exact.')
    actual_D = {name for index in b.LAYERS for name in b.runtime.decoder_parameters(model, index)}
    b.require(actual_D == set(report['trained_decoder_names'])
        and b.fit.proof_from_records(actual, actual_D) == report['source_frozen_complement'],
        'The fresh actual decoder union or frozen complement differs from source evidence.')
    b.deployment.deployment_contract(model)
    b.deployment.budget_contract(budget, report['source_C0_binding']['bit_budget'])
    b.require(budget == report['export']['bit_budget'], 'The complete fresh mixed ledger differs from the saved endpoint.')
    fixture = b.torch.load(directory/'same_input_output.pt', map_location='cpu', weights_only=True)
    inputs = {key: fixture[key] for key in ('input_ids', 'attention_mask', 'position_ids')}
    b.require(fixture['schema'] == 'BR_BF16_same_input_v1' and b.digest(inputs) == fixture['input_signature'],
              'The actual original complete input fixture changed.')
    warm = b.runtime.prepare_full_deployment(model, 'cuda:0')
    logits = probe(model, fixture)
    compared = comparison(fixture['logits'], logits)
    b.write(directory/'fresh_comparison.json', compared)
    b.require(compared['exact'], 'Same BF16 deployment/group8/full input fresh logits differ.')
    b.runtime.offload_full_deployment(model)
    b.require(b.fit.state_records(model) == expected, 'Deployment prewarm or readout changed a persistent field.')
    value = dict(schema='BR36_H96_32_fresh_audit_v1', status='verified', validation_only=False, prototype_only=False,
        checkpoint_id=meta['checkpoint_id'], source_C0_checkpoint_id=report['source_C0_binding']['checkpoint_id'],
        full_state_exact=True, persistent_tensors_compared=len(actual), strict_missing_keys=list(loaded.missing_keys),
        strict_unexpected_keys=list(loaded.unexpected_keys), frozen_complement_exact=True,
        source_frozen_complement=report['source_frozen_complement'], trained_D_leaves=len(actual_D),
        same_complete_input_BF16_logits_exact=True, finite=True, comparison=compared,
        parent_weights_loaded=False, teacher_loaded=False, bit_budget=budget, full_deployment_prewarm=warm,
        training_report=b.file_binding(report_path), export_meta=b.file_binding(directory/'final_model'/b.checkpoint.META_FILE),
        source_state=report['source_state'], expected_state=report['export']['expected_state'], fixture=report['export']['fixture'],
        block_reports=report['block_reports'], source_geometry_sha256=report['source_geometry_sha256'],
        input_tokens=fixture['input_ids'].shape[-1], physical_GPUs=[gpu], elapsed_seconds=time.monotonic()-started,
        scope='Independent full BR36 saved-state and deployment readback; no quality or exact-resume claim.')
    b.write(directory/'fresh_audit.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=range(8), required=True)
    args = parser.parse_args()
    args.directory = args.directory.resolve(strict=True)
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu)
        and b.torch.cuda.device_count() == 1, 'Use caller bitvae and the identical assigned single GPU.')
    b.require(not (args.directory/'fresh.exit').exists() and not (args.directory/'fresh_audit.json').exists(),
              'Never overwrite or retry an independent fresh result.')
    b.sequence.check_resources([str(args.gpu)])
    b.torch.set_num_threads(4)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    code = 1
    try:
        result = audit(args.directory, args.gpu)
        code = 0
        print(json.dumps(dict(status=result['status'], checkpoint_id=result['checkpoint_id'])), flush=True)
    except BaseException:
        b.write(args.directory/'fresh_failure.json', dict(error=traceback.format_exc()))
        raise
    finally:
        (args.directory/'fresh.exit').write_text(str(code)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
