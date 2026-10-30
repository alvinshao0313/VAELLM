"""Explicit unchanged build/export/consumer steps for the actual aligned06 source."""
import argparse
import json
import os
import sparse07_bindings as b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'export', 'consume'))
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    prepared = b.read_prepared()
    if args.command == 'export':
        b.c.cpu_only()
        b.builder.export(b.OUTPUT / 'prepared.json', b.ARTIFACT)
        _, audit, _ = b.check_export()
        containers = dict(native_base_tensor_bytes=prepared['source_bit_budget']['full_model_tensor_bytes'],
            native_base_file_bytes=audit['native_base_file_bytes'], Rice_tensor_bytes=audit['payload_tensor_bytes'],
            Rice_file_bytes=audit['payload_file_bytes'])
        containers['native_base_container_bytes'] = containers['native_base_file_bytes']-containers['native_base_tensor_bytes']
        containers['Rice_container_bytes'] = containers['Rice_file_bytes']-containers['Rice_tensor_bytes']
        b.require(containers['native_base_container_bytes'] >= 0 and containers['Rice_container_bytes'] >= 0,
            'Actual weight container size cannot be smaller than its complete tensor storage.')
        audit.update(aligned06_source_audit=prepared['source_native_audit'], container_accounting=containers,
            packed_main_code_padding_bits=audit['bit_budget']['code_padding_bits'])
        b.write(b.OUTPUT / 'audit.json', audit)
        b.write(b.CHECKPOINT / 'assembly_report.json', audit)
        result = dict(status=audit['status'], checkpoint_id=audit['checkpoint_id'], bit_budget=audit['bit_budget'],
            quality_evaluation_started=False, measured_new_source=True)
    else:
        b.require(args.gpu is not None and os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu)
            and b.torch.cuda.device_count() == 1, 'Use exactly one explicitly assigned visible GPU.')
        if args.command == 'build':
            result = b.builder.build(b.OUTPUT / 'prepared.json', b.ARTIFACT)
        else:
            checkpoint, audit, _ = b.check_export()
            if audit['status'] == 'over_budget':
                result = dict(status='over_budget', quality_evaluation_started=False, consumer_started=False)
            else:
                b.require(prepared['source_checkpoint_id'] == audit['source_checkpoint_id'], 'Actual06 source changed.')
                b.torch.set_num_threads(4)
                b.torch.manual_seed(137)
                b.torch.backends.cuda.matmul.allow_tf32 = False
                b.torch.backends.cudnn.allow_tf32 = False
                result = b.runtime.verify(checkpoint, b.OUTPUT / 'gpu_validation.json')
    print(json.dumps(dict(status=result['status'], output=str(b.OUTPUT), actual06_source_bound=True)), flush=True)


if __name__ == '__main__':
    main()
