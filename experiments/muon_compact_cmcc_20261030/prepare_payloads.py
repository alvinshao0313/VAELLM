"""CPU-only adaptation of72 completed layer payloads; never assemble a C1 model."""
import io
import json
import os

from . import bindings as b
from . import optimizer as opt
from .prepare import exact
from experiments.adaptive_class_20261030 import artifact


def main():
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not b.torch.cuda.is_initialized(), 'CPU payload preparation only.')
    b.torch.set_num_threads(2)
    source = b.OUTPUT.parent/'adaptive_class_20261030'
    files = []
    for number in (1,2,3):
        run = source/f'formal_gpu{number}'
        b.require(run.with_suffix('.exit').read_text().strip() == '0', 'Actual formal36 fit must be complete.')
        files += sorted(run.glob('group_*/layer_*/*.pt'))
    files = [path for path in files if path.name in ('dynamic3.pt','fixed2_rice7.pt')]
    b.require(len(files) == 72, 'Need all actual36 dynamic and36 Rice layer payloads.')
    rows = []
    for path in files:
        module,raw = artifact.load(path)
        name = raw['spec']['geometry']['name']
        parameters = {name+'.decoders.'+key:value for key,value in module.decoders.named_parameters()}
        layout = {}
        for stage,(decoder,spec) in enumerate(zip(module.decoders,raw['spec']['stages'],strict=True)):
            item = spec['decoder_spec']
            suffixes = tuple(part+('.weight' if isinstance(getattr(decoder,part),b.torch.nn.Linear) else '.conv.weight')
                for part in ('linear_in','linear_out'))
            for suffix,shape in zip(suffixes,((item['hidden_dim'],item['in_dim']),(item['out_dim'],item['hidden_dim'])),strict=True):
                key = name+f'.decoders.{stage}.'+suffix
                value = parameters[key]
                b.require(tuple(value.shape) == (shape if isinstance(getattr(decoder,suffix.split('.')[0]),b.torch.nn.Linear) else (*shape,1)),
                    'Actual future compact matrix shape differs from metadata.')
                layout[key] = dict(stages=1,matrix_shape=[1,*shape],packed_shape=list(value.shape),elements=value.numel(),
                    consumer=path.stem,decoder_type=item['decoder_type'])
        matrices,affine = opt.split(parameters,layout)
        frozen_keys = [key for key in raw['state'] if not key.startswith('decoders.')]
        frozen_before = {key:b.core.tensor_record(module.state_dict()[key]) for key in frozen_keys}
        module.enable_decoder_training()
        b.require(all(value.dtype == b.torch.float32 for value in module.decoders.parameters())
            and module._decoded_cache is None and module._cache_versions is None,
            'Actual dynamic/Rice decoder FP32 training/cache interface failed.')
        for parameter in module.decoders.parameters(): parameter.data = parameter.data.bfloat16()
        b.require(frozen_before == {key:b.core.tensor_record(module.state_dict()[key]) for key in frozen_keys},
            'Dynamic lengths/code/protection/rotation or Rice payload changed.')
        exact(module.state_dict(),raw['state'],path.stem+' actual layer BF16 roundtrip')
        data = artifact.serialize(module.metadata(),{key:value.detach().cpu() for key,value in module.state_dict().items()})
        fresh = b.torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
        b.require(fresh['spec'] == raw['spec'], 'Serializer metadata changed.')
        exact(fresh['state'],raw['state'],path.stem+' serializer')
        rows.append(dict(path=str(path),source=b.binding(path),layer_name=name,consumer=path.stem,
            decoder_stages=len(module.decoders),matrix_leaves=len(matrices),affine_leaves=len(affine),
            actual_decoder_elements=sum(value.numel() for value in parameters.values()),layout=layout,
            persistent_tensors=len(raw['state']),frozen_tensors=len(frozen_keys),frozen_state_exact=True,
            BF16_FP32_BF16_decoder_values_exact=True,serializer_full_state_exact=True,serialized_bytes=len(data)))
    b.require(sum(row['consumer'] == 'dynamic3' for row in rows) == 36
        and sum(row['consumer'] == 'fixed2_rice7' for row in rows) == 36 and not b.torch.cuda.is_initialized(), 'Incomplete CPU payload coverage.')
    result = dict(schema='future_compact_payload_CPU_interfaces_v1',status='CPU_verified_full_model_GPU_pending',
        actual_completed_dynamic_layers=36,actual_completed_Rice_layers=36,CUDA_initialized=False,
        scope='Actual completed layer payloads only; no C1 assembly, no GPU pass, no final parent or quality claim',
        next_dependency='Each real complete dynamic/Rice final_model before its full-model OPD lifecycle short2',layers=rows)
    b.core.write(b.OUTPUT/'future_payloads_CPU.json',result)
    print(json.dumps({key:value for key,value in result.items() if key != 'layers'}),flush=True)


if __name__ == '__main__':
    main()
