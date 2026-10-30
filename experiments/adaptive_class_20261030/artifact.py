"""One real torch serializer, including all metadata and every deployed tensor."""
from io import BytesIO
from pathlib import Path

import torch

from experiments.adaptive_native_model_20261029.codec import decode_lengths
from experiments.adaptive_native_model_20261029.consumer import CompactRVQLinear
from experiments.adaptive_native_model_20261029.spec import resolve
from experiments.adaptive_vector_rvq import common as c
from .sparse import Fixed2RiceLinear

SCHEMA = 'whole_gate_compact_artifact_v1'


def dynamic_serialize(bundle, persistent_fields):
    c.require(not persistent_fields, 'No extra deployment fields are enabled in this fixed same-K comparison.')
    lengths = decode_lengths(bundle['length_frame'], bundle['n_vectors'])
    spec = resolve(bundle, lengths)
    state = {'length_frame': bundle['length_frame']}
    state.update({f'codes_{index}': stage['codes'] for index, stage in enumerate(bundle['stages'])})
    state.update({f'decoders.{index}.{name}': tensor for index, stage in enumerate(bundle['stages'])
                  for name, tensor in stage['decoder_state'].items()})
    state.update({name: value for name, value in bundle['fixed'].items() if not name.startswith('rotation_')})
    state.update({f'weight_rotation.{axis}_signs': bundle['fixed'][f'rotation_{axis}_signs'] for axis in ('input', 'output')})
    return serialize(spec, state)


def serialize(spec, state):
    stream = BytesIO()
    torch.save(dict(schema=SCHEMA, spec=spec, state=state), stream)
    return stream.getvalue()


def save_module(module, path):
    state = {name: value.detach().cpu().contiguous().clone() for name, value in module.state_dict().items()}
    data = serialize(module.metadata(), state)
    with Path(path).open('xb') as stream:
        stream.write(data)
    return len(data)


def load(path):
    value = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    c.require(value['schema'] == SCHEMA, 'Use this comparison\'s actual complete serializer.')
    kind = value['spec'].get('consumer')
    module = (Fixed2RiceLinear if kind == 'fixed2_rice7' else CompactRVQLinear).from_state(value['spec'], value['state'])
    c.require(c.digest(module.state_dict()) == c.digest(value['state']), 'Fresh complete layer fields differ.')
    return module, value
