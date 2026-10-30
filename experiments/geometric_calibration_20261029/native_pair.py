"""Read-only reconstruction of the audited G01/G02 native projection pair."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/home/star/VAELLM')
HERE = Path(__file__).resolve().parent
SNAPSHOT = ROOT / 'result/qwen3_8b_mmlu65_20261004/rotation_factor_probe/source_snapshot_20261005_01'
sys.path.insert(0, str(SNAPSHOT))
sys.dont_write_bytecode = True

import torch
from train_utils import checkpoint_v6

SOURCE_ROOT = Path('/data/star/VAELLM/cat_aligned_recovery_20261028/G')
OUTPUT_ROOT = Path('/data/star/VAELLM/geometric_calibration_20261029')
STAGES = ('01_qk_fineweb2000', '02_online_opd32')
SOURCE_IDS = ('44ed6d15-abf0-46b0-a627-998a6406087a', '38a7461e-a2f1-4586-80e2-b99883ef2628')
PROJECTIONS = ('model.layers.18.self_attn.k_proj', 'model.layers.18.mlp.gate_proj',
               'model.layers.18.mlp.down_proj')
DECODER_PREFIX = '_parallel_stage_decoder.'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_record(value):
    value = value.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())


def schema(module):
    return {name: [list(value.shape), str(value.dtype)] for name, value in module.state_dict().items()}


def state_record(state):
    return {name: tensor_record(value) for name, value in state.items()}


class NativePair:
    def __init__(self):
        require(Path(checkpoint_v6.__file__).resolve().is_relative_to(SNAPSHOT),
                'The audited snapshot must supply the native checkpoint implementation.')
        self.audits = [read_json(SOURCE_ROOT / stage / 'native_audit.json') for stage in STAGES]
        require(all(item['status'] == 'verified' for item in self.audits), 'Source audits must be verified.')
        require(self.audits[1]['source_checkpoint_id'] == self.audits[0]['checkpoint_id'],
                'G02 must be the actual recovery child of G01.')
        self.directories = [Path(item['checkpoint_dir']) for item in self.audits]
        self.metas = [read_json(path / 'checkpoint_meta.json') for path in self.directories]
        require(tuple(item['checkpoint_id'] for item in self.metas) == SOURCE_IDS,
                'Unexpected audited G01/G02 identity.')
        self.check_source_stats()
        self.mapped = [torch.load(directory / meta['state_dict_file'], map_location='cpu',
                                  weights_only=True, mmap=True)
                       for directory, meta in zip(self.directories, self.metas, strict=True)]
        layers = sorted({int(item['name'].split('.')[2]) for item in self.metas[0]['converted_modules']})
        require(layers[len(layers) // 2] == 18, 'The preselected representative depth changed.')

    def check_source_stats(self):
        for audit, directory, meta in zip(self.audits, self.directories, self.metas, strict=True):
            require(meta['checkpoint_id'] == audit['checkpoint_id'], 'Audit/meta identity mismatch.')
            stat = (directory / meta['state_dict_file']).stat()
            expected = audit['checkpoint_weight_stat']
            require((stat.st_size, stat.st_mtime_ns, stat.st_ino) ==
                    (expected['size'], expected['mtime_ns'], expected['inode']),
                    'The audited source file changed; stop rather than silently rebinding.')

    def projection(self, name):
        require(name in PROJECTIONS, 'This bounded prototype permits only the three preselected matrices.')
        specs = [next(item for item in meta['converted_modules'] if item['name'] == name)
                 for meta in self.metas]
        require(specs[0] == specs[1], 'Parent/child projection schemas differ.')
        prefix = name + '.'
        states = [{key[len(prefix):]: value for key, value in mapped.items() if key.startswith(prefix)}
                  for mapped in self.mapped]
        require(states[0].keys() == states[1].keys(), 'Parent/child persistent field names differ.')
        frozen = [key for key in states[0] if not key.startswith(DECODER_PREFIX)]
        require(all(torch.equal(states[0][key], states[1][key]) for key in frozen),
                'Codes/protection/rotation/other persistent fields must be exactly fixed.')
        require(any(not torch.equal(states[0][key], states[1][key]) for key in states[0]
                    if key.startswith(DECODER_PREFIX)), 'The genuine recovery decoder update is missing.')
        return specs[0], states

    def manifest(self):
        return dict(source_ids=list(SOURCE_IDS), source_paths=[str(p) for p in self.directories],
                    source_weight_stats=[item['checkpoint_weight_stat'] for item in self.audits],
                    selection_rule='Preselected median depth L18 complete K, gate, down matrices; not a final layer selector',
                    projections=list(PROJECTIONS), native_snapshot=str(SNAPSHOT),
                    native_source_sha256={str(p.relative_to(SNAPSHOT)): sha256(p) for p in
                        (SNAPSHOT / 'train_utils/checkpoint_v6.py', SNAPSHOT / 'litebsq/vae_linear.py')})


def rebuild(spec, state, device):
    model = torch.nn.Module()
    model.projection = torch.nn.Linear(spec['in_features'], spec['out_features'], bias=spec['has_bias'])
    checkpoint_v6._rebuild_converted_modules(model, [dict(spec, name='projection')])
    model.projection.load_state_dict(state, strict=True)
    model = model.to(device=device).eval().requires_grad_(False)
    checkpoint_v6.refresh_vae_linear_runtime_after_state_load(model)
    require(schema(model.projection) == {key: [list(value.shape), str(value.dtype)]
                                         for key, value in state.items()},
            'Fresh strict rebuild changed persistent schema or dtype.')
    return model


@torch.no_grad()
def deployed_weight(model):
    module = model.projection
    module.eval().disable_trainable_decode()
    module._decoder_compute_dtype = torch.bfloat16
    module.clear_decoded_weight_cache()
    with torch.autocast(next(module.parameters()).device.type, enabled=False):
        weight = module._decode_weight(dtype=torch.bfloat16)
    require(weight.dtype == torch.bfloat16 and bool(torch.isfinite(weight).all()),
            'Native deployed BF16 decode returned a nonfinite weight or wrong dtype.')
    return weight
