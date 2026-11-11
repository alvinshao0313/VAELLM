"""Bind complete compressed sources to the already selected category-aware N runtime."""
from pathlib import Path
import re

# This module closes all four namespace search paths before legacy data helpers
# prepend the public tree. Do not edit the runtime used by live experiments.
from experiments.fresh_fixed2_cmcc_20261030 import sources as N

import torch

from experiments.adaptive_native_model_20261029 import checkpoint
from experiments.adaptive_native_model_20261029.consumer import CompactRVQLinear
from experiments.adaptive_vector_rvq import common as c
from litebsq.vae_linear import VAELinear, prime_named_vae_linear_cache
from train_utils.distill_precision import install_precision_runtime

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_B = Path('/data1/cmcc/VAELLM/artifacts/cat_structure_full_20261029/B/native/home_star_models_Qwen_Qwen3-8B_20261029_134231/final_model')
SOURCE_B_ID = '87571461-7310-46e5-bc2e-904baf1818ed'
OUTPUT = Path('/data1/cmcc/VAELLM/artifacts/mlp_learnability_cmcc_20261030')
HF = N.HF
MLP_PATTERN = re.compile(r'^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)$')
require, read, write, sha, digest = c.require, c.read, c.write, c.sha, c.digest


def file_binding(path):
    path = Path(path).resolve(strict=True)
    return {**c.stat(path), 'sha256': sha(path)}


def runtime_bindings():
    paths = {Path(__file__).resolve(), Path(N.__file__).resolve(), Path(checkpoint.__file__).resolve(),
             Path(__import__(CompactRVQLinear.__module__, fromlist=['x']).__file__).resolve()}
    paths.update(N.common.s.SNAPSHOT.rglob('*.py'))
    return {str(path): sha(path) for path in sorted(paths)}


def source(directory, *, prototype_only=False):
    """Bind real v6 B for mechanics, or a real complete mixed endpoint for learning."""
    directory = Path(directory).resolve(strict=True)
    mixed = directory / checkpoint.META_FILE
    native = directory / 'checkpoint_meta.json'
    require(mixed.is_file() != native.is_file(), 'Require one actual native or mixed manifest, never fabricated metadata.')
    path = mixed if mixed.is_file() else native
    meta = read(path)
    if mixed.is_file():
        require(meta['format'] == checkpoint.FORMAT and meta['schema_version'] == 2
                and meta['checkpoint_kind'] == 'mixed_final_model' and not meta['pending_dense_modules']
                and len(meta['native_modules']) + len(meta['compact_modules']) == 252,
                'Learning requires a complete compressed mixed endpoint with no pending dense projection.')
        require(not prototype_only, 'Only the retained native B is the prototype source; a mixed learning parent must be genuine fresh C0.')
        from experiments.fresh_fixed2_cmcc_20261030 import evaluate as c0
        from experiments.cmcc_runtime_20261030 import adaptive_evaluation as shared
        support = OUTPUT.parent/'adaptive_class_cmcc_20261030/evaluation_support'
        shared.setup(support)
        c0_provenance, verified = c0.source(directory,directory.parent/'report.json')
        require(verified == meta, 'Fresh C0 validation inspected another actual manifest.')
        source_format = 'mixed'
    else:
        require(prototype_only and directory == SOURCE_B.resolve() and meta['checkpoint_id'] == SOURCE_B_ID,
                'Native B is allowed only as this explicit mechanics prototype; it cannot replace the fresh C0 learning parent.')
        require(meta['format'] == checkpoint.native.FORMAT_V6 and meta['checkpoint_kind'] == 'final_model'
                and len(meta['converted_modules']) == 252 and not meta['pending_dense_targets']
                and meta['lora_config'] is None and not meta['skip_targets'], 'Require the real complete native B endpoint.')
        source_format = 'native'
        c0_provenance = None
    weight = directory / meta['state_dict_file']
    return dict(schema='all_MLP_compressed_source_v1', path=str(directory), source_format=source_format,
        checkpoint_id=meta['checkpoint_id'], prototype_only=bool(prototype_only),
        manifest=file_binding(path), config=file_binding(directory/'config.json'), weight=file_binding(weight),
        runtime_sha256=runtime_bindings(), fresh_C0=c0_provenance)


def check_source(binding):
    require(binding == source(binding['path'], prototype_only=binding['prototype_only']),
            'The original compressed source, metadata or bound runtime changed.')


def load_source(binding):
    """Strict real loader; weights always come from the bound compressed directory."""
    check_source(binding)
    directory = Path(binding['path'])
    if binding['source_format'] == 'mixed':
        model, meta, loaded, measured = checkpoint.load(directory)
    else:
        meta = read(directory/'checkpoint_meta.json')
        model, state, loaded = checkpoint.load_parent(directory, meta)
        del state
        measured = None
    require(not loaded.missing_keys and not loaded.unexpected_keys, 'The complete compressed source did not load strictly.')
    groups = checkpoint.inventories(model)
    require(not groups['pending_dense'] and sum(map(len, groups.values())) == 252, 'Incomplete compressed source topology.')
    for name,module in model.named_modules():
        if isinstance(module,(VAELinear,CompactRVQLinear)):
            validate_supported(module,name)
    mlp_inventory(model)
    return model, meta, measured


def validate_supported(module,name):
    require(type(module) in (VAELinear,CompactRVQLinear), 'Additional compact/native subclasses need a separate reviewed deployment contract: '+name)
    forbidden = ('sparse_', 'low_rank_', 'protected_output_', 'protected_residual_',
                 '_protected_residual_', 'original_weight', 'protected_input_weight')
    extra = [key for key,value in module.state_dict().items()
             if key.startswith(forbidden) and torch.is_tensor(value) and value.numel()]
    require(not extra, 'Uncovered sparse/low-rank/output/protected-residual/original payload: '+name+' '+str(extra))
    if isinstance(module,VAELinear):
        require(module.original_weight is None and module.low_rank_a is module.low_rank_b is None
                and not module.has_sparse_residual() and not module.has_protected_residual_vae()
                and int(module.protected_residual_stages) == 0 and not module.always_use_original and module.temporary,
                'Uncovered native extra topology: '+name)


@torch.no_grad()
def prewarm_deployment(model):
    """The real group8 algorithm groups all native targets locally on each device."""
    native,compact = {},[]
    for name,module in model.named_modules():
        if not isinstance(module,(VAELinear,CompactRVQLinear)):
            continue
        validate_supported(module,name)
        require(all(p.dtype == torch.bfloat16 and not p.requires_grad for p in module.parameters()),
                'Prewarm only actual frozen BF16 deployment modules: '+name)
        if isinstance(module,VAELinear):
            require(module.cache_decoded_weight and not module.trainable_decode
                    and module._decoder_compute_dtype == torch.bfloat16, 'Native deployment policy not set before prewarm: '+name)
            device = next(module.parameters()).device
            native.setdefault(str(device),[]).append((name,module))
        else:
            compact.append((name,module))
    rows = {}
    for device,targets in native.items():
        result = prime_named_vae_linear_cache(targets,dtype=torch.bfloat16,clear_existing=True,
                                             group_size=8,compute_device=device)
        require(result == dict(total=len(targets),warmed=len(targets),skipped=0,failed=0), 'Actual device-local group8 prewarm failed: '+device)
        for name,module in targets:
            require(module._cached_weight is not None and module._cached_weight.dtype == torch.bfloat16
                    and not module._cached_weight.requires_grad and not torch.is_inference(module._cached_weight),
                    'Native cache must be ordinary no_grad BF16, usable for hidden backward: '+name)
        rows[device] = result
    for name,module in compact:
        module.clear_decoded_weight_cache()
        module(torch.empty((0,module.in_features),dtype=torch.bfloat16,device=module.length_frame.device))
        require(module._decoded_cache is not None and not torch.is_inference(module._decoded_cache), 'Compact deployment cache missing: '+name)
    return dict(native_by_device=rows,group_size=8,compact=len(compact),ordinary_no_grad_cache=True)


def mlp_inventory(model):
    require(model.config.num_hidden_layers == 36 and model.config.hidden_size == 4096
            and model.config.intermediate_size == 12288, 'This prototype covers the actual Qwen3-8B geometry.')
    rows = []
    for name, module in model.named_modules():
        match = MLP_PATTERN.fullmatch(name)
        if match is None:
            continue
        require(isinstance(module, (VAELinear, CompactRVQLinear)), 'All MLP projections must come from actual compressed consumers: '+name)
        protected = module.protected_input_indices
        require(protected.dtype == torch.int64 and protected.ndim == 1 and protected.numel() > 0
                and protected.unique().numel() == protected.numel()
                and int(protected.min()) >= 0 and int(protected.max()) < module.in_features,
                'Invalid actual protected-input index set: '+name)
        layer, category = int(match[1]), match[2]
        expected = (4096, 12288) if category == 'down_proj' else (12288, 4096)
        require((module.out_features, module.in_features) == expected, 'Unexpected actual MLP projection shape: '+name)
        rows.append(dict(name=name, layer=layer, category=category,
            consumer='compact' if isinstance(module, CompactRVQLinear) else 'native',
            shape=list(expected), protected_columns=protected.numel(),
            active_shape=[module.out_features, module.in_features-protected.numel()],
            active_elements=module.out_features*(module.in_features-protected.numel()),
            full_BF16_template_bytes=module.out_features*module.in_features*2))
    expected_names = {f'model.layers.{layer}.mlp.{category}' for layer in range(36)
                      for category in ('gate_proj', 'up_proj', 'down_proj')}
    require({row['name'] for row in rows} == expected_names and len(rows) == 108, 'Require all108 MLP projections across all36 layers.')
    require(sum(row['active_elements'] for row in rows) == 5341446144,
            'Actual source protection differs from this approved all-MLP parameter/resource budget.')
    return rows
