"""Real compressed sources and the original BF16 teacher for block reconstruction."""
from pathlib import Path

from experiments.mlp_learnability_cmcc_20261030 import bindings as original
from experiments.adaptive_native_model_20261029.native_prefix import from_native

torch, checkpoint, N = original.torch, original.checkpoint, original.N
VAELinear, CompactRVQLinear = original.VAELinear, original.CompactRVQLinear
require, read, write, sha, digest = original.require, original.read, original.write, original.sha, original.digest
file_binding = original.file_binding
HF, SOURCE_B = original.HF, original.SOURCE_B
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = Path('/data1/cmcc/VAELLM/artifacts/block_reconstruction_cmcc_20261030')
CATEGORIES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
install_precision_runtime = original.install_precision_runtime
prime_named_vae_linear_cache = original.prime_named_vae_linear_cache


def source_binding(directory, *, prototype_only=False):
    return original.source(directory, prototype_only=prototype_only)


def runtime_bindings():
    return {**N.source_pins(), **original.runtime_bindings(), str(Path(original.__file__).resolve()): sha(original.__file__),
        str(Path(from_native.__code__.co_filename).resolve()): sha(from_native.__code__.co_filename),
        **{str(path.resolve()): sha(path) for path in HERE.glob('*.py')}}


@torch.no_grad()
def load_student(binding, baseline_dir):
    model, meta, budget = original.load_source(binding)
    model.eval().requires_grad_(False)
    parent_id = meta['checkpoint_id']
    if binding['prototype_only']:
        require(binding['source_format'] == 'native', 'The prototype starts only at the preserved complete native B.')
        before = {key: value for key, value in model.state_dict().items() if '.mlp.gate_proj.' not in key}
        before_digest = digest(before)
        gates = [row for row in meta['converted_modules'] if row['name'].endswith('.mlp.gate_proj')]
        require(len(gates) == 36, 'Use every real B gate, without selecting a partial family.')
        for row in gates:
            name = row['name']
            module = model.get_submodule(name)
            bundle, lengths, _ = from_native(row, module.state_dict(), dict(
                source_checkpoint_id=parent_id, source_module=name,
                purpose='BR mixed interface prototype only; no new C0 or quality claim'))
            checkpoint.native.set_module_by_name(model, name, CompactRVQLinear(bundle, lengths))
        after = {key: value for key, value in model.state_dict().items() if '.mlp.gate_proj.' not in key}
        require(digest(after) == before_digest, 'The mechanics conversion changed a non-gate source field.')
        del before, after
        model.eval().requires_grad_(False)
        meta, budget, state = checkpoint.save(model, Path(baseline_dir), provenance=dict(
            schema='BR_mixed_prototype_baseline_v1', prototype_only=True, source_binding=binding,
            converted_gates=36, additional_fit_updates=0, original_non_gate_exact=True,
            scope='Existing native-prefix fixed2 conversion; its consumer differs from native B. Not a formal C0.'))
        require(digest(state) == digest(model.state_dict()), 'The real converted baseline save changed persistent fields.')
        del state
        baseline_path = Path(baseline_dir).resolve()
        conversion = dict(converted_gates=36, additional_fit_updates=0,
            non_gate_digest=before_digest, original_non_gate_exact=True,
            consumer_boundary='Native gate -> existing fixed2 compact. No quality or native-functional-equality claim.')
    else:
        require(binding['source_format'] == 'mixed' and binding['fresh_C0'], 'Only a verified complete new C0 is a formal parent.')
        baseline_path = Path(binding['path'])
        conversion = dict(converted_gates=0, additional_fit_updates=0, original_non_gate_exact=True)
    groups = checkpoint.inventories(model)
    require([len(groups[key]) for key in ('native', 'compact', 'pending_dense')] == [216, 36, 0]
        and budget['budget_passed'], 'BR requires the complete 216-native/36-fixed2/0-dense mixed baseline within2.6.')
    for name, module in model.named_modules():
        if isinstance(module, (VAELinear, CompactRVQLinear)):
            original.validate_supported(module, name)
    install_precision_runtime(model, torch.bfloat16)
    from experiments.muon_compact_cmcc_20261030 import inputs as teacher_inputs
    teacher_identity = teacher_inputs.teacher_identity(model.state_dict(),
        [name for group in groups.values() for name in group])
    from .runtime import remember_deployment_flags
    flags = remember_deployment_flags(model)
    return model, meta, dict(source_binding=binding, original_parent_id=parent_id,
        baseline_checkpoint_id=meta['checkpoint_id'], baseline_path=str(baseline_path),
        post_conversion_budget=budget, conversion=conversion, deployment_flags=flags,
        teacher_identity=teacher_identity,
        baseline_state_digest=digest(model.state_dict()))


def load_teacher():
    from transformers import AutoModelForCausalLM
    require(torch.cuda.is_available(), 'The actual BF16/FA2 teacher constructor requires the real GPU execution environment.')
    model = AutoModelForCausalLM.from_pretrained(str(HF), torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', local_files_only=True, low_cpu_mem_usage=True)
    model.eval().requires_grad_(False)
    require(model.config.model_type == 'qwen3' and len(model.model.layers) == 36
        and all(p.dtype == torch.bfloat16 and not p.requires_grad for p in model.parameters())
        and model.model.rotary_emb.inv_freq.dtype == torch.float32,
        'Use the real original Qwen3-8B BF16 teacher with its native FP32 RoPE.')
    return model
