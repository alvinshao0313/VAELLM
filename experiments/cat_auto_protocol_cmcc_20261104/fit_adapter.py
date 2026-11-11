"""Isolated PUBLIC single-category fit, with audited semantic call-site hooks.

The native residual loop/loss/pooled BSQ/conversion execute unchanged. A private
AST-derived function adds only the missing runtime hooks; no shared globals mutate.
"""
import ast
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import random
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from .adapter import category_projections, file_sha, public_consumer_identity
from .contracts import digest, inventory_id, require, validate_recipe, validate_run_identity
from .geometry import architecture_args, component_plan, rotation
from .identity import artifact_id, candidate_id, semantic_seed
from .vae_muon import VAEMuon, tensor_digest

NATIVE_SHA256 = 'd9921dc341cf8383f729171b481056e8bd7ae2d1afafb279041abfa44b17257b'


def fit_code_identity():
    names = ('fit_adapter.py', 'vae_muon.py', 'adapter.py', 'geometry.py', 'config.py', 'contracts.py', 'identity.py')
    return dict(schema='cat_auto_fit_actual_sources_v0', source_sha256={str(Path(__file__).with_name(name).resolve()):
        file_sha(Path(__file__).with_name(name)) for name in names})


def _validate_actual_identity(identity):
    actual = public_consumer_identity()
    require(identity['consumer_identity'] == actual, 'Declared consumer identity differs from the actual selected PUBLIC family/files.')
    required = fit_code_identity()['source_sha256']
    declared = identity['code_identity'].get('source_sha256', {}) if isinstance(identity['code_identity'], dict) else {}
    require(all(declared.get(path) == sha for path, sha in required.items()), 'Declared code identity omits or changes actual fit/semantic/Muon sources.')


def _formal_evidence(calibration_bundle, inventory, identity, recipe, technical_steps, device):
    if technical_steps is not None or not str(device).startswith('cuda'):
        return False
    require(inventory['model_identity'].get('schema') == 'dense_model_content_identity_v0'
        and inventory.get('source_evidence', {}).get('verification'), 'A CPU fixture inventory cannot become a formal CUDA candidate.')
    certificate = calibration_bundle.get('formal_recipe_certificate')
    require(certificate and certificate.get('phase') == 'formal' and certificate.get('formal_eligible') is True,
        'Full CUDA fitting requires its frozen formal recipe/data registry certificate.')
    registry_path = Path(certificate['registry_path'])
    require(file_sha(registry_path) == certificate['registry_sha256'], 'Formal recipe registry changed.')
    registry = json.loads(registry_path.read_text())
    require(registry['phase'] == 'formal' and registry['run_identity'] == identity and registry.get('technical_steps') is None
        and any(row['candidate_id'] == candidate_id(identity, recipe) and row['recipe'] == recipe for row in registry['candidates']),
        'Formal registry does not contain this exact own-parent resolved candidate.')
    prepared = torch.load(registry['data_plan'], map_location='cpu', weights_only=False)
    require(file_sha(registry['data_plan']) == registry['data_plan_sha256'] and digest(prepared['manifest']) == prepared['data_identity'] == identity['data_identity']
        and prepared['manifest']['model_identity'] == inventory['model_identity'] and prepared['manifest']['calibration'] == calibration_bundle['manifest']
        and all(torch.equal(a, b) for a, b in zip(prepared['calibration'], calibration_bundle['input_ids'], strict=True)),
        'Formal calibration manifest/tensors are not the frozen actual root data plan.')
    require(prepared['manifest'].get('version') and prepared['manifest'].get('sources') and prepared['manifest'].get('updates')
        and prepared['manifest'].get('proposal') and not any(str(prepared['manifest'].get(key, '')).startswith('cpu_fixture')
            for key in ('phase', 'verification_scope')), 'A CPU fixture data manifest cannot become a formal CUDA fit.')
    return True


class _Hooks(ast.NodeTransformer):
    def __init__(self):
        self.counts = dict(factory_import=0, shuffle=0, observe=0)

    def visit_ImportFrom(self, node):
        if node.module == 'litebsq.llm_vae':
            require([item.name for item in node.names] == ['MultiLayerVAE'], 'Native factory import changed.')
            self.counts['factory_import'] += 1
            return ast.copy_location(ast.Pass(), node)
        return node

    def visit_BinOp(self, node):
        if ast.dump(node, include_attributes=False) == ast.dump(ast.parse('int(shuffle_seed) + int(stage_idx)', mode='eval').body, include_attributes=False):
            self.counts['shuffle'] += 1
            return ast.copy_location(ast.Call(func=ast.Name(id='_semantic_shuffle_seed', ctx=ast.Load()), args=[ast.Name(id='stage_idx', ctx=ast.Load())], keywords=[]), node)
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if ast.unparse(node) == 'loss.backward()':
            self.counts['observe'] += 1
            observe = ast.parse('_observe_batch(vae, loss_dict, step, stage_idx)').body[0]
            return [ast.copy_location(observe, node), node]
        return self.generic_visit(node)


def _private_pipeline(hooks):
    from train_utils import cat_train_pipeline as native
    require(file_sha(native.__file__) == NATIVE_SHA256, 'PUBLIC pipeline changed; re-review hook structure before fitting.')
    tree = ast.parse(inspect.getsource(native.train_group_vae_payload))
    patch = _Hooks()
    tree = patch.visit(tree)
    require(patch.counts == dict(factory_import=1, shuffle=4, observe=1), 'Native hook sites changed; refuse implicit fallback.')
    ast.fix_missing_locations(tree)
    namespace = dict(native.train_group_vae_payload.__globals__)
    namespace.update(hooks)
    exec(compile(tree, str(Path(__file__).resolve()) + ':private_PUBLIC_fit', 'exec'), namespace)
    return namespace['train_group_vae_payload'], patch.counts


def _flat_gradient(loss, parameters):
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(parameter) if value is None else value).float().reshape(-1)
        for parameter, value in zip(parameters, values, strict=True)])


def observe_bsq(vae, loss_dict, h):
    """Read actual stage batch contributions without touching .grad/optimizer/RNG."""
    quantizer = vae.model.quantizer
    result, breakdown = quantizer(h, return_loss_breakdown=True)
    scale = vae.model.lfq_weight * vae.model.num_models
    contributions = dict(commitment=breakdown.commitment * quantizer.commitment_loss_weight * scale,
        entropy=(quantizer.gamma0 * breakdown.per_sample_entropy - quantizer.gamma * breakdown.batch_entropy)
            * quantizer.zeta / quantizer.inv_temperature * quantizer.entropy_loss_weight * scale)
    parameters = tuple(vae.model.encoder.parameters())
    recon = _flat_gradient(loss_dict['train/recon_loss'], parameters)
    denominator = torch.linalg.vector_norm(recon).double()
    require(denominator.item() > 0, 'Actual reconstruction encoder gradient norm is zero; diagnostic undefined.')
    norms = {}
    for name, loss in contributions.items():
        gradient = _flat_gradient(loss, parameters)
        norm = torch.linalg.vector_norm(gradient).double()
        norms[name] = dict(ratio=float(norm / denominator), cosine=float(torch.dot(gradient.double(), recon.double()) / (norm * denominator)) if norm.item() else 0.)
    quant_input = torch.nn.functional.normalize(quantizer.project_in(h).reshape(*h.shape[:-1], quantizer.num_codebooks, -1), dim=-1)
    if quantizer.force_quantization_f32:
        quant_input = quant_input.float()
    _, _, probabilities = quantizer.soft_entropy_loss(quant_input)
    marginal = probabilities[..., 1]
    return dict(contributions=norms, bit_marginal_extreme_fraction=float(((marginal < .01) | (marginal > .99)).float().mean()),
        observer='Actual same-batch encoder autograd contributions; weighted native loss; pooled B/M bit marginal',
        gradient_accumulation_unchanged=True)


class _AuditedOptimizer:
    """Use a real Optimizer instance: install its bound step, not a scheduler proxy."""
    @staticmethod
    def install(optimizer, stage, vae, total_steps):
        original = optimizer.step
        named = dict(vae.named_parameters())
        step = 0

        def audited(_optimizer, *args, **kwargs):
            nonlocal step
            step += 1
            require(all(parameter.grad is not None for parameter in named.values())
                and torch.stack([torch.isfinite(parameter.grad).all() for parameter in named.values()]).all().item(),
                'Nonfinite/missing actual stage gradient before update.')
            audit = step in (1, 2, total_steps)
            actual_lrs = [float(group['lr']) for group in optimizer.param_groups]
            if step <= 2:
                expected_lrs = [float(group['initial_lr']) * (1. - (step - 1) / 10000.) for group in optimizer.param_groups]
                require(all(math.isclose(actual, expected, rel_tol=0., abs_tol=1e-12) for actual, expected in zip(actual_lrs, expected_lrs, strict=True)),
                    'First two actual optimizer LRs must be the original linear10000 prefix, not a shortened schedule.')
            before = {name: parameter.detach().clone() for name, parameter in named.items()} if audit else None
            output = original(*args, **kwargs)
            if audit:
                changed = {name: int(torch.count_nonzero(parameter != before[name])) for name, parameter in named.items()}
                states = []
                for name, parameter in named.items():
                    state = optimizer.state[parameter]
                    require(parameter.dtype == torch.bfloat16 and torch.isfinite(parameter).all().item(), 'Updated parameters must be finite BF16.')
                    require(int(state['step']) == step, 'Actual optimizer step does not match the real update count.')
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            require(torch.isfinite(value).all().item(), 'Nonfinite actual optimizer state.')
                            if key in ('exp_avg', 'exp_avg_sq'):
                                require(value.dtype == torch.bfloat16, 'Native Adam moments changed precision.')
                            if key == 'momentum_buffer':
                                require(value.dtype == torch.float32, 'Muon matrix momentum changed precision.')
                    states.append(dict(name=name, step=step, dtypes={key: str(value.dtype) for key, value in state.items() if isinstance(value, torch.Tensor)}))
                require(step > 2 or any(changed.values()), 'No actual parameter moved in the first two updates.')
                stage.setdefault('updates', []).append(dict(step=step, actual_learning_rates=actual_lrs,
                    expected_original_linear10000_learning_rates=expected_lrs if step <= 2 else None,
                    changed_elements=changed, optimizer_states=states))
                print(dict(event='actual_VAE_update_audit', stage=stage['stage_index'], step=step,
                    actual_learning_rates=actual_lrs, original_schedule_steps=10000,
                    changed_elements=sum(changed.values()), actual_state_steps=[row['step'] for row in states]), flush=True)
            stage['actual_steps'] = step
            return output
        optimizer.step = MethodType(audited, optimizer)
        return optimizer


def fit_category(model, *, inventory, recipe, run_identity, calibration_bundle, activation_bundle=None,
        device, output_dir, technical_steps=None):
    from litebsq.llm_vae import MultiLayerVAE
    from train_utils import cat_train_pipeline as native
    from train_utils.activation_utils import ActivationCalibrationCache, collect_activation_stats_for_linears, activation_stats_to_views, _normalize_calibration_dataset_mix
    from train_utils.cat_category_runtime import ResolvedCategoryRuntimeConfig
    from train_utils.cat_data_prep import compute_channel_rank_score, select_outlier_channel_indices_from_scores
    from train_utils.utils import LinearRef, set_seed
    from rotation.weight_preprocess import rotation_spec

    public_consumer_identity()
    recipe, identity = validate_recipe(recipe), validate_run_identity(run_identity)
    _validate_actual_identity(identity)
    require(identity['model'] == inventory['model_identity'], 'Fit identity uses a different model root.')
    require(not getattr(model, 'peft_config', None), 'Fit requires caller-exported own live weights with no active PEFT adapters.')
    require(technical_steps is None or technical_steps in (1, 2), 'Only explicit1/2-step CPU/pilot mechanism mode; never counted as an offline fit.')
    steps = recipe['steps'] if technical_steps is None else technical_steps
    formal = _formal_evidence(calibration_bundle, inventory, identity, recipe, technical_steps, device)
    plan = component_plan(inventory, recipe)
    require(plan['capability']['constructible'], 'Geometry not executable: ' + str(plan['failures']))
    rows = category_projections(inventory, recipe['category_id'])
    require([row['actual_name'] for row in rows] == recipe['group_signature'], 'All actual category projections must share one ordered M group.')
    modules = dict(model.named_modules())
    refs = []
    for row in rows:
        name = row['actual_name']
        require(type(modules[name]) is torch.nn.Linear and list(modules[name].weight.shape) == row['shape'], 'Current target must be the actual dense own-parent projection.')
        refs.append(LinearRef(name=name, module=modules[name], category=recipe['category_id'],
            transpose=recipe['transpose'][name] if isinstance(recipe['transpose'], dict) else recipe['transpose']))
    names = set(recipe['group_signature'])
    before = tensor_digest({ref.name: ref.module.weight for ref in refs})
    frozen = tensor_digest({key: value for key, value in model.state_dict().items() if not any(key.startswith(name + '.') for name in names)})
    ids = calibration_bundle['input_ids']
    require(ids and all(value.device.type == 'cpu' and value.ndim == 2 and value.shape[0] == 1 for value in ids), 'Calibration must supply actual ordered CPU input blocks, not a request to resample.')
    require(technical_steps is not None or len(ids) == 512 and all(tuple(value.shape) == (1, 256) for value in ids), 'Formal activation statistics use the frozen512×256 calibration.')
    token_digest = tensor_digest({str(index): value for index, value in enumerate(ids)})
    binding = dict(source_parent_uuid=identity['parent_uuid'], source_weights_digest=identity['parent_weights_digest'],
        target_digest=before, calibration_token_digest=token_digest, calibration_data_identity=calibration_bundle['data_identity'])
    if activation_bundle is not None:
        require(all(activation_bundle.get(key) == value for key, value in binding.items()), 'Activation cache parent/target/data identity mismatch.')
        activation_runtime = activation_bundle['activation_runtime']
    else:
        cache = ActivationCalibrationCache(dataset=_normalize_calibration_dataset_mix(calibration_bundle['dataset_spec']), model_path=calibration_bundle['model_path'],
            nsamples=len(ids), seqlen=ids[0].shape[1], seed=31, input_ids=ids)
        stats, returned_cache = collect_activation_stats_for_linears(model=model, linear_items=[(ref.name, ref.module) for ref in refs],
            model_path=cache.model_path, access_token=None, dataset=cache.dataset, nsamples=cache.nsamples,
            seqlen=cache.seqlen, seed=31, device=str(device), cache=cache)
        require(returned_cache is cache, 'Native collector resampled instead of consuming the supplied frozen calibration.')
        activation_runtime = dict(cache=cache, stats_by_linear=stats, dataset=cache.dataset, model_path=cache.model_path,
            nsamples=cache.nsamples, seqlen=cache.seqlen, seed=31, device=str(device), access_token=None, log_every=0)
    stats = activation_runtime['stats_by_linear']
    flattened_stats = {name + '/' + key: value for name, row in stats.items() for key, value in row.items() if isinstance(value, torch.Tensor)}
    activation_bundle = dict(binding, activation_runtime=activation_runtime, native_activation_stats=stats,
        global_adaptive_plan=None, statistics_identity=tensor_digest(flattened_stats))
    actmax, actmean, _ = activation_stats_to_views(stats)
    channel_plan = {}
    for ref in refs:
        count = recipe['protection']['counts'][ref.name]
        score = compute_channel_rank_score(metric='channel_weight_actmean_abs', weight=ref.module.weight, residual=None,
            act_max=actmax[ref.name], act_mean=actmean[ref.name], act_sq_mean=None, axis=recipe['protection']['axis'],
            transpose=ref.transpose, linear_name=ref.name, expected_in_features=ref.module.in_features, expected_out_features=ref.module.out_features)
        selected, _ = select_outlier_channel_indices_from_scores(scores_by_name={ref.name: score}, linear_names=[ref.name],
            channel_protect_count=count, channel_scope='layer')
        channel_plan.update(selected)
    report = dict(candidate_id=candidate_id(identity, recipe), resolved_recipe=recipe, parent_identity=identity,
        actual_targets=recipe['group_signature'], current_category=recipe['category_id'], M=len(refs),
        target_before_digest=before, frozen_prefix_before_digest=frozen, native_source_sha256=NATIVE_SHA256,
        adapter_sha256=file_sha(__file__), calibration_token_digest=token_digest, activation_statistics_identity=activation_bundle['statistics_identity'],
        formal_eligible=formal, verification_scope='GPU_model_fit' if str(device).startswith('cuda') else 'CPU_mechanism_fixture',
        technical_steps=technical_steps, original_schedule_steps=10000, actual_stage_steps=steps, stages=[],
        protection_indices={name: value.tolist() for name, value in channel_plan.items()})
    pending, hidden = [], {}

    def stage_seed(stage, purpose):
        return semantic_seed(identity['model'], recipe['category_id'], 'residual_stage_' + str(stage + 1), purpose, identity['search_seed'])

    def construct(args):
        stage_index = len(report['stages'])
        require(not pending and stage_index < recipe['S'], 'Each real stage factory must have one matching optimizer.')
        seed = stage_seed(stage_index, 'vae_initialization')
        set_seed(seed)
        vae = MultiLayerVAE(args)
        row = dict(stage_index=stage_index, init_seed=seed, shuffle_seed=stage_seed(stage_index, 'vae_shuffle'),
            initialization_digest=tensor_digest(dict(vae.named_parameters())))
        report['stages'].append(row)
        vae.model.encoder.register_forward_hook(lambda _module, _inputs, output: hidden.update(h=output))
        pending.append((vae, row))
        return vae

    def optimizer_factory(parameters, args, lr):
        vae, row = pending.pop()
        require([id(value) for value in parameters] == [id(value) for value in vae.parameters()], 'Optimizer factory owner mismatch.')
        if recipe['optimizer']['kind'] == 'vae_muon':
            optimizer = VAEMuon(vae, multiplier=recipe['optimizer']['multiplier'], evidence=row)
        else:
            optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, betas=(.9, .95), eps=1e-8, weight_decay=0.)
        return _AuditedOptimizer.install(optimizer, row, vae, steps)

    def per_ref_rotation(_mode, _block, _seed, name):
        level = recipe['rotation'].get('per_layer', {}).get(name, recipe['rotation']['level'])
        spec = rotation(level)
        return rotation_spec(spec['kind'], spec['block_size'], stage_seed(0, 'weight_rotation'), name)

    def observe(vae, loss_dict, step, stage_idx):
        if stage_idx == 0 and step + 1 in (5000, 10000):
            report.setdefault('BSQ_diagnostics', []).append(dict(step=step + 1, **observe_bsq(vae, loss_dict, hidden['h'])))
        hidden.clear()

    function, counts = _private_pipeline(dict(MultiLayerVAE=construct, create_optimizer=optimizer_factory,
        rotation_spec=per_ref_rotation, _semantic_shuffle_seed=lambda stage: stage_seed(stage, 'vae_shuffle'), _observe_batch=observe))
    report['private_AST_hook_counts'] = counts
    args = architecture_args(recipe)
    for key, value in dict(parallel_layers=len(refs), vae_weight_dtype='bf16', vae_autocast_dtype='bf16', normalize_weight=True,
            new_quant=True, quantizer_type='BSQ', optimizer='adamw', beta1=.9, beta2=.95, weight_decay=0.,
            lr=.003 * recipe['optimizer']['multiplier'], lr_scheduler='linear', lr_warmup_steps=0,
            weight_rotation=recipe['rotation']['kind'], weight_rotation_block_size=recipe['rotation'].get('block_size', 0),
            **{key: value for key, value in recipe['loss'].items() if key not in ('reconstruction', 'quantizer')}).items():
        setattr(args, key, value)
    runtime = ResolvedCategoryRuntimeConfig(category=recipe['category_id'], residual_stages=recipe['S'], steps=steps,
        intra_part_sort_mode='none', codebook_bits=recipe['K'], codebook_dim=recipe['D'], intra_parallel=(1, 1),
        channel_protect_count=max(recipe['protection']['counts'].values()), recon_loss_type='mse', base_ch=recipe['encoder']['hidden_dim'],
        num_res_blocks=0, decoder_base_ch=recipe['decoder']['hidden_dim'], decoder_num_res_blocks=0,
        norm_type='layer', activation_type='swish', decoder_type='asymmetric')
    # Technical execution consumes the first1/2 original10K updates, never a compressed schedule.
    if technical_steps is not None:
        # Native passes stage_steps to its scheduler. Keep the native10K schedule
        # by specializing only the actual get_scheduler call in this private AST.
        source = inspect.getsource(native.train_group_vae_payload)
        require(source.count('num_training_steps=int(stage_steps)') == 1, 'Native scheduler call changed.')
        # A private globals module proxy is insufficient for a local import;
        # instead set stage_steps only at scheduler call through an AST constant.
        tree = ast.parse(source)
        patch = _Hooks()
        tree = patch.visit(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == 'num_training_steps':
                node.value = ast.Constant(value=10000)
        ast.fix_missing_locations(tree)
        namespace = dict(function.__globals__)
        exec(compile(tree, str(Path(__file__).resolve()) + ':technical_original_schedule', 'exec'), namespace)
        function = namespace['train_group_vae_payload']
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        payload = function(model=model, group_refs=refs, group_tag=recipe['category_id'], runtime_cfg=runtime, vae_args=args,
            training_args=SimpleNamespace(bf16=True, fp16=False, seed=stage_seed(0, 'weight_rotation')), train_device=str(device),
            convert_device=str(device), do_convert=True, batch_size=recipe['B'], log_every=1000, eval_every=0, eval_blocks=0,
            gpu_resident_data=False, activation_runtime=activation_runtime, channel_protect_mode='channel', channel_plan=channel_plan,
            channel_scope='layer', channel_rank_metric='channel_weight_actmean_abs', channel_axis=recipe['protection']['axis'],
            channel_quant='int8', deterministic=True, shuffle_seed=stage_seed(0, 'vae_shuffle'))
        native.apply_group_vae_payload(model=model, group_refs=refs, group_tag=recipe['category_id'], payload=payload, convert_device=str(device))
    finally:
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    after_state = model.state_dict()
    after_frozen = tensor_digest({key: value for key, value in after_state.items() if not any(key.startswith(name + '.') for name in names)})
    require(after_frozen == frozen, 'Non-target/prefix persistent tensors changed during single-category fitting.')
    report.update(target_after_digest=tensor_digest({key: value for key, value in after_state.items() if any(key.startswith(name + '.') for name in names)}),
        frozen_prefix_after_digest=after_frozen, payload_consumer='PUBLIC apply_group_vae_payload / native packed VAE Linear',
        payload_stages=len(payload['all_stage_bits']), exit_code=0, status='completed')
    report['runtime_resolution'] = dict(stage_evidence=report['stages'], kappa_owner_stage_evidence=[row.get('calibration') for row in report['stages']])
    report['artifact_id'] = artifact_id(report['candidate_id'], report['runtime_resolution'], report['target_after_digest'])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'fit_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    bank = extract_category_bank(model, inventory=inventory, recipe=recipe, fit_report=report, output_dir=output_dir / 'category_bank')
    return dict(model=model, fit_report=report, activation_bundle=activation_bundle, category_bank=bank)


def extract_category_bank(model, *, inventory, recipe, fit_report, output_dir):
    from train_utils import checkpoint_v6 as v6
    names = set(recipe['group_signature'])
    specs = [spec for spec in v6._collect_vae_linear_specs(model) if spec['name'] in names]
    require({spec['name'] for spec in specs} == names, 'Single-class bank must contain every actual converted target.')
    state = {key: value for key, value in model.state_dict().items() if any(key.startswith(name + '.') for name in names)}
    require(all(value.device.type == 'cpu' for value in state.values()), 'Native apply leaves deployment modules on CPU; retain actual storage aliases before saving.')
    require(not any(key.endswith(('.weight', '.original_weight')) and any(key == name + '.weight' or key == name + '.original_weight' for name in names) for key in state),
        'A compressed bank must not keep dense original target backups.')
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / 'native_bank.pt'
    torch.save(state, state_path)
    loaded = torch.load(state_path, map_location='cpu', weights_only=True, mmap=True)
    require(tensor_digest(loaded) == tensor_digest(state), 'Saved packed single-class bank changed tensor fields.')
    storages = {}
    for key, value in loaded.items():
        storage = value.untyped_storage()
        storages.setdefault((storage.data_ptr(), storage.nbytes()), []).append(key)
    record = dict(schema='cat_auto_single_category_native_bank_v0', category_id=recipe['category_id'],
        candidate_id=fit_report['candidate_id'], artifact_id=fit_report['artifact_id'], recipe=recipe,
        inventory_identity=inventory_id(inventory), model_identity=inventory['model_identity'], parent_identity=fit_report['parent_identity'],
        target_before_digest=fit_report['target_before_digest'], tensor_digest=tensor_digest(loaded),
        state_file=str(state_path.resolve()), state_sha256=file_sha(state_path), converted_modules=specs,
        tensor_bytes=sum(key[1] for key in storages), aliases=[values for values in storages.values() if len(values) > 1],
        metadata_bytes=None, formal_eligible=fit_report['formal_eligible'], verification_scope=fit_report['verification_scope'])
    meta = output_dir / 'bank.json'
    meta.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    record['metadata_bytes'] = meta.stat().st_size
    return record


def install_category_bank(model, bank, *, inventory, expected_parent_identity):
    from train_utils import checkpoint_v6 as v6
    require(bank['schema'] == 'cat_auto_single_category_native_bank_v0' and bank['inventory_identity'] == inventory_id(inventory)
        and bank['parent_identity'] == expected_parent_identity, 'Joint banks require the exact common model/parent/data/code/consumer identity.')
    require(file_sha(bank['state_file']) == bank['state_sha256'], 'Persisted bank file changed.')
    names = [row['actual_name'] for row in category_projections(inventory, bank['category_id'])]
    require(set(names) == {spec['name'] for spec in bank['converted_modules']}, 'Bank omits actual category modules.')
    before = tensor_digest({name: model.get_submodule(name).weight for name in names})
    require(before == bank['target_before_digest'], 'Joint installation target differs from the fitted own parent.')
    frozen = tensor_digest({key: value for key, value in model.state_dict().items() if not any(key.startswith(name + '.') for name in names)})
    state = torch.load(bank['state_file'], map_location='cpu', weights_only=True, mmap=True)
    require(tensor_digest(state) == bank['tensor_digest'], 'Actual bank tensor identity changed.')
    v6._rebuild_converted_modules(model, bank['converted_modules'])
    missing, unexpected = model.load_state_dict(state, strict=False)
    require(not unexpected and not any(any(key.startswith(name + '.') for name in names) for key in missing), 'Actual bank payload incomplete or unconsumed.')
    v6.refresh_vae_linear_runtime_after_state_load(model)
    require(frozen == tensor_digest({key: value for key, value in model.state_dict().items() if not any(key.startswith(name + '.') for name in names)}), 'Installing one bank changed another class or frozen fields.')
    return model
