"""Reuse bound D16 evidence; prove the batch-only configuration and CPU stage0."""
import ast
from copy import deepcopy
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import sys
import time

from . import bindings as b


def prepare():
    b.cpu_environment()
    b.require(not (b.RESULT / 'prepared.json').exists(), 'Never replace prepared evidence.')
    from . import fit, run, evaluate, sequence, observation
    old = b.read(b.ORIGINAL_PREPARED)
    b.check_pins(old['source_sha256'])
    source = b.sources()
    b.require(source == old['source_binding'] and old['status'] == 'cpu_verified'
        and old['live_down_source_digest'] == 'c2ffa5bbaffa477fd0f1b7e9be3a8efb4e32557ee000d8cb45fa4633eba26c05',
        'Reuse the verified genuine live source and frozen prefix, never raw D16 as training input.')
    parsed, identity, baseline = b.recipe(b.ARM, b.OUTPUT / b.ARM / 'formal')
    reference = b.read(b.CONTROL_REPORT)
    expected = deepcopy(reference['runtime_identity'])
    expected['experiment_runtime']['category_vector_batches']['down_proj'] = 8192
    b.require(identity == expected and reference['source_binding'] == source,
        'The actual runtime differs from original D16 only in down vector batch.')
    cfg, vae, native = parsed[-2]['down_proj'], parsed[3], b.recovery.native
    b.require(json.loads(json.dumps(asdict(cfg))) == old['recipes']['d16']['category_configs']['down_proj']
        and native._resolve_train_dtype(parsed[2]) == b.torch.bfloat16
        and vae.vae_weight_dtype == vae.vae_autocast_dtype == 'bf16'
        and vae.optimizer == 'adamw' and vae.beta1 == .9 and vae.beta2 == .95
        and vae.weight_decay == 0 and vae.lr == .003
        and vae.lr_scheduler == 'linear' and vae.lr_warmup_steps == 0,
        'Keep D16 architecture, loss, dtype, AdamW and linear10K schedule.')
    call = next(node for node in ast.walk(ast.parse(inspect.getsource(fit.execute)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'train_group_vae_payload')
    batch = next(item.value for item in call.keywords if item.arg == 'batch_size')
    b.require(ast.unparse(batch) == "settings['batch']" and b.ARMS[b.ARM]['batch'] == 8192,
        'The actual native call must receive B8192 from the fixed settings.')
    args = native._clone_namespace(vae, parallel_layers=36, residual_stages=2,
        codebook_bits=24, codebook_dim=16, base_ch=128, num_res_blocks=0,
        norm_type=cfg.norm_type, activation_type=cfg.activation_type, decoder_type=cfg.decoder_type,
        decoder_base_ch=cfg.decoder_base_ch, decoder_num_res_blocks=cfg.decoder_num_res_blocks,
        recon_loss_type=cfg.recon_loss_type)
    native.apply_autoencoder_arch_defaults(args)
    from litebsq.llm_vae import MultiLayerVAE
    from experiments.adaptive_class_20261030.initialization import observe
    native.set_seed(31)
    model = MultiLayerVAE(args)
    with observe(native) as rows:
        optimizer = native.create_optimizer(model.parameters(), args, args.lr)
    initial = b.initialization_reference()
    b.require(all(rows[0][key] == initial[0][key] for key in
        ('initial_parameter_digest','parameter_shapes','parameter_elements','parameter_dtypes','initial_learning_rates')),
        'CPU stage0 must match the actual original D16 initialization.')
    b.require(type(optimizer) is b.torch.optim.AdamW and optimizer.defaults['eps'] == 1e-8,
        'Keep the original AdamW default epsilon.')
    for path in b.HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    imported = {name:module.__file__ for name,module in sys.modules.items()
        if name.split('.')[0] in ('train_utils','litebsq','rotation','e2e_common') and getattr(module,'__file__',None)}
    b.require(all(Path(path).resolve().is_relative_to(b.SNAPSHOT) for path in imported.values()),
        'All native imports must remain in the bound N snapshot.')
    quality = b.read(b.CONTROL_FULL8)
    readout = b.read(b.CONTROL_READOUT)
    b.require(quality['checkpoint_id'] == readout['checkpoint_id'] == reference['raw_checkpoint_id']
        and quality['leaf_count'] == 64 and quality['sample_count'] == 24742
        and readout['input_ids_sha256'] == b.INPUT_SHA,
        'Bind the existing complete original D16 control without evaluating it again.')
    b.write(b.RESULT / 'prepared.json', dict(status='cpu_verified', exit_code=0,
        command=[sys.executable,*sys.argv], created_unix=time.time(), source_binding=source,
        source_sha256=b.pins(), original_prepared_sha256=b.sha(b.ORIGINAL_PREPARED),
        live_down_source_digest=old['live_down_source_digest'], frozen_prefix_digest=old['frozen_prefix_digest'],
        original_D16_report=str(b.CONTROL_REPORT), original_D16_full8=str(b.CONTROL_FULL8),
        original_D16_readout=str(b.CONTROL_READOUT), runtime_identity=identity, effective_settings=b.ARMS[b.ARM],
        category_config=asdict(cfg), CPU_stage0_initialization=rows[0], actual_stage_initialization_references=initial,
        training_dtype=str(native._resolve_train_dtype(parsed[2])), optimizer_epsilon=optimizer.defaults['eps'],
        actual_native_batch_argument=ast.unparse(batch), native_import_paths=imported,
        scalar_exposure=dict(control_per_layer_update=262144, candidate_per_layer_update=131072,
            control_M36_S2_10K_total=188743680000, candidate_M36_S2_10K_total=94371840000),
        reused_engineering_evidence=str(b.frozen.OUTPUT / 'd16/short/report.json'),
        CUDA_initialized=b.torch.cuda.is_initialized(),
        boundary='CPU proves stage0/configuration/source; actual formal factory must verify both original stage digests. '
            'No simulated stage1 RNG, GPU short, resume audit, control retraining or control re-evaluation.'))
    print(dict(status='cpu_verified', output=str(b.RESULT / 'prepared.json'), batch=8192,
        stage0_digest=rows[0]['initial_parameter_digest'], CUDA_initialized=b.torch.cuda.is_initialized()), flush=True)


if __name__ == '__main__':
    b.torch.set_num_threads(8)
    prepare()
