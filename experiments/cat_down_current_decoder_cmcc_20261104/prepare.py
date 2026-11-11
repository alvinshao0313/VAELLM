"""CPU proof of real ownership, original optimizer and original500 data prefix."""
import ast
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time

from . import bindings as b, training, evaluate
from experiments.adaptive_native_model_20261029.handoff_validation import frozen_digest
from transformers import Trainer, TrainingArguments
from transformers.optimization import get_scheduler


def data_proof(model, stage, vae):
    r = b.recovery
    tokenizer, data = r.canonical._prepare_after_category_dataset(model=model, vae_args=vae, cfg=stage.config)
    reference = b.torch.load(b.original.REFERENCE_ARGS, map_location='cpu', weights_only=False)
    fields = reference.to_dict()
    # CPU inspects the exact data consumer, without performing model computation.
    fields.update(output_dir=str(b.RESULT/'cpu_data_only'), use_cpu=True, bf16=False, fp16=False,
        dataloader_pin_memory=False, report_to=[], group_by_length=False)
    args = type(reference)(**fields)
    collator = r.canonical.build_distill_data_collator(tokenizer, model_max_length=1024, dynamic_padding=True)
    trainer = Trainer(model=model, args=args, train_dataset=data.train_dataset,
        data_collator=collator, processing_class=tokenizer)
    b.require(data.is_iterable and not data.group_by_length and not args.group_by_length
        and args.dataloader_num_workers == 16 and args.per_device_train_batch_size == 4
        and args.seed == 37 and args.data_seed == 31 and args.gradient_accumulation_steps == 1,
        'Rebuild the original lazy iterable500 stream:stage37/data31/B4/GAS1/workers16/group_by_lengthFalse.')
    loader = trainer.get_train_dataloader()
    iterator = iter(loader)
    trace, peak, first, peak_index = [], None, None, None
    for index in range(500):
        batch = {key:value.detach().cpu().clone() for key,value in next(iterator).items()}
        shape = list(batch['input_ids'].shape)
        b.require(shape[0] == 4 and shape[1] <= 1024 and shape[1] % 8 == 0, 'Original actual B4/cap1024 prefix changed.')
        trace.append(dict(batch=index+1, shape=shape, digest=b.c.digest(batch),
            valid_tokens=batch['attention_mask'].sum(-1).tolist()))
        if first is None:
            first = batch
        if peak is None or shape[1] > peak['input_ids'].shape[1]:
            peak, peak_index = batch, index+1
    old_probe_path = b.CONTROL_OUTPUT/'same_input_deployment.pt'
    probe = b.torch.load(old_probe_path, map_location='cpu', weights_only=True)['inputs']
    b.require(all(key in first and b.torch.equal(value, first[key]) for key,value in probe.items())
        and list(first['input_ids'].shape) == [4,456], 'Reconstructed first batch differs from retained original500 actual input probe.')
    b.require(peak['input_ids'].shape[1] == max(row['shape'][1] for row in trace), 'Peak fixture must come from this real original stream.')
    b.torch.save(dict(first_batch=first, peak_batch=peak, peak_batch_index=peak_index), b.DATA_FIXTURE)
    # Verify fixture recollation rather than assuming padding/label behavior.
    rows = [{key:value[i] for key,value in peak.items()} for i in range(4)]
    recollated = collator(rows)
    b.require(set(recollated) == set(peak) and all(b.torch.equal(recollated[key],value) for key,value in peak.items()),
        'The technical-only peak replay must preserve every real collated token/mask/label exactly.')
    proof = dict(is_iterable=data.is_iterable, actual_group_by_length=False, workers=args.dataloader_num_workers,
        data_config=asdict(stage.config.data), source_stats=data.source_stats, dataset_mix_spec=data.dataset_mix_spec,
        original_saved_args_sha256=b.sha(b.original.REFERENCE_ARGS), original_first_probe=str(old_probe_path),
        original_first_probe_sha256=b.sha(old_probe_path), first_batch_shape=list(first['input_ids'].shape),
        first_batch_identity_exact=True, prefix_batch_count=500, prefix_examples=2000,
        prefix_trace=trace, peak_batch_index=peak_index, peak_batch_shape=list(peak['input_ids'].shape),
        peak_batch_digest=b.c.digest(peak), peak_recollation_exact=True, fixture_sha256=b.sha(b.DATA_FIXTURE),
        CPU_only_argument_overrides=['output_dir','use_cpu','bf16','fp16','dataloader_pin_memory','report_to'],
        boundary='This records a newly rebuilt original500 stream and first-batch identity. Historical original500 per-batch hashes were never saved; no full historical exact-order claim.')
    del iterator, loader, trainer, first, peak
    gc.collect()
    return proof


def main():
    b.cpu_environment()
    b.torch.set_num_threads(4)
    b.require(not b.PREPARED.exists() and not b.OUTPUT.exists() and not b.DATA_FIXTURE.exists(), 'One fresh independent CPU preparation only.')
    started = time.time()
    binding = b.source()
    cat,hf,_train,vae,runtime,stage = b.recipe(b.OUTPUT/'formal')
    model,meta,strict,budget = b.checkpoint.load(b.RAW)
    b.require(meta['checkpoint_id'] == b.RAW_ID and not strict.missing_keys and not strict.unexpected_keys, 'Load only the exact originalV raw.')
    selected,native,compact,lr,inventory = training.selection(model,stage)
    active = {name:p for name,p in model.named_parameters() if p.requires_grad}
    frozen = frozen_digest(model,{id(p) for p in active.values()})
    non_down = b.c.digest({name:value for name,value in model.state_dict().items()
        if not any(name.startswith(prefix+'.') for prefix,_ in native)})
    reference = b.torch.load(b.original.REFERENCE_ARGS,map_location='cpu',weights_only=False)
    cls,kwargs = training.MixedCatTrainer.get_optimizer_cls_and_kwargs(reference, model)
    kwargs = {key:value for key,value in kwargs.items() if key not in ('params','model','optimizer_dict')}
    groups = b.recovery.build_model_level_param_groups(selected,lr_config=lr,model=model)
    optimizer = cls(groups,**kwargs)
    b.require(cls is b.torch.optim.AdamW and kwargs == dict(lr=1e-4,betas=(.9,.999),eps=1e-8)
        and len(groups) == 1 and groups[0]['lr'] == 1e-5 and groups[0]['weight_decay'] == 0.
        and not optimizer.state, 'Original actual freshAdamW defaults/group semantics only; scope changes parameter list.')
    scheduler = get_scheduler('cosine',optimizer=optimizer,num_warmup_steps=5,num_training_steps=500)
    old_groups = b.torch.load(b.original.REFERENCE_ARGS.with_name('optimizer.pt'),map_location='cpu',weights_only=True)['param_groups']
    actual = optimizer.state_dict()['param_groups']
    b.require([{k:v for k,v in g.items() if k!='params'} for g in actual] == [{k:v for k,v in g.items() if k!='params'} for g in old_groups]
        and len(actual[0]['params']) == 216 and len(old_groups[0]['params']) == 1728,
        'All original group fields remain exact except intended parameter inventory.')
    effective = dict(cls=cls.__module__+'.'+cls.__name__, constructor_kwargs=b.original.normalized(kwargs),
        defaults=b.original.normalized(optimizer.defaults), param_groups=b.original.normalized(actual), fresh_state_entries=0,
        original_group_fields_exact_except_params=True, decoder_base_lr=1e-5, scheduled_initial_lr=0.,
        standard_scheduler_initial_state=scheduler.state_dict())
    data = data_proof(model,stage,vae)
    b.require(frozen_digest(model,{id(p) for p in active.values()}) == frozen, 'CPU data reconstruction changed frozen fields.')
    for path in b.HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    record = dict(schema='current36_down_CPU_prepared_v1',status='verified_CPU_current36_down_and_original_data_contract',
        command=[sys.executable,*sys.argv],pid=__import__('os').getpid(),created_unix=time.time(),contract=b.CONTRACT,
        source_binding=binding, source_budget=budget, original_stage=asdict(stage), original_stage_exact=True,
        original_runtime_identity=runtime['runtime_identity'],effective_optimizer=effective, actual_inventory=inventory,
        frozen_state=frozen, frozen_non_down_deployment_digest=non_down, data=data,
        CUDA_initialized=b.torch.cuda.is_initialized(),elapsed_seconds=time.time()-started,source_sha256=b.pins())
    b.require(not record['CUDA_initialized'], 'CPU must not initialize CUDA.')
    b.OUTPUT.mkdir(parents=True)
    b.write(b.PREPARED,record)
    print(json.dumps(dict(status=record['status'],trainable_leaves=216,trainable_elements=767232,
        first_batch=data['first_batch_shape'],peak=data['peak_batch_shape'],peak_index=data['peak_batch_index'],seconds=record['elapsed_seconds'])),flush=True)


if __name__ == '__main__':
    main()
