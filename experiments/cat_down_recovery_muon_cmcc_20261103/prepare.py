"""CPU proof from the real raw-down tensors and actual saved AdamW arguments."""
from dataclasses import asdict
import gc
import inspect
import json
import time

from . import bindings as b
from .optimizer import CategoryMuon
from . import training
from experiments.adaptive_native_model_20261029.handoff_validation import frozen_digest


def main():
    b.cpu_environment()
    b.torch.set_num_threads(4)
    b.require(not b.PREPARED.exists() and not b.OUTPUT.exists(),'Never overwrite preparation or optimizer comparison.')
    started=time.monotonic()
    source=b.source()
    cat,hf,training_args,vae,runtime,stage=b.recipe(b.OUTPUT/'formal')
    reference=b.torch.load(b.REFERENCE_ARGS,map_location='cpu',weights_only=False)
    actual_args={name:getattr(reference,name) for name in ('adam_beta1','adam_beta2','adam_epsilon','learning_rate',
        'weight_decay','max_steps','warmup_ratio','max_grad_norm','seed','data_seed','per_device_train_batch_size')}
    b.require(actual_args==dict(adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,learning_rate=1e-4,
        weight_decay=0.,max_steps=500,warmup_ratio=.01,max_grad_norm=1.5,seed=37,data_seed=31,per_device_train_batch_size=4),
        'Actual original category Trainer defaults/configuration differ from the fixed reference.')
    model,meta,strict,budget=b.checkpoint.load(b.RAW)
    b.require(meta['checkpoint_id']==b.RAW_ID and not strict.missing_keys and not strict.unexpected_keys,
        'Load the complete actual raw C0 strictly.')
    selected,native,compact,lr,info=b.recovery.selection(model,stage)
    active={name:p for name,p in selected.peft_model.named_parameters() if p.requires_grad}
    b.require(len(active)==1728 and not selected.lora_parameters and not selected.norm_parameters
        and not selected.lm_head_parameters and not selected.residual_lora_parameters
        and {id(p) for p in active.values()}=={id(p) for p in selected.decoder_parameters.values()}
        and all(p.dtype==b.torch.float32 for p in active.values()),'Use only original actual FP32 all252 D leaves.')
    layout=b.mature.derive_layout(model,meta)
    matrices,auxiliary=b.mature.split(active,layout)
    b.require(len(matrices)==576 and len(auxiliary)==1152 and sum(row['stages'] for row in layout.values())==1008,
        'Real matrix/stage/auxiliary inventory differs.')
    for name,p in matrices.items():
        pieces=b.mature.stages(p,layout[name]['stages'])
        b.require(pieces.is_contiguous() and pieces.numel()==p.numel()
            and all(pieces[index].data_ptr()==p.data_ptr()+index*pieces[0].numel()*p.element_size()
                for index in range(len(pieces))),'Actual independent stage matrices overlap or reorder: '+name)
    cls,kwargs=training.MuonMixedCatTrainer.get_optimizer_cls_and_kwargs(reference,model)
    b.require(cls is b.torch.optim.AdamW,'Original actual reference optimizer class changed.')
    kwargs={key:value for key,value in kwargs.items() if key not in ('params','model','optimizer_dict')}
    optimizer=CategoryMuon(active,layout,lr.resolved_decoder_lr(),kwargs)
    state=optimizer.state_dict()
    b.require(set(state)=={'state','param_groups'} and not state['state'] and len(state['param_groups'])==2
        and optimizer.adam.state is optimizer.state and optimizer.adam.param_groups[0] is optimizer.param_groups[1]
        and all(group['lr']==1e-5 for group in optimizer.param_groups),'HF-standard shared Muon/AdamW state/group contract failed.')
    frozen=frozen_digest(model,{id(p) for p in active.values()})
    implementation=dict(canonical=inspect.getsourcefile(b.recovery.canonical),
        base_trainer=inspect.getsourcefile(training.original.MixedCatTrainer),
        grouping=inspect.getsourcefile(training.build_model_level_param_groups),
        mature_matrix_math=inspect.getsourcefile(b.mature),
        effective_trainer=inspect.getsourcefile(training.MuonMixedCatTrainer))
    b.require(all(str(b.original.common.s.SNAPSHOT) in implementation[key] for key in ('canonical','grouping')),
        'Keep the actual selected N canonical data/loss/optimizer grouping runtime.')
    record=dict(schema='C0_down_category_Muon_prepared_v1',status='verified_CPU_actual_layout_and_original_contract',
        source_binding=source,contract=b.CONTRACT,original_stage=asdict(stage),
        original_saved_training_args=actual_args,original_optimizer_enum=str(reference.optim),
        original_lr_scheduler_enum=str(reference.lr_scheduler_type),
        original_decoder_group_lr=lr.resolved_decoder_lr(),original_optimizer_kwargs=kwargs,
        original_stage_optimizer_is_reference_only=True,effective_optimizer=optimizer.effective_contract(),
        actual_inventory=info,layout=layout,trainable_leaves=len(active),trainable_elements=sum(p.numel() for p in active.values()),
        real_stage_matrices_independent=True,frozen_state=frozen,source_budget=budget,
        original_data_config=asdict(stage.config.data),original_loss_config=asdict(stage.config.loss),
        original_runtime_identity=runtime['runtime_identity'],implementation=implementation,
        stage_comparison=dict(all_original_fields_JSON_normalized_exact=True,only_ignored_field='output_dir',
            reference_report_sha256=b.sha(b.C0/'report.json')),
        source_sha256=b.pins(),CUDA_initialized=b.torch.cuda.is_initialized(),elapsed_seconds=time.monotonic()-started,
        boundary='Actual CPU layout/reference/source proof only; new HF standard Muon lifecycle2/save/fresh still required. No optimizer update or GPU forward.')
    b.require(not record['CUDA_initialized'],'CPU proof initialized CUDA.')
    b.OUTPUT.mkdir(parents=True)
    b.write(b.PREPARED,record)
    print(json.dumps(dict(status=record['status'],trainable_leaves=len(active),matrix_leaves=len(matrices),
        independent_matrices=1008,auxiliary_leaves=len(auxiliary),seconds=record['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    main()
