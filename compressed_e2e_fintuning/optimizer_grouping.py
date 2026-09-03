import logging

from torch import nn


logger = logging.getLogger(__name__)


def create_decoder_grouped_optimizer(trainer):
    """Create the continuous E2E optimizer with decoder-specific learning rate groups."""
    decoder_param_ids = frozenset(int(v) for v in getattr(trainer, "decoder_param_ids", ()))
    if not decoder_param_ids:
        raise RuntimeError("create_decoder_grouped_optimizer requires non-empty decoder_param_ids.")
    decoder_lr = getattr(trainer, "decoder_lr", None)
    if decoder_lr is None:
        raise RuntimeError("decoder_param_ids were provided without a resolved decoder_lr.")
    if trainer.optimizer is not None:
        return trainer.optimizer

    opt_model = getattr(trainer, "model_wrapped", None) or trainer.model
    decay_parameters = set(trainer.get_decay_parameter_names(opt_model))
    nondecoder_decay = []
    nondecoder_no_decay = []
    decoder_decay = []
    decoder_no_decay = []
    trainable_ids = set()

    for name, param in opt_model.named_parameters():
        if not bool(param.requires_grad):
            continue
        param_id = id(param)
        trainable_ids.add(param_id)
        is_decoder = param_id in decoder_param_ids
        has_decay = name in decay_parameters
        if is_decoder and has_decay:
            decoder_decay.append(param)
        elif is_decoder:
            decoder_no_decay.append(param)
        elif has_decay:
            nondecoder_decay.append(param)
        else:
            nondecoder_no_decay.append(param)

    grouped_lists = (nondecoder_decay, nondecoder_no_decay, decoder_decay, decoder_no_decay)
    grouped_ids = {id(param) for params in grouped_lists for param in params}
    if grouped_ids != trainable_ids or sum(len(params) for params in grouped_lists) != len(grouped_ids):
        raise RuntimeError("E2E optimizer grouping produced duplicate or missing trainable parameters.")
    missing_decoder = decoder_param_ids - trainable_ids
    if missing_decoder:
        raise RuntimeError(
            "Decoder optimizer group contains ids that are not trainable model parameters: "
            + ",".join(str(v) for v in sorted(missing_decoder))
        )

    optimizer_grouped_parameters = []
    if nondecoder_decay:
        optimizer_grouped_parameters.append(
            {
                "group_name": "nondecoder_decay",
                "params": nondecoder_decay,
                "weight_decay": trainer.args.weight_decay,
            }
        )
    if nondecoder_no_decay:
        optimizer_grouped_parameters.append(
            {
                "group_name": "nondecoder_no_decay",
                "params": nondecoder_no_decay,
                "weight_decay": 0.0,
            }
        )
    if decoder_decay:
        optimizer_grouped_parameters.append(
            {
                "group_name": "decoder_decay",
                "params": decoder_decay,
                "lr": float(decoder_lr),
                "weight_decay": trainer.args.weight_decay,
            }
        )
    if decoder_no_decay:
        optimizer_grouped_parameters.append(
            {
                "group_name": "decoder_no_decay",
                "params": decoder_no_decay,
                "lr": float(decoder_lr),
                "weight_decay": 0.0,
            }
        )

    if trainer.optimizer_cls_and_kwargs is not None:
        optimizer_cls, optimizer_kwargs = trainer.optimizer_cls_and_kwargs
    else:
        optimizer_cls, optimizer_kwargs = trainer.get_optimizer_cls_and_kwargs(trainer.args, opt_model)
    optimizer_kwargs = dict(optimizer_kwargs)
    if "params" in optimizer_kwargs:
        optimizer_grouped_parameters = optimizer_kwargs.pop("params")
    if "model" in optimizer_kwargs:
        optimizer_grouped_parameters = optimizer_kwargs.pop("model")
    if "optimizer_dict" in optimizer_kwargs:
        optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

    trainer.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    if optimizer_cls.__name__ == "Adam8bit":
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
        skipped = 0
        for module in opt_model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                manager.register_module_override(module, "weight", {"optim_bits": 32})
        logger.info("Adam8bit embedding fp32 override: skipped=%sM params", skipped / 2**20)

    return trainer.optimizer
