import os
import sys
from typing import Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from opencompass.models import HuggingFaceBaseModel
from opencompass.registry import MODELS
from opencompass.utils import get_logger


@MODELS.register_module()
class VAELLMOpenCompassModel(HuggingFaceBaseModel):
    def __init__(
        self,
        checkpoint_dir: str,
        adapter_dir: Optional[str] = None,
        base_model_path: Optional[str] = None,
        access_token: Optional[str] = None,
        map_location: str = "cpu",
        strict: bool = True,
        eval_device: str = "cuda",
        prewarm_group_size: int = 8,
        **kwargs,
    ):
        if not checkpoint_dir:
            raise ValueError("checkpoint_dir is required.")
        self.checkpoint_dir = checkpoint_dir
        self.adapter_dir = adapter_dir or None
        self.base_model_path = base_model_path
        self.access_token = access_token
        self.map_location = map_location
        self.strict = bool(strict)
        self.eval_device = eval_device
        self.prewarm_group_size = int(prewarm_group_size)
        tokenizer_path = kwargs.pop("tokenizer_path", None) or base_model_path
        path = base_model_path or checkpoint_dir
        super().__init__(path=path, tokenizer_path=tokenizer_path, **kwargs)

    def _load_model(self, path: str, kwargs: dict, peft_path: Optional[str] = None, peft_kwargs: dict = dict()):
        if peft_path is not None:
            raise ValueError("Use adapter_dir for VAELLMOpenCompassModel, not peft_path.")

        logger = get_logger()
        from tools.cat_eval import (
            _prepare_model_for_eval,
            _read_checkpoint_meta,
            _read_dense_adapter_meta,
            _resolve_adapter_dir,
            _resolve_checkpoint_dir,
            _resolve_checkpoint_loader,
            _resolve_eval_device,
            _validate_adapter_checkpoint_match,
        )

        ckpt_dir = _resolve_checkpoint_dir(self.checkpoint_dir)
        meta_preview = _read_checkpoint_meta(ckpt_dir)
        checkpoint_loader = _resolve_checkpoint_loader(meta_preview)
        adapter_dir = None if self.adapter_dir is None else _resolve_adapter_dir(self.adapter_dir)

        if adapter_dir is not None:
            if checkpoint_loader != "v6":
                raise ValueError(
                    "adapter_dir requires a compressed cat checkpoint. "
                    f"Current checkpoint loader detected: {checkpoint_loader}."
                )
            adapter_meta = _read_dense_adapter_meta(adapter_dir)
            fingerprint = _validate_adapter_checkpoint_match(
                checkpoint_dir=ckpt_dir,
                checkpoint_meta=meta_preview,
                adapter_meta=adapter_meta,
            )
            logger.info(
                "Adapter match check passed: adapter_dir=%s meta_sha256=%s state_sha256=%s",
                adapter_dir,
                fingerprint["meta_sha256"],
                fingerprint["state_sha256"],
            )

            from e2e_common.compressed_dense_bridge import build_dense_model_from_checkpoint
            from e2e_common.adapter_loading import (
                adapter_has_post_norm_head_linear,
                assert_adapter_load_result_clean,
                build_peft_model_for_adapter_load,
                detach_tied_lm_head_weight_if_needed,
                read_adapter_weight_keys,
                validate_adapter_modules_to_save,
            )
            from e2e_common.post_norm_head import (
                ensure_post_norm_head_linear,
                fuse_post_norm_head_linear,
                has_post_norm_head_linear,
            )

            adapter_keys = read_adapter_weight_keys(adapter_dir)
            adapter_config = validate_adapter_modules_to_save(adapter_dir, adapter_keys)
            adapter_has_post_norm_head = adapter_has_post_norm_head_linear(adapter_keys)
            logger.info(
                "Adapter precheck passed: weight_keys=%d modules_to_save=%s has_post_norm_head_linear=%s",
                len(adapter_keys),
                adapter_config.get("modules_to_save"),
                str(adapter_has_post_norm_head),
            )
            model, meta, _resolved_ckpt_dir = build_dense_model_from_checkpoint(
                ckpt_dir,
                access_token=self.access_token,
                base_model_path=self.base_model_path,
                logger=logger,
                decode_group_size=self.prewarm_group_size,
                decode_device=_resolve_eval_device(self.eval_device, logger),
            )
            if adapter_has_post_norm_head:
                attached = ensure_post_norm_head_linear(model)
                logger.info("Attached post_norm_linear before adapter load: %s", str(attached))
            peft_model = build_peft_model_for_adapter_load(model, adapter_dir)
            adapter_load_result = peft_model.load_adapter(adapter_dir, adapter_name="default", is_trainable=False)
            assert_adapter_load_result_clean(adapter_load_result)
            model = peft_model.merge_and_unload(safe_merge=True)
            if adapter_has_post_norm_head:
                detach_tied_lm_head_weight_if_needed(model, logger)
                fused_post_norm_head = fuse_post_norm_head_linear(model)
                if not fused_post_norm_head:
                    raise RuntimeError("Adapter has lm_head.post_norm_linear weights, but post_norm_linear fusion failed.")
                if has_post_norm_head_linear(model):
                    raise RuntimeError("post_norm_linear fusion returned success, but LMHeadWithPostNormLinear remains.")
            checkpoint_loader = "cat+adapter"
        else:
            from train_utils.v6_model_loader import load_v6_model_checkpoint

            model, meta, load_result = load_v6_model_checkpoint(
                ckpt_dir,
                access_token=self.access_token,
                base_model_path=self.base_model_path,
                map_location=self.map_location,
                strict=self.strict,
            )
            logger.info(
                "Loaded cat checkpoint for OpenCompass: missing_keys=%d unexpected_keys=%d",
                len(getattr(load_result, "missing_keys", [])),
                len(getattr(load_result, "unexpected_keys", [])),
            )

        resolved_base_model_path = self.base_model_path or meta.get("base_model_path")
        if not resolved_base_model_path:
            raise ValueError("Cannot determine base_model_path. Provide base_model_path.")
        logger.info(
            "Loaded VAELLM OpenCompass model: checkpoint_loader=%s checkpoint_dir=%s base_model_path=%s adapter_dir=%s",
            checkpoint_loader,
            ckpt_dir,
            resolved_base_model_path,
            adapter_dir,
        )
        model.eval()
        self.model = model
        self.model.generation_config.do_sample = False
        _prepare_model_for_eval(
            self.model,
            self.eval_device,
            self.prewarm_group_size,
            logger,
            {},
        )
