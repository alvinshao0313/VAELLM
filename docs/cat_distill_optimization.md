# CAT 恢复训练当前架构

CAT online 与 checkpoint-distill 共用 `train_utils.cat_after_category_common`：同一套目标 inventory、数据、五种模型级 loss、plain full-space LoRA、decoder finalize 和 v6 exact-resume 逻辑。

在线 CAT 有六种 `after_category_mode`；checkpoint-distill 只允许 current-family 三种。两条入口必须显式声明调用来源，不再根据目标数量或模型拓扑猜测。

稳定 checkpoint 与训练 step 分离：

- `training_step` 保存 optimizer/scheduler/RNG、当前组件状态和 `round_base` 身份，供精确续训。
- `category_boundary`/`final_model` 保存 finalize 后完整模型，不携带活动 PEFT proxy，`lora_config=null`。

旧 parser、compressed-subspace、模型级 DoRA/RSLoRA/AdaLoRA、旧 loss family 和 block-distill 路径不再属于 active stack。正式脚本见 `scripts/catlora_simple2.sh`、`scripts/catlora_distill_from_checkpoint.sh`。
