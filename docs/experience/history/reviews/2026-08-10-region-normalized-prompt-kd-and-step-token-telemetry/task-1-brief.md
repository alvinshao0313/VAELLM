> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-1-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 1: Replace Fractional Prompt Mask with Two Binary Region Masks

**Files:** `train_utils/distill_losses.py`, `tests/test_distill_losses.py`

Restore `build_distill_token_mask()` to the response-only causal helper with exactly three inputs: labels, attention mask, and reference logits. Remove the prompt-weight argument from this helper.

Add exactly the frozen dataclass `DistillTokenRegions` with fields `response_mask: torch.Tensor` and `prompt_mask: torch.Tensor`. Add function `build_distill_token_regions` with keyword-only inputs `labels: Optional[torch.Tensor]`, `attention_mask: Optional[torch.Tensor]`, `reference_logits: torch.Tensor`, returning `DistillTokenRegions`.

- [ ] First add a failing single-turn test for labels `[-100,-100,-100,A,B,EOS]`, expecting response mask `[0,0,1,1,1,0]` and prompt mask `[1,1,0,0,0,0]`.
- [ ] Add a padding test. For labels `[-100,-100,A,EOS,-100,-100]` and attention `[1,1,1,1,0,0]`, expect response `[0,1,1,0,0,0]` and prompt `[1,0,0,0,0,0]`.
- [ ] Add a multi-turn/interleaved test. For labels `[-100,A,-100,B,EOS]`, expect response `[1,0,1,1,0]` and prompt `[0,1,0,0,0]`.
- [ ] Add labels-none fallback test: response remains the current shifted attention/no-metadata mask; prompt is all zero because prompt/response cannot be inferred.
- [ ] Add invariants: both masks float32, binary, same shape/device, disjoint, final position zero.
- [ ] Restore `build_distill_token_mask()` to pre-feature behavior: when labels exist use `labels != -100`; otherwise use attention validity or all-ones; causal-shift left; final position zero. Preserve current shape validation.
- [ ] Implement `build_distill_token_regions()` so `response_mask` directly reuses `build_distill_token_mask()`. For prompt, use `labels == -100`, AND with attention validity when attention exists, causal-shift left, final position zero. With labels absent return zero prompt mask.
- [ ] Remove tests whose intended contract is that prompt weighting inserts fractional values such as 0.1 into the training mask. Low-level reducers may keep arbitrary-float-mask tests if independently useful, but prompt weighting must not depend on them.
- [ ] Run `pytest tests/test_distill_losses.py -q`.

Task 1 acceptance: no production prompt-weight path creates a fractional causal mask.

---

