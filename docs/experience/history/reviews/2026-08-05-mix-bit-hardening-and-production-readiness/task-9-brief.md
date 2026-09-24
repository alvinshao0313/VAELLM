> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-9-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 9: Introduce Tokenizer Fingerprint v2 and Save a Self-Contained Final Tokenizer

**Files:**
- Modify: `mix_bit/calibration.py`
- Modify: `mix_bit/model_adapter.py`
- Modify: `mix_bit/assembler.py`
- Modify: `mix_bit/validation.py`
- Modify: `mix_bit/cli/assemble_mixed_model.py`
- Modify: `mix_bit/tests/test_calibration.py`
- Modify: `mix_bit/tests/test_assembler.py`
- Modify: `mix_bit/tests/test_validation.py`
- Modify: `mix_bit/tests/test_tiny_integration.py`
- Modify: `mix_bit/README.md`

**Constants and interfaces:**

```python
TOKENIZER_FINGERPRINT_VERSION = 2


def build_tokenizer_fingerprint_payload(tokenizer: Any) -> dict[str, Any]:
    """Return versioned provenance plus content fields used by the digest."""


def compute_tokenizer_config_sha256(tokenizer: Any) -> str:
    """Hash only version and content; reported path is provenance-only."""
```

保留现有函数名 `compute_tokenizer_config_sha256`，避免调用方改名；其内部改为 v2。

在 `mix_bit/model_adapter.py` 新增并统一使用：

```python
def normalize_tokenizer_for_mix_bit(tokenizer: Any, *, source_label: str) -> Any:
    """Force right padding and normalize a missing pad token to eos exactly once."""
```

实现顺序固定：先设置 `padding_side="right"`；再把 `mix_bit_pad_token_normalized_from_eos` 设为 False；如果 `pad_token_id is None`，要求 `eos_token_id` 存在，令 `pad_token_id=eos_token_id` 并把标记设为 True；返回同一 tokenizer 对象。`GenericDecoderAdapter.load_tokenizer`、assembler 的 local reload 和 validation 的 local reload 都必须调用该 helper，不得复制三套 normalization 逻辑。

`stable_init_kwargs` 必须排除以下不稳定或路径/鉴权相关键：`name_or_path`、`tokenizer_file`、`vocab_file`、`merges_file`、`special_tokens_map_file`、`tokenizer_config_file`、`added_tokens_file`、`cache_dir`、`local_files_only`、`revision`、`token`、`use_auth_token`。该集合定义为模块常量 `TOKENIZER_INIT_KWARGS_EXCLUDED`。

### Tokenizer fingerprint v2 payload

payload 固定分成 provenance 与 content 两部分：

```json
{
  "version": 2,
  "reported_name_or_path": "Qwen/Qwen3-8B",
  "content": {
    "class_name": "Qwen2TokenizerFast",
    "vocab_size": 151936,
    "model_max_length": 32768,
    "padding_side": "right",
    "truncation_side": "right",
    "bos_token_id": null,
    "eos_token_id": 151645,
    "pad_token_id": 151645,
    "unk_token_id": null,
    "chat_template": "<full template or null>",
    "special_tokens_map": {},
    "added_vocab": [],
    "core_kind": "backend_tokenizer_json",
    "core_sha256": "<64 lowercase hex characters>",
    "stable_init_kwargs": {}
  }
}
```

示例中的具体 token id 只说明字段类型，测试和实现必须读取 tokenizer 实际值，不得硬编码 Qwen 数值。`reported_name_or_path` 只用于 provenance，不参与 digest。`compute_tokenizer_config_sha256` 必须只 hash：

```python
payload = build_tokenizer_fingerprint_payload(tokenizer)
digest_payload = {
    "version": payload["version"],
    "content": payload["content"],
}
return _sha256_bytes(_canonical_json_bytes(digest_payload))
```

因此 source tokenizer 的 `name_or_path="Qwen/Qwen3-8B"` 与 reload tokenizer 的 `name_or_path=<final_model_dir>` 不会造成伪 mismatch；除路径以外的任何 content 变化仍必须改变 SHA。

### Core tokenizer hash

优先：

```python
backend = getattr(tokenizer, "backend_tokenizer", None)
if backend is not None and callable(getattr(backend, "to_str", None)):
    core_kind = "backend_tokenizer_json"
    core_bytes = backend.to_str().encode("utf-8")
```

fallback：

```python
get_vocab = getattr(tokenizer, "get_vocab", None)
if not callable(get_vocab):
    raise ValueError(
        "Tokenizer exposes neither backend_tokenizer.to_str nor get_vocab"
    )
vocab_items = sorted(
    (str(token_text), int(token_id))
    for token_text, token_id in get_vocab().items()
)
core_kind = "sorted_vocab"
core_bytes = _canonical_json_bytes(vocab_items)
```

如果两种方式都不可用，明确失败；不得退回仅 `vocab_size`。

### Recursive JSON normalization

只允许 null、bool、int、finite float、str、list/tuple、dict with stringified keys。`Path` 转 str。对于 Hugging Face `AddedToken` 或具有 `content` 字段的同类对象，固定序列化为：

```json
{
  "content": "<token text>",
  "single_word": false,
  "lstrip": false,
  "rstrip": false,
  "normalized": true,
  "special": false
}
```

上述布尔值必须从对象同名属性读取，缺失时使用示例中的默认值。其他对象记录为：

```json
{"unsupported_type": "ClassName"}
```

不得调用对象 `repr()`，避免地址造成非确定 hash。`special_tokens_map`、`added_vocab` 和过滤后的 `stable_init_kwargs` 均使用这一 normalization。`added_vocab` 必须转成按 token 文本排序的 `[token_text, token_id]` 列表。

### Calibration manifest

新增字段：

```json
"tokenizer_fingerprint_version": 2
```

existing calibration resume 必须要求 version=2 和 SHA 一致。旧 manifest 缺少 version 时必须失败并提示重新生成 calibration，不允许静默接受。

### Final checkpoint tokenizer save

以下两个 assembler 接口都新增 keyword-only 参数，并逐层原样传递：

```python
def assemble_optimal_mixed_checkpoint(
    *,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    inventory_path: str,
    pool_index: CandidatePoolIndex,
    allocation_path: str,
    device: str,
    allow_suboptimal: bool = False,
    overwrite: bool = False,
    output_dir: str | None = None,
    access_token: str | None = None,
) -> dict[str, Any]:
    """Assemble, save and verify the selected mixed checkpoint."""


def save_full_checkpoint_from_assignments(
    *,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    pool_index: CandidatePoolIndex,
    assignments: Mapping[str, str],
    output_dir: str,
    device: str,
    mix_bit_provenance: Mapping[str, Any],
    access_token: str | None = None,
) -> dict[str, Any]:
    """Save the standalone checkpoint and its tokenizer."""
```

除新增 `access_token` 外，不得增加、删除、重排或重命名参数。

流程固定：

1. 通过当前 profile adapter 加载 source tokenizer；该 adapter 已调用 `normalize_tokenizer_for_mix_bit`；
2. 计算 source tokenizer fingerprint v2 和完整 payload；
3. 把 fingerprint version、content SHA 和 source reported path 写入 `extra_meta["mix_bit"]`；字段名固定为 `tokenizer_fingerprint_version`、`tokenizer_fingerprint_sha256`、`source_tokenizer_reported_name_or_path`；
4. 调用 `save_model_checkpoint`，必须显式传入 source tokenizer；
5. 从最终 output_dir local-only 重载：

调用 `AutoTokenizer.from_pretrained` 时固定传入最终 `output_dir`、`local_files_only=True`、`trust_remote_code=False`；返回对象必须立即传给 `normalize_tokenizer_for_mix_bit`，其中 `source_label` 固定为 `str(output_dir)`。

6. 计算 reload content fingerprint，必须等于 source；
7. 不相等则失败并保留 state、meta 和 tokenizer 文件供排查；当前 assembler 没有 completed marker，因此不得虚构或删除不存在的 marker；
8. return payload 新增 `tokenizer_fingerprint_sha256` 和 `tokenizer_reported_name_or_path`。

`assemble_mixed_model.py` 新增 `--access_token` 并传递。

### Final validation

`validate_mixed_model` 必须：

- 从 final dir local-only 加载 tokenizer；
- 验证 checkpoint meta 中 tokenizer fingerprint version=2；
- 重算 fingerprint 并比较；
- validation report 新增：

```json
"tokenizer": {
  "fingerprint_version": 2,
  "fingerprint_sha256": "<64 lowercase hex characters>",
  "reported_name_or_path": "/absolute/final_model/path",
  "local_reload_passed": true
}
```

- 下游 evaluator 的 tokenizer 必须来自 final dir；这里禁止的是 tokenizer 再依赖原模型路径，模型 checkpoint 本身仍按现有 `base_model_path` 机制重建 backbone。

### Existing-output tokenizer skip contract

Task 6 的 identical-output state 检查通过后，还必须：

1. 从 final dir local-only 加载 tokenizer 并调用统一 normalization；
2. 从 checkpoint meta 读取 version 和 expected content SHA；
3. 重算 fingerprint 并比较；
4. tokenizer 文件缺失、旧 version、local reload 失败或 SHA mismatch 时抛 `ValueError` 并要求显式 `--overwrite`；
5. 不得从原模型路径补载 tokenizer，不得自动调用 `save_pretrained` 修复旧目录。

只有上述检查和 Task 6 state fingerprint 检查都通过，才能 `skipped_identical=True`。

- [ ] **Step 1: Add deterministic tokenizer fingerprint tests**

使用 lightweight fake tokenizer，覆盖：

- 相同 backend JSON/hash 相同；
- content 完全相同但 `name_or_path` 不同，hash 必须相同且 payload 的 reported path 不同；
- vocab 内容变化但 vocab_size 相同，hash 必须变化；
- chat template 变化，hash 必须变化；
- added token 变化，hash 必须变化；
- name/path 相同但 core 变化，hash 必须变化；
- 不支持对象不会把内存地址写入 payload。

- [ ] **Step 2: Add calibration version tests**

完整实现：

- `test_calibration_manifest_records_tokenizer_fingerprint_v2`
- `test_calibration_resume_rejects_legacy_tokenizer_fingerprint`
- `test_calibration_resume_rejects_same_vocab_size_changed_core`

- [ ] **Step 3: Add final save/reload tokenizer test**

使用 tiny local tokenizer fixture；断言 final dir 可以 `local_files_only=True` 加载，fingerprint 与 source 相同。同时覆盖：provenance 相同但 tokenizer 文件缺失时不得 skip；meta 为 fingerprint v1 时不得 skip；tokenizer 完整且 fingerprint 相同时才允许 `skipped_identical=True`。

不得让测试访问 Hugging Face 网络。

- [ ] **Step 4: Add validation negative test**

修改 final tokenizer file 后，validation 必须失败于 fingerprint mismatch，而不是继续跑 KL。

- [ ] **Step 5: Run tests and confirm old code fails**

- [ ] **Step 6: Implement fingerprint v2 in calibration module**

保持旧函数名，更新所有 manifest writer/reader。

- [ ] **Step 7: Save tokenizer in assembler and add CLI token**

不得修改 `save_model_checkpoint` API；它已经支持 tokenizer 参数。

- [ ] **Step 8: Validate local tokenizer in final validation**

下游 evaluator 获取 tokenizer 时优先使用 final dir。

- [ ] **Step 9: Run focused tests**

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest \
  mix_bit/tests/test_calibration.py \
  mix_bit/tests/test_assembler.py \
  mix_bit/tests/test_validation.py \
  mix_bit/tests/test_tiny_integration.py -q
```

- [ ] **Step 10: Commit Task 9 files**

```bash
git add \
  mix_bit/calibration.py \
  mix_bit/model_adapter.py \
  mix_bit/assembler.py \
  mix_bit/validation.py \
  mix_bit/cli/assemble_mixed_model.py \
  mix_bit/tests/test_calibration.py \
  mix_bit/tests/test_assembler.py \
  mix_bit/tests/test_validation.py \
  mix_bit/tests/test_tiny_integration.py \
  mix_bit/README.md
git commit -m "fix: persist and verify final tokenizer"
```

---

