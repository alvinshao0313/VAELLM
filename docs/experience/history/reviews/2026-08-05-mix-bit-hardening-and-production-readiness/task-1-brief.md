> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-1-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 1: Implement the Shared Candidate Mode Contract

**Files:**
- Create: `mix_bit/candidate_contract.py`
- Create: `mix_bit/tests/test_candidate_contract.py`
- Read only: `litebsq/bitpack.py`
- Read only: `train_utils/model_checkpoint_io.py::_collect_vae_linear_specs`

**Interfaces:**

```python
from collections.abc import Mapping, Sequence
from mix_bit.schema import CandidateMode


def candidate_mode_from_payload(payload: Mapping[str, object], *, label: str) -> CandidateMode:
    """Parse exactly the five required mode fields and reject missing/invalid values."""


def validate_mode_payload(
    payload: Mapping[str, object],
    expected: CandidateMode,
    *,
    label: str,
) -> None:
    """Require all five mode fields to equal expected; nominal_bit tolerance is 1e-12."""


def validate_module_spec_mode_contract(
    spec: Mapping[str, object],
    mode: CandidateMode,
    *,
    label: str,
) -> None:
    """Require actual VAE structure, VQ storage and decoder dimensions to match mode."""
```

### Exact validation behavior

`candidate_mode_from_payload` must:

- require a mapping;
- require keys `name`, `nominal_bit`, `codebook_bits`, `codebook_dim`, `residual_stages`;
- convert using `str`, `float`, `int`, `int`, `int`;
- reject empty name;
- reject non-finite nominal bit;
- reject `codebook_bits < 1`、`codebook_dim < 1`、`residual_stages < 1`;
- 计算 `derived_nominal_bit = residual_stages * codebook_bits / codebook_dim`，要求 `abs(nominal_bit - derived_nominal_bit) <= 1e-12`；这里的 nominal bit 只表示 VQ sign stream，不计 decoder/metadata overhead，与现有 MILP 定义一致；
- instantiate and return existing `CandidateMode`;
- allow unrelated future metadata keys but never use them in equality decisions.

`validate_mode_payload` must compare:

```python
str(actual.name) == str(expected.name)
abs(float(actual.nominal_bit) - float(expected.nominal_bit)) <= 1e-12
int(actual.codebook_bits) == int(expected.codebook_bits)
int(actual.codebook_dim) == int(expected.codebook_dim)
int(actual.residual_stages) == int(expected.residual_stages)
```

`validate_module_spec_mode_contract` must support both `parallel_parts == 1` and `parallel_parts > 1`:

- `parallel_parts` missing时按 1；小于 1 失败。
- `residual_stages` 必须存在且与 mode 相等。
- `codebook_dim` 必须存在且与 mode 相等。
- `stage_codebook_dims` 必须是 list/tuple，长度等于 stages，所有元素都等于 mode codebook dim；不允许当前 checkpoint loader 的“单元素复制”兼容逻辑进入候选 contract。
- 当 stages > 1：
  - `stage_vq_weights` 和 `stage_decoders` 必须存在；
  - 二者长度都等于 stages；
  - 每个 stage 若 parallel_parts=1 必须是一个 dict；若 >1 必须是长度等于 parallel_parts 的 list/tuple。
- 当 stages == 1：
  - 使用 legacy-compatible `vq_weights` 和 `decoders`；
  - 二者必须是长度等于 parallel_parts 的 list/tuple；
  - `stage_vq_weights`/`stage_decoders` 可以为 null，但若非 null 必须与 stage 0 legacy 内容结构一致。
- 每个 VQ storage spec 必须先调用现有 `validate_bitpack_u8_spec`。
- normalized VQ spec 必须满足：

```python
normalized["storage_format"] == "bitpack_u8"
normalized["dtype"] == "uint8"
normalized["logical_dtype"] == "bool"
int(normalized["pack_bits"]) == 8
int(normalized["logical_shape"][-1]) == mode.codebook_bits
```

- `logical_shape` 为空或不是 list/tuple 时失败。
- 每个 decoder spec 必须是 dict，并满足：

```python
int(decoder["in_dim"]) == mode.codebook_bits
int(decoder["out_dim"]) == mode.codebook_dim
```

- 错误信息必须包含 `label`、字段名、actual、expected。

- [ ] **Step 1: Write failing parsing and metadata tests**

在 `test_candidate_contract.py` 按以下名称逐个实现完整 fixture、调用和断言：

- `test_mode_payload_requires_all_five_fields`
- `test_mode_payload_rejects_non_finite_nominal_bit`
- `test_mode_payload_rejects_nominal_bit_inconsistent_with_structure`
- `test_validate_mode_payload_rejects_same_name_wrong_nominal_bit`
- `test_validate_mode_payload_rejects_same_name_wrong_codebook_bits`
- `test_validate_mode_payload_rejects_same_name_wrong_codebook_dim`
- `test_validate_mode_payload_rejects_same_name_wrong_residual_stages`

Expected: import/function missing。

- [ ] **Step 2: Add reusable spec builders to the test file**

测试 builder 固定输出 packed storage：

```python
def packed_vq_spec(bits: int) -> dict[str, object]:
    return {
        "storage_format": "bitpack_u8",
        "dtype": "uint8",
        "logical_dtype": "bool",
        "pack_bits": 8,
        "logical_shape": [8, 1, bits],
        "shape": [8, 1, (bits + 7) // 8],
    }


def decoder_spec(bits: int, dim: int) -> dict[str, object]:
    return {
        "in_dim": bits,
        "out_dim": dim,
        "hidden_dim": 8,
        "num_res_blocks": 0,
        "norm_type": "layer",
        "activation_type": "swish",
        "decoder_type": "linear",
        "use_checkpoint": False,
        "param_dtype": "float32",
    }
```

- [ ] **Step 3: Write failing module-contract tests**

必须完整实现：

- `test_s2_single_part_contract_accepts_exact_structure`
- `test_s2_parallel_parts_contract_accepts_exact_structure`
- `test_contract_rejects_wrong_residual_stages`
- `test_contract_rejects_wrong_codebook_dim`
- `test_contract_rejects_short_stage_codebook_dims`
- `test_contract_rejects_wrong_stage_codebook_dim`
- `test_contract_rejects_wrong_stage_count`
- `test_contract_rejects_wrong_parallel_part_count`
- `test_contract_rejects_vq_logical_bits_mismatch`
- `test_contract_rejects_non_bitpacked_storage`
- `test_contract_rejects_decoder_in_dim_mismatch`
- `test_contract_rejects_decoder_out_dim_mismatch`
- `test_s1_legacy_fields_are_validated_without_stage_fields`

- [ ] **Step 4: Run the focused tests and confirm failure**

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest mix_bit/tests/test_candidate_contract.py -q
```

Expected: FAIL because module/functions do not exist。

- [ ] **Step 5: Implement `candidate_contract.py` exactly as specified**

Do not import `candidate_pool.py` or `checkpoint_pool.py`; this module must stay dependency-light and cycle-free。

- [ ] **Step 6: Run focused tests**

Expected: all candidate-contract tests PASS。

- [ ] **Step 7: Run schema and candidate-space regressions**

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest \
  mix_bit/tests/test_candidate_space.py \
  mix_bit/tests/test_candidate_contract.py -q
```

- [ ] **Step 8: Commit only Task 1 files**

```bash
git add mix_bit/candidate_contract.py mix_bit/tests/test_candidate_contract.py
git commit -m "fix: enforce mixed-bit candidate mode contract"
```

---

