> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.ai-bridge/CAT_DISTILL_IMPLEMENTATION_STOP_REASON.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../README.md) 查阅。

# Implementation Stop Reason

## Failed task
Task 0 baseline tests before Task 1: parser / mode / CLI contract.

## Exact command
```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate bitvae && cd /home/shaoyuantian/program/VAELLM && export PYTHONPATH=. && pwd && printf 'PYTHONPATH=%s\n' "$PYTHONPATH" && which python && python -V && pytest -q tests/test_cat_inline_remaining_lora.py tests/test_cat_inline_distributed.py tests/test_cat_eval_adapter_match.py tests/test_temporary_switch_residency.py tests/test_distill_losses.py tests/test_distill_dynamic_padding.py tests/test_lora_distill_token_stats_callback.py tests/test_e2e_checkpoint_io_legacy.py
```

## Environment
```text
which python: /home/shaoyuantian/anaconda3/envs/bitvae/bin/python
python -V: Python 3.11.13
CUDA_VISIBLE_DEVICES:
WORLD_SIZE:
RANK:
LOCAL_RANK:
PYTHONPATH: .
pwd: /home/shaoyuantian/program/VAELLM
```

## Full error
```text
/home/shaoyuantian/program/VAELLM
PYTHONPATH=.
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python
Python 3.11.13
....F................................................................... [ 40%]
........................................................................ [ 81%]
................................                                         [100%]
=================================== FAILURES ===================================
_______ CatInlineDistributedTests.test_gpu_launcher_validation_and_count _______

self = <test_cat_inline_distributed.CatInlineDistributedTests testMethod=test_gpu_launcher_validation_and_count>

    def test_gpu_launcher_validation_and_count(self):
        script = "scripts/catlora_simple copy.sh"
        prefix = "source <(sed -n '3,27p' \"$1\"); printf '%s:%s' \"$CUDA_VISIBLE_DEVICES\" \"$NPROC_PER_NODE\""
        for value, expected in (("5", "5:1"), ("5,6,7,8", "5,6,7,8:4"), ("0,2,4", "0,2,4:3")):
            result = subprocess.run(
                ["bash", "-c", prefix, "bash", script],
                env={**os.environ, "DISTILL_GPUS": value},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
>           self.assertEqual(result.stdout, expected)
E           AssertionError: ':' != '5:1'
E           - :
E           + 5:1

tests/test_cat_inline_distributed.py:171: AssertionError
=========================== short test summary info ============================
FAILED tests/test_cat_inline_distributed.py::CatInlineDistributedTests::test_gpu_launcher_validation_and_count
1 failed, 175 passed in 27.32s
```

## What had been changed before failure
No source files were modified before this baseline failure.

Only `.ai-bridge/CAT_DISTILL_IMPLEMENTATION_STOP_REASON.md` was overwritten after the failure, as required by the plan.

## Reproduction
From the repository root:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bitvae
cd /home/shaoyuantian/program/VAELLM
export PYTHONPATH=.
which python
python -V
pytest -q tests/test_cat_inline_remaining_lora.py tests/test_cat_inline_distributed.py tests/test_cat_eval_adapter_match.py tests/test_temporary_switch_residency.py tests/test_distill_losses.py tests/test_distill_dynamic_padding.py tests/test_lora_distill_token_stats_callback.py tests/test_e2e_checkpoint_io_legacy.py
```

## Current hypothesis
The baseline failure is in `tests/test_cat_inline_distributed.py::CatInlineDistributedTests::test_gpu_launcher_validation_and_count`. The test sources lines 3-27 from `scripts/catlora_simple copy.sh` and expects `CUDA_VISIBLE_DEVICES` and `NPROC_PER_NODE` to be set from `DISTILL_GPUS`; the actual captured stdout was `:`.
