> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-3-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 3: Pin Candidate Training to the Parent Python Interpreter

**Files:**
- Modify: `mix_bit/candidate_pool.py`
- Modify: `mix_bit/scripts/train_candidate_single.sh`
- Modify: `mix_bit/tests/test_candidate_pool.py`
- Modify: `mix_bit/README.md`

**Command contract:**

```text
argv[0] = absolute path to train_candidate_single.sh
argv[1] = gpu id
argv[2] = absolute resolved sys.executable
argv[3:] = cat_train arguments
```

### Shell behavior

`train_candidate_single.sh` must be exactly equivalent to：

```bash
if [ "$#" -lt 2 ]; then
  echo "Usage: bash $0 <CUDA_VISIBLE_DEVICES> <PYTHON_EXECUTABLE> [cat_train_arguments]" >&2
  exit 2
fi

GPU_ID="$1"
PYTHON_EXECUTABLE="$2"
shift 2

if [ ! -x "${PYTHON_EXECUTABLE}" ]; then
  echo "Python executable is not executable: ${PYTHON_EXECUTABLE}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
exec "${PYTHON_EXECUTABLE}" tools/cat_train.py "$@"
```

保留现有环境变量：

```text
PYTHONPATH=.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHONHASHSEED=31
CUBLAS_WORKSPACE_CONFIG=:4096:8
TOKENIZERS_PARALLELISM=false
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

### Python behavior

在 `build_trial_command`：

```python
python_executable = str(Path(sys.executable).resolve())
if not Path(python_executable).is_file():
    raise FileNotFoundError(
        f"Current Python executable does not exist: {python_executable}"
    )
command = [script_path, gpu_id, python_executable, *cat_train_args]
```

`scheduler_meta.json` 和 `trial_spec.json` 新增：

```json
"python_executable": "/absolute/path/to/bitvae/bin/python"
```

该字段只记录 provenance，不参与旧 artifact resume 判定；artifact 的模型/recipe/hash 契约已经覆盖结果有效性。

- [ ] **Step 1: Write command-construction tests**

完整实现以下测试，并在第一个测试中 monkeypatch `candidate_pool.sys.executable`：

- `test_trial_command_passes_resolved_sys_executable_as_second_argument`
- `test_scheduler_meta_records_python_executable`
- `test_trial_spec_records_python_executable`

测试中 monkeypatch `candidate_pool.sys.executable` 到一个临时可执行文件，并断言命令位置，不得只搜索字符串。

- [ ] **Step 2: Write shell contract tests**

使用 `subprocess.run` 完整实现：

- `test_candidate_shell_requires_python_argument`
- `test_candidate_shell_rejects_non_executable_python`
- `test_candidate_shell_executes_explicit_interpreter`

第三个测试创建一个临时 executable shell stub，它把收到的 argv 写到文件；调用正式 `train_candidate_single.sh` 后必须看到第一个参数是 `tools/cat_train.py`，后面参数原样保留。

- [ ] **Step 3: Run tests and confirm old code fails**

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest mix_bit/tests/test_candidate_pool.py -q
```

- [ ] **Step 4: Modify `build_trial_command` and metadata**

不得通过修改父进程 PATH 解决；必须显式传绝对 interpreter。

- [ ] **Step 5: Modify the shell script**

不得保留任何 `exec python` 或 `python tools/cat_train.py`。

- [ ] **Step 6: Run focused tests**

- [ ] **Step 7: Run a no-model shell smoke test**

```bash
mix_bit/scripts/train_candidate_single.sh 4 /bin/echo --sentinel
```

Expected stdout contains exactly：

```text
tools/cat_train.py --sentinel
```

- [ ] **Step 8: Update README execution requirement**

明确说明父 CLI 必须由 `bitvae` Python 启动，而子任务会自动固定到同一解释器。

- [ ] **Step 9: Commit Task 3 files**

```bash
git add mix_bit/candidate_pool.py mix_bit/scripts/train_candidate_single.sh mix_bit/tests/test_candidate_pool.py mix_bit/README.md
git commit -m "fix: pin candidate workers to parent python"
```

---

