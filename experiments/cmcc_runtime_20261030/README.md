# cmcc 固定接续入口

`adaptive_pipeline.py` 和 `adaptive_evaluation.py` 只重绑迁入材料的路径及身份，保留原压缩／恢复和完整九项协议。原材料与历史报告不回写。当前决策以[唯一执行计划](../../docs/experience/records/w25_transfer/current_plan.md)为准。

`sequence.py` 是本轮已固定实验的一次性编排入口。它不搜索新参数、不自动重试、不占卡、不创建监控服务。调用者先激活 `bitvae`；driver 在 CPU 上运行，子进程按固定资源执行。

1. 绑定 `short_trl01_launch.json` 中 dynamic／Rice 原 worker 与 shell 的 PID／start_ticks，以 Linux pidfd 阻塞等待；进程已结束时只接受真实 exit0 和完整终态，不重启。
2. 两臂经 `verify_adaptive_short.py` 核实三类别、216 native／36 compact／0 dense、BF16 strict 保存重载、实际预算以及 up／down 两阶段初始化精确配对后，GPU0／1 分别执行自己的原 500 恢复、live up 两阶 10K＋500、live down 两阶 10K＋500。第三阶36层码流复用 `formal_gpu1/2/3`，不重训。
3. 各臂 formal 成品经真实消费端 SHA／UUID／strict 重载／账本核验后，各自立即开始 full8→LongBench21／4750。dynamic 使用 GPU0/2/3，Rice 使用 GPU1/4，互不冲突。
4. H192 在独立分支用已验证的新接续入口复用原成功 full8，仅运行 GPU5/6/7 的完整 LongBench 和 strict 九项汇总；原失败输出保留。它不阻断 adaptive 分支。
5. 完成后机械汇总九项、均分、BF16 差值、位宽和耗时；dynamic／Rice 互相比较，H192 与绑定真实旧 Muon 报告比较。不据分数自动调整 K、评分或恢复方法。质量 `not_met` 是有效结果；技术失败或无效／未完成证据停止受影响依赖。

准备时写出完整、可审查的 `plan.json`，固定实际 argv、环境、所有输入及源码 SHA、GPU 和独立输出路径。执行时持有一个全局 OS flock；已执行的 plan 不可再次运行。每次启动前只检查一次指定 GPU，有其它 compute 占用就停止该链，不等待或重试。已有 short 则等待其真实 shell 终止，保证退出文件已写入。

```bash
# 已在调用者 shell 激活 bitvae；handoff 由 H192 接续验证生成。
CUDA_VISIBLE_DEVICES="" python -B -m experiments.cmcc_runtime_20261030.sequence \
  --prepare result/qwen3_8b_mmlu65_20261004/cmcc_sequence_20261030 \
  --h192-handoff /data1/cmcc/VAELLM/artifacts/g09_hotpot_coverage_cmcc_20261030/continuation_handoff.json

# 检查 plan 后以 tmux/nohup 后台执行，无需人工逐阶段接续。
CUDA_VISIBLE_DEVICES="" nohup python -B -u -m experiments.cmcc_runtime_20261030.sequence \
  --plan result/qwen3_8b_mmlu65_20261004/cmcc_sequence_20261030/plan.json \
  > result/qwen3_8b_mmlu65_20261004/cmcc_sequence_20261030/sequence.log 2>&1 < /dev/null &
```

主动查询该目录的 `status.json` 即可查看当前实际阶段；`metadata.json` 保存 driver 身份、源码、计划 SHA、环境及依赖版本，各 stage 保存 stdout/stderr 日志和真实退出码；最终 `result.json`／`sequence.exit` 区分程序闭合与质量通过。停止人工监控不会停止它。若 driver 被中断、源码变化或任一链失败，不复用此入口自动恢复，应保留实际状态再判断具体接续。

CPU 回归位于 `test_sequence.py`，使用真实完整 strict 当前口径报告验证有效 `not_met` 和错误 UUID，用真实未完成 H192 报告验证拒绝，用真实短生命周期子进程验证 pidfd、退出码／失败传播和第二次启动拒绝，并核全局 flock 互斥。它不生成 GPU 通过报告，不重复 GPU 数值验证。

2026-10-30 实测五项 CPU 回归全部通过，原始证据在[单一验证目录](../../result/qwen3_8b_mmlu65_20261004/cmcc_sequence_validation_20261030/launch.json)和[核心日志](../../result/qwen3_8b_mmlu65_20261004/cmcc_sequence_validation_20261030/tests.log)。首次环境检查发现本机 `bitvae` Python 未导出 `os.pidfd_open`，已按本机 x86_64 Linux 头文件的 `__NR_pidfd_open=434` 直接调用同一内核接口；后续真实子进程回归通过，仍以 pidfd 阻塞等待。此入口限定当前 Linux x86_64 服务器，不安装依赖；没有新增质量或压缩算法经验。
