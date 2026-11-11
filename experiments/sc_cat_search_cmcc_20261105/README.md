# SC-CAT新标准离线执行器

本包只实现用户`call_81e0e6704b6f430680438a62984cc201`批准的Qwen3-8B七类离线搜索：保护→转置（A/B及至多一个逐层C）→旋转→D→encoder→decoder→optimizer→倍率。每类最多20个实际不同配置含anchor，S2各10K、全M36及完整八项；只并行各类当前轮，精确同分先actual logical/packed费用再保留进入本轮的current。新合法C完整fresh重拟合，不拼接bank。

原source9拟合与PUBLIC消费者保持不变；旧19点、21 profiles、8 joint及其raw不改不删。新运行以完整resolved配方查询旧批自然终态库存，保留原ID／来源，继承新轮目标保护比例。正式CAT/live/KD、LongBench、最终九项和BF16重放不在本包范围。

实际授权、唯一结果记录和产物根：

- `artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/authorization.json`
- `docs/experience/records/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31_results.md`
- `artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/`

调用者从仓库根激活`bitvae`并核Python；脚本不自行激活环境。CPU验证用tmux/nohup并隐藏CUDA。所有新源码与授权核证完成后，实际prepare命令为：

```bash
CUDA_VISIBLE_DEVICES='' python -B -m experiments.sc_cat_search_cmcc_20261105.run \
  --component prepare \
  --output artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31 \
  --authorization artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31/authorization.json \
  --old-manifest artifacts/cat_sequential_search_cmcc_20261105/confirmed_order_20261105_1049/manifest.json \
  --result-document docs/experience/records/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31_results.md
```

后台运行入口：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.sc_cat_search_cmcc_20261105.run \
  --component run --wait-for-prior \
  --output artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31
```

`--wait-for-prior`只对授权中的一个旧controller核cmdline/output/start_ticks，以既有pidfd阻塞等待退出，不轮询日志、不重试；`prior_wait.json`明确记录等待且GPU消费0。退出后仍核旧search、实际worker均终态及GPU0–5资源，再封存最终`source_cache.json`和导入库存。等待不是新GPU已启动；无此参数时旧批仍活跃会明确拒绝。GPU6/7始终排除。

可选`--drain-file <path>`只影响本新批：文件出现后停止新派发，已启动candidate完整fit/full8或readout收尾，再以`drained`退出。移除文件后用同manifest的`--resume`接已结束任务边界；不接管正在运行／半途失败worker、不自动重试。`--resource-receipt`可记录主线程现有实际资源收据，不新增用户批准步骤。`report`只从已有search生成三表。

`search.json`记录冻结当前轮、每份实际请求／进程／退出引用、实际拥有窗与下一步；candidate350 GPUh、readout3.5 GPUh分别预留完整依赖，失败费用照计、不互借额度。D8／16／其它formal窗7200／5400／3600秒再乘两侧宽度和/256；full8为900秒，新technical为900或1800秒；每类A/B共同读出≤1800秒。新完整模型不是全部永久保留：仅在完整证据导入后，按已持久依赖图自动释放本新批无后继raw state/runtime，保留所有bank、metadata和原始指标；anchor/current/最佳不同于anchor、待A/B、active/验证引用始终保护。先记精确路径／大小／原因，严格检查只在本批candidates内，再删再核；不清旧/profile/joint。

`evidence.py`提供显式历史证据导入与loadability分离；`layerwise.py`用原BF16同层输入、固定512 train及有效mask做双方向FP32正式输出误差和原坐标W-MSE，FP64仅作诊断。旧仅FP64累加统计不冒充新正式归约；适用fit/full8仍复用。`selected_settings.json`及唯一Markdown包含候选明细、类别配置和未启动的正式恢复边界；完成时交付anchor／最后current冠军／最佳不同于anchor及共同BF16下降排序。

尚未完成模型GPU混合路径之前不宣称技术通过；离线搜索完成也不代表正式CAT、九项达标或跨模型通过。CPU验证报告和实际启动记录由本批结果文档关联。
