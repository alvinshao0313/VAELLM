# LongBench 固定高敏任务 CPU 评分

本入口只提供长预算节点的 partial screen，最终验收仍需要官方全部 21 项、4750 题及完整九项。高敏任务由 BF16→固定 a0e5 的分项绝对掉点 `>=5pp` 预先确定，不根据新候选重新筛选。分类来源和所有 21 项分数绑定在 [固定 manifest](../../result/qwen3_8b_mmlu65_20261004/longbench_sensitive_screen_20261012.json)，SHA256 `be201d967e9aae091fd668c50bb3db54896bfbf60aacbb508ea5db9845d10837`。

高敏按官方顺序为 `narrativeqa qasper multifieldqa_en multifieldqa_zh hotpotqa 2wikimqa musique triviaqa passage_retrieval_en passage_retrieval_zh lcc repobench-p`，12 项、2950 题。低敏余 9 项、1800 题。本入口只读取高敏预测文件；低敏未评分，不记为零。

## CPU 调用

调用者先激活 `bitvae`，在 `/home/star/VAELLM` 运行，使用 fresh Python subprocess，避免 snapshot/local 模块遮蔽 PUBLIC 评分来源：

```bash
source /home/star/miniconda3/etc/profile.d/conda.sh
conda activate bitvae
CUDA_VISIBLE_DEVICES='' python -B -m experiments.longbench_sensitive_screen.run \
  --selection result/qwen3_8b_mmlu65_20261004/longbench_sensitive_screen_20261012.json \
  --archive result/qwen3_8b_longbench_20261004/data/manifest.json \
  --prediction-dir /absolute/path/to/candidate/run \
  --output result/qwen3_8b_mmlu65_20261004/candidate_sensitive_partial.json
```

`prediction-dir` 是原 runner 输出根目录，包含 `config.json` 和 `predictions/<task>.jsonl`。config 必须为原 full 模式、同一完整输入 archive/协议和模型系列；每项必须保留该任务全部官方条目。复用 PUBLIC `data.load_archive`、`scoring.check_prediction`、`scoring.score_example`：校验数量、canonical 顺序、row/_id、input hash、原答案/classes/length、config SHA 和 `diagnostic=False`，然后按原官方指标重算。无需加载完整模型。输出路径必须不存在，避免覆盖已有 metrics。

输出 schema 为 `longbench_v1_sensitive_screen_v1`、status 为 `partial_screen_complete`、`is_full_benchmark=false`、`kind=partial`，含 12 项分数、2950 样本、`high_macro_mean`（未舍入的分项等权均值）、配置/模型身份、输入/预测/实际评分源码 SHA。它不生成完整 scores schema、低敏分数、完整 LB 均值或虚构九项。

## 未来预测调用

初次2950题完整高敏入口未新增 GPU runner，也未启动预测；它复用已有 full runner 的任务分片能力。下文H50另有隔离样本runner。完整高敏入口的实际 CLI 是 `--datasets nargs='+'`，所以任务名以空格分隔，不能传逗号单 token：

```bash
python -B -m experiments.longbench_v1.predict \
  --config /absolute/path/to/candidate/full_config.json \
  --output /absolute/path/to/candidate/sensitive_run \
  --datasets narrativeqa qasper multifieldqa_en multifieldqa_zh hotpotqa 2wikimqa musique triviaqa passage_retrieval_en passage_retrieval_zh lcc repobench-p
```

预测仍用原 full config 和官方 v1/32K 总窗口/greedy/原 task cap，不限制各任务题数。之后显式调用本 CPU scorer。最终完整验收继续原 `experiments.longbench_v1.score`：其 `score_directory` 遍历全部 `DATASETS`，缺低敏文件会拒绝；该完整入口和 `scoring.py` 未修改。没有自动调度、等待空卡、重试或监控。

## 实际验证

2026-10-12，后台唯一 tmux `longbench_sensitive_screen_CPU_20261012` 完成，实际 CPU 10.744685522 秒、outer.exit0，CUDA 未初始化、0FW、未加载模型权重。两次真实 CLI 重评原 BF16 和 a0e5 完整预测，12 项的完整记录（分数/样本/metric/舍入值）及等权均值均精确复现完整 scores：

| 来源 | 高敏 12 项均值 |
| --- | --- |
| BF16 | 62.841532191570174 |
| a0e5 | 50.81178369159549 |

内存内的少行、错 config SHA、diagnostic=True、错 input identity 四种输入均被拒绝，没有复制大数据或留下临时拒绝文件。CLI help/源码 AST 正常；完整评分仍遍历全部 21 项仅按代码事实确认，未补 GPU 测试。原预测和 metrics 未改写。

最小证据保留在 [验证目录](../../result/qwen3_8b_mmlu65_20261004/longbench_sensitive_screen_validation_20261012/)：

- [report.json](../../result/qwen3_8b_mmlu65_20261004/longbench_sensitive_screen_validation_20261012/report.json)：实际调用 argv、源指标 SHA、精确复现和拒绝证据、源码 SHA、CPU 边界。
- [bf16_partial.json](../../result/qwen3_8b_mmlu65_20261004/longbench_sensitive_screen_validation_20261012/bf16_partial.json)、[source_partial.json](../../result/qwen3_8b_mmlu65_20261004/longbench_sensitive_screen_validation_20261012/source_partial.json)：真实独立 partial 输出。
- [core.log](../../result/qwen3_8b_mmlu65_20261004/longbench_sensitive_screen_validation_20261012/core.log)、`outer.exit`：原始核心 stdout/stderr 和终态。

初次2950题完整高敏入口仅新增 [core.py](core.py)、[run.py](run.py) 和当时的 README；没有修改现有活跃代码、依赖或分类 manifest。下文H50为随后新增的独立样本路径。partial/full 边界遵循 [实验设计经验](../../docs/experience/lessons/experiment_design.md)。

## H50：固定600题的独立粗筛

2026-10-12获准增加更便宜的阶段验证，最终完整21项/4750题与九项口径不变。新增[sample_core.py](sample_core.py)、[sample_predict.py](sample_predict.py)、[sample_score.py](sample_score.py)，原core/run、原predict main、冻结S及训练代码不改。

永久[H50 manifest](../../result/qwen3_8b_mmlu65_20261004/longbench_h50_screen_20261012.json) SHA `24c7df3c93de5bda9b3cc24c19ece1fcfa93c9fd45207b3f915059792eb35af7`，绑定旧高敏12/2950名单、原archive、抽样seed137、每项50及600个原row index、历史回放来源/hash。每任务按`sha256(UTF8(f'137:{dataset}:{row_index}')).digest()`排序取前50，只由任务与原row index决定，所有候选相同，不按答案或预测挑题。索引mapping保留hash排名顺序，以`json.dumps(indices,sort_keys=True,separators=(',',':')).encode('utf-8')`取hash，SHA `e3dec00d4f095ebd856554eb675e0d12ba53eb30277b6c78ab82c9f15231d9a4`。实际执行按原archive的canonical row顺序，评分按固定hash排名聚合，复现研究原数值。

H50全部记录标为`sample_partial`、`is_full_benchmark=false`，终态为`sample_partial_complete`，不会写`partition_complete`或完整LB/nine状态。配置仍使用原full config作为模型/推理协议载体；`sample_manifest.json`另行固定覆盖范围，不能因config.mode为full就把600题称为全量。新预测同时保留原dataset/index、输入hash、答案/classes/length、config SHA和`diagnostic=false`，另写sample manifest SHA/partial标记；评分拒绝未知/重复/漏row、错config或错partial标记。

候选H50只减同H50的a0e5基线 **53.692011958798304**，不减完整高敏2950题的50.81178369159549。输出使用`h50_macro_mean`、`paired_source_h50_delta_pp`，没有完整任务宏均值、低敏填零或虚构九项。样本manifest还绑定同一a0来源UUID及原预测/config/full scores hash。

预测只能由调用者激活bitvae，在冻结`S=/home/star/VAELLM/result/qwen3_8b_mmlu65_20261004/rotation_factor_probe/source_snapshot_20261005_01`下用绝对入口运行，强制依赖解析到S。它直接调用原`runner.generate_one/read_input`与`models.check_config/load_model`，保留原32K总窗口、官方task caps、non-thinking greedy、FA2/group8、BF16/FP32 RoPE和推理seed42。只减少rows，不截短input或generation。以下GPU0和候选路径是未来显式派卡后的调用示例，本轮未执行GPU：

```bash
source /home/star/miniconda3/etc/profile.d/conda.sh
conda activate bitvae
cd /home/star/VAELLM/result/qwen3_8b_mmlu65_20261004/rotation_factor_probe/source_snapshot_20261005_01
PYTHONPATH=/home/star/VAELLM/result/qwen3_8b_mmlu65_20261004/rotation_factor_probe/source_snapshot_20261005_01 \
  CUDA_VISIBLE_DEVICES=0 python -B -u \
  /home/star/VAELLM/experiments/longbench_sensitive_screen/sample_predict.py \
  --config /absolute/path/to/candidate/full_config.json \
  --manifest /home/star/VAELLM/result/qwen3_8b_mmlu65_20261004/longbench_h50_screen_20261012.json \
  --output /data/star/VAELLM/result/candidate/h50 --gpu 0
```

没有自动派卡、重试或等待。需续接时由root显式提供同config/manifest/output和`--resume`；可选`--max-runtime-seconds`仅在完整样本之间检查，不改变官方caps。600题完成也只表示H50完成。最终少量入选点及预定长预算末点直接原生重跑完整21；多600题约20分钟的历史生成开销，暂不为去重增加补齐/合并体系。

CPU评分须从PUBLIC fresh进程调用；可读取真实H50预测，也可从已有完整高敏任务预测选择相同50题，后者先核每任务完整官方行数/身份。下面是本轮实际执行过的a0来源CLI；重跑应换新的output，原指标不覆盖：

```bash
cd /home/star/VAELLM
PYTHONPATH=/home/star/VAELLM CUDA_VISIBLE_DEVICES='' python -B \
  /home/star/VAELLM/experiments/longbench_sensitive_screen/sample_score.py \
  --manifest /home/star/VAELLM/result/qwen3_8b_mmlu65_20261004/longbench_h50_screen_20261012.json \
  --prediction-dir /home/star/VAELLM/result/qwen3_8b_mmlu65_20261004/general_lm_qwen_k38_qkonly_s2000_longbench21/merged \
  --output /home/star/VAELLM/result/qwen3_8b_mmlu65_20261004/longbench_h50_screen_20261012/a0_partial.json
```

## H50实测证据与排序边界

复用原177.420292秒、11个真实完整端点/9种预算的[研究FINAL_REPORT](../../result/qwen3_8b_mmlu65_20261004/longbench_h50_screen_20261012/analysis.json)和[原核心log副本](../../result/qwen3_8b_mmlu65_20261004/longbench_h50_screen_20261012/calibration.log)，不重复全11回放。原/tmp文件未发现活动进程使用，本轮只复制，未删除；后续manifest加载只依赖永久副本。validate.py的原准备命令和CPU证据仍引用原/tmp来源路径，因此本轮保留该源log。

a0历史H50累计生成1122.098677秒，完整H2950为6342.948688秒，full21为19755.716407秒；H50分别为17.69%/5.68%，不含新模型加载成本。H50相对完整H的配对收益误差MAE .221100776pp、最大 .408267136pp；55对中1对反转，完整H差距仅 .058620043pp。相对full21有3对反转，其中2对原H本身已与full21反向；小样本误差与未测低敏变化均可能改变排序，以上只是旧端点观察，不能当置信保证或固定晋级门槛。H25出现较大反转，未采用。即使H50排序正常，八项变化仍可改变九项排序。

本轮新增CPU验证由唯一tmux`longbench_h50_cpu_20261012`后台完成，outer **4067911/start111403385**、worker **4067917/start111403417**，outer.exit0，实际27.206751秒，0FW/CUDA未初始化/未加载模型。新CLI分别重评a0/F_A/F_A_HOT为 **53.692011958798304 / 55.99814570806964 / 55.906648954019566**，与旧研究逐数完全一致。9类错误输入均拒绝；冻结S实际核全部600原token切片，共6,042,700 tokens，输入hash/形状/32K限制与四个依赖模块路径通过。CLI help/源码AST通过，原core/run及S实际依赖SHA保持。

证据集中在[同一目录](../../result/qwen3_8b_mmlu65_20261004/longbench_h50_screen_20261012/)的cpu_report.json、三份partial原指标、core.log/outer.exit、analysis.json/calibration.log和复现validate.py；真实命令、源hash、拒绝结果及S输入报告已归并CPU报告。尚未运行GPU，CPU验证不证明新样本runner已实际完成生成或最终质量达标。
