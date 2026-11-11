"""Prepare the approved deterministic batch, retaining offline evidence and spent fees."""
import argparse
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from . import EXECUTION, io, search
from .identity import source_pins


def prepare(root):
    root = Path(root).resolve()
    require(not root.exists(), 'Never overwrite a prior batch.')
    require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'Prepare with the confirmed environment and hidden CUDA.')
    root.mkdir(parents=True)
    (root/'validation').mkdir()
    old = Path('artifacts/sc_cat/qwen--qwen3-8b/2026-11-08_172625_s31').resolve()
    source = io.read(old/'manifest_kappa_final.json')
    stopped = io.read(old/'user_stop_20261109.json')
    require(stopped['controller_exited'] and not stopped['remaining_owned_processes'], 'Prior batch is still active.')
    handoff = io.read(io.bound_path(source['selected_settings']))
    io.write(root/'selected_settings.json', handoff)
    # Only entry wiring and cost/reporting code changed. Record the exact unchanged
    # mechanisms for the prior native resume, joint QAT and export proof reuse.
    unchanged = []
    for name in ('builder.py', 'optimizer.py', 'boundary.py', 'settings.py', 'selection.py',
                 'raw.py', 'search.py', 'remaining.py', 'terminal.py'):
        before = (Path('experiments/sc_cat_recovery_cmcc_20261108')/name).read_text()
        after = (Path(__file__).parent/name).read_text()
        require(after == before.replace('sc_cat_recovery_cmcc_20261108', 'sc_cat_recovery_cmcc_20261110'),
                'Reused mechanism changed: '+name)
        unchanged.append(name)
    io.write(root/'validation/reuse.json', dict(unchanged_mechanisms=unchanged,
        previous_full_GPU=io.binding(old/'technical_import_fix/validation.json'),
        previous_fixed_kappa_GPU=io.binding(old/'technical_kappa_fix/validation.json'),
        changed_path='FA2 deterministic backward binding before Transformers import', execution=EXECUTION))
    for module, filename, extras in (
        ('audit_cpu', 'artifact_cpu.json', ['--handoff', str(root/'selected_settings.json')]),
        ('validate_cpu', 'implementation_cpu.json', ['--resume']),
        ('validate_remaining_cpu', 'remaining_cpu.json', [])):
        with (root/'validation'/f'{module}.log').open('w') as log:
            subprocess.run([sys.executable, '-B', '-u', '-m', __package__+'.'+module,
                            '--output', str(root/'validation'/filename), *extras],
                           stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=True)
        print(dict(event='CPU_check_passed', module=module), flush=True)
    first_old = io.read(old/'authorization.json')
    second_old = io.read(old/'authorization_second.json')
    history = dict(previous_total_GPUh=stopped['total_actual_GPUh'],
        previous_first_GPUh=stopped['first_batch_actual_GPUh'], previous_second_GPUh=stopped['second_batch_actual_GPUh'],
        first_approved_cap_GPUh=first_old['total_GPU_hours'], second_approved_cap_GPUh=second_old['new_budget_GPUh'],
        settlement=io.binding(old/'user_stop_20261109.json'), first_authorization=io.binding(old/'authorization.json'),
        second_authorization=io.binding(old/'authorization_second.json'), offline_cost_separate=True)
    keep=('schema','model_tag','search_seed','frozen_order','parent','model_identity','data_identity',
          'inventory','data_plan','baseline','protocol_identity','offline_source_manifest','input_contract')
    manifest={k:source[k] for k in keep}
    record=Path('docs/experience/records/sc_cat/qwen--qwen3-8b')/(root.name+'_results.md')
    manifest.update(run_id=root.name, created_Asia_Shanghai=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        status='approved_deterministic_recovery_ready', output_root=str(root), result_document=str(record.resolve()),
        selected_settings=io.binding(root/'selected_settings.json'), cost_history=history,
        artifact_cpu=io.binding(root/'validation/artifact_cpu.json'), implementation_cpu=io.binding(root/'validation/implementation_cpu.json'),
        remaining_cpu=io.binding(root/'validation/remaining_cpu.json'),
        reused_GPU_validation=io.binding(old/'technical_import_fix/validation.json'),
        reused_kappa_validation=io.binding(old/'technical_kappa_fix/validation.json'),
        prior_second_authorization=io.binding(old/'authorization_second.json'),
        numerical_policy=dict(FA2=EXECUTION, generic_Trainer_deterministic=False,
            search_and_fixed_replay_identical=True, old_recovery_nodes_reused=False),
        recovery_source_sha256=source_pins(), code_identity=digest(source_pins()),
        protocol_contracts={name:dict(source=io.binding(Path('docs')/name), frozen_utf8=(Path('docs')/name).read_text())
                            for name in ('sc_cat_protocol.md','sc_cat_execution.md')},
        resources=dict(gpus=[4,5,6,7], maximum_parallel_GPU_tasks=4,
            free_output_bytes=shutil.disk_usage(root).free), GPU_authorized=True)
    io.write(root/'manifest.json', manifest)
    auth=deepcopy(first_old)
    cap=history['first_approved_cap_GPUh']-history['previous_first_GPUh']
    auth.update(run_id=root.name, status='approved_deterministic_first_class', manifest=io.binding(root/'manifest.json'),
        output_root=str(root), total_GPU_hours=cap, budget_by_component={k:cap for k in first_old['budget_by_component']},
        component_limit_meaning='Individual ceilings; shared total is the binding aggregate cap, never their sum.',
        execution_policy=second_old['execution_policy'], historical_costs=history,
        authorization_reference=dict(current_user_instruction='开始逐类恢复的搜索吧',
            original_first=history['first_authorization'], original_second=history['second_authorization'],
            original_second_confirmation='确认689 GPUh上限并启动', resources='GPU4–7，最大并行度',
            no_budget_increase=True),
        confirmation='Current restart instruction; previously approved caps less settled actual fees.',
        second_batch='Already approved: remaining six categories, full nine, actual bitwidth and fixed BF16 replay.')
    io.write(root/'authorization_first.json',auth)
    state=search.new_state(handoff,root.name)
    state.update(cost_history=history,cumulative_GPUh=history['previous_total_GPUh'],new_run_GPUh=0)
    io.write(root/'search.json',state)
    lines=[f'# Qwen3 新版 SC-CAT 确定性逐类恢复：{root.name}', '',
        '状态：CPU 检查通过；按已确认预算启动。正式质量结果尚未产生，整链未验收。', '',
        f"离线 VAE 搜索与完整八项、数据身份、费用原账复用：[{source['run_id']}](2026-11-08_172625_s31_results.md#candidate-results)。本次不重做离线广搜。",
        '新正式链以原 BF16 为根；K 的两个离线 bank 仅在完整身份验证通过后安装成新 raw；后续每类在本链真实父模型上拟合。',
        'FA2 确定性反向在 Transformers 导入前启用，搜索与固定配方重放相同。旧恢复节点不作为新父。', '',
        '两份默认 DL 各 500 步、四节点完整八项，仅赢家进入规范组合／范围／LoRA／优化器／三轮学习率搜索；逐项校准短筛，条件 1000 步与全节点选优沿用正文。',
        '末类两份合法终点完成九项择优、实际位宽验收，再从 BF16 固定配方重放。A–E 数据搜索保留在二阶段。', '',
        f"费用：历史恢复 {history['previous_total_GPUh']:.12f} GPUh（首类 {history['previous_first_GPUh']:.12f}；第二批 {history['previous_second_GPUh']:.12f}）。",
        f"本次 K 可用上限 {cap:.12f} GPUh；后六类及验收／重放可用上限 {689-history['previous_second_GPUh']:.12f} GPUh。已用费用不清零，不增加已批准上限。",
        '费用包括验证、训练、保存导出和评测，按进程实际占用窗计入分项；运行中窗口尚未结算。GPU4–7 最大四任务并行，类别按父依赖串行。', '',
        '<a id="candidate-results"></a>', '## VAE 候选交接', '',
        '| 类别 | 合法完整不同配置 | Top1／S8 | Top2／S8 |', '| --- | ---: | --- | --- |']
    for category in handoff['frozen_order']:
        item=handoff['categories'][category]
        cells=[f"`{i}` / {handoff['candidates'][i]['full8_mean']:.11f}" for i in item['top2']]
        lines.append(f"| {category} | {item['distinct_legal_complete']} | {cells[0]} | {cells[1]} |")
    lines += ['',f'完整配置、八项原值、来源与身份：[selected_settings.json]({root}/selected_settings.json)。', '',
        '<a id="category-settings"></a>', '<a id="cat-recovery"></a>', '## 恢复节点与采用依据', '',
        '<!-- recovery-live-start -->','尚无正式节点。','<!-- recovery-live-end -->','',
        f'复现入口：`python -B -u -m {__package__}.run --root {root}`。',
        f'CPU 检查与复用依据：`{root}/validation/`；配置与硬预算：`manifest.json`、`authorization_first.json`；实际费用：`technical/budget.json`、`formal/budget.json`，后续继承到 `remaining/budget.json`。', '']
    require(not record.exists(),'Preserve existing result records.')
    record.write_text('\n'.join(lines),encoding='utf-8')
    alias=Path('artifacts/sc_cat/qwen--qwen3-8b')/root.name
    require(not alias.exists(),'Do not overwrite an artifact alias.')
    alias.symlink_to(root,target_is_directory=True)
    print(dict(event='preparation_complete',root=str(root),first_cap_GPUh=cap,history=history),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    prepare(parser.parse_args().root)
