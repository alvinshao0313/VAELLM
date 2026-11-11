"""Update the one result document from actual new-batch node/decision records."""
from pathlib import Path
import os

from .io import read, bound_path


def update(manifest,state,budget):
    path=Path(manifest['result_document'])
    text=path.read_text(encoding='utf-8')
    start,end='<!-- recovery-live-start -->','<!-- recovery-live-end -->'
    assert text.count(start)==text.count(end)==1
    lines=[f"状态：`{state['status']}`；已结束 GPU 拥有窗累计 **{budget['total_actual_GPU_hours']:.9f} GPUh**（含新技术验证，不含历史离线）。",
        '', '| 候选／轨迹 | 参数／日程 | 节点／来源 | 完整八项均分 | 八项原分数／checkpoint | 采用依据 | 已结束相关 GPUh |',
        '| --- | --- | --- | ---: | --- | --- | ---: |']
    selected={(block.get('certificate') or {}).get('selected_node_id') for block in state.get('categories',{}).values()}
    def link(label,target):
        return f'[{label}]({os.path.relpath(target,path.parent)})'
    for source,item in sorted(state.get('nodes',{}).items(),key=lambda pair:(pair[1]['candidate_id'],pair[1]['trajectory_id'],pair[1]['step'])):
        score=read(bound_path(item['score']))
        settings=item['settings']
        parameters=f"{settings['groups']}; D/P={settings['decoder_scope']}/{settings['protection_scope']}; r/a/drop={settings['lora_rank']}/{settings['lora_alpha']}/{settings['lora_dropout']}; {settings['bound_optimizer']}/{settings['lora_optimizer']}; LR={settings['decoder_lr']}/{settings['protection_multiplier']}/{settings['lora_lr']}; T={settings['schedule_max']}"
        cost=sum(t.get('actual_seconds',0) for task,t in budget['tasks'].items() if task.startswith(item['task_prefix']+'/'))/3600
        mean=100*sum(score['task_scores'].values())/8
        lines.append(f"| {item['candidate_id'][:12]} / {item['trajectory_id'][:12]} | {parameters} | {item['step']} / `{source}` | {mean:.11f} | {link('八项原值',item['score']['path'])}；{link('导出与fresh',item['report']['path'])} | {'最终接续' if source in selected else '已测合法节点；待或参与全节点比较'} | {cost:.8f} |")
    if not state.get('nodes'):
        lines.append('| 尚未运行 | 默认 DL，r12/a24/drop0.1，AdamW | 原 BF16 → 两份独立 raw | — | — | 待首类预算确认 | 0 |')
    lines += ['', '| 搜索项 | 校准状态 |', '| --- | --- |']
    lines += [f"| {name} | {value['status']} |" for name,value in state['short_screen'].items()]
    lines += ['', f"全模型已用 1000 步名额：{state['long_categories']}；后续预算批准类别：{state['approved_categories']}。"]
    body='\n'.join(lines)+'\n'
    path.write_text(text.split(start)[0]+start+'\n'+body+end+text.split(end)[1],encoding='utf-8')
