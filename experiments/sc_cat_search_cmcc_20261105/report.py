"""Generate the three standard result views from the actual persistent search."""
from datetime import datetime
from pathlib import Path

from experiments.cat_sequential_search_cmcc_20261105.report import stage_amplitudes
from . import bindings as b


def result_row(manifest,reference):
    result = b.checked_result(reference['path'],manifest)
    recipe = result['recipe'];inventory = b.read(manifest['inventory']['path'])['inventory']
    projections = {row['actual_name']:row for row in inventory['projections']}
    return dict(candidate_id=result['candidate_id'],canonical=reference,recipe=recipe,
        full8_mean=result['full8_mean'],task_metrics=result['full8_evidence']['task_metrics'],
        full8_evidence=result['full8_evidence'],joint_KL=result['joint_KL'],selection_data_identity=result['selection_data_identity'],
        logical_bits=result['logical_bits'],packed_bits=result['packed_bits'],bank=result['bank'],endpoint=result['endpoint'],
        actual_stage_amplitudes=stage_amplitudes(result),
        actual_protection={name:dict(count=count,D=recipe['D'],n=count//recipe['D'],actual_fraction=count/projections[name]['shape'][1])
            for name,count in recipe['protection']['counts'].items()})


def _section(text,start,end,body):
    first = text.index(start)+len(start);last = text.index(end,first)
    return text[:first]+'\n\n'+body.rstrip()+'\n\n'+text[last:]


def summarize(manifest,state,output,*,update_markdown=False):
    results = {};categories = [];details = [];layerwise = []
    baseline = b.load_baseline(manifest['baseline']['path'])
    def evidence(reference):
        if reference['path'] not in results:results[reference['path']] = result_row(manifest,reference)
        return results[reference['path']]
    for name,category in state['categories'].items():
        current = evidence(category['current']['result']);anchor = evidence(category['anchor']['result'])
        different = evidence(category['best_different']['result']) if category.get('best_different') else None
        for index,stage in enumerate(category['rounds']):
            base = evidence(stage['input']['result'])
            for row in stage['candidates']:
                value = evidence(row['result']) if row.get('result') else None
                details.append(dict(category_id=name,round_index=index,parameter=stage['parameter'],round_status=stage['status'],
                    parameter_base_id=stage['input']['candidate_id'],actual_parent=manifest['parent'],
                    candidate={key:row[key] for key in ('candidate_id','planned_candidate_id','requested_value','settings','recipe','status','reason','error',
                        'predicted_logical_bits','predicted_packed_bits','result','execution','reuse_kind','mixed_evidence','dedicated_mixed_slot') if key in row},
                    full8_mean=value['full8_mean'] if value else None,delta_vs_round_current=value['full8_mean']-base['full8_mean'] if value else None,
                    delta_vs_BF16=value['full8_mean']-baseline['mean_percent'] if value else None,
                    selected=stage.get('selected',{}).get('candidate_id') == row.get('candidate_id'),
                    current_configuration_at_round_entry=stage['input']['candidate_id']))
            readout = stage.get('readout',{})
            if readout.get('status') == 'completed':
                ref = readout['evidence'];b.require(b.sha(ref['path']) == ref['sha256'],'Original layerwise readout changed.')
                layerwise.append(dict(category_id=name,evidence=ref,readout=b.read(ref['path'])))
        n = manifest['caps'][name]['N'];recipe = current['recipe']
        categories.append(dict(category_id=name,status=category['status'],completed_parameters=[s['parameter'] for s in category['rounds'] if s['status'] == 'completed'],
            actual_settings=category['current']['settings'],resolved_recipe=recipe,anchor_id=anchor['candidate_id'],
            champion_id=current['candidate_id'] if category['status'] == 'completed' else None,current_best_id=current['candidate_id'],
            best_different_from_anchor_id=different['candidate_id'] if different else None,
            anchor=category['anchor']['result'],champion=category['current']['result'] if category['status'] == 'completed' else None,
            current_result=category['current']['result'],best_different=category.get('best_different'),
            full8_mean=current['full8_mean'],loss_vs_BF16=baseline['mean_percent']-current['full8_mean'],
            logical_bits=current['logical_bits'],packed_bits=current['packed_bits'],class_N=n,
            class_logical_bpw=current['logical_bits']/n,class_packed_bpw=current['packed_bits']/n,cap_bits=manifest['caps'][name]['cap_bits'],
            actual_protection=current['actual_protection'],distinct_completed_configurations=len(category['known']),
            actual_stage_amplitudes=current['actual_stage_amplitudes']))
    value = dict(schema='sc_cat_standard_selected_settings_v1',status=state['status'],manifest_identity=manifest['manifest_identity'],
        manifest=b.bound(Path(manifest['output_root'])/'manifest.json'),authorization=manifest['authorization'],
        baseline=manifest['baseline'],protocol_identity=manifest['protocol_identity'],run_identity=manifest['run_identity'],
        category_GPU_hours=state['candidate_GPU_hours'],readout_GPU_hours=state['readout_GPU_hours'],
        total_owned_GPU_hours=state['candidate_GPU_hours']+state['readout_GPU_hours'],rules=manifest['rules'],
        candidate_details=details,categories=categories,canonical_results=list(results.values()),layerwise=layerwise,
        frozen_order=state.get('frozen_order'),payload_retention=state.get('payload_retention'),raw_releases=state.get('raw_releases',[]),
        scope='Measured best in the declared offline search. Live CAT, KD, LongBench, final acceptance and BF16 replay have not started.')
    b.write(output,value)
    if not update_markdown:return value
    target = Path(manifest['result_document']);original = target.read_text(encoding='utf-8')
    stamp = datetime.now().astimezone().isoformat(timespec='seconds')
    common = '机器更新：'+stamp+'；状态`'+state['status']+'`。权威状态[search.json]('+str(Path(manifest['output_root'])/'search.json')+')，完整原精度分项与resolved配方见[selected_settings.json]('+str(Path(output).absolute())+')。'
    candidate_text = [common,'','每轮固定进入时的current；只继承目标比例与完整设置，拟合父始终是原BF16。复用保留原candidate ID／来源，未完成不填分数。',
        '', '| 类别 | 参数／轮基底 | 候选ID／取值 | 状态 | S8 | 对轮初current差pp | 对BF16差pp | 本轮选择 |',
        '| --- | --- | --- | --- | ---: | ---: | ---: | --- |']
    for row in details:
        candidate = row['candidate'];score = row['full8_mean']
        candidate_text.append('| '+' | '.join((row['category_id'],row['parameter']+'／'+row['parameter_base_id'][:12],
            candidate.get('candidate_id','未构造')[:12]+'／'+str(candidate['requested_value']),candidate['status'],
            repr(score) if score is not None else '未完成',repr(row['delta_vs_round_current']) if score is not None else '—',
            repr(row['delta_vs_BF16']) if score is not None else '—','选中' if row['selected'] else ''))+' |')
    candidate_text.extend(['','### 逐层转置依据','',
        '| 类别 | 层ID／投影 | A／B来源 | A输出MSE | B输出MSE | A权重MSE | B权重MSE | C方向／同分保留 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | --- |'])
    if not layerwise:candidate_text.append('| 未完成 | 尚无已完成的本轮A/B共同FP32读出 | — | — | — | — | — | 未生成 |')
    for block in layerwise:
        readout = block['readout']
        for row in readout['rows']:
            candidate_text.append('| '+' | '.join((block['category_id'],str(row['layer_id'])+'／'+row['projection'],
                row['A_candidate_id'][:12]+'／'+row['B_candidate_id'][:12],repr(row['A']['MSE']),repr(row['B']['MSE']),
                repr(row['A_weight']['MSE']),repr(row['B_weight']['MSE']),str(row['transpose'])+('／精确同分保留' if row['exact_tie'] else '')))+' |')
        candidate_text.extend(['','`'+block['category_id']+'`逐层完整SSE／计数与输入来源：[layerwise.json]('+block['evidence']['path']+')。',''])
    candidate_text.extend(['','原件包含每层输出SSE／元素数／MSE、原坐标W-MSE、原BF16输入／样本token身份、FP32正式归约及FP64诊断。方向按全512输出MSE，权重误差仅记录；唯一C完整全M重拟合。',
        '', '本批已结算候选拥有窗 '+repr(state['candidate_GPU_hours'])+' GPUh；读出 '+repr(state['readout_GPU_hours'])+' GPUh。运行中的未结算窗另见原任务记录，未与子phase重复相加；历史费用单列。'])
    original = _section(original,'## 候选明细','## 代码缺口、补齐与必要验证','\n'.join(candidate_text))
    category_text = [common,'','未完成类别仅报告当前最好已测；champion与顺序在规定依赖闭合后冻结。最佳不同的参照是anchor，冠军非anchor时两者可为同一配置。','',
        '| 类别 | 完成状态／覆盖 | 主码率／cap bits | 目标保护／实际count | T／旋转 | D/K/S/B/M | encoder／decoder | optimizer／倍率 | S8／BF16下降pp | 实际logical／packed bits | anchor／冠军／不同入围 | 顺序 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    order = value.get('frozen_order',{});order = order['ordered_categories'] if order else []
    for row in categories:
        recipe = row['resolved_recipe'];counts = sorted({p['count'] for p in row['actual_protection'].values()})
        category_text.append('| '+' | '.join((row['category_id'],row['status']+'／'+str(len(row['completed_parameters']))+'/8',
            recipe['nominal_main_rate']+'／'+str(row['cap_bits']),row['actual_settings']['protection']+'／'+str(counts),
            str(recipe['transpose'])+'／'+recipe['rotation']['level'],'/'.join(str(recipe[k]) for k in ('D','K','S','B','linear_group_size')),
            str(recipe['encoder']['hidden_dim'])+'／'+str(recipe['decoder']['hidden_dim']),recipe['optimizer']['kind']+'／'+str(recipe['optimizer']['multiplier']),
            repr(row['full8_mean'])+'／'+repr(row['loss_vs_BF16']),str(row['logical_bits'])+'／'+str(row['packed_bits']),
            row['anchor_id'][:12]+'／'+(row['champion_id'] or '未冻结')[:12]+'／'+(row['best_different_from_anchor_id'] or '无合法不同配置')[:12],
            str(order.index(row['category_id'])+1) if order else '未冻结'))+' |')
    category_text.extend(['','实际各层count／输入轴比例／n=count/D及Muonκ／真实stage幅度见机器总表。类别bpw按本类N计算，全模型分母不改变；含其余dense的partial费用不冒充最终成品。',
        '', '冻结顺序：'+(' → '.join(order) if order else '尚未冻结；仍有未闭合类别。'),
        '', '**正式CAT、恢复、LongBench和BF16重放未开始，均不在本批离线授权内。**'])
    original = _section(original,'## 类别配置汇总','<a id="cat-recovery"></a>','\n'.join(category_text))
    lines = original.splitlines()
    for index,line in enumerate(lines):
        if line.startswith('状态：'):
            lines[index] = '状态：**已批准离线批；机器状态`'+state['status']+'`。** 更新时间'+stamp+'。运行与等待分别以真实search／prior_wait记录为准；正式CAT及最终质量目标未完成。';break
    target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return value
