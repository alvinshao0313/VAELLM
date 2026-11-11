"""One JSON settings table and one generated section in the existing result record."""
from datetime import datetime
from pathlib import Path
from . import bindings as b

MD = b.REPO/'docs/experience/records/w25_transfer/2026-11-05_cat_selected_settings.md'
MARKER = '<!-- sequential-search-selected-settings -->'


def stage_amplitudes(result):
    reference = result['artifact_evidence']['completed']
    b.require(b.sha(reference['path']) == reference['sha256'], 'Actual fit evidence changed.')
    completed = b.read(reference['path']); row = next(row for row in completed['candidates'] if row['candidate_id'] == result['candidate_id'])
    output = []
    for stage in row['fit_report']['stages']:
        first = stage['updates'][0]; calibration = stage.get('calibration')
        learning_rates = first['actual_learning_rates']
        output.append(dict(stage_index=stage['stage_index'], actual_steps=stage['actual_steps'],
            first_actual_optimizer_group_LRs=learning_rates, kappa_calibration=calibration, owner_layout=stage.get('owner_layout'),
            initial_matrix_effective_LR=learning_rates[0]*calibration['kappa'] if calibration else learning_rates[0],
            schedule='linear10000; actual recorded first/second/final updates in completed fit_report'))
    return output


def summarize(manifest, state, output, *, update_markdown=False):
    rows = []
    for category, data in state['categories'].items():
        reference=data['current']['result']; b.require(b.sha(reference['path'])==reference['sha256'],'Selected canonical result changed.')
        result = b.checked_result(data['current']['result']['path'], manifest); recipe = result['recipe']
        coverage = [{key:row[key] for key in ('parameter','requested_value','settings','candidate_id','status','reason','error','result','execution',
            'predicted_logical_bits','predicted_packed_bits','protection_fallback_reason') if key in row}
            for stage in data['rounds'] for row in stage['candidates']]
        attempted={row['parameter'] for row in data['rounds']}
        failed={row['parameter'] for row in data['rounds'] if row['status']=='failed'}
        if data.get('failure_parameter'): failed.add(data['failure_parameter']); attempted.add(data['failure_parameter'])
        n = manifest['caps'][category]['N']
        rows.append(dict(category_id=category, status=data['status'], selected_candidate_id=result['candidate_id'],
            selected_result=data['current']['result'], selected_recipe=recipe, requested_settings=data['current']['settings'],
            actual_protection_counts=recipe['protection']['counts'], D=recipe['D'], K=recipe['K'], S=recipe['S'], B=recipe['B'],
            M=recipe['linear_group_size'], nominal_main_rate=recipe['nominal_main_rate'], full8_mean=result['full8_mean'],
            full8_evidence=result['full8_evidence'], class_N=n, logical_bits=result['logical_bits'], packed_bits=result['packed_bits'],
            class_logical_bpw=result['logical_bits']/n, class_packed_bpw=result['packed_bits']/n,
            cap_bits=manifest['caps'][category]['cap_bits'], cap_evidence=manifest['cap_evidence'],
            optimizer=recipe['optimizer'], actual_stage_amplitudes=stage_amplitudes(result),
            completed_parameters=[row['parameter'] for row in data['rounds'] if row['status']=='completed'],
            failed_parameters=[name for name in manifest['rules']['order'] if name in failed],
            unstarted_parameters=[name for name in manifest['rules']['order'] if name not in attempted],
            distinct_completed_full_fits=len(data['known']), coverage=coverage,
            evidence_scope='Best completed recipe within declared same-BF16 sequential comparisons; uncompleted stages are not selected'))
    report = dict(schema='cat_sequential_selected_settings_v1', status=state['status'], manifest_identity=manifest['manifest_identity'],
        rules=manifest['rules'], source9_fit_identity=manifest['run_identity']['code_identity'], run_identity=manifest['run_identity'],
        protocol_identity=manifest['protocol_identity'], baseline=manifest['baseline'], cap_evidence=manifest['cap_evidence'],
        owned_GPU_hours=state['owned_GPU_hours'], categories=rows,
        scope='Observed best in this finite single-BF16/category/seed search only; not global optimum, live CAT result or final nine-task acceptance')
    b.write(output,report)
    if update_markdown:
        original = MD.read_text(encoding='utf-8').split(MARKER)[0].rstrip()
        # Keep the original21-profile evidence, replacing its obsolete status and placeholder table.
        original=original.split('## 类设置交付表：')[0].rstrip()
        body=original[original.index('## 数据来源与比较条件'):]
        body=body.replace('## 七类基准与已比较K三档','## 原21份profile：历史固定控制与K三档')
        body=body.replace('几何、优化器等尚不能称为该类最优。','')
        body=body.replace('这里不生成联合提名、winner或cap。各类当前最高配置也不代表完成了该类转置／旋转／D／优化器／宽度／保护等顺序对照。',
            '该历史表只覆盖三档K的固定控制；本轮实际联合cap及逐参数取舍见下表。')
        original='# 各类别设置结果表：已评范围内入选配方\n\n本记录保留原21份profile事实来源，并原位汇总本轮实际联合cap及逐参数结果；运行状态为`'+state['status']+'`。不同类别未完成时分别列明。\n\n'+body
        text = [MARKER, '', '## 逐参数搜索实际入选设置', '',
            '更新时间：'+datetime.now().astimezone().isoformat(timespec='seconds')+'。运行状态：`'+state['status']+'`。', '',
            '权威总表：['+str(Path(output).name)+']('+str(Path(output).resolve())+')。只继承超参，每份新拟合均从同一原始BF16开始；所有合法实际候选均按完整64叶／24742题S8比较。下表仅为本次已评集合内入选，未完成类别不称搜索完成。', '',
            '| 类别 | 状态／已完成参数 | D/K/S/B | T／旋转 | E／D宽度 | 优化器／倍率 | 实际保护count | S8 | 类logical／packed bpw | cap bits |',
            '| --- | --- | --- | --- | --- | --- | --- | ---: | --- | ---: |']
        for row in rows:
            recipe=row['selected_recipe']; counts=sorted(set(row['actual_protection_counts'].values()))
            text.append('| '+ ' | '.join((row['category_id'],row['status']+'／'+str(len(row['completed_parameters']))+'/8',
                '/'.join(str(row[key]) for key in ('D','K','S','B')),str(recipe['transpose'])+'／'+recipe['rotation']['level'],
                str(recipe['encoder']['hidden_dim'])+'／'+str(recipe['decoder']['hidden_dim']),
                recipe['optimizer']['kind']+'／'+str(recipe['optimizer']['multiplier']),str(counts),
                repr(row['full8_mean']),str(row['class_logical_bpw'])+'／'+str(row['class_packed_bpw']),str(row['cap_bits'])))+' |')
        for row in rows:
            if row['failed_parameters'] or row['unstarted_parameters']:
                text.extend(['', '`'+row['category_id']+'`：失败参数 '+str(row['failed_parameters'])+'；尚未启动 '+str(row['unstarted_parameters'])+'。'])
        text.extend(['', '实际各stage首步LR、Muon κ与矩阵有效幅度、完整projection counts、未覆盖／非法／超cap／失败原因及每点证据均在同一JSON总表。费用含主码、decoder、保护值／indices／scales、rotation和适用bias；类bpw按本类实际N归一。', '',
            '累计本批候选实际拥有窗为'+repr(state['owned_GPU_hours'])+' GPUh；逐候选和分phase窗口互相重叠，不相加收费。R1／R2、既有21 profile费用单列。顺序搜索完成也不等于正式live逐类恢复链、九项验收或跨模型验证完成。', ''])
        MD.write_text(original+'\n\n'+'\n'.join(text),encoding='utf-8')
    return report
