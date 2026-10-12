"""Scale-search contract around the unchanged native protection replay."""
from pathlib import Path

from experiments.long_context_recovery.contract import require
from experiments.longbench_v1.common import check_hashes, check_stat
from experiments.protected_int4_probe.measure import SOURCES
from experiments.protected_int4_scale_search.search import RATIOS
from experiments.protected_int4_quality.common import read, SHARDS, verify_shard, summarize as source_summary
from experiments.protected_int4_quality.protection import (payload_from, immutable_inventory, replay,
                                                          verify_native_cache)

ARM = 'output_scale_int4_scale_search'
SCHEMA = 'protected_int4_scale_search_expanded_bf16_v1'


def check(record):
    require(record['schema'] == SCHEMA and record['arm'] == ARM and record['model_label'] in SOURCES
            and record['source_checkpoint_id'] == SOURCES[record['model_label']][1]
            and not record['queued'] and not record['automatic_retry'], 'Unexpected isolated scale-search identity.')
    require(record['scale_search']['ratios'] == list(RATIOS) and record['scale_search']['row_chunk'] == 512
            and record['scale_search']['candidate_zero'] == 'exact_same_source_plain_payload'
            and record['scale_search']['objective'] == 'per_output_BF16_decode_weight_SSE'
            and record['scale_search']['tie_rule'] == 'keep_earlier_candidate', 'Scale search protocol changed.')
    check_hashes(record['source_sha256']); check_hashes(record['input_sha256'])
    for item in record['protected_stats']:
        check_stat(item)
    require(read(Path(record['source_checkpoint']) / 'checkpoint_meta.json')['checkpoint_id']
            == record['source_checkpoint_id'], 'Source checkpoint changed.')
    plain = record['plain_control']
    require(plain['source_checkpoint_id'] == record['source_checkpoint_id'] and plain['arm'] == 'output_scale_int4'
            and plain['accounting']['candidate_projected_linear_bits'] == record['accounting']['candidate_projected_linear_bits']
            and plain['accounting']['expanded_simulation_linear_bits'] == record['accounting']['expanded_simulation_linear_bits'],
            'Primary plain control source or storage differs.')


def summarize(output, record):
    result = source_summary(output, record)
    plain = record['plain_control']
    for shard, _tasks, batch in SHARDS:
        verify_shard(read(Path(output) / ('lm_eval_results_' + shard + '.json')),
                     read(plain['raw_result_paths'][shard]), batch)
    result.update(primary_control='same_source_plain_INT4', secondary_control='source_native_INT8',
        plain_control_mean_percent=plain['mean_percent'],
        mean_delta_vs_plain_pp=result['mean_percent'] - plain['mean_percent'],
        task_delta_vs_plain_pp={task: 100 * (score - plain['task_metrics'][task])
                               for task, score in result['task_metrics'].items()})
    return result
