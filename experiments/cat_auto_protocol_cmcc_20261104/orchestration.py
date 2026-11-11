"""Finite manifest transitions and reuse; deliberately no background daemon."""
from copy import deepcopy
from .contracts import digest, require
from .identity import reusable_result


def ready_tasks(graph):
    by_id = {node['id']: node for node in graph['nodes']}
    return [deepcopy(node) for node in graph['nodes'] if node['status'] == 'planned_CPU_only'
        and all(by_id[key]['status'] in ('completed', 'matched_reuse') for key in node['parent_dependencies'])]


def record_completion(graph, node_id, result):
    require(result.get('exit_code') == 0 and result.get('status') == 'completed' and result.get('evidence_identity'), 'PID disappearance or an empty result is not completion.')
    nodes = {node['id']: node for node in graph['nodes']}
    require(node_id in nodes and nodes[node_id]['status'] == 'running', 'Only the actual running task may close.')
    node = nodes[node_id]
    require(result.get('reuse_key') == node['reuse_key'], 'Result belongs to a different parent/config/data/purpose identity.')
    node.update(status='completed', result=result)


def require_GPU_execution(graph, *, authorization, capability_records):
    require(authorization and authorization.get('model_identity') == graph['model_identity']
        and authorization.get('upper_gpu_hours', 0) > 0 and authorization.get('absolute_deadline')
        and authorization.get('physical_GPU_ids'), 'No model-specific GPU authorization/budget/deadline is present.')
    require(not graph.get('not_an_execution_budget') and not graph['unknown_nodes'], 'An unknown/scenario budget cannot execute a formal search.')
    require(all(row['status'] == 'verified' and row['verification_scope'] == 'GPU_model_path'
        for row in capability_records), 'Required formal model execution adapters are not yet GPU-verified.')
    require(graph['total_upper_gpu_hours'] <= authorization['upper_gpu_hours'], 'Full DAG exceeds the confirmed model GPU-hour budget.')
    return dict(model_budget_confirmed=True, authorization_identity=digest(authorization))


def live_recipe_requests(identity, anchor, offline_winner):
    # Both planned IDs are rebuilt on the *actual* live parent, never an old
    # BF16 bank. Equal actual recipes consume one raw, not a third alternative.
    from .identity import candidate_id
    result = []
    for role, recipe in (('anchor', anchor), ('offline_winner', offline_winner)):
        identifier = candidate_id(identity, recipe)
        if identifier not in {row['candidate_id'] for row in result}:
            result.append(dict(role=role, candidate_id=identifier, recipe=recipe, run_identity=identity,
                must_fresh_fit_own_parent=True, maximum_live_raws=2, formal_eligible=False))
    return result
