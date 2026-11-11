"""Full-eight selection, honest250 screening, independent calibration and raw0."""
from dataclasses import asdict, dataclass

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.nodes import FullEightScore, NodeScore
from .settings import RecoverySettings


@dataclass(frozen=True)
class TrajectoryResult:
    category_id: str
    parent_id: str
    raw_id: str
    recipe_id: str
    candidate_id: str
    trajectory_id: str
    logical_bits: int
    packed_bits: int
    settings: RecoverySettings
    raw_score: FullEightScore
    nodes: tuple[NodeScore, ...]
    status: str = 'complete'

    def __post_init__(self):
        require(self.status in ('complete', 'screened250', 'pending250'), 'Failed or missing results cannot select nodes.')
        expected = self.settings.node_steps if self.status == 'complete' else (125, 250)
        require(tuple(n.step for n in self.nodes) == expected and (self.status == 'complete' or self.settings.schedule_max == 500),
            'Wrong measured node grid or schedule.')
        require(self.raw_score.evidence['source_checkpoint_id'] == self.raw_id, 'Raw score source differs.')
        protocol = ('protocol_id', 'phase', 'expected_leaf_count', 'expected_sample_count')
        require(all(set(n.score.task_scores) == set(self.raw_score.task_scores) and
            all(n.score.evidence[k] == self.raw_score.evidence[k] for k in protocol) for n in self.nodes), 'Node protocol differs.')
        require(len({self.raw_id, *(n.checkpoint_id for n in self.nodes)}) == len(self.nodes)+1, 'Repeated node identity.')

    @property
    def best_node(self):
        return max(self.nodes, key=lambda n: (n.score.mean, n.step))


def _key(row, node):
    return (-node.score.mean, row.logical_bits, row.packed_bits, row.candidate_id, -node.step, row.trajectory_id)


def compare(rows, *, early=False):
    require(rows and len({(r.category_id, r.parent_id) for r in rows}) == 1, 'Compare only a common true parent/category.')
    require(len({(r.raw_score.evidence['protocol_id'], r.raw_score.evidence['phase']) for r in rows}) == 1, 'Common evaluation required.')
    pairs = [(r, max((n for n in r.nodes if not early or n.step <= 250), key=lambda n: (n.score.mean, n.step))) for r in rows]
    return sorted(pairs, key=lambda pair: _key(*pair))


def calibration(rows, existing_complete_ids):
    require(len(rows) >= 3 and all(r.status == 'complete' and r.settings.schedule_max == 500 for r in rows),
        'Calibrate only >=3 distinct fully measured500 trajectories.')
    ids = {r.trajectory_id for r in rows}
    require(len(ids) == len(rows) and set(existing_complete_ids) <= ids, 'Freeze actual pre-round controls, not new calibration completions.')
    early = [r.trajectory_id for r, n in compare(rows, early=True)]
    keep = set(early[:2]) | set(existing_complete_ids)
    full, node = compare(rows)[0]
    simulated, snode = compare([r for r in rows if r.trajectory_id in keep])[0]
    passed = (full.trajectory_id, node.checkpoint_id) == (simulated.trajectory_id, snode.checkpoint_id)
    return dict(status='passed' if passed else 'failed', category_id=full.category_id, parent_id=full.parent_id,
        existing_complete_ids=sorted(existing_complete_ids), early_ranking=early, retained=sorted(keep),
        full_winner=[full.trajectory_id, node.checkpoint_id], simulated_winner=[simulated.trajectory_id, snode.checkpoint_id])


def screen(rows, existing_complete_ids, calibration_state):
    require(calibration_state.get('status') == 'passed' or len(rows) <= 2, 'This search item has not passed model-specific calibration.')
    rank = compare(rows, early=True)
    keep = {r.trajectory_id for r, n in rank[:2]} | set(existing_complete_ids)
    return [r.trajectory_id for r in rows if r.trajectory_id in keep and r.status != 'complete']


def long_check(rows, searched_recipe_id, used_categories):
    require(len(set(used_categories)) == len(used_categories) <= 2, 'Persistent long-check quota is invalid.')
    row, _node = compare([r for r in rows if r.recipe_id == searched_recipe_id])[0]
    if row.category_id in used_categories or len(used_categories) == 2 or row.status != 'complete' or row.settings.schedule_max != 500:
        return None
    last = row.nodes[-1]
    return row if last.score.mean > max(row.raw_score.mean, *(n.score.mean for n in row.nodes[:-1])) else None


def select_category(rows):
    require(all(r.status in ('complete', 'screened250') for r in rows), 'Pending continuation cannot finalize a category.')
    best, node = compare(rows)[0]
    raw = min(rows, key=lambda r: (-r.raw_score.mean, r.logical_bits, r.packed_bits, r.candidate_id, r.trajectory_id))
    zero = raw.raw_score.mean > node.score.mean
    chosen = raw if zero else best
    result = dict(category_id=chosen.category_id, parent_id=chosen.parent_id,
        selected_trajectory_id='raw0:'+chosen.raw_id if zero else chosen.trajectory_id,
        selected_node_id=chosen.raw_id if zero else node.checkpoint_id, selected_round_base_id=chosen.raw_id,
        selected_step=0 if zero else node.step, selected_kind='raw_zero' if zero else 'recovery_node',
        raw_zero_strictly_above_all_recovery=zero, all_announced_cells_complete=True,
        phase=chosen.raw_score.evidence['phase'])
    result['selection_id'] = digest(result)
    return chosen, result


def terminal_endpoints(rows):
    """One best recovery endpoint per VAE; only the global qualifying raw may replace it."""
    global_chosen, global_certificate = select_category(rows)
    endpoints=[]
    for candidate in sorted({r.candidate_id for r in rows}):
        own=[r for r in rows if r.candidate_id==candidate]
        row,node=compare(own)[0]
        if global_certificate['selected_step']==0 and global_chosen.candidate_id==candidate:
            row,certificate=global_chosen,dict(global_certificate)
        else:
            certificate=dict(category_id=row.category_id,parent_id=row.parent_id,
                selected_trajectory_id=row.trajectory_id,selected_node_id=node.checkpoint_id,
                selected_round_base_id=row.raw_id,selected_step=node.step,selected_kind='recovery_node',
                raw_zero_strictly_above_all_recovery=False,all_announced_cells_complete=True,phase=row.raw_score.evidence['phase'])
        certificate.pop('selection_id',None)
        certificate['terminal_endpoint']=True
        certificate['selection_id']=digest(certificate)
        endpoints.append((row,certificate))
    require(1<=len(endpoints)<=2,'Only the original Top2 live recipes may enter final nine-task selection.')
    return endpoints


@dataclass(frozen=True)
class SelectedCategory:
    category_id: str
    recipe_id: str
    source_candidate_id: str
    settings: RecoverySettings
    selected_step: int
    source_trajectory_id: str
    source_node_id: str
    source_round_base_id: str
    seed_identity: dict
    phase: str

    def training_plan(self):
        return asdict(self)
