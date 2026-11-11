"""Last-category endpoint eligibility, then actual same-artifact nine-task choice."""
from __future__ import annotations

from dataclasses import asdict

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.nodes import NodeScore, best_trajectory
from experiments.sc_cat_live_cmcc_20261106.selection import RecoveryQuota


def terminal_endpoints(results):
    recovered, recovery_node = best_trajectory(results)
    baselines = [row for row in results if row.settings.decoder_lr == 1e-5
        and row.settings.schedule_max == 500 and row.settings.scope == "remaining_lora_prefix_decoder"]
    require(len(baselines) == 2 and len({row.raw_id for row in baselines}) == 2, "Keep the two actual distinct original raw recipes at the last category.")
    raw = min(baselines, key=lambda row: (-row.raw_score.mean, not row.is_anchor,
        row.logical_bits, row.packed_bits, row.candidate_id, row.raw_id))
    qualified_zero = raw.raw_score.mean > recovery_node.score.mean
    endpoints = []
    for baseline in sorted(baselines, key=lambda row: (not row.is_anchor, row.candidate_id)):
        own = [row for row in results if row.raw_id == baseline.raw_id]
        winner, node = best_trajectory(own)
        if qualified_zero and baseline.raw_id == raw.raw_id:
            winner, node = baseline, NodeScore(0, baseline.raw_score, baseline.raw_id)
        endpoints.append((winner, node))
    return endpoints, dict(best_raw_id=raw.raw_id, best_raw_mean=str(raw.raw_score.mean),
        best_recovery_node_id=recovery_node.checkpoint_id, best_recovery_mean=str(recovery_node.score.mean),
        global_raw_zero_strictly_qualified=qualified_zero)


class ContinuationQuota(RecoveryQuota):
    def __init__(self, previous, terminal_chooser):
        self.__dict__.update(previous.__dict__)
        self.terminal_chooser = terminal_chooser

    def select_completed_category(self, results):
        if results[0].category_id != self.categories[-1]:
            return super().select_completed_category(results)
        self._complete_results(results)
        endpoints, eligibility = terminal_endpoints(results)
        winner, node, reported = self.terminal_chooser(endpoints)
        require(any(winner is row and node == point for row, point in endpoints), "Commit an actual measured eligible endpoint, never a rebuilt substitute.")
        zero = node.step == 0
        certificate = dict(category_id=winner.category_id, parent_id=winner.parent_id,
            selected_trajectory_id=winner.trajectory_id, selected_node_id=node.checkpoint_id,
            selected_round_base_id=winner.raw_id, selected_recipe_id=winner.recipe_id,
            selected_candidate_id=winner.candidate_id, selected_step=node.step,
            selected_kind="raw_zero" if zero else "recovery_node", raw_zero_strictly_above_all_recovery=zero,
            phase=winner.raw_score.evidence["phase"], required_trajectories=sorted(row.trajectory_id for row in results),
            evaluation_digests=sorted({point.score.evidence["evaluation_digest"] for row in results for point in row.nodes}
                | {row.raw_score.evidence["evaluation_digest"] for row in results}),
            all_announced_cells_complete=True, selection_rule="two_actual_endpoints_S9",
            terminal_raw_zero_eligibility=eligibility, nine_task_comparison=reported)
        certificate.update({key: value for key, value in eligibility.items() if key != "global_raw_zero_strictly_qualified"})
        certificate["selection_id"] = digest(certificate)
        self.selected_categories.append(winner.category_id)
        return winner, node, certificate
