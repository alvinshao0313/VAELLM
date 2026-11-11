"""SC-CAT §6–7 choices over complete, actual own-parent recovery evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest
from experiments.cat_auto_protocol_cmcc_20261104.nodes import (
    FullEightScore, NodeScore, RecoverySettings, ThirdArm, TrajectoryResult,
    RecoveryQuota as _RecoveryQuota, best_trajectory, load_complete_eight_score,
    trajectory_identity,
)


class RecoveryQuota(_RecoveryQuota):
    """Keep the existing accounting API, using the confirmed curve-only rule."""

    def decide(self, results: Sequence[TrajectoryResult], *, prefix_targets: Sequence[str],
               current_targets: Sequence[str], future_dense_targets: Sequence[str]) -> ThirdArm | None:
        winner, _node = best_trajectory(results)
        category = winner.category_id
        expected = self.categories[len(self.selected_categories)] if len(self.selected_categories) < len(self.categories) else None
        if category != expected or category in self.decisions:
            raise ValueError("Allocate optional arms once, in frozen category order.")
        if len(results) not in (1, 2) or any(row.settings != RecoverySettings() for row in results):
            raise ValueError("Complete the one or two original 500-step baselines before nomination.")
        if len({row.raw_id for row in results}) != len(results) or len({row.recipe_id for row in results}) != len(results):
            raise ValueError("The live recipe/raw pairs must actually differ.")
        prefix, current, future = set(prefix_targets), set(current_targets), set(future_dense_targets)
        if (not current or not current <= prefix or prefix & future
                or any(len(set(names)) != len(names) for names in (prefix_targets, current_targets, future_dense_targets))):
            raise ValueError("Use explicit unique current/prefix/future owners with disjoint dense targets.")
        early = max(item.score.mean for item in winner.nodes[:-1])
        endpoint = winner.nodes[-1].score.mean
        setting = reason = None
        if endpoint > max(winner.raw_score.mean, early) and len(self.long_categories) < 2:
            setting, reason = replace(winner.settings, schedule_max=1000), "strict_500_peak_new_1000_schedule"
        elif winner.raw_score.mean >= winner.best_node.score.mean:
            if current != prefix:
                scope = "remaining_lora_current_decoder" if future else "current_decoder"
                setting, reason = replace(winner.settings, scope=scope), "raw_not_lower_change_decoder_scope"
        elif early > endpoint:
            setting, reason = replace(winner.settings, decoder_lr=.5e-5), "early_node_higher_half_decoder_lr"
        elif endpoint > early:
            setting, reason = replace(winner.settings, decoder_lr=2e-5), "late_node_higher_double_decoder_lr"
        self.baselines[category] = tuple(results)
        if setting is None:
            self.decisions[category] = None
            return None
        eligible = len(results) == 2 and setting.schedule_max == 500 and len(self.interaction_categories) < 2
        arm = ThirdArm(category, winner.raw_id, winner.recipe_id, setting, str(reason), eligible)
        if setting.schedule_max == 1000:
            self.long_categories.append(category)
        if eligible:
            self.interaction_categories.append(category)
        self.decisions[category] = arm
        return arm

    def _complete_results(self, results: Sequence[TrajectoryResult]):
        winner, node = best_trajectory(results)
        category = winner.category_id
        expected_category = self.categories[len(self.selected_categories)] if len(self.selected_categories) < len(self.categories) else None
        if category != expected_category or category not in self.decisions:
            raise ValueError("Declare the next category's optional-arm decision before final selection.")
        baselines = self.baselines[category]
        expected = {(row.raw_id, row.settings) for row in baselines}
        arm = self.decisions[category]
        if arm is not None:
            expected.add((arm.raw_id, arm.settings))
            if arm.interaction_eligible:
                expected.add((self.interaction_raw(arm, baselines), arm.settings))
        actual = {(row.raw_id, row.settings) for row in results}
        if len(actual) != len(results) or actual != expected or len({row.trajectory_id for row in results}) != len(results):
            raise ValueError("Finish every announced baseline/third/interaction full-eight trajectory before selection, including raw0.")
        by_raw = {row.raw_id: row for row in baselines}
        fields = ("category_id", "parent_id", "recipe_id", "candidate_id", "is_anchor", "logical_bits", "packed_bits", "raw_score")
        for row in results:
            if any(getattr(row, name) != getattr(by_raw[row.raw_id], name) for name in fields):
                raise ValueError("All variants must preserve their actual raw recipe, raw0 score, parent and fees.")
        for baseline in baselines:
            if baseline not in results:
                raise ValueError("Do not replace an already completed baseline with another trajectory.")
        return winner, node, baselines

    def select_completed_category(self, results: Sequence[TrajectoryResult]) -> tuple[TrajectoryResult, NodeScore, dict]:
        recovered, recovery_node, baselines = self._complete_results(results)
        raw = min(baselines, key=lambda row: (-row.raw_score.mean, not row.is_anchor,
            row.logical_bits, row.packed_bits, row.candidate_id, row.raw_id))
        zero = raw.raw_score.mean > recovery_node.score.mean
        winner, node = ((raw, NodeScore(0, raw.raw_score, raw.raw_id)) if zero else (recovered, recovery_node))
        certificate = dict(category_id=winner.category_id, parent_id=winner.parent_id,
            selected_trajectory_id=winner.trajectory_id, selected_node_id=node.checkpoint_id,
            selected_round_base_id=winner.raw_id, selected_recipe_id=winner.recipe_id,
            selected_candidate_id=winner.candidate_id, selected_step=node.step,
            selected_kind="raw_zero" if zero else "recovery_node",
            raw_zero_strictly_above_all_recovery=zero,
            best_raw_id=raw.raw_id, best_raw_mean=str(raw.raw_score.mean),
            best_recovery_node_id=recovery_node.checkpoint_id,
            best_recovery_mean=str(recovery_node.score.mean), phase=winner.raw_score.evidence["phase"],
            required_trajectories=sorted(row.trajectory_id for row in results),
            evaluation_digests=sorted({item.score.evidence["evaluation_digest"] for row in results for item in row.nodes}
                | {row.raw_score.evidence["evaluation_digest"] for row in baselines}),
            all_announced_cells_complete=True)
        certificate["selection_id"] = digest(certificate)
        self.selected_categories.append(winner.category_id)
        return winner, node, certificate


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
    seed_identity: Mapping
    phase: str = "formal"

    def __post_init__(self):
        legal = (0, *self.settings.node_steps) if self.phase == "formal" else (0, 1, 2) if self.phase in ("cpu_fixture", "technical_validation") else ()
        if isinstance(self.selected_step, bool) or self.selected_step not in legal:
            raise ValueError("Select a measured native node or the globally qualified actual raw0.")
        if not all((self.category_id, self.recipe_id, self.source_candidate_id,
                self.source_trajectory_id, self.source_node_id, self.source_round_base_id)):
            raise ValueError("Preserve complete selected source provenance.")
        if self.selected_step == 0 and self.source_node_id != self.source_round_base_id:
            raise ValueError("Zero-KD selection originates from the actual raw, never a training_step UUID.")
        if self.selected_step > 0 and self.source_node_id == self.source_round_base_id:
            raise ValueError("A positive training node must differ from its actual raw UUID.")

    def training_plan(self) -> dict:
        return dict(category_id=self.category_id, recipe_id=self.recipe_id,
            recovery=asdict(self.settings), schedule_max=self.settings.schedule_max,
            stop_after_step=self.selected_step, seed_identity=dict(self.seed_identity),
            skip_kd=self.selected_step == 0, repeat_search=False, repeat_node_selection=False,
            training_exact_resume=False)


def selected_category(winner: TrajectoryResult, node: NodeScore, *, seed_identity: Mapping) -> SelectedCategory:
    """Freeze exactly the outcome returned by the complete-category chooser."""
    return SelectedCategory(winner.category_id, winner.recipe_id, winner.candidate_id, winner.settings,
        node.step, winner.trajectory_id, node.checkpoint_id, winner.raw_id, seed_identity,
        phase=winner.raw_score.evidence["phase"])
