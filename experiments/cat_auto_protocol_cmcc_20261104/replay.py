"""A single immutable prefix and fixed-recipe replay with original LR time domains."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from transformers import TrainerCallback

from .nodes import RecoverySettings
from .contracts import digest, validate_boundary


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
        legal = self.settings.node_steps if self.phase == "formal" else (1, 2) if self.phase in ("cpu_fixture", "technical_validation") else ()
        if self.selected_step not in legal:
            raise ValueError("Replay stops at the selected measured 125-step node, never raw0.")
        if not all((self.category_id, self.recipe_id, self.source_candidate_id, self.source_trajectory_id, self.source_node_id, self.source_round_base_id)):
            raise ValueError("A fixed recipe must preserve full selection provenance.")

    def training_plan(self) -> dict:
        return dict(category_id=self.category_id, recipe_id=self.recipe_id,
            recovery=asdict(self.settings), schedule_max=self.settings.schedule_max,
            stop_after_step=self.selected_step, seed_identity=dict(self.seed_identity),
            repeat_search=False, repeat_node_selection=False,
            training_exact_resume=False)


class StrictMainChain:
    def __init__(self, *, bf16_root_id: str, categories: Sequence[str]):
        self.bf16_root_id, self.categories = bf16_root_id, tuple(categories)
        if not bf16_root_id or not self.categories or len(self.categories) != len(set(self.categories)):
            raise ValueError("A chain requires an identified BF16 root and a frozen unique category order.")
        self.boundaries: list[dict] = []

    @property
    def live_parent_id(self) -> str:
        return self.boundaries[-1]["boundary_id"] if self.boundaries else self.bf16_root_id

    def commit(self, selected: SelectedCategory, *, live_parent_id: str, boundary_reference: Mapping,
               selection_certificate: Mapping, actual_round_base_id: str,
               restored_node_id: str, runtime_identity: Mapping) -> dict:
        boundary = validate_boundary(boundary_reference)
        boundary_id = boundary["uuid"]
        if selected.phase != "formal":
            raise ValueError("Technical state fixtures cannot become selected formal chain recipes.")
        index = len(self.boundaries)
        if index >= len(self.categories) or selected.category_id != self.categories[index]:
            raise ValueError("Only the next frozen category may extend the sole main chain.")
        if live_parent_id != self.live_parent_id:
            raise ValueError("A candidate fitted on another prefix cannot enter this chain.")
        if restored_node_id != selected.source_node_id or actual_round_base_id != selected.source_round_base_id:
            raise ValueError("The boundary must come from this exact selected node and its complete round_base.")
        certificate = dict(selection_certificate)
        certificate_id = certificate.pop("selection_id", None)
        if digest(certificate) != certificate_id or not certificate.get("all_announced_cells_complete") or any(
                certificate.get(key) != value for key, value in (("category_id", selected.category_id),
                ("parent_id", live_parent_id), ("selected_trajectory_id", selected.source_trajectory_id),
                ("selected_node_id", restored_node_id), ("selected_step", selected.selected_step))):
            raise ValueError("Final commit requires a complete selection certificate for every announced recovery cell.")
        if boundary["completed_prefix"] != list(self.categories[:index + 1]):
            raise ValueError("The real deployment runtime must advance exactly the frozen category prefix.")
        export, fresh = boundary["export_evidence"], boundary["fresh_evidence"]
        if not boundary["mutable_manifest"] or export.get("source_node_id") != restored_node_id or export.get("round_base_id") != actual_round_base_id or not export.get("all_mutable_restored") or fresh.get("checkpoint_id") != boundary_id or not fresh.get("state_exact"):
            raise ValueError("The real boundary must prove complete mutable restoration and its own fresh exact load.")
        if certificate["phase"] == "cpu_fixture" and boundary["formal_eligible"]:
            raise ValueError("CPU rule/state fixtures cannot enter a formal main chain.")
        if not boundary["formal_eligible"]:
            raise ValueError("A formal main chain requires validated model-scale export/fresh/ledger evidence.")
        used = {self.bf16_root_id, actual_round_base_id, restored_node_id, *(row["boundary_id"] for row in self.boundaries)}
        if not boundary_id or boundary_id in used or not runtime_identity:
            raise ValueError("Commit requires a new finalized UUID, all mutable provenance and actual runtime identity.")
        row = dict(parent_id=live_parent_id, boundary_id=boundary_id, selected=asdict(selected),
            boundary_reference=boundary, selection_id=certificate_id, runtime_identity=dict(runtime_identity),
            checkpoint_kind="category_boundary", training_exact_resume=False)
        self.boundaries.append(row)
        return row


class StopAtSelectedNode(TrainerCallback):
    """Stop a fresh replay prefix without rescaling the original scheduler maximum."""
    def __init__(self, *, schedule_max: int, selected_step: int):
        if schedule_max not in (500, 1000) or selected_step not in range(125, schedule_max + 1, 125):
            raise ValueError("Use the frozen original schedule_max and a measured selected_step.")
        self.schedule_max, self.selected_step = schedule_max, selected_step

    def on_train_begin(self, args, state, control, **kwargs):
        if args.max_steps != self.schedule_max or state.global_step != 0:
            raise ValueError("Fixed-recipe replay starts fresh and keeps its original schedule_max.")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if args.max_steps != self.schedule_max or state.global_step > self.selected_step:
            raise ValueError("Replay changed schedule_max or passed its frozen selected node.")
        if state.global_step == self.selected_step:
            control.should_save = True
            control.should_training_stop = True
        return control
