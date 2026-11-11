"""Bounded CAT recovery choices, with complete-grid scores and explicit quotas."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import json
import math
from pathlib import Path
import hashlib
from types import MappingProxyType
from typing import Mapping, Sequence

from .contracts import digest


SCOPES = ("remaining_lora_prefix_decoder", "remaining_lora_current_decoder", "current_decoder")


def _fraction(value) -> Fraction:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Recovery selection requires finite, unrounded scores.")
    return Fraction(str(value))


@dataclass(frozen=True)
class RecoverySettings:
    scope: str = "remaining_lora_prefix_decoder"
    decoder_lr: float = 1e-5
    schedule_max: int = 500
    lora_rank: int = 12
    lora_alpha: int = 24
    lora_dropout: float = .1
    lora_lr: float = 1e-4

    def __post_init__(self):
        if self.scope not in SCOPES or self.schedule_max not in (500, 1000):
            raise ValueError("Unsupported core recovery scope or schedule_max.")
        if self.decoder_lr not in (.5e-5, 1e-5, 2e-5):
            raise ValueError("The core decoder LR is restricted to the confirmed three values.")
        if (self.lora_rank, self.lora_alpha, self.lora_dropout, self.lora_lr) != (12, 24, .1, 1e-4):
            raise ValueError("Keep the fixed future-dense rank12/alpha24/dropout.1/LR1e-4 recipe.")

    @property
    def node_steps(self) -> tuple[int, ...]:
        return tuple(range(125, self.schedule_max + 1, 125))


@dataclass(frozen=True)
class FullEightScore:
    task_scores: Mapping[str, object]
    evaluation_id: str
    evidence: Mapping

    def __post_init__(self):
        if len(self.task_scores) != 8 or not self.evaluation_id:
            raise ValueError("Each node requires all eight tasks and its evaluation identity.")
        for value in self.task_scores.values():
            if not 0 <= _fraction(value) <= 1:
                raise ValueError("Complete-eight task metrics use their original [0,1] scale.")
        required = {"checkpoint_id", "source_checkpoint_id", "protocol_id", "evaluation_digest",
            "leaf_count", "sample_count", "expected_leaf_count", "expected_sample_count", "phase"}
        if not required <= set(self.evidence) or not all(self.evidence[key] for key in
                ("checkpoint_id", "source_checkpoint_id", "protocol_id", "evaluation_digest")):
            raise ValueError("Bind the score to actual checkpoint, complete evaluation and protocol identities.")
        if self.evidence["leaf_count"] != self.evidence["expected_leaf_count"] or self.evidence["sample_count"] != self.evidence["expected_sample_count"]:
            raise ValueError("Partial evaluation coverage cannot select a node.")
        if any(not isinstance(self.evidence[key], int) or isinstance(self.evidence[key], bool) or self.evidence[key] <= 0
                for key in ("leaf_count", "sample_count", "expected_leaf_count", "expected_sample_count")):
            raise ValueError("Complete evaluation counts must be positive explicit integers.")
        if self.evidence["phase"] not in ("formal", "cpu_fixture"):
            raise ValueError("Declare actual formal evidence or an explicitly ineligible CPU rule fixture.")
        if self.evidence["phase"] == "formal" and not self.evidence.get("validated_complete_summary"):
            raise ValueError("Formal selection requires a validated bound complete-eight summary.")
        object.__setattr__(self, "task_scores", MappingProxyType(dict(self.task_scores)))
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))

    @property
    def mean(self) -> Fraction:
        return sum((_fraction(value) for value in self.task_scores.values()), Fraction()) / 8


def load_complete_eight_score(*, summary_path: str | Path, evaluation_path: str | Path,
                              checkpoint_dir: str | Path, source_checkpoint_id: str,
                              protocol_manifest: Mapping, artifact_path_map: Mapping[str, str] | None = None) -> FullEightScore:
    """Read existing complete evaluator evidence; never infer quality from a log."""
    summary_path, evaluation_path, checkpoint_dir = Path(summary_path), Path(evaluation_path), Path(checkpoint_dir)
    def sha(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))
    plan_path = Path(protocol_manifest["plan_path"])
    bindings = {str(path): sha(path) for path in (summary_path, evaluation_path, checkpoint_dir / "checkpoint_meta.json", plan_path)}
    summary, evaluation, meta = read(summary_path), read(evaluation_path), read(checkpoint_dir / "checkpoint_meta.json")
    if evaluation.get("status") != "complete_original_full8" or evaluation.get("exit_code") != 0:
        raise ValueError("The eight-task evaluator is incomplete or failed.")
    if bindings[str(plan_path)] != protocol_manifest["plan_sha256"] or evaluation.get("source_plan", {}).get("sha256") != protocol_manifest["plan_sha256"]:
        raise ValueError("The completed evaluator differs from its frozen original execution plan.")
    if bindings[str(checkpoint_dir / "checkpoint_meta.json")] != protocol_manifest["checkpoint_meta_sha256"]:
        raise ValueError("The completed evaluator's bound checkpoint metadata changed.")
    if summary["checkpoint_id"] != meta["checkpoint_id"] or evaluation["checkpoint_id"] != meta["checkpoint_id"]:
        raise ValueError("The complete summary/evaluation belongs to another deployment UUID.")
    origin = (meta.get("extra_meta") or {}).get("cat_auto_selected_node", {}).get("source_node_id", meta["checkpoint_id"])
    if origin != source_checkpoint_id or (meta.get("extra_meta") or {}).get("cat_auto_phase") == "cpu_fixture":
        raise ValueError("Formal score evidence requires the actual source node and cannot use a CPU fixture boundary.")
    for actual, expected in ((summary["leaf_count"], protocol_manifest["leaf_count"]),
            (summary["sample_count"], protocol_manifest["sample_count"]),
            (evaluation["leaf_tasks"], protocol_manifest["leaf_count"]),
            (evaluation["sample_count"], protocol_manifest["sample_count"])):
        if actual != expected:
            raise ValueError("Complete evaluation coverage differs from the frozen model-specific protocol.")
    if summary["task_metric_keys"] != protocol_manifest["task_metric_keys"] or set(summary["task_metrics"]) != set(protocol_manifest["task_metric_keys"]):
        raise ValueError("Eight-task metric selection differs from the frozen protocol.")
    if not summary.get("input_results") or not evaluation.get("stages") or any(stage.get("exit_code") != 0 for stage in evaluation["stages"]):
        raise ValueError("All original complete-evaluation result shards and successful stages are required.")
    if len(evaluation["stages"]) != len(protocol_manifest["expected_stages"]):
        raise ValueError("The evaluator did not complete every original planned stage.")
    for actual, expected in zip(evaluation["stages"], protocol_manifest["expected_stages"], strict=True):
        if any(actual[key] != expected[key] for key in ("name", "argv", "sample_count")):
            raise ValueError("A completed evaluator stage changed its original command or coverage.")
    path_map = dict(artifact_path_map or {})
    if set(path_map) - {row["path"] for row in summary["input_results"]}:
        raise ValueError("Explicit artifact path mappings may resolve only original bound result shards.")
    resolutions = []
    for row in summary["input_results"]:
        path = Path(path_map.get(row["path"], row["path"]))
        if sha(path) != row["sha256"]:
            raise ValueError("A bound complete-eight result shard changed.")
        bindings[str(path)] = row["sha256"]
        resolutions.append(dict(recorded_path=row["path"], actual_read_path=str(path), sha256=row["sha256"]))
    mean_percent = math.fsum(summary["task_metrics"].values()) * 100 / 8
    if not math.isclose(mean_percent, summary["mean_percent"], rel_tol=0, abs_tol=1e-12) or not math.isclose(summary["mean_percent"], evaluation["full8_mean_percent"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("Complete-eight summary arithmetic or evaluation binding differs.")
    if any(sha(Path(path)) != value for path, value in bindings.items()):
        raise ValueError("Evaluation evidence changed while selecting.")
    evidence = dict(checkpoint_id=meta["checkpoint_id"], source_checkpoint_id=source_checkpoint_id,
        protocol_id=protocol_manifest["protocol_id"], evaluation_digest=digest(bindings),
        leaf_count=summary["leaf_count"], sample_count=summary["sample_count"],
        expected_leaf_count=protocol_manifest["leaf_count"], expected_sample_count=protocol_manifest["sample_count"],
        phase="formal", validated_complete_summary=True, artifacts=bindings, explicit_artifact_resolutions=resolutions)
    return FullEightScore(summary["task_metrics"], str(evaluation_path), evidence)


@dataclass(frozen=True)
class NodeScore:
    step: int
    score: FullEightScore
    checkpoint_id: str
    fixed_train_kd: object | None = None

    def __post_init__(self):
        if self.step < 0 or not self.checkpoint_id:
            raise ValueError("A node needs a nonnegative step and an actual checkpoint identity.")
        if self.fixed_train_kd is not None:
            _fraction(self.fixed_train_kd)
        if self.score.evidence["source_checkpoint_id"] != self.checkpoint_id:
            raise ValueError("The evaluation deployment must originate from this actual training node.")


@dataclass(frozen=True)
class TrajectoryResult:
    category_id: str
    parent_id: str
    raw_id: str
    recipe_id: str
    candidate_id: str
    trajectory_id: str
    is_anchor: bool
    logical_bits: int
    packed_bits: int
    settings: RecoverySettings
    raw_score: FullEightScore
    nodes: tuple[NodeScore, ...]
    fixed_train_readout_id: str | None = None

    def __post_init__(self):
        if not all((self.category_id, self.parent_id, self.raw_id, self.recipe_id, self.candidate_id, self.trajectory_id)):
            raise ValueError("A trajectory must bind category, live parent, raw, recipe and trajectory identities.")
        if self.logical_bits < 0 or self.packed_bits < 0:
            raise ValueError("Storage costs must be nonnegative.")
        if tuple(node.step for node in self.nodes) != self.settings.node_steps:
            raise ValueError("Select only after the whole unchanged schedule and every 125-step full-eight node finish.")
        tasks = set(self.raw_score.task_scores)
        if any(set(node.score.task_scores) != tasks for node in self.nodes):
            raise ValueError("Raw and node evaluations must cover the same complete eight tasks.")
        if self.raw_score.evidence["source_checkpoint_id"] != self.raw_id:
            raise ValueError("Raw0 diagnostics must belong to this trajectory's raw checkpoint.")
        keys = ("protocol_id", "phase", "expected_leaf_count", "expected_sample_count")
        protocol = tuple(self.raw_score.evidence[key] for key in keys)
        if any(tuple(node.score.evidence[key] for key in keys) != protocol for node in self.nodes):
            raise ValueError("Raw and nodes must share one evaluation protocol and evidence phase.")
        ids = [self.raw_id, *(node.checkpoint_id for node in self.nodes)]
        if len(ids) != len(set(ids)):
            raise ValueError("Raw and saved node identities must be distinct.")

    @property
    def best_node(self) -> NodeScore:
        return max(self.nodes, key=lambda node: (node.score.mean, node.step))


def best_trajectory(results: Sequence[TrajectoryResult]) -> tuple[TrajectoryResult, NodeScore]:
    if not results:
        raise ValueError("No complete recovery trajectory is available.")
    if len({(row.category_id, row.parent_id) for row in results}) != 1:
        raise ValueError("Recovery trajectories may compete only within one category and its own live parent.")
    if len({tuple(sorted(row.raw_score.task_scores)) for row in results}) != 1:
        raise ValueError("All competing trajectories require the same eight evaluation tasks.")
    if len({tuple(row.raw_score.evidence[key] for key in ("protocol_id", "phase", "expected_leaf_count", "expected_sample_count")) for row in results}) != 1:
        raise ValueError("Different evaluation protocols, coverage contracts or fixture/formal phases cannot compete.")
    winner = min(results, key=lambda row: (-row.best_node.score.mean, not row.is_anchor,
        row.logical_bits, row.packed_bits, row.candidate_id, row.trajectory_id))
    return winner, winner.best_node


@dataclass(frozen=True)
class ThirdArm:
    category_id: str
    raw_id: str
    recipe_id: str
    settings: RecoverySettings
    reason: str
    interaction_eligible: bool


class RecoveryQuota:
    """In-order allocation; the two interaction slots are committed before results."""
    def __init__(self, categories: Sequence[str]):
        self.categories = tuple(categories)
        if not self.categories or len(set(self.categories)) != len(self.categories):
            raise ValueError("Freeze a nonempty unique semantic category order first.")
        self.decisions: dict[str, ThirdArm | None] = {}
        self.long_categories: list[str] = []
        self.interaction_categories: list[str] = []
        self.started: list[dict] = []
        self.baselines: dict[str, tuple[TrajectoryResult, ...]] = {}
        self.selected_categories: list[str] = []

    def decide(self, results: Sequence[TrajectoryResult], *, prefix_targets: Sequence[str],
               current_targets: Sequence[str], future_dense_targets: Sequence[str]) -> ThirdArm | None:
        winner, node = best_trajectory(results)
        category = winner.category_id
        expected = self.categories[len(self.selected_categories)] if len(self.selected_categories) < len(self.categories) else None
        if category != expected or category in self.decisions:
            raise ValueError("Allocate optional arms once, in frozen category order.")
        if len(results) not in (1, 2) or any(row.settings != RecoverySettings() for row in results):
            raise ValueError("Third-arm generation follows one or two original 500-step baseline trajectories only.")
        if len({row.raw_id for row in results}) != len(results) or len({row.recipe_id for row in results}) != len(results):
            raise ValueError("The two live recipes and their raw checkpoints must actually differ.")
        self.baselines[category] = tuple(results)
        prefix, current, future = set(prefix_targets), set(current_targets), set(future_dense_targets)
        if not current or not current <= prefix or len(prefix) != len(tuple(prefix_targets)) or len(current) != len(tuple(current_targets)):
            raise ValueError("Current decoder owners must be an explicit nonempty subset of the compressed prefix.")
        if prefix & future:
            raise ValueError("Compressed decoder owners and future dense owners must be disjoint.")
        early = max(item.score.mean for item in winner.nodes[:-1])
        endpoint = winner.nodes[-1].score.mean
        setting = reason = None
        if endpoint > max(winner.raw_score.mean, early) and len(self.long_categories) < 2:
            setting, reason = replace(winner.settings, schedule_max=1000), "strict_500_peak_new_1000_schedule"
            self.long_categories.append(category)
        elif winner.raw_score.mean >= max(item.score.mean for item in winner.nodes):
            if current != prefix:
                scope = "remaining_lora_current_decoder" if future else "current_decoder"
                setting, reason = replace(winner.settings, scope=scope), "raw_not_lower_change_decoder_scope"
        elif early > endpoint:
            setting, reason = replace(winner.settings, decoder_lr=.5e-5), "early_node_higher_half_decoder_lr"
        elif endpoint > early and winner.fixed_train_readout_id:
            last_two = winner.nodes[-2:]
            if all(item.fixed_train_kd is not None for item in last_two) and _fraction(last_two[1].fixed_train_kd) < _fraction(last_two[0].fixed_train_kd):
                setting, reason = replace(winner.settings, decoder_lr=2e-5), "late_node_higher_fixed_train_kd_decreases"
        if setting is None:
            self.decisions[category] = None
            return None
        eligible = len(results) == 2 and setting.schedule_max == 500 and len(self.interaction_categories) < 2
        arm = ThirdArm(category, winner.raw_id, winner.recipe_id, setting, str(reason), eligible)
        if eligible:
            self.interaction_categories.append(category)
        self.decisions[category] = arm
        return arm

    def interaction_raw(self, arm: ThirdArm, results: Sequence[TrajectoryResult]) -> str | None:
        if self.decisions.get(arm.category_id) != arm:
            raise ValueError("Interaction eligibility must be recorded before the third-arm result exists.")
        if not arm.interaction_eligible:
            return None
        peers = [row for row in results if row.raw_id != arm.raw_id]
        if len(peers) != 1 or peers[0].category_id != arm.category_id:
            raise ValueError("A four-cell check reuses exactly the other already-fitted raw.")
        return peers[0].raw_id

    def counts(self) -> dict:
        return dict(max_trajectories=3 * len(self.categories) + 2,
            actual_started=len(self.started), actual_node_evaluations=sum(row["settings"].schedule_max // 125 for row in self.started),
            extra_1000_steps=sum(row["settings"].schedule_max - 500 for row in self.started),
            allocated_third=sum(value is not None for value in self.decisions.values()),
            long_categories=list(self.long_categories), interaction_categories=list(self.interaction_categories))

    def select_completed_category(self, results: Sequence[TrajectoryResult]) -> tuple[TrajectoryResult, NodeScore, dict]:
        winner, node = best_trajectory(results)
        category = winner.category_id
        expected_category = self.categories[len(self.selected_categories)] if len(self.selected_categories) < len(self.categories) else None
        if category != expected_category:
            raise ValueError("Finalize one category once, in frozen single-chain order.")
        if category not in self.decisions:
            raise ValueError("Declare the bounded optional-arm decision before final selection.")
        expected = {(row.raw_id, row.settings) for row in self.baselines[category]}
        arm = self.decisions[category]
        if arm is not None:
            expected.add((arm.raw_id, arm.settings))
            if arm.interaction_eligible:
                expected.add((self.interaction_raw(arm, self.baselines[category]), arm.settings))
        actual = {(row.raw_id, row.settings) for row in results}
        if len(actual) != len(results) or actual != expected:
            raise ValueError("Finish every announced baseline/third/fourth-cell trajectory before entering the next category.")
        certificate = dict(category_id=category, parent_id=winner.parent_id,
            selected_trajectory_id=winner.trajectory_id, selected_node_id=node.checkpoint_id,
            selected_step=node.step, phase=winner.raw_score.evidence["phase"],
            required_trajectories=sorted(row.trajectory_id for row in results),
            evaluation_digests=sorted(item.score.evidence["evaluation_digest"] for row in results for item in row.nodes),
            all_announced_cells_complete=True)
        certificate["selection_id"] = digest(certificate)
        self.selected_categories.append(category)
        return winner, node, certificate

    def register_start(self, *, category_id: str, raw_id: str, settings: RecoverySettings, kind: str) -> None:
        if category_id not in self.categories or kind not in ("baseline", "third", "interaction"):
            raise ValueError("Unknown recovery category or arm kind.")
        expected = self.categories[len(self.selected_categories)] if len(self.selected_categories) < len(self.categories) else None
        if category_id != expected:
            raise ValueError("The next category depends on the sole fully selected previous category.")
        existing = [row for row in self.started if row["category_id"] == category_id]
        if any(row["raw_id"] == raw_id and row["settings"] == settings for row in existing):
            raise ValueError("Do not launch a duplicate recovery trajectory.")
        if kind == "baseline":
            if settings != RecoverySettings() or sum(row["kind"] == kind for row in existing) >= 2 or category_id in self.decisions:
                raise ValueError("Each category starts at most two distinct original 500-step baselines before decisions.")
        else:
            arm = self.decisions.get(category_id)
            if arm is None or settings != arm.settings or any(row["kind"] == kind for row in existing):
                raise ValueError("Optional starts must match the recorded single third/interaction decision.")
            if kind == "third" and raw_id != arm.raw_id:
                raise ValueError("The third arm starts fresh from the selected baseline's raw.")
            if kind == "interaction" and (not arm.interaction_eligible or raw_id == arm.raw_id
                    or not any(row["kind"] == "baseline" and row["raw_id"] == raw_id for row in existing)):
                raise ValueError("The fourth cell reuses the other existing raw and its preallocated 500-step setting.")
        if len(self.started) >= 3 * len(self.categories) + 2:
            raise ValueError("The global 3C+2 recovery limit is exhausted.")
        self.started.append(dict(category_id=category_id, raw_id=raw_id, settings=settings, kind=kind))


def trajectory_identity(*, category_id: str, parent_id: str, raw_id: str, recipe_id: str,
                        settings: RecoverySettings, seed_identity: Mapping) -> str:
    payload = dict(category_id=category_id, parent_id=parent_id, raw_id=raw_id,
        recipe_id=recipe_id, settings=asdict(settings), seed_identity=dict(seed_identity))
    return digest(payload)
