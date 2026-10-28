"""Run the frozen native CAT recipe from an explicitly prepared boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def read(path):
    return Path(path).read_text(encoding="utf-8")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def weight_stat(path):
    item = Path(path).stat()
    return dict(bytes=item.st_size, inode=item.st_ino, device=item.st_dev, mtime_ns=item.st_mtime_ns)


def write(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class BoundaryComplete(Exception):
    """The native saver has finished the requested complete category boundary."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--profile", choices=("C", "G", "D"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resume-boundary", type=Path)
    parser.add_argument("--source-run-report", type=Path)
    parser.add_argument("--stop-after-up", action="store_true")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Refuse to overwrite an existing execution report.")
    prepared = json.loads(read(args.preparation))
    if prepared["status"] != "cpu_verified":
        parser.error("The actual preparation must have passed its strict state verification.")
    if args.stop_after_up and (args.profile != "C" or args.resume_boundary):
        parser.error("Only the initial C shared gate/up run may stop after up.")
    if bool(args.resume_boundary) != bool(args.source_run_report):
        parser.error("C continuation requires both its boundary and actual source execution report.")
    row = prepared["profiles"][args.profile]
    snapshot = Path(row["training_cwd"]).resolve()
    argv = list(row["training_argv"][2:])
    # Both save events already exist in the fixed 500-step recipe; retain them.
    argv[argv.index("--save_total_limit") + 1] = "2"
    if args.resume_boundary:
        if args.profile != "C":
            parser.error("Only C may continue its own actual after_up without a new fork.")
        argv[argv.index("--resume_from_checkpoint") + 1] = str(args.resume_boundary.resolve())
    if Path.cwd().resolve() != snapshot:
        parser.error("Run from the frozen training_cwd in the preparation report.")
    for path, expected in prepared["snapshot_source_sha256"].items():
        actual = sha(path)
        if actual != expected:
            raise ValueError("Frozen native source changed: " + path)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status="starting", profile=args.profile, pid=os.getpid(),
                  command=[sys.executable, *sys.argv], native_argv=argv, cwd=str(snapshot),
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  preparation=str(args.preparation.resolve()),
                  source_checkpoint_id=row["checkpoint_id"],
                  implementation_sha256={str(p): sha(p) for p in Path(__file__).parent.glob("*.py")},
                  started_unix=time.time(), saved_boundaries=[])
    write(args.report, report)
    native = original_save = None
    try:
        sys.path.insert(0, str(snapshot))
        from train_utils.cat_runtime_adapter import parse_cat_runtime_args
        from train_utils import cat_train_pipeline as native
        from train_utils.checkpoint_v6 import load_v6_meta

        if not Path(native.__file__).resolve().is_relative_to(snapshot):
            raise RuntimeError("Native imports escaped the frozen snapshot.")
        boundary = Path(argv[argv.index("--resume_from_checkpoint") + 1])
        metadata = load_v6_meta(str(boundary))
        expected_completed = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if args.resume_boundary or args.profile == "D":
            expected_completed += ["gate_proj", "up_proj"]
        if (metadata["checkpoint_kind"] != "category_boundary"
                or metadata["completed_categories"] != expected_completed
                or metadata["extra_meta"]["category"] != expected_completed[-1]):
            raise ValueError("Unexpected starting category boundary or position.")
        binding = row
        if args.source_run_report:
            common = json.loads(read(args.source_run_report))
            binding = common["saved_boundaries"][-1]
            if (common["status"] != "category_boundary_complete" or common["profile"] != "C"
                    or Path(binding["path"]).resolve() != boundary.resolve()):
                raise ValueError("C boundary is not the recorded real shared after_up.")
        if metadata["checkpoint_id"] != binding["checkpoint_id"]:
            raise ValueError("Checkpoint UUID differs from its preparation or actual save.")
        for name, key in (("checkpoint_meta.json", "metadata_sha256"), ("cat_runtime_state.pt", "runtime_sha256")):
            if sha(boundary / name) != binding[key]:
                raise ValueError("Checkpoint metadata or runtime changed after verification.")
        actual_stat = weight_stat(boundary / metadata["state_dict_file"])
        if any(actual_stat[key] != binding["weight_stat"][key] for key in actual_stat):
            raise ValueError("Checkpoint weight file changed after verification.")
        report["actual_source_checkpoint_id"] = metadata["checkpoint_id"]
        report["actual_source"] = str(boundary)
        report["completed_at_start"] = expected_completed
        original_save = native.save_cat_v6_full_checkpoint

        def save_and_record(*positional, **kwargs):
            result = original_save(*positional, **kwargs)
            kind = kwargs.get("checkpoint_kind")
            if kind in ("category_boundary", "final_model"):
                path = Path(result["output_dir"])
                saved = load_v6_meta(str(path))
                event = dict(path=str(path), checkpoint_id=saved["checkpoint_id"],
                             kind=kind, completed_categories=saved["completed_categories"],
                             metadata_sha256=sha(path / "checkpoint_meta.json"),
                             runtime_sha256=sha(path / "cat_runtime_state.pt"),
                             weight_stat=weight_stat(path / saved["state_dict_file"]), saved_unix=time.time())
                report["saved_boundaries"].append(event)
                write(args.report, report)
                if args.stop_after_up and kind == "category_boundary" and kwargs.get("category") == "up_proj":
                    # Both down runs restart from this complete native boundary,
                    # under the same seed rule rather than different RNG continuations.
                    raise BoundaryComplete()
            return result

        native.save_cat_v6_full_checkpoint = save_and_record
        cat, hf, training, vae = parse_cat_runtime_args(argv)
        training.seed = int(cat.seed)
        report["status"] = "running"
        write(args.report, report)
        native.run_cat_train(cat_args=cat, hf_args=hf, training_args=training, vae_args=vae)
        if args.stop_after_up:
            raise RuntimeError("Native execution returned without the requested after_up boundary.")
        if not any(row["kind"] == "final_model" for row in report["saved_boundaries"]):
            raise RuntimeError("Native execution did not save its final model.")
        report.update(status="completed", exit_code=0)
    except BoundaryComplete:
        report.update(status="category_boundary_complete", exit_code=0)
    except BaseException as error:
        report.update(status="failed", exit_code=1, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        if original_save is not None:
            native.save_cat_v6_full_checkpoint = original_save
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - report["started_unix"])
        write(args.report, report)


if __name__ == "__main__":
    main()
