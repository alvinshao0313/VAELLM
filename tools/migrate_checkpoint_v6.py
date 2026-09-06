from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rotation.model_utils import get_model
from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model, save_v6_full_checkpoint
from train_utils.legacy_checkpoint_io import (
    inspect_legacy_checkpoint,
    load_legacy_checkpoint_for_migration,
    normalize_legacy_model_for_v6,
)


def _bool(raw: object) -> bool:
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {raw!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Migrate one supported legacy VAELLM checkpoint to v6.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dry_run", type=_bool, default=True)
    parser.add_argument("--access_token", default=None)
    return parser


def migrate_checkpoint_v6(
    *, source: str, output_dir: str, dry_run: bool = True, access_token: str | None = None
) -> dict:
    inspection = inspect_legacy_checkpoint(source)
    output = os.path.abspath(output_dir)
    report = {
        "source": inspection.checkpoint_dir,
        "output_dir": output,
        "dry_run": bool(dry_run),
        "base_model_path": inspection.base_model_path,
        "status": "validated",
    }
    if dry_run:
        return report
    if os.path.exists(output):
        raise FileExistsError(f"migration output already exists: {output}")

    model, inspection = load_legacy_checkpoint_for_migration(
        inspection.checkpoint_dir, access_token=access_token
    )
    compressed, original_only = normalize_legacy_model_for_v6(model)
    result = save_v6_full_checkpoint(
        model,
        output,
        checkpoint_kind="final_model",
        compressed_targets=compressed,
        pending_dense_targets=(),
        skip_targets=original_only,
        legacy_original_only_sources=original_only,
        train_mode="none",
        lora_config=None,
        completed_categories=inspection.meta.get("completed_categories") or (),
        compression_categories=inspection.meta.get("compression_categories") or (),
        target_layers=inspection.meta.get("target_layers"),
        target_modules=inspection.meta.get("target_modules") or (),
        finalized_status={
            "lora_finalized": True,
            "decoder_finalized": True,
            "aux_finalized": True,
            "inference_ready": True,
        },
        runtime_audit={"runtime": "tools.migrate_checkpoint_v6"},
        base_model_path=inspection.base_model_path,
        extra_meta={
            "migration": {
                "source": inspection.checkpoint_dir,
                "source_format": inspection.meta.get("format"),
                "legacy_original_only_sources": list(original_only),
            }
        },
    )

    fresh = get_model(inspection.base_model_path, access_token)
    _fresh, loaded_meta, load_result = load_v6_full_checkpoint_into_model(
        fresh, output, expected_kind="final_model", map_location="cpu", strict=True
    )
    if getattr(load_result, "missing_keys", None) or getattr(load_result, "unexpected_keys", None):
        raise RuntimeError("fresh v6 migration reload reported missing/unexpected keys")
    report.update(
        {
            "status": "written_and_reloaded",
            "checkpoint_id": loaded_meta["checkpoint_id"],
            "compressed_targets": list(compressed),
            "skip_targets": list(original_only),
            "checkpoint": result["output_dir"],
        }
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    report = migrate_checkpoint_v6(
        source=args.source,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        access_token=args.access_token,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
