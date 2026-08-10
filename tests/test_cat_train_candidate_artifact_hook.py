from __future__ import annotations

import inspect
from unittest import mock

import pytest


def test_candidate_only_save_requires_spec_and_output_dir():
    from train_utils.cat_train_args import process_cat_train_args

    with pytest.raises(ValueError, match="candidate_artifact_spec|candidate_artifact_output_dir"):
        process_cat_train_args(
            [
                "--convert",
                "--save_candidate_artifact",
                "--target_categories",
                "q_proj",
                "--model_path",
                "toy",
            ]
        )


def test_candidate_artifact_args_rejected_when_disabled():
    from train_utils.cat_train_args import process_cat_train_args

    with pytest.raises(ValueError, match="candidate_artifact"):
        process_cat_train_args(
            [
                "--convert",
                "--candidate_artifact_spec",
                "/tmp/spec.json",
                "--target_categories",
                "q_proj",
                "--model_path",
                "toy",
            ]
        )


def test_save_model_path_unchanged_without_candidate_flag():
    from train_utils.cat_train_args import process_cat_train_args

    cat_args, _hf, _train, _vae = process_cat_train_args(
        [
            "--convert",
            "--save_model",
            "--target_categories",
            "q_proj",
            "--model_path",
            "toy",
            "--eval_ppl",
            "false",
        ]
    )
    assert cat_args.save_model is True
    assert cat_args.save_candidate_artifact is False


def test_pipeline_candidate_branch_skips_full_model_checkpoint():
    import train_utils.cat_train_pipeline as pipeline

    source = inspect.getsource(pipeline.run_cat_train)
    assert "if cat_args.save_candidate_artifact:" in source
    assert "elif cat_args.save_model:" in source
    assert "save_candidate_artifact_from_model" in source

    # Candidate branch index must precede the full-model save_model_checkpoint call
    # in the final-save section.
    candidate_idx = source.index("if cat_args.save_candidate_artifact:")
    elif_idx = source.index("elif cat_args.save_model:", candidate_idx)
    assert candidate_idx < elif_idx

    # Ensure the candidate arm does not call save_model_checkpoint before the elif.
    candidate_arm = source[candidate_idx:elif_idx]
    assert "save_model_checkpoint" not in candidate_arm
    assert "model.state_dict()" not in candidate_arm
    assert "save_pretrained" not in candidate_arm


def test_candidate_only_save_rejects_save_model_combination():
    from train_utils.cat_train_args import process_cat_train_args

    with pytest.raises(ValueError, match="mutually exclusive|save_candidate_artifact|save_model"):
        process_cat_train_args(
            [
                "--convert",
                "--save_candidate_artifact",
                "--save_model",
                "--candidate_artifact_spec",
                "/tmp/spec.json",
                "--candidate_artifact_output_dir",
                "/tmp/artifact",
                "--target_categories",
                "q_proj",
                "--model_path",
                "toy",
            ]
        )
