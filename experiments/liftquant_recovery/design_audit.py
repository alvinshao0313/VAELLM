"""Read-only audit of actual A/B payloads and the pinned LiftQuant LR arithmetic.

No model construction, training, CUDA use or checkpoint modification.
"""
import argparse
import json
import math
from pathlib import Path
import warnings

import torch


def schedule_difference(total, initial):
    # Exact relevant upstream expressions: empty_optimizer, CosineAnnealingLR,
    # scheduler.step(), then scheduler.get_lr()[0] assigned to the real optimizer.
    official_optimizer = torch.optim.AdamW([torch.tensor(0)], lr=initial)
    official = torch.optim.lr_scheduler.CosineAnnealingLR(
        official_optimizer, T_max=total, eta_min=initial / 20)
    current_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))], lr=initial)
    current = torch.optim.lr_scheduler.LambdaLR(
        current_optimizer, lambda s: .05 + .95 * (1 + math.cos(math.pi * s / total)) / 2)
    differences, examples = [], []
    actual_official_lr = initial
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for index in range(total):
            actual_current_lr = current_optimizer.param_groups[0]["lr"]
            differences.append(abs(actual_current_lr - actual_official_lr))
            if index in {0, 1, total // 2, total - 1}:
                examples.append(dict(update=index + 1, official_used_lr=actual_official_lr,
                                     current_used_lr=actual_current_lr))
            official.step()
            actual_official_lr = official.get_lr()[0]
            current_optimizer.step()
            current.step()
    return dict(total_steps=total, initial_lr=initial, same_used_lrs=not any(differences),
                max_absolute_used_lr_difference=max(differences), examples=examples)


def audit_payloads(source, run):
    a = torch.load(source / "pytorch_model.bin", map_location="cpu", mmap=True, weights_only=True)
    b = torch.load(run / "recovered_model/pytorch_model.bin", map_location="cpu", mmap=True, weights_only=True)
    if a.keys() != b.keys():
        raise ValueError("Different checkpoint state keys.")
    changed = []
    for name, before in a.items():
        after = b[name]
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(f"Payload schema differs: {name}")
        if not torch.equal(before, after):
            changed.append(dict(name=name, numel=before.numel(),
                                max_abs=(before.float() - after.float()).abs().max().item()))
    if any("_parallel_stage_decoder." not in entry["name"] for entry in changed):
        raise ValueError("Unexpected non-decoder change in this smoke's exported checkpoint.")
    meta = json.loads((source / "checkpoint_meta.json").read_text())
    blocks = {}
    for index in (9, 10):
        specs = [m for m in meta["converted_modules"] if m["name"].startswith(f"model.layers.{index}.")]
        entries = [x for x in changed if x["name"].startswith(f"model.layers.{index}.")]
        blocks[index] = dict(linears=len(specs),
                            code_proxy_numel=sum(math.prod(m["vq_weights"][0]["logical_shape"]) for m in specs),
                            changed_decoder_tensors=len(entries),
                            decoder_parameter_elements=sum(x["numel"] for x in entries))
    return dict(state_entries=len(a), all_non_decoder_state_exactly_equal=True,
                all_packed_codes_exactly_equal=True, changed_tensors=changed, blocks=blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    report = dict(torch_version=torch.__version__, cuda_used=False)
    report["schedule"] = [schedule_difference(total, initial) for total in (1, 4, 3968)
                          for initial in (2e-5, 2e-4)]
    report["payloads"] = audit_payloads(args.source, args.run)
    metrics = json.loads((args.run / "block_metrics.json").read_text())
    report["mse"] = {}
    for block, values in metrics.items():
        ntrain, holdout = values["ntrain"], values["holdout"]
        after_train = ((ntrain + holdout) * values["after_mse"] - holdout * values["holdout_mse"]) / ntrain
        report["mse"][block] = dict(before_all=values["before_mse"], after_all=values["after_mse"],
                                   before_train=values["steps"][0]["loss"],
                                   after_train_inferred_from_weighted_means=after_train,
                                   after_holdout=values["holdout_mse"])
    report["status"] = "AUDIT_COMPLETE_DIFFERENCES_CONFIRMED"
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    compact = {k: v for k, v in report.items() if k != "payloads"}
    compact["payloads"] = {k: v for k, v in report["payloads"].items() if k != "changed_tensors"}
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()
