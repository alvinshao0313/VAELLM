
"""Merge isolated linear-output worker artifacts into one native v6 model."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from experiments.linear_output.artifacts import load_linear, tensor_digest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen3-8B")
    p.add_argument("--workers", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--expected_target_count", type=int, default=252)
    args=p.parse_args()
    torch.set_num_threads(4)
    if Path(args.output_dir).exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    manifests=[]
    for w in args.workers:
        path=Path(w)
        m=json.loads((path/"manifest.json").read_text())
        if m.get("status") != "COMPLETE":
            raise RuntimeError(f"{path} status is {m.get('status')!r}")
        if set(m.get("modules", [])) != {r["module"] for r in m.get("records", [])}:
            raise RuntimeError(f"Module/record coverage mismatch in {path}")
        if (m.get("codebook_bits"), m.get("codebook_dim")) != (64, 32):
            raise RuntimeError(f"Wrong W2 configuration in {path}")
        manifests.append((path,m))
    if len({m.get("objective") for _,m in manifests}) != 1:
        raise RuntimeError("Cannot merge different objectives")
    names=[]
    records={}
    for path,m in manifests:
        for rec in m.get("records",[]):
            name=rec["module"]
            if rec.get("steps") != m.get("steps") or rec.get("code_payload_bpw") != 2.0:
                raise RuntimeError(f"Incomplete training or incorrect bit rate: {name}")
            if name in records: raise RuntimeError(f"duplicate module {name}")
            records[name]=(path/"linears"/name/"packed",rec)
            names.append(name)
    if len(names) != args.expected_target_count:
        raise RuntimeError(f"expected {args.expected_target_count} modules, found {len(names)}")
    if len(set(names)) != len(names): raise RuntimeError("duplicate modules")
    names.sort()
    model_root=Path(args.model_path)
    if not model_root.is_dir():
        from huggingface_hub import snapshot_download
        model_root=Path(snapshot_download(repo_id=args.model_path,local_files_only=True))
    model=AutoModelForCausalLM.from_pretrained(
        str(model_root), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True
    ).cpu()
    if args.expected_target_count == 252:
        categories={"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
        expected={n for n,m in model.named_modules()
                  if isinstance(m,torch.nn.Linear) and n.rsplit(".",1)[-1] in categories}
        if set(names) != expected:
            raise RuntimeError("Packed modules do not cover exactly the model's seven projection categories")
    for i,name in enumerate(names,1):
        original=model.get_submodule(name)
        packed,rec=records[name]
        if tensor_digest(original.weight.detach().cpu().float()) != rec["source_weight_sha256"]:
            raise RuntimeError(f"Base model source weight differs: {name}")
        if not packed.is_dir(): raise RuntimeError(f"missing packed checkpoint {packed}")
        compressed=load_linear(name,original,packed).cpu()
        parent_name,attr=name.rsplit(".",1)
        setattr(model.get_submodule(parent_name),attr,compressed)
        if i % 16 == 0: print(f"MERGE {i}/{len(names)}",flush=True)
    out=Path(args.output_dir)
    save_v6_full_checkpoint(
        model, str(out), checkpoint_kind="final_model", compressed_targets=names,
        base_model_path=str(model_root), save_config=True,
        extra_meta={
            "algorithm":"independent_linear_output_shared_teacher_sweep",
            "objective": manifests[0][1].get("objective"),
            "transpose":False, "residual_stages":1, "codebook_bits":64,
            "codebook_dim":32, "worker_count":len(manifests)
        }
    )
    json.dump({"status":"COMPLETE","modules":names,"worker_count":len(manifests)},
              (out/"merge_manifest.json").open("w"), indent=2)
    print(json.dumps({"status":"COMPLETE","modules":len(names),"output":str(out)}))

if __name__=="__main__":
    main()
