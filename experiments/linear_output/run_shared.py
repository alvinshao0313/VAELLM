"""Shared teacher-sweep runner for the isolated one-stage W2 experiment."""
from __future__ import annotations
import argparse, json, os, random, re, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_scheduler
from litebsq.llm_vae import MultiLayerVAE
from train_utils.cat_data_prep import LinearPrepRef, prepare_group_linear_entries, materialize_prepared_group_data
from train_utils.cat_train_data import compute_stage_norm_stats, apply_stage_norm, restore_stage_norm
from train_utils.cat_train_pipeline import _fuse_norm_into_decoder, _fuse_q_scale_into_decoder
from train_utils.cat_train_pipeline import apply_group_vae_payload
from train_utils.cat_train_data import restore_stage_norm
from train_utils.train_args import create_optimizer
from train_utils.distill_data import build_distill_data_collator
from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from .artifacts import dump_json, export_linear, load_linear, tensor_digest, state_digest
from .calibration import build_bundle
from .config import parser, validate, vae_arguments
from .objectives import full_vae_forward
from .output_kernel import output_mse
from .incremental import LinearTrainer
from .final_state import finalize_training, load_final_training_state, export_saved_state
from train_utils.cat_train_pipeline import _fuse_norm_into_decoder, _fuse_q_scale_into_decoder

def _module_layer(name):
    m=re.search(r"model\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1

class SharedTrainer:
    def __init__(self, linear: nn.Linear, name: str, args, out: Path):
        self.name=name; self.args=args; self.linear=linear; self.out=out
        self.original=linear.weight.detach().cpu().float().contiguous()
        prep=prepare_group_linear_entries(
            group_refs=[LinearPrepRef(name,self.original,linear.in_features,linear.out_features,False)],
            activation_weight_by_linear=None, channel_protect_count=0, channel_axis="input",
            recon_loss_type="mse", apply_outlier_channel_removal=False)
        data=materialize_prepared_group_data(
            prepared_entries=prep,intra_parallel=(1,1),codebook_dim=32,
            batch_size=args.vae_chunk_vectors,normalize_weight=False,recon_loss_type="mse",
            train_device="cpu",split_weights_by_linear=[self.original],shuffle_seed=args.seed)
        self.split_metas=data.split_metas
        self.blocks=data.stacked_data.detach().clone().contiguous()
        if not torch.equal(self.blocks.reshape_as(self.original), self.original):
            raise RuntimeError(f"layout changed for {name}")
        self.mean,self.scale=compute_stage_norm_stats(self.blocks) if args.normalize_weight else (torch.zeros(1,1),torch.ones(1,1))
        self.normalized=apply_stage_norm(self.blocks,mean=self.mean,scale=self.scale)
        self.vae=MultiLayerVAE(vae_arguments(args)).to(args.device).train()
        self.optimizer=create_optimizer(self.vae.parameters(),self.vae.args,args.vae_learning_rate)
        self.scheduler=get_scheduler("linear",self.optimizer,num_warmup_steps=0,num_training_steps=args.steps)
        self.initial_state=state_digest(self.vae)
        self.log=(out/"training.jsonl").open("w",encoding="utf-8")
        self.start=time.perf_counter()
    def update(self, inputs: torch.Tensor, step: int):
        dev=torch.device(self.args.device)
        blocks=self.normalized.to(dev, dtype=torch.bfloat16 if self.args.vae_autocast_dtype=="bf16" and dev.type=="cuda" else torch.float32)
        target=self.original.to(dev)
        x=inputs.to(dev)
        self.optimizer.zero_grad(set_to_none=True)
        decoded,aux,_=full_vae_forward(self.vae,blocks,chunk_vectors=self.args.vae_chunk_vectors)
        restored=restore_stage_norm(decoded.float(),mean=self.mean.to(dev),scale=self.scale.to(dev))
        if self.args.objective=="linear_output_mse":
            main=output_mse(restored,target,x,use_triton=True)
        else:
            main=(restored.float()-blocks.float()).square().mean()
        loss=main*self.vae.model.l1_weight*self.vae.model.num_models+aux
        if not torch.isfinite(loss): raise RuntimeError(f"nonfinite {self.name} step {step}")
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(self.vae.parameters(),float("inf"),error_if_nonfinite=True)
        self.optimizer.step(); self.scheduler.step()
        item={"module":self.name,"stage":0,"step":step,"total_steps":self.args.steps,"objective":self.args.objective,"loss":float(loss.detach()),"main_loss":float(main.detach()),"auxiliary_loss":float(aux.detach()),"normalized_weight_mse":float((restored.float()-target.reshape_as(restored).float()).square().mean().detach()),"grad_norm":float(grad),"valid_tokens":len(x)}
        self.log.write(json.dumps(item)+"\n"); self.log.flush()
        del blocks,target,x,decoded,restored,main,aux,loss
        if dev.type=="cuda": torch.cuda.empty_cache()
    @torch.no_grad()
    def export(self):
        self.vae.eval(); decoded=[]
        dev=torch.device(self.args.device)
        for block in self.normalized.to(dev,dtype=torch.bfloat16 if self.args.vae_autocast_dtype=="bf16" and dev.type=="cuda" else torch.float32).split(self.args.vae_chunk_vectors):
            y,_=self.vae(block,is_train=False); decoded.append(y.float().cpu())
        stage=restore_stage_norm(torch.cat(decoded),mean=self.mean,scale=self.scale)
        expected=stage.reshape_as(self.original)
        decoder=self.vae.model.decoder.get_sub_decoder(0)
        _fuse_q_scale_into_decoder(decoder,q_scale=1/(self.args.codebook_bits**0.5) if self.args.new_quant else 1.0)
        _fuse_norm_into_decoder(decoder,mean=float(self.mean.item()),std=float(self.scale.item()))
        bits=[]
        for block in self.normalized.split(self.args.vae_chunk_vectors):
            _,b=self.vae(block.to(dev,dtype=torch.bfloat16 if self.args.vae_autocast_dtype=="bf16" and dev.type=="cuda" else torch.float32),is_train=False); bits.append(b.detach().cpu())
        payload={"format":"vaellm_group_vae_payload","version":1,"target_common_split_metas":self.split_metas,"parts_per_linear":1,"row_parts":1,"col_parts":1,"residual_stages":1,"all_stage_bits":[torch.cat(bits)],"all_stage_decoders":[[decoder.cpu()]],"all_stage_codebook_dims":[32],"all_stage_split_metas":[self.split_metas],"protected_channel_quant_format":"none","weight_rotation_specs":[None]}
        packed=self.out/"packed"; parity=export_linear(self.name,self.linear,payload,expected,packed,self.args)
        rec={"module":self.name,"shape":list(self.original.shape),"objective":self.args.objective,"transpose":False,"rotation":"none","protection":"none","steps":self.args.steps,"batch_size":self.args.batch_size,"stages":[{"stage":0,"steps":self.args.steps,"initial_state_sha256":self.initial_state,"normalization_mean":float(self.mean.item()),"normalization_scale":float(self.scale.item())}],"source_weight_sha256":tensor_digest(self.linear.weight),"elapsed_seconds":time.perf_counter()-self.start,**parity}
        dump_json(self.out/"record.json",rec); self.log.close(); return rec

class SharedCapture:
    def __init__(self,model,tok,bundle,args,names):
        self.model=model; self.args=args; self.names=names
        self.loader=DataLoader(bundle.train_dataset,batch_size=args.batch_size,shuffle=not bundle.is_iterable,drop_last=True,num_workers=0,generator=torch.Generator().manual_seed(args.data_seed),collate_fn=build_distill_data_collator(tok,model_max_length=args.model_max_length,dynamic_padding=args.dynamic_padding))
        self.it=iter(self.loader)
    def next(self):
        try: batch=next(self.it)
        except StopIteration: self.it=iter(self.loader); batch=next(self.it)
        mask=batch["attention_mask"].to(self.model.device).bool()
        caps={}
        handles=[]
        for name in self.names:
            layer=self.model.get_submodule(name)
            def hook(_m,inp,n=name):
                x=inp[0]
                if x.ndim!=3 or x.shape[:2]!=mask.shape: raise ValueError(f"bad input {n} {tuple(x.shape)}")
                caps[n]=x.detach()[mask].float().cpu().contiguous()
            handles.append(layer.register_forward_pre_hook(hook))
        try:
            with torch.no_grad(): self.model(input_ids=batch["input_ids"].to(self.model.device),attention_mask=mask,use_cache=False)
        finally:
            for h in handles: h.remove()
        if set(caps)!=set(self.names): raise RuntimeError("not all target linears executed")
        return caps

def select(model,args):
    if getattr(args,"target_linears",None):
        names=list(args.target_linears)
        for n in names:
            if not isinstance(model.get_submodule(n),nn.Linear):
                raise ValueError(f"not a Linear: {n}")
    else:
        cats={x.strip() for x in args.compression_categories.split(",") if x.strip()}
        names=[n for n,m in model.named_modules() if isinstance(m,nn.Linear) and n.rsplit(".",1)[-1] in cats]
    lo=getattr(args,"layer_start",None); hi=getattr(args,"layer_end",None)
    if lo is not None: names=[n for n in names if _module_layer(n)>=lo]
    if hi is not None: names=[n for n in names if _module_layer(n)<hi]
    if not names: raise ValueError("no target linears")
    return names

SharedTrainer = LinearTrainer

def run_shared(model,tok,args):
    validate(args); model.eval().requires_grad_(False); names=select(model,args)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=False)
    bundle=build_bundle(args,tok); trainers={}
    for i,n in enumerate(names):
        d=out/"linears"/n; d.mkdir(parents=True)
        trainers[n]=SharedTrainer(model.get_submodule(n),n,args,d)
        print(f"INIT {i+1}/{len(names)} {n}",flush=True)
    cap=SharedCapture(model,tok,bundle,args,names)
    manifest={"algorithm":"independent_linear_output_shared_teacher_sweep","status":"TRAINING","objective":args.objective,"modules":names,"steps":args.steps,"codebook_bits":args.codebook_bits,"codebook_dim":args.codebook_dim,"teacher_sweep":"one frozen teacher forward per calibration batch; current batch activations held only until all Linear updates finish","layer_range":[getattr(args,"layer_start",None),getattr(args,"layer_end",None)],"records":[]}
    manifest["config"]=vars(args).copy()
    dump_json(out/"manifest.json",manifest)
    for step in range(1,args.steps+1):
        xs=cap.next()
        for n in names: trainers[n].step(xs[n],step)
        if step==1 or step%args.log_every==0 or step==args.steps: print(json.dumps({"step":step,"modules":len(names)}),flush=True)
        del xs
    # No deployment export may start until every final VAE/optimizer is durable.
    finalize_training(trainers,args,manifest,out)
    model.cpu()
    return manifest

def main():
    p=parser()
    p.add_argument("--layer_start",type=int,default=None)
    p.add_argument("--layer_end",type=int,default=None)
    p.add_argument("--export_only",action="store_true",help="Export final saved VAE states without calibration or training.")
    p.add_argument("--training_state",type=Path,default=None,help="Defaults to OUTPUT_DIR/final_training_state.pt with --export_only.")
    args=p.parse_args()
    if args.training_state is not None and not args.export_only:
        p.error("--training_state requires --export_only")
    checkpoint=None
    if args.export_only:
        state_path=args.training_state or Path(args.output_dir)/"final_training_state.pt"
        checkpoint=load_final_training_state(state_path)
        output_dir,device=args.output_dir,args.device
        args=argparse.Namespace(**checkpoint["config"])
        args.output_dir,args.device=output_dir,device
    validate(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(4)
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM,AutoTokenizer
    root=Path(args.model_path)
    if not root.is_dir(): root=Path(snapshot_download(repo_id=args.model_path,local_files_only=True))
    args.model_path=str(root.resolve())
    if checkpoint is not None:
        model=AutoModelForCausalLM.from_pretrained(args.model_path,torch_dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True,low_cpu_mem_usage=True)
        export_saved_state(model,checkpoint,Path(args.output_dir),args.device)
        return
    tok=AutoTokenizer.from_pretrained(args.model_path,use_fast=True,local_files_only=True)
    if tok.pad_token_id is None: tok.add_special_tokens({"pad_token":tok.eos_token})
    tok.padding_side="right"
    model=AutoModelForCausalLM.from_pretrained(args.model_path,torch_dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True,low_cpu_mem_usage=True).to(args.device)
    run_shared(model,tok,args)

if __name__=="__main__": main()
