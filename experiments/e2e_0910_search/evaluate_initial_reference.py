import argparse
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
import torch

REPO = Path('/home/shaoyuantian/program/VAELLM')
sys.path.insert(0, str(REPO))
CANDIDATE = Path('/home/shaoyuantian/program/VAELLM-e2e-0910-20260924/litebsq/fused_multistage_decoder.py')
EXPECTED_HASH = 'ac09ece5e46bb71463e6e63de0eb2d05107db4f2bf48029ae2ba4b94f9cd7abd'
actual_hash = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
if actual_hash != EXPECTED_HASH:
    raise RuntimeError(f'Candidate changed: {actual_hash}')
spec = importlib.util.spec_from_file_location('e2e_initial_ref_candidate', CANDIDATE)
candidate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = candidate
spec.loader.exec_module(candidate)
import litebsq
sys.modules["litebsq.fused_multistage_decoder"] = candidate
litebsq.fused_multistage_decoder = candidate
import litebsq.vae_linear as live_linear
live_linear.fused_decode_packed_symmetric_decoder = candidate.fused_decode_packed_symmetric_decoder
live_linear.packed_symmetric_decoder_supports_fuse = candidate.packed_symmetric_decoder_supports_fuse
live_linear._TRITON_AVAILABLE = candidate._TRITON_AVAILABLE

from compressed_e2e_fintuning.mid_eval import run_e2e_lm_eval
from compressed_e2e_fintuning.runtime_v6 import _sync_model_padding_config
from e2e_common.data import build_tokenizer
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from train_utils.lora_utils import ensure_distill_process_group_initialized, get_distill_local_device, is_distill_main_process
from train_utils.v6_model_loader import load_v6_model_checkpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('initial_reference_fixed_decoder')
configure_e2e_determinism(False)
set_e2e_seed(0)
ensure_distill_process_group_initialized()
try:
    checkpoint = REPO / 'result/catlora/Qwen_Qwen3-8B_20260910_094022/final_model'
    output = REPO / 'result/compressed_e2e_fintuning/e2e_0910_search_20260924/initial_reference_fixed_decoder'
    model_path = 'Qwen/Qwen3-8B'
    device = get_distill_local_device(fallback='cuda')
    tokenizer = build_tokenizer(model_path)
    model, meta, _ = load_v6_model_checkpoint(str(checkpoint), map_location='cpu', strict=True, expected_kind='final_model')
    if meta['checkpoint_id'] != '01457bf3-ef22-49e8-847f-dc721287c2d6':
        raise RuntimeError(f'Unexpected initial checkpoint: {meta["checkpoint_id"]}')
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    _sync_model_padding_config(model, tokenizer)
    args = argparse.Namespace(
        eval_tasks='boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu',
        eval_num_fewshot=0, eval_lm_batch_size='auto', eval_lm_limit=None,
        eval_hif4_act=False, eval_device='cuda',
    )
    if is_distill_main_process():
        source_paths = ['compressed_e2e_fintuning/mid_eval.py', 'train_utils/eval_utils.py', 'train_utils/v6_model_loader.py', 'train_utils/checkpoint_v6.py', 'e2e_common/data.py', 'litebsq/vae_linear.py', 'litebsq/vae_linear_prewarm.py']
        config = dict(checkpoint=str(checkpoint), checkpoint_id=meta['checkpoint_id'], candidate_kernel=str(CANDIDATE), candidate_sha256=actual_hash, source_sha256={name: hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in source_paths}, entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), model_path=model_path, tokenizer_use_fast=True, seed=0, full_determinism=False, physical_gpus=[6,7], world_size=2, checkpoint_parameter_dtypes_preserved=True, cache_decoded_weight=False, torch_version=torch.__version__, **vars(args))
        (output/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        log.info('CONFIG %s', json.dumps(config, ensure_ascii=False))
    log.info('Loaded fixed initial checkpoint %s; device=%s; candidate=%s', meta['checkpoint_id'], device, actual_hash)
    result = run_e2e_lm_eval(model=model, tokenizer=tokenizer, args=args, base_model_path=model_path, output_dir=str(output), log=log, eval_tag='initial_reference_fixed_decoder', move_to_device=True, cache_decoded_weight=False)
    if is_distill_main_process():
        log.info('INITIAL_REFERENCE_COMPLETE')
finally:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
