import os
import tqdm
import json
import re
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import torch.nn as nn

import lpips
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch.distributed as dist
from torch.multiprocessing import spawn
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

from bitvae.data import ImageData
from bitvae.utils.arguments import MainArgs, add_model_specific_args
from bitvae.evaluation import calculate_frechet_distance
from bitvae.evaluation import InceptionV3

torch.set_num_threads(32)


def calculate_batch_codebook_usage_percentage_bit(batch_encoding_indices):
    if isinstance(batch_encoding_indices, list):
        all_indices = []
        for one_encoding_indices in batch_encoding_indices:
            all_indices.append(one_encoding_indices.flatten(0, -2)) # [bhw, d]
        all_indices = torch.cat(all_indices, dim=0) # [sigma(bhw), d]
    else:
        # Flatten the batch of encoding indices into a single 1D tensor
        raise NotImplementedError
    all_indices = all_indices.detach().cpu()

    codebook_usage = torch.sum(all_indices, dim=0) # (d, )
    
    return codebook_usage, len(all_indices), all_indices.numpy()

def default_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqgan_ckpt', type=str, default=None)
    parser.add_argument('--inference_type', type=str, choices=["image"])
    parser.add_argument('--save', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda", choices=["cpu", "cuda"])
    parser = MainArgs.add_main_args(parser)
    parser = ImageData.add_data_specific_args(parser)
    args, unknown = parser.parse_known_args()
    args, parser, d_vae_model = add_model_specific_args(args, parser)
    args = parser.parse_args()
    return args, d_vae_model


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main():
    args, d_vae_model = default_parse_args()
    os.makedirs(args.default_root_dir, exist_ok=True)

    # init resolution
    args.resolution = (args.resolution[0], args.resolution[0]) if len(args.resolution) == 1 else args.resolution # init resolution

    d_vae = None
    num_codes = None
    
    if args.tokenizer in ["flux"]:
        print('args: ',args)
        d_vae = d_vae_model(args)
        num_codes = args.codebook_size
        state_dict = torch.load(args.vqgan_ckpt, map_location=torch.device("cpu"), weights_only=True)
        d_vae.load_state_dict(state_dict["vae"])
    else:
        raise NotImplementedError


    world_size = 1 if args.debug else torch.cuda.device_count()
    manager = torch.multiprocessing.Manager()
    return_dict = manager.dict()

    if args.debug:
        inference_eval(0, world_size, args, d_vae_model, d_vae, num_codes, return_dict)
    else:
        spawn(inference_eval, args=(world_size, args, d_vae_model, d_vae, num_codes, return_dict), nprocs=world_size, join=True)

    pred_xs, pred_recs, lpips_alex, lpips_vgg, ssim_value, psnr_value, num_iter, total_usage, total_usage_bit, total_num_token, all_bit_indices_cat = [], [], 0, 0, 0, 0, 0, 0, 0, 0, []
    for rank in range(world_size):
        pred_xs.append(return_dict[rank]['pred_xs'])
        pred_recs.append(return_dict[rank]['pred_recs'])
        lpips_alex += return_dict[rank]['lpips_alex']
        lpips_vgg += return_dict[rank]['lpips_vgg']
        ssim_value += return_dict[rank]['ssim_value']
        psnr_value += return_dict[rank]['psnr_value']
        num_iter += return_dict[rank]['num_iter']
        total_usage += return_dict[rank]['total_usage']
        if not args.disable_codebook_usage_bit:
            total_usage_bit += return_dict[rank]['total_usage_bit']
            total_num_token += return_dict[rank]['total_num_token']
            all_bit_indices_cat.append(return_dict[rank]['all_bit_indices_cat'])
    pred_xs = np.concatenate(pred_xs, 0)
    pred_recs = np.concatenate(pred_recs, 0)

    result_str = image_eval(pred_xs, pred_recs, lpips_alex, lpips_vgg, ssim_value, psnr_value, num_iter, total_usage, num_codes, total_usage_bit, total_num_token)
    print(result_str)
    # save result_str to exp_dir
    if args.tokenizer == "flux":
        basename = os.path.basename(args.vqgan_ckpt)
        match = re.search(r'model_step_(\d+)\.ckpt', basename)
        iter_num = match.group(1) if match else None
        ckpt_dir = os.path.dirname(args.vqgan_ckpt)
        save_dir = os.path.join(ckpt_dir, "evaluation")
        os.makedirs(save_dir, exist_ok=True)
        if args.random_flip:
            flip_prob = int(args.flip_prob * 10)
            result_name = os.path.join(save_dir, f"result_{args.dataset_list}_{iter_num}_{args.schedule_mode}_{args.resolution}_max_flip_lvl_{args.max_flip_lvl}_flip_prob_{flip_prob}.txt")
        elif args.random_flip_1lvl:
            result_name = os.path.join(save_dir, f"result_{args.dataset_list}_{iter_num}_{args.schedule_mode}_flip_lvl_{args.flip_lvl_idx}.txt")
        elif args.drop_when_test:
            result_name = os.path.join(save_dir, f"result_{args.dataset_list}_{iter_num}_{args.schedule_mode}_drop_lvl_idx_{args.drop_lvl_idx}_drop_lvl_num_{args.drop_lvl_num}.txt")
        else:
            result_name = os.path.join(save_dir, f"result_{args.dataset_list}_{iter_num}_{args.schedule_mode}_{args.resolution}.txt")
    else:
        raise NotImplementedError
    with open(result_name, "w") as f:
        f.write(result_str)
    # print('Usage = %.2f'%((total_usage > 0.).sum() / num_codes))

def inference_eval(rank, world_size, args, d_vae_model, d_vae, num_codes, return_dict):
    # Don't remove this setup!!! dist.init_process_group is important for building loader (data.distributed.DistributedSampler)
    setup(rank, world_size) 

    device = torch.device(f"cuda:{rank}")

    for param in d_vae.parameters():
        param.requires_grad = False
    d_vae.to(device).eval()

    save_dir = 'results/%s'%(args.save)
    print('generating and saving image to %s...'%save_dir)
    os.makedirs(save_dir, exist_ok=True)

    data = ImageData(args)

    loader = data.val_dataloader()

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx]).to(device)
    inception_model.eval()

    loader_iter = iter(loader)

    pred_xs = []
    pred_recs = []
    all_bit_indices_cat = []
    # LPIPS score related
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)   # closer to "traditional" perceptual loss, when used for optimization
    lpips_alex = 0.0
    lpips_vgg = 0.0

    # SSIM score related
    ssim_value = 0.0

    # PSNR score related
    psnr_value = 0.0
    
    num_images = len(loader)
    print(f"Testing {num_images} files")
    num_iter = 0

    total_usage = 0.0
    total_usage_bit = 0.0
    total_num_token = 0

    for batch_idx in tqdm(range(num_images)):
        batch = next(loader_iter)
            
        with torch.no_grad():
            x = batch['image']
            if args.tokenizer in ["flux"]:
                torch.cuda.empty_cache()
                # x: [-1, 1]
                x_recons, vq_output = d_vae(x.to(device), 2, 0, is_train=False)
                x_recons = x_recons.cpu()
            else:
                raise NotImplementedError

        if not args.disable_codebook_usage_bit:
            bit_indices = vq_output["bit_encodings"]
            codebook_usage_bit, num_token, bit_indices_cat = calculate_batch_codebook_usage_percentage_bit(bit_indices)
            total_usage_bit += codebook_usage_bit
            total_num_token += num_token
            all_bit_indices_cat.append(bit_indices_cat)
        
        paths = batch["path"]
        assert len(paths) == x.shape[0]

        for p, input_ori, recon_ori in zip(paths, x, x_recons):
            path = os.path.join(save_dir, "input_recon", os.path.basename(p))
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            
            input_ori = input_ori.unsqueeze(0).to(device)
            input_ = (input_ori + 1) / 2 # [-1, 1] -> [0, 1]
            
            pred_x = inception_model(input_)[0]
            pred_x = pred_x.squeeze(3).squeeze(2).cpu().numpy()

            recon_ori = recon_ori.unsqueeze(0).to(device)
            recon_ = (recon_ori + 1) / 2 # [-1, 1] -> [0, 1]
            # recon_ = recon_.permute(1, 2, 0).detach().cpu()
            with torch.no_grad():
                pred_rec = inception_model(recon_)[0]
            pred_rec = pred_rec.squeeze(3).squeeze(2).cpu().numpy()

            pred_xs.append(pred_x)
            pred_recs.append(pred_rec)

            # calculate lpips
            with torch.no_grad():
                lpips_alex += loss_fn_alex(input_ori, recon_ori).sum() # [-1, 1]
                lpips_vgg += loss_fn_vgg(input_ori, recon_ori).sum() # [-1, 1]

            #calculate PSNR and SSIM
            rgb_restored = (recon_ * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rgb_gt = (input_ * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rgb_restored = rgb_restored.astype(np.float32) / 255.
            rgb_gt = rgb_gt.astype(np.float32) / 255.
            ssim_temp = 0
            psnr_temp = 0
            B, _, _, _ = rgb_restored.shape
            for i in range(B):
                rgb_restored_s, rgb_gt_s = rgb_restored[i], rgb_gt[i]
                with torch.no_grad():
                    ssim_temp += ssim_loss(rgb_restored_s, rgb_gt_s, data_range=1.0, channel_axis=-1)
                    psnr_temp += psnr_loss(rgb_gt, rgb_restored)
            ssim_value += ssim_temp / B
            psnr_value += psnr_temp / B
            num_iter += 1
        
    pred_xs = np.concatenate(pred_xs, axis=0)
    pred_recs = np.concatenate(pred_recs, axis=0)
    temp_dict = {
        'pred_xs':pred_xs,
        'pred_recs':pred_recs,
        'lpips_alex':lpips_alex.cpu(),
        'lpips_vgg':lpips_vgg.cpu(),
        'ssim_value': ssim_value,
        'psnr_value': psnr_value,
        'num_iter': num_iter,
        'total_usage': total_usage,
        'total_usage_bit': total_usage_bit,
        'total_num_token': total_num_token,
    }
    if not args.disable_codebook_usage_bit:
        all_bit_indices_cat = np.concatenate(all_bit_indices_cat, axis=0)
        temp_dict['all_bit_indices_cat'] = all_bit_indices_cat
    return_dict[rank] = temp_dict

    if dist.is_initialized():
        dist.barrier()
    cleanup()

def image_eval(pred_xs, pred_recs, lpips_alex, lpips_vgg, ssim_value, psnr_value, num_iter, total_usage, num_codes, total_usage_bit, total_num_token):
    mu_x = np.mean(pred_xs, axis=0)
    sigma_x = np.cov(pred_xs, rowvar=False)
    mu_rec = np.mean(pred_recs, axis=0)
    sigma_rec = np.cov(pred_recs, rowvar=False)
    
    fid_value = calculate_frechet_distance(mu_x, sigma_x, mu_rec, sigma_rec)
    lpips_alex_value = lpips_alex / num_iter
    lpips_vgg_value = lpips_vgg / num_iter
    ssim_value = ssim_value / num_iter
    psnr_value = psnr_value / num_iter
    if total_num_token != 0:
        bit_distribution = total_usage_bit / total_num_token
        bit_distribution_str = '\n'.join(f'{value:.4f}' for value in bit_distribution)

    # usage_0 = (total_usage > 0.).sum() / num_codes * 100
    # usage_10 = (total_usage > 10.).sum() / num_codes * 100

    result_str = f"""
    FID = {fid_value:.4f}
    LPIPS_VGG: {lpips_vgg_value.item():.4f}
    LPIPS_ALEX: {lpips_alex_value.item():.4f}
    SSIM: {ssim_value:.4f}
    PSNR: {psnr_value:.3f}
    """
    if total_num_token != 0:
        result_str += f"""
        Bit_Distribution: {bit_distribution_str}
        """
    # Usage(>0): {usage_0:.2f}%
    # Usage(>10): {usage_10:.2f}%
    return result_str
if __name__ == '__main__':
    main()