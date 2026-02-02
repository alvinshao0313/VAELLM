import os
import argparse
import time
import logging
import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, get_scheduler, AutoTokenizer
from rotation.model_rotation import prepare_model, remove_rotate_hooks
from rotation.model_utils import get_model
import transformers
try:
    import wandb
except ImportError:
    wandb = None

from bitvae.utils.logger import create_logger

from bitvae.models.llm_vae import AutoEncoder, MultiLayerVAE
from bitvae.modules.loss import get_disc_loss, adopt_weight
from bitvae.utils.llm_arguments import LLMArgs
from bitvae.data.llm_dataset import WeightMatrixDataset
from bitvae.models.llm_discriminator import MLPDiscriminator

logger = logging.getLogger(__name__)


def create_optimizer(params, args, lr):
    opt_name = args.optimizer.lower()
    if opt_name == 'adam':
        return torch.optim.Adam(params, lr=lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif opt_name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif opt_name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=args.beta1, weight_decay=args.weight_decay)
    elif opt_name == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def main():
    parser = argparse.ArgumentParser()
    # 添加模型和LLM相关参数
    parser = AutoEncoder.add_model_specific_args(parser)
    parser = LLMArgs.add_llm_args(parser)

    args = parser.parse_args()

    transformers.set_seed(args.seed)

    # 设置日志
    os.makedirs(args.default_root_dir, exist_ok=True)
    logger = create_logger(args.default_root_dir)
    logger.info("Training arguments:")
    for k, v in sorted(vars(args).items()):
        logger.info(f"  {k}: {v}")

    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb not installed")
        os.environ["WANDB_SILENT"] = "true"  # 静默 wandb 的上传进度条和版本提示
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)

    device = torch.device(args.device)

    # 1. 加载模型，初始化权重
    logger.info(f"Loading LLM from {args.model_path}...")
    llm = get_model(args.model_path)
    if args.rotate:
        llm, _ = prepare_model(llm, args)

    # 2. 确定并行训练的目标层
    # 策略：
    #   外层循环：遍历模块类型 (q_proj, k_proj, etc.)
    #   内层循环：同时处理该类型下的 'parallel_layers' 个层

    # "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    transpose_modules = ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    logger.info(f"Target modules for compression: {target_modules}")
    logging.info(f"Transpose modules: {transpose_modules}")
    layers = getattr(llm.model, 'layers', [])
    num_llm_layers = len(layers)

    # 验证
    if args.parallel_layers > num_llm_layers:
        logger.warning(
            f"parallel_layers ({args.parallel_layers}) > num_llm_layers ({num_llm_layers}). Cap at {num_llm_layers}")
        args.parallel_layers = num_llm_layers

    # 按 'parallel_layers' 分批处理

    param_groups = []  # 权重列表: [[(L0.q, L1.q...), name], [(L0.k, L1.k...), name]]

    for mod_name in target_modules:
        # 收集所有层的同类权重
        group_weights = []
        group_names = []

        for i in range(num_llm_layers):
            if mod_name == "down_proj" and i == 1:
                continue
            layer = layers[i]
            # 查找模块 (e.g. self_attn.q_proj)
            found = False
            for n, m in layer.named_modules():
                if n.endswith(mod_name) and isinstance(m, torch.nn.Linear):
                    # FIX: Always append the Parameter itself, not a view/transpose.
                    # We handle transposition during data stacking.
                    group_weights.append(m.weight)

                    group_names.append(f"L{i}.{n}")
                    found = True
                    break
            if not found:
                logger.warning(f"Module {mod_name} not found in Layer {i}")

        # 分割成 batch
        for i in range(0, len(group_weights), args.parallel_layers):
            batch_w = group_weights[i: i + args.parallel_layers]
            batch_n = group_names[i: i + args.parallel_layers]
            if len(batch_w) == args.parallel_layers:
                param_groups.append((batch_w, batch_n, mod_name))
            else:
                # 处理剩余层
                if len(batch_w) > 0:
                    logger.warning(f"Remaining {len(batch_w)} layers for {mod_name} processed separately.")
                    param_groups.append((batch_w, batch_n, mod_name))

    current_global_step = 0

    # -------------------------------------------------------
    # 大循环：遍历处理每个模块组
    # -------------------------------------------------------
    for (weight_list, name_list, group_tag) in param_groups:
        num_current_group = len(weight_list)
        logger.info(f"=== Group: {group_tag} | Layers: {name_list[0]} ... {name_list[-1]} ({num_current_group}) ===")

        # 1. Preprocess & Stack Data
        # All weights in list are [Out, In]. We need [N_Chunks, Out_Dim].
        # Stack them: [num_models, Out, In] -> flatten -> [num_models, Total_Chunks, Chunk_Size]
        # Then transpose to [Total_Chunks, num_models, Chunk_Size] for DataLoader to iterate chunks

        orig_shapes = []

        # 为当前 batch 创建 MultiLayerVAE
        args.parallel_layers = num_current_group
        multi_vae = MultiLayerVAE(args).to(device)

        # 批处理权重 - 向量化预处理

        # 1. 堆叠原始权重
        # FIX: Handle transposition here
        if group_tag in transpose_modules:
            # m.weight is [Out, In]. We need [In, Out] for processing if transposed.
            stacked_w = torch.stack([w.detach().t().float() for w in weight_list]).cpu()
        else:
            stacked_w = torch.stack([w.detach().float() for w in weight_list]).cpu()

        orig_shapes = [w.shape for w in weight_list]

        # 展平 [32, N]
        stacked_flat = stacked_w.reshape(stacked_w.shape[0], -1)

        # 2. 计算统计量 [32, 1]
        d_mean = stacked_flat.mean(dim=1, keepdim=True)
        d_std = stacked_flat.std(dim=1, keepdim=True)

        # 3. 截断 (Clip)
        if args.clip_threshold > 0:
            limit = args.clip_threshold * d_std
            stacked_flat = torch.clamp(stacked_flat, min=d_mean - limit, max=d_mean + limit)
            # 重新计算统计量
            d_mean = stacked_flat.mean(dim=1, keepdim=True)
            d_std = stacked_flat.std(dim=1, keepdim=True)

        # 4. 归一化 (Normalize)
        if args.normalize_data:
            stacked_flat = (stacked_flat - d_mean) / (d_std + 1e-6)

        # 5. 维度检查：必须整除 codebook_dim
        numel = stacked_flat.shape[1]
        if numel % args.codebook_dim != 0:
            raise ValueError(
                f"Flattened elements ({numel}) not divisible by codebook_dim ({args.codebook_dim})."
            )

        # 6. 重塑形状
        # [32, Total_Elements] -> [32, N_Chunks, Chunk_Size] -> [N_Chunks, 32, Chunk_Size]
        stacked_data = stacked_flat.view(stacked_flat.shape[0], -1, args.codebook_dim).permute(1, 0, 2)

        # 直接从预处理后的 Tensor 创建 DataLoader
        # stacked_data: [N_Chunks, num_models, Chunk_Size]
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(stacked_data),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        # 准备评估 Loader (相同数据, 顺序, 全量Batch)
        eval_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(stacked_data),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        optimizer = create_optimizer(multi_vae.parameters(), args, args.lr)

        # Add Scheduler
        lr_scheduler = None
        if args.lr_scheduler != 'none':
            lr_scheduler = get_scheduler(
                name=args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=args.max_steps,
            )

        # GAN (Discriminator) 逻辑
        discriminator = None
        optimizer_d = None
        if args.image_gan_weight > 0:
            # 使用并行 Discriminator (每个模型有自己的判别器)
            discriminator = MLPDiscriminator(
                input_dim=args.codebook_dim,
                hidden_dim=args.disc_hidden_dim,
                num_models=num_current_group
            ).to(device)
            optimizer_d = create_optimizer(discriminator.parameters(), args, args.disc_lr)

        # 训练循环
        multi_vae.train()
        start_time = time.time()

        train_iter = iter(train_loader)
        for step in range(args.max_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # batch is list [Tensor], Tensor shape [B, num_models, C]
            x_batch = batch[0].to(device, non_blocking=True)

            # --- VAE 更新 ---
            optimizer.zero_grad()
            x_recon_train, _, loss_dict = multi_vae(x_batch)
            loss = loss_dict["loss"]

            # GAN 生成器 (Generator) Loss
            if discriminator is not None and step >= args.discriminator_iter_start:
                # 传入 [B, num_models, C]，由 Discriminator 内部处理并行
                logits_fake = discriminator(x_recon_train)

                # G Loss: 希望 logits_fake 像 Real
                g_weight = adopt_weight(current_global_step + step,
                                        args.discriminator_iter_start, 0.0, 0) * args.image_gan_weight

                if g_weight > 0:
                    g_loss_d = -torch.mean(logits_fake)
                    loss = loss + g_weight * g_loss_d
                    loss_dict["train/g_loss"] = g_loss_d.item()

            loss.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            # --- 判别器 (Discriminator) 更新 ---
            if discriminator is not None and step >= args.discriminator_iter_start:
                optimizer_d.zero_grad()

                with torch.no_grad():
                    fake_d = x_recon_train.detach()
                    real_d = x_batch

                logits_real_d = discriminator(real_d)
                logits_fake_d = discriminator(fake_d)

                # D Loss
                d_loss_fn = get_disc_loss(args.disc_loss_type)
                d_loss = d_loss_fn(logits_real_d, logits_fake_d)

                d_loss.backward()
                optimizer_d.step()

                loss_dict["train/d_loss"] = d_loss.item()
                loss_dict["train/logits_real"] = logits_real_d.mean().item()
                loss_dict["train/logits_fake"] = logits_fake_d.mean().item()

            if (step + 1) % args.log_every == 0:
                if args.use_wandb:
                    wandb.log(loss_dict, step=current_global_step + step + 1)

                iter_speed = (time.time() - start_time) / args.log_every

                # Construct detailed log message
                log_msg = f"[{group_tag}] Step {step+1}: "
                log_msg += f"Loss={loss.item():.5f} | "

                # Automatically log all other metrics in loss_dict
                for k, v in loss_dict.items():
                    if k == "loss":
                        continue

                    # helper to get float value
                    val = v.item() if hasattr(v, 'item') else v

                    # shorten key name, e.g. train/recon_loss -> recon_loss
                    short_k = k.replace("train/", "")

                    log_msg += f"{short_k}={val:.5f} | "

                log_msg += f"Speed: {iter_speed:.4f}s/it"

                logger.info(log_msg)
                start_time = time.time()

            # --- 评估循环：全量矩阵 MSE ---
            if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
                logger.info(f"[{group_tag}] Evaluating full matrix reconstruction at step {step+1}...")
                multi_vae.eval()

                # 1. 收集所有重建的 Chunks
                all_recon_chunks = []
                with torch.no_grad():
                    for batch in eval_loader:
                        x_in = batch[0].to(device)
                        x_recon = multi_vae(x_in, is_train=False)
                        all_recon_chunks.append(x_recon.cpu())

                # 2. 重新拼接
                full_recon = torch.cat(all_recon_chunks, dim=0)

                # 3. 维度变换 & 展平 [num_models, Total_Padded_Elements]
                full_recon_flat = full_recon.permute(1, 0, 2).reshape(num_current_group, -1)

                # 4. 逆归一化 (Un-normalize)
                if args.normalize_data:
                    # OOM Fix: 计算全量 MSE 时移至 CPU，避免挤爆显存
                    full_recon_flat = full_recon_flat.float().cpu()  # 确保是 float32
                    cur_mean = d_mean.cpu()
                    cur_std = d_std.cpu()
                    full_recon_flat = full_recon_flat * (cur_std + 1e-6) + cur_mean

                # 5. 截断到原始元素数量（无 padding 情况下等同于原长度）
                full_recon_valid = full_recon_flat[:, :numel]

                # 6. 与原始权重对比
                # stacked_w 在加载时已经是 CPU Tensor
                original_flat = stacked_w.view(num_current_group, -1).cpu()

                matrix_mse = F.mse_loss(full_recon_valid, original_flat)
                relative_l1 = F.l1_loss(full_recon_valid, original_flat, reduction='sum') / \
                    (original_flat.abs().sum() + 1e-10)

                logger.info(
                    f"[{group_tag}] Step {step+1} Full Matrix MSE: {matrix_mse.item():.4e}, Relative L1: {relative_l1.item():.4e}")
                if args.use_wandb:
                    wandb.log({"eval/matrix_mse": matrix_mse.item(),
                               "eval/relative_l1": relative_l1.item()
                               }, step=current_global_step + step + 1)

                multi_vae.train()
                # Clean up eval tensors
                del full_recon, full_recon_flat, full_recon_valid, original_flat
                torch.cuda.empty_cache()
        # -------------------------------------------------------
        # 重建 & 替换权重
        # -------------------------------------------------------
        logger.info(f"Reconstructing {group_tag}...")
        multi_vae.eval()

        # Inference Loader (Sequential)
        infer_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(stacked_data),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        all_recon = []
        with torch.no_grad():
            for batch in infer_loader:
                x_in = batch[0].to(device)
                # forward(..., is_train=False) -> [B, num_models, C]
                ret = multi_vae(x_in, is_train=False)
                if isinstance(ret, tuple):
                    recon_batch = ret[0]
                else:
                    recon_batch = ret

                all_recon.append(recon_batch.cpu())

        # 拼接
        full_recon = torch.cat(all_recon, dim=0)

        # 1. 维度还原 [num_models, Total_Chunks, C]
        full_recon = full_recon.permute(1, 0, 2)

        # 向量化后处理
        # 2. 展平
        full_recon_flat = full_recon.reshape(full_recon.shape[0], -1)

        # 3. 逆归一化
        if args.normalize_data:
            # 直接使用之前计算好的 d_mean 和 d_std，注意设备匹配
            means = d_mean.to(full_recon.device)
            stds = d_std.to(full_recon.device)
            full_recon_flat = full_recon_flat * (stds + 1e-6) + means

        # 4. 赋值回权重
        for i, weight_param in tqdm.tqdm(enumerate(weight_list), total=num_current_group, desc=f"Replacing weights for {group_tag}"):
            if group_tag in transpose_modules:
                # recon is [In, Out] (flattened).
                # We need to reshape to [In, Out] then transpose to [Out, In]
                # orig_shapes[i] is [Out, In]
                rows, cols = orig_shapes[i]  # [Out, In]
                recon_matrix = full_recon_flat[i].reshape(cols, rows).t()
            else:
                recon_matrix = full_recon_flat[i].reshape(orig_shapes[i])

            # 赋值
            weight_param.data = recon_matrix.to(weight_param.device, dtype=weight_param.dtype)

        logger.info(f"Replaced {num_current_group} layers for {group_tag}.")

        # Cleanup
        del multi_vae
        del stacked_data
        del optimizer
        torch.cuda.empty_cache()

        from llm_eval import calculate_ppl
        llm.eval().to(device)
        setattr(args, 'limit', -1)
        ppl = calculate_ppl(llm, args)
        llm.cpu()
        logger.info(f"After reconstruction PPL evaluated: {ppl}")

        # Update global step for next group
        current_global_step += args.max_steps

    # Save
    if args.default_root_dir:
        try:
            save_path = os.path.join(args.default_root_dir, f"compressed-{args.model_path.split('/')[-1]}")
            llm.save_pretrained(save_path)
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            tokenizer.save_pretrained(save_path)
            logger.info(f"Compressed model and tokenizer saved to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save compressed model and tokenizer: {e}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
