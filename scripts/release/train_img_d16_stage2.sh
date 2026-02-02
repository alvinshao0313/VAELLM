WORKER_GPU=8
NODE_NUM=4
NUM_WORKERS=12

if [[ "$*" == *"--debug"* ]]; then
    WORKER_GPU=1
    NODE_NUM=1
    NUM_WORKERS=0
fi

torchrun \
    --nproc_per_node=$WORKER_GPU \
    --nnodes=$NODE_NUM --master_addr=$WORKER_0_HOST \
    --node_rank=$NODE_ID --master_port=$PORT \
    train.py --num_workers $NUM_WORKERS \
    --patch_size 16 \
    --base_ch 128 --encoder_ch_mult 1 2 4 4 4 --decoder_ch_mult 1 2 4 4 4 \
    --codebook_dim 16 \
    --optim_type AdamW --lr 1e-4 --disable_sch --dis_lr_multiplier 1 --max_steps 200000 \
    --resolution 1024 1024 --batch_size 8 --dataset_list "openimages" --dataaug "resizecrop" \
    --disc_layers 3 --discriminator_iter_start 0 \
    --l1_weight 1 --perceptual_weight 1 --image_disc_weight 1 --image_gan_weight 0.3 --gan_feat_weight 0 --lfq_weight 4 \
    --codebook_size 65536 --entropy_loss_weight 0.1 --diversity_gamma 1 \
    --default_root_dir "bitvae_results/infinity_d16_stage2" --log_every 20 --ckpt_every 5000 --visu_every 5000 \
    --new_quant --lr_drop 150000 \
    --remove_residual_detach --use_lecam_reg_zero --base_ch_disc 128 --dis_lr_multiplier 2.0 --use_checkpoint \
    --schedule_mode "dense" --use_stochastic_depth --drop_rate 0.5 --keep_last_quant --tokenizer 'flux' --quantizer_type 'MultiScaleBSQ' \
    --pretrained "bitvae_results/infinity_d16_stage1/checkpoints/model_step_499999.ckpt" --not_load_optimizer --multiscale_training $@