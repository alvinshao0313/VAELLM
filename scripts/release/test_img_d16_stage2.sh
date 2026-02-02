python3 eval.py --tokenizer 'flux' --inference_type 'image' --patch_size 16 \
    --base_ch 128 --encoder_ch_mult 1 2 4 4 4 --decoder_ch_mult 1 2 4 4 4 \
    --codebook_dim 16 --codebook_size 65536 \
    --vqgan_ckpt "bitvae_results/infinity_d16_stage2/checkpoints/model_step_199999.ckpt" \
    --batch_size 1 --dataset_list "imagenet" --save ./imagenet_256_dim16_stage2 --dataaug "resizecrop" --resolution 256 256 --num_workers 0 \
    --default_root_dir "test" --save_prediction --quantizer_type 'MultiScaleBSQ' \
    --new_quant --schedule_mode "dynamic" --remove_residual_detach --disable_codebook_usage \
    $@

python3 eval.py --tokenizer 'flux' --inference_type 'image' --patch_size 16 \
    --base_ch 128 --encoder_ch_mult 1 2 4 4 4 --decoder_ch_mult 1 2 4 4 4 \
    --codebook_dim 16 --codebook_size 65536 \
    --vqgan_ckpt "bitvae_results/infinity_d16_stage2/checkpoints/model_step_199999.ckpt" \
    --batch_size 1 --dataset_list "imagenet" --save ./imagenet_512_dim16_stage2 --dataaug "resizecrop" --resolution 512 512 --num_workers 0 \
    --default_root_dir "test" --save_prediction --quantizer_type 'MultiScaleBSQ' \
    --new_quant --schedule_mode "dynamic" --remove_residual_detach --disable_codebook_usage \
    $@
