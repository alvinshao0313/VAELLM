# Bitwise Visual Tokenizer
The training and inference code of bitwise tokenizer used by [Infinity](https://github.com/FoundationVision/Infinity).

### BitVAE Model ZOO
We provide Infinity models for you to play with, which are on <a href='https://huggingface.co/FoundationVision/infinity'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20weights-FoundationVision/Infinity-yellow'></a> or can be downloaded from the following links:

### Visual Tokenizer

|   vocabulary    | stride |   IN-256 rFID $\downarrow$    | IN-256 PSNR $\uparrow$ | IN-512 rFID $\downarrow$ | IN-512 PSNR $\uparrow$ | HF weights🤗                                                                        |
|:----------:|:-----:|:--------:|:---------:|:-------:|:-------:|:------------------------------------------------------------------------------------|
|  $V_d=2^{16}$   |  16  |   1.22   |  20.9   |    0.31    |  22.6   | [infinity_vae_d16.pth](https://huggingface.co/FoundationVision/infinity/blob/main/infinity_vae_d16.pth) |
|  $V_d=2^{24}$   |  16  |   0.75   |  22.0   |    0.30    |  23.5   | [infinity_vae_d24.pth](https://huggingface.co/FoundationVision/infinity/blob/main/infinity_vae_d24.pth) |
|  $V_d=2^{32}$   |  16  |   0.61   |  22.7   |    0.23    |  24.4   | [infinity_vae_d32.pth](https://huggingface.co/FoundationVision/infinity/blob/main/infinity_vae_d32.pth) |
|  $V_d=2^{64}$   |  16  |   0.33   |  24.9   |     0.15     |  26.4   | [infinity_vae_d64.pth](https://huggingface.co/FoundationVision/infinity/blob/main/infinity_vae_d64.pth) |
| $V_d=2^{32}$ |  16  | 0.75 |  21.9   |     0.32     |  23.6   | [infinity_vae_d32_reg.pth](https://huggingface.co/FoundationVision/Infinity/blob/main/infinity_vae_d32reg.pth) |

### Environment installation
```
bash scripts/prepare.sh
```

Download `checkpoints` and `labels` from [Google Drive](https://drive.google.com/drive/folders/15VCFUpcv1ktU7RR3Yw_LqI4vuR5Q2r0y?usp=sharing) and put them under the project folder. If you want to use our trained model weights, please also download `bitvae_results`.
We expect that the data is organized as below.
```
${PROJECT_ROOT}
    -- bitvae
    -- bitvae_results
        -- Infinity_d16_stage1
        -- Infinity_d16_stage2
        -- Infinity_d32_stage1
        -- Infinity_d32_stage2
    -- checkpoints
    -- labels
    -- scripts
    -- test
    ...
```


### Training
Before training, please generate a `labels/openimages/train.txt` according to our provided `labels/imagenet/val_example.txt`. please replace <REAL_PATH> with the real path on your system.

Tokenizer with hidden dimension 16
```
bash scripts/release/train_img_d16_stage1.sh # stage 1: single-scale pre-training
bash scripts/release/train_img_d16_stage2.sh # stage 2: multi-scale fine-tuning
```
Tokenizer with hidden dimension 32
```
bash scripts/release/train_img_d32_stage1.sh # stage 1: single-scale pre-training
bash scripts/release/train_img_d32_stage2.sh # stage 2: multi-scale fine-tuning
```

### Testing & evaluation
Before testing, please generate a `labels/imagenet/val.txt` according to our provided `labels/imagenet/val_example.txt`. please replace <REAL_PATH> with the real path on your system.

Tokenizer with hidden dimension 16
```
bash scripts/release/test_img_d16_stage1.sh
bash scripts/release/test_img_d16_stage2.sh
``` 
Tokenizer with hidden dimension 32
```
bash scripts/release/test_img_d32_stage1.sh
bash scripts/release/test_img_d32_stage2.sh
``` 
### 📖 Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:

```
@misc{Infinity,
    title={Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis}, 
    author={Jian Han and Jinlai Liu and Yi Jiang and Bin Yan and Yuqi Zhang and Zehuan Yuan and Bingyue Peng and Xiaobing Liu},
    year={2024},
    eprint={2412.04431},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2412.04431}, 
}
```

```
@misc{VAR,
      title={Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction}, 
      author={Keyu Tian and Yi Jiang and Zehuan Yuan and Bingyue Peng and Liwei Wang},
      year={2024},
      eprint={2404.02905},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.02905}, 
}
```

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
