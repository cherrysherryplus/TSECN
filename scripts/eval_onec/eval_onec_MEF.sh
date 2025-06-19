#!/bin/bash
# MEF 35
command="python tools/eval_onec.py \
        --dataroot datasets/test \
        --which_dataset MEF \
        --name  TSECN \
        --save_dir_name TSECN \
        --model single \
        --which_direction AtoB \
        --no_dropout \
        --dataset_mode unaligned \
        --which_model_netG sid_unet_resize_v2 \
        --skip 1 \
        --use_norm 1 \
        --use_wgan 0 \
        --self_attention \
        --cerm \
        --tpe \
        --use_pdm True \
        --start 0 \
        --end 2.05 \
        --step 0.05 \
        --times_residual \
        --instance_norm 0 \
        --resize_or_crop='no' \
        --gpu_ids 0 \
        --which_dataset MEF \
        --which_epoch 35"


$command
