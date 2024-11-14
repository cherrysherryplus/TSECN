#!/bin/bash

# LOL-v1-real
command="python eval_o2mif_v2.py \
        --dataroot datasets/test \
        --which_dataset LOL-v1-real \
        --name  enlightening_wI3C_wTPE_wGray_woI3CAgain_wVggReLU51L1_wRandomSample \
        --model single \
        --which_direction AtoB \
        --which_epoch 100 \
        --no_dropout \
        --dataset_mode unaligned \
        --which_model_netG sid_unet_resize_v2 \
        --skip 1 \
        --use_norm 1 \
        --use_wgan 0 \
        --self_attention \
        --i3c \
        --tpe \
        --concat_gray \
        --times_residual \
        --instance_norm 0 \
        --resize_or_crop='no' \
        --gpu_ids 0"
        #--i3c_again(not use)

$command