#!/bin/bash

command="python eval_o2mif.py \
        --step 0.05
        --dataroot datasets/test \
        --which_dataset DICM \
        --name  enlightening_wI3C_wTPE_wGray_woI3CAgain_wVggReLU51L1 \
        --model single \
        --which_direction AtoB \
        --which_epoch 170 \
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