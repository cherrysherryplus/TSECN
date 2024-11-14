#!/bin/bash
# NOTE --i3c --tpe --concat_gray
# BEST: enlightening_wI3C_wTPE_wGray_woI3CAgain_wVggReLU51L1
base_command="python predict_eval.py \
        --dataroot datasets/test \
        --which_dataset LOL-v1-real \
        --name  enlightening_wI3C_wTPE_wGray_woI3CAgain_wVggReLU51L1_wRandomSample-1-1 \
        --model single \
        --which_direction AtoB \
        --no_dropout \
        --dataset_mode unaligned \
        --which_model_netG sid_unet_resize_v2 \
        --skip 1 \
        --use_norm 1 \
        --use_wgan 0 \
        --self_attention \
        --i3c \
        --i3c_again \
        --tpe \
        --concat_gray \
        --times_residual \
        --instance_norm 0 \
        --resize_or_crop='no' \
        --gpu_ids 0"
        #--i3c_again(not use)

# Check if which_epoch is specified
if [ -n "$1" ]; then
  # If which_epoch is specified, run only that epoch
  which_epoch=$1
  command="$base_command --which_epoch $which_epoch"
  echo "Running Epoch $which_epoch"
  $command
else
  # If which_epoch is not specified, loop over epochs with a step size of 20
  for (( epoch=200; epoch>=10; epoch-=10 )); do
    # Construct the full command with the current epoch
    command="$base_command --which_epoch $epoch"

    # Execute the command
    echo "Running Epoch $epoch"
    $command
  done
fi

echo "Finished processing epochs."