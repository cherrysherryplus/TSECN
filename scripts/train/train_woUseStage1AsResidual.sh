# changelogs
# BEST: enlightening_wI3C_wTPE_wGray_woI3CAgain_wVggReLU51L1_wTVlossOnStage1
# enlightening_wI3C_wTPE_wGray_woUseStage1AsResidual_wVggReLU51L1	指定not_use_stage1_as_residual，修正stage1+illu*gray的公式问题
python train.py \
		--dataroot datasets/train/EnlightenGAN \
		--no_dropout \
		--name enlightening_wI3C_wTPE_wGray_woUseStage1AsResidual_wVggReLU51L1 \
		--model single \
		--dataset_mode unaligned \
		--which_model_netG sid_unet_resize_v2 \
        --which_model_netD no_norm_4 \
        --patchD \
        --patch_vgg \
        --patchD_3 5 \
        --n_layers_D 5 \
        --n_layers_patchD 4 \
		--fineSize 320 \
        --patchSize 64 \
		--skip 1 \
		--batchSize 4 \
    	--lr 1e-4 \
        --self_attention \
		--i3c \
		--tpe \
		--concat_gray \
		--not_use_stage1_as_residual \
		--use_norm 1 \
		--use_wgan 0 \
        --use_ragan \
        --hybrid_loss \
        --times_residual \
		--instance_norm 0 \
		--vgg 1 \
        --vgg_choose relu5_1 \
		--gpu_ids 0 \
		--save_epoch_freq 10 \
		--display_port=8097
		# --i3c_again \
		# --tv 1 \