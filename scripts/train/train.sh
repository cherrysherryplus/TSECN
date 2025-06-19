# Since the training of GANs is unstable, it is common to get unsatisfactory results
# 1."experiment_name" should be replaced with a meaningful name for your experiment.
# 2."dataset_root" should be replaced with the path to your training dataset.
python tools/train.py \
		--dataroot dataset_root \
		--no_dropout \
		--name experiment_name \
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
		--batchSize 8 \
    	--lr 1e-4 \
		--lr_factor 1.0 \
        --self_attention \
		--cerm \
		--tpe \
		--use_norm 1 \
		--use_wgan 0 \
        --use_ragan \
        --hybrid_loss \
        --times_residual \
		--instance_norm 0 \
		--vgg 1.0 \
        --vgg_choose relu5_1 \
		--gpu_ids 0 \
		--save_epoch_freq 5 \
		--display_port=8097 \
		--niter 200 \
		--niter_decay 100
		# --tv 0.01
		