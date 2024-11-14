1.use predict_eval.sh or predict_eval.py to predict and evaluate in one go. The parameters that need specification are:
--which_dataset <DATASET_NAME>
--name <EXPERIMENT_NAME>
--which_epoch <EPOCH_NUM or 'latest'>
--i3c   #whether use i3c
--tpe   #whether use tpe
--concat_gray   #whether concat gray in the 2nd stage
--i3c_again     #whether update illumination constraint

2.argparse usage
--which_model_netD no_norm_4    :   WHICH discriminator to use
--patchD \                      :   WHETHER use the patch discriminator (local discriminator)
--patch_vgg \                   :   WHETHER use vgg loss between each patch (ONLY function if self.opt.vgg is TRUE and self.opt.patchD is TRUE)
--patchD_3 5 \                  :   # image patches for local discriminator
--n_layers_D 5 \                :   # layers of global discriminator
--n_layers_patchD 4 \           :   # layers of local discriminator
--fineSize 320 \                :   Image size
--patchSize 64 \                :   Image size of patch
--skip 1 \                      :   WHETHER use skip connection in the ouput; B = net.forward(A) + skip*A
--batchSize 8 \                 :   batch size
--lr 1e-4 \                     :   lr of generator and discriminator
--self_attention \              :   WHETHER use illumination constraint in EnlightenGAN
--i3c \                         :   WHETHER use I3C
--tpe \                         :   WHETHER use TPE
--concat_gray \                 :   WHETHER concat gray with the first stage output in the second stage
--use_norm 1 \                  :   WHETHER use BatchNorm in generator
--use_wgan 0 \                  :   use wgan
--use_ragan \                   :   use ragan
--hybrid_loss \                 :   use lsgan and ragan separately
--times_residual \              :   output = input (low-light image) + residual (latent image?) * attention (illumination constraint) [latent = latent*gray]
--instance_norm 0 \             :   use instance normalization (NEVER USED!!!)
--vgg 1 \                       :   use perceptrual loss
--vgg_choose relu5_1 \          :   WHICH layers' features are used for perceptrual loss computation
--gpu_ids 0 \                   :   gpus usage
--save_epoch_freq 10 \          :   save checkpoint epoch intervals


3. "resize_and_crop" is "crop" during training with the crop size equal to "opt.fineSize", "no" during inference

4. In fact, the loss_D_P uses 6 crops (if patchD_3 is set to 5) [self.loss_D_P = loss_D_P/float(self.opt.patchD_3 + 1)]

5. 把networks.py里面的stage2的输入改了一下，之前input被改了，导致最后stage2输出的时候，其实没有用到最原始的lowlight输入

6. visdom ssh穿透命令
ssh -CNgv -L 8097:127.0.0.1:8097 -p 40192 root@region-42.seetacloud.com

7. Currently, all exps are based on "enlightening_wI3C_wTPE_wGray_wI3CAgain_wVggReLU51L1", so the naming of exps only highlight configs different
from the base one. For example:
"enlightening_wI3C-GradMultiply_wTPE_wGray_wI3CAgain_wVggReLU51L1" in train_wI3C-GradMultiply => Baseline + I3C-GradMultiply



WIP
1. add gradient in the second stage input (concat original i3c + stage1 output + gradient of original i3c)
2. experiment 'not_use_stage1_as_residual' again
3. experiment o2mif with \theta in -1~1.0