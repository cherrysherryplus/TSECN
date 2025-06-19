import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
import random
import numpy as np
from pdb import set_trace as st
from . import networks
from . import loss_vgg
from . import loss_gan
from .loss_tv import tv_loss
from .loss_color import color_loss
from .loss_oa import OutlierAwareLoss


class SingleModel(BaseModel):
    def name(self):
        return 'SingleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A_gray = self.Tensor(nb, 1, size, size)

        if opt.vgg > 0:
            self.vgg_loss = loss_vgg.PerceptualLoss(opt)
            if self.opt.IN_vgg:
                self.vgg_patch_loss = loss_vgg.PerceptualLoss(opt)
                self.vgg_patch_loss.cuda()
            self.vgg_loss.cuda()
            self.vgg = loss_vgg.load_vgg16("./pretrained_weights", self.gpu_ids)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

        skip = True if opt.skip > 0 else False
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=skip, opt=opt)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, False)
            if self.opt.patchD:
                self.netD_P = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_patchD, opt.norm, use_sigmoid, self.gpu_ids, True)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                if self.opt.patchD:
                    self.load_network(self.netD_P, 'D_P', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            if opt.use_wgan:
                self.criterionGAN = loss_gan.DiscLossWGANGP()
            else:
                self.criterionGAN = loss_gan.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            if opt.use_mse:
                self.criterionCycle = torch.nn.MSELoss()
            else:
                self.criterionCycle = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # NOTE newly defined loss
            self.criterionOA = OutlierAwareLoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr * opt.lr_factor, betas=(opt.beta1, 0.999))
            # 0.01
            if self.opt.patchD:
                self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr * opt.lr_factor, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        if self.isTrain:
            networks.print_network(self.netD_A)
            if self.opt.patchD:
                networks.print_network(self.netD_P)
        if opt.isTrain:
            self.netG_A.train()
        else:
            self.netG_A.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_img = input['input_img']
        input_A_gray = input['A_gray']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A_gray.resize_(input_A_gray.size()).copy_(input_A_gray)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    
    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.real_A_gray = Variable(self.input_A_gray)
            if self.opt.noise > 0:
                self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
                self.real_A = self.real_A + self.noise
            if self.opt.input_linear:
                self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
            if self.opt.skip == 1:
                self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray)
            else:
                self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray)
            self.real_B = Variable(self.input_B)


    def predict(self, exposure_level=1.0):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.real_A_gray = Variable(self.input_A_gray)
            if self.opt.noise > 0:
                self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
                self.real_A = self.real_A + self.noise
            if self.opt.input_linear:
                self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
            if self.opt.skip == 1:
                if self.opt.tpe:
                    self.fake_B, self.latent_real_A, self.stage1, self.stage1_gray = self.netG_A.forward(self.real_A, self.real_A_gray)
                    gray = util.atten2im(self.real_A_gray.data)
                    self.fake_B += (exposure_level-1) * self.latent_real_A
                else:
                    self.fake_B, self.latent_real_A, self.real_A_gray = self.netG_A.forward(self.real_A, self.real_A_gray)
                    self.fake_B += (exposure_level-1) * self.latent_real_A
            else:
                self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray)
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            # only for latent visualization
            latent_real = util.tensor2im(self.latent_real_A.data)
            latent_show = util.latent2im(self.latent_real_A.data)
            # st()
            if self.opt.tpe:
                stage1 = util.tensor2im(self.stage1)
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('stage1', stage1), ('latent_real', latent_real), ('latent_show', latent_show), ('gray', gray)])
            else:
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])
            
    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake, use_ragan):
        # Real
        pred_real = netD.forward(real)
        pred_fake = netD.forward(fake.detach())
        if self.opt.use_wgan:
            loss_D_real = pred_real.mean()
            loss_D_fake = pred_fake.mean()
            loss_D = loss_D_fake - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD, 
                                                real.data, fake.data)
        elif self.opt.use_ragan and use_ragan:
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B = self.fake_B
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, True)
        self.loss_D_A.backward()
    
    def backward_D_P(self):
        if self.opt.hybrid_loss:
            loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch, self.fake_patch, False)
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P += self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i], False)
                self.loss_D_P = loss_D_P/float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P = loss_D_P
        else:
            loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch, self.fake_patch, True)
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P += self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i], True)
                self.loss_D_P = loss_D_P/float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P = loss_D_P
        if self.opt.D_P_times2:
            self.loss_D_P = self.loss_D_P*2
        self.loss_D_P.backward()
        
    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_A_gray = Variable(self.input_A_gray)
        self.real_img = Variable(self.input_img)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        if self.opt.skip == 1:
            if self.opt.tpe:
                self.fake_B, self.latent_real_A, self.stage1, self.gray = self.netG_A.forward(self.real_img, self.real_A_gray)
            else:
                self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_img, self.real_A_gray)
        else:
            self.fake_B = self.netG_A.forward(self.real_img, self.real_A_gray)
        if self.opt.patchD:
            w = self.real_A.size(3)
            h = self.real_A.size(2)
            w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))

            self.fake_patch = self.fake_B[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]
            self.real_patch = self.real_B[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]
            self.input_patch = self.real_A[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]
        if self.opt.patchD_3 > 0:
            self.fake_patch_1 = []
            self.real_patch_1 = []
            self.input_patch_1 = []
            w = self.real_A.size(3)
            h = self.real_A.size(2)
            for i in range(self.opt.patchD_3):
                w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
                h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
                self.fake_patch_1.append(self.fake_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])
                self.real_patch_1.append(self.real_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])
                self.input_patch_1.append(self.real_A[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])

    def backward_G(self, epoch):
        pred_fake = self.netD_A.forward(self.fake_B)
        if self.opt.use_wgan:
            self.loss_G_A = -pred_fake.mean()
        elif self.opt.use_ragan:
            pred_real = self.netD_A.forward(self.real_B)

            self.loss_G_A = (self.criterionGAN(pred_real - torch.mean(pred_fake), False) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), True)) / 2
            
        else:
            self.loss_G_A = self.criterionGAN(pred_fake, True)
        
        loss_G_A = 0
        if self.opt.patchD:
            pred_fake_patch = self.netD_P.forward(self.fake_patch)
            if self.opt.hybrid_loss:
                loss_G_A += self.criterionGAN(pred_fake_patch, True)
            else:
                pred_real_patch = self.netD_P.forward(self.real_patch)
                
                loss_G_A += (self.criterionGAN(pred_real_patch - torch.mean(pred_fake_patch), False) +
                                      self.criterionGAN(pred_fake_patch - torch.mean(pred_real_patch), True)) / 2
        if self.opt.patchD_3 > 0:   
            for i in range(self.opt.patchD_3):
                pred_fake_patch_1 = self.netD_P.forward(self.fake_patch_1[i])
                if self.opt.hybrid_loss:
                    loss_G_A += self.criterionGAN(pred_fake_patch_1, True)
                else:
                    pred_real_patch_1 = self.netD_P.forward(self.real_patch_1[i])
                    
                    loss_G_A += (self.criterionGAN(pred_real_patch_1 - torch.mean(pred_fake_patch_1), False) +
                                        self.criterionGAN(pred_fake_patch_1 - torch.mean(pred_real_patch_1), True)) / 2
                    
            if not self.opt.D_P_times2:
                self.loss_G_A += loss_G_A/float(self.opt.patchD_3 + 1)
            else:
                self.loss_G_A += loss_G_A/float(self.opt.patchD_3 + 1)*2
        else:
            if not self.opt.D_P_times2:
                self.loss_G_A += loss_G_A
            else:
                self.loss_G_A += loss_G_A*2
                
        if epoch < 0:
            vgg_w = 0
        else:
            vgg_w = 1
        if self.opt.vgg > 0:
            self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg, 
                    self.fake_B, self.real_A) * self.opt.vgg if self.opt.vgg > 0 else 0
            if self.opt.patch_vgg:
                if not self.opt.IN_vgg:
                    loss_vgg_patch = self.vgg_loss.compute_vgg_loss(self.vgg, 
                    self.fake_patch, self.input_patch) * self.opt.vgg
                else: 
                    loss_vgg_patch = self.vgg_patch_loss.compute_vgg_loss(self.vgg, 
                    self.fake_patch, self.input_patch) * self.opt.vgg
                if self.opt.patchD_3 > 0:
                    for i in range(self.opt.patchD_3):
                        if not self.opt.IN_vgg:
                            loss_vgg_patch += self.vgg_loss.compute_vgg_loss(self.vgg, 
                                self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                        else:
                            loss_vgg_patch += self.vgg_patch_loss.compute_vgg_loss(self.vgg, 
                                self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                    self.loss_vgg_b += loss_vgg_patch/float(self.opt.patchD_3 + 1)
                else:
                    self.loss_vgg_b += loss_vgg_patch
            self.loss_G = self.loss_G_A + self.loss_vgg_b*vgg_w
        # NOTE newly defined losses
        if self.opt.tv > 0:
            self.loss_tv = tv_loss(self.fake_B) + tv_loss(self.stage1)
            for i in range(self.opt.patchD_3):
                self.loss_tv += tv_loss(self.fake_patch_1[i]) * 0.2
            self.loss_G = self.loss_G + self.opt.tv * self.loss_tv
        if self.opt.color > 0:
            self.loss_color = color_loss(self.fake_B) + color_loss(self.stage1)
            for i in range(self.opt.patchD_3):
                self.loss_color += color_loss(self.fake_patch_1[i]) * 0.2
            self.loss_G = self.loss_G + self.opt.color * self.loss_color
        if self.opt.oa > 0:
            self.loss_oa = self.criterionOA(self.fake_B, self.real_B) + self.criterionOA(self.stage1, self.real_B)
            for i in range(self.opt.patchD_3):
                self.loss_oa += self.criterionOA(self.fake_patch_1[i], self.real_patch_1[i]) * 0.2
            self.loss_G = self.loss_G + self.opt.oa * self.loss_oa
        self.loss_G.backward()

    def optimize_parameters(self, epoch):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G(epoch)
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        if not self.opt.patchD:
            self.optimizer_D_A.step()
        else:
            # self.forward()
            self.optimizer_D_P.zero_grad()
            self.backward_D_P()
            self.optimizer_D_A.step()
            self.optimizer_D_P.step()


    def get_current_errors(self, epoch):
        # first stage
        
        # second stage
        D_A = self.loss_D_A.item()
        D_P = self.loss_D_P.item() if self.opt.patchD else 0
        G_A = self.loss_G_A.item()
        
        loss_dict = OrderedDict([('D_A', D_A), ('G_A', G_A), ("D_P", D_P)])
        # NOTE newly defined losses
        if self.opt.tv > 0:
            tv = self.loss_tv.data.item() if self.opt.tv > 0 else 0
            loss_dict['tv'] = tv * self.opt.tv
        if self.opt.color > 0:
            color = self.loss_color.data.item() if self.opt.color > 0 else 0
            loss_dict['color'] = color * self.opt.color
        if self.opt.oa > 0:
            oa = self.loss_oa.data.item() if self.opt.oa > 0 else 0
            loss_dict['oa'] = oa * self.opt.oa
        if self.opt.vgg > 0:
            vgg = self.loss_vgg_b.data.item() if self.opt.vgg > 0 else 0
            loss_dict['vgg'] = vgg * self.opt.vgg
        return loss_dict
        

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        if self.opt.tpe:
            # stage1
            stage1 = util.tensor2im(self.stage1.data)
        if self.opt.skip > 0:
            latent_real_A = util.tensor2im(self.latent_real_A.data)
            latent_show = util.latent2im(self.latent_real_A.data)
            if self.opt.patchD:
                fake_patch = util.tensor2im(self.fake_patch.data)
                real_patch = util.tensor2im(self.real_patch.data)
                if self.opt.patch_vgg:
                    input_patch = util.tensor2im(self.input_patch.data)
                    if not self.opt.self_attention:
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch), ('input_patch', input_patch)])
                    else:
                        self_attention = util.atten2im(self.real_A_gray.data)
                        if self.opt.tpe:
                            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                    ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                    ('fake_patch', fake_patch), ('input_patch', input_patch), ('self_attention', self_attention), ('stage1',stage1)])
                        else:
                            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                    ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                    ('fake_patch', fake_patch), ('input_patch', input_patch), ('self_attention', self_attention)])
                else:
                    if not self.opt.self_attention:
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch)])
                    else:
                        self_attention = util.atten2im(self.real_A_gray.data)
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch), ('self_attention', self_attention)])
            else:
                if not self.opt.self_attention:
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B)])
                else:
                    self_attention = util.atten2im(self.real_A_gray.data)
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
                                    ('latent_real_A', latent_real_A), ('latent_show', latent_show),
                                    ('self_attention', self_attention)])
        else:
            if not self.opt.self_attention:
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])
            else:
                self_attention = util.atten2im(self.real_A_gray.data)
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
                                    ('self_attention', self_attention)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        if self.opt.patchD:
            self.save_network(self.netD_P, 'D_P', label, self.gpu_ids)

    def update_learning_rate(self):
        
        if self.opt.new_lr:
            lr = self.old_lr/2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr * self.opt.lr_factor
        if self.opt.patchD:
            for param_group in self.optimizer_D_P.param_groups:
                param_group['lr'] = lr * self.opt.lr_factor
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
