import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from pdb import set_trace as st


###############################################################################
# Functions
###############################################################################
def pad_tensor(input):
    
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:,:, pad_top: height - pad_bottom, pad_left: width - pad_right]


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    return num_params


##############################################################################
# Discriminators
##############################################################################
def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[], patch=False):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
    if which_model_netD == 'no_norm':
        netD = NoNormDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'no_norm_4':
        netD = NoNormDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    if len(gpu_ids) >= 0:
        netD.cuda(device=gpu_ids[0])
        netD = torch.nn.DataParallel(netD, gpu_ids)
    netD.apply(weights_init)
    return netD


class NoNormDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, gpu_ids=[]):
        super(NoNormDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


##############################################################################
# Generators
##############################################################################
def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], skip=False, opt=None):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
    # Final network (+CERM+TPE): sid_unet_resize_v2
    if opt.which_model_netG == 'sid_unet_resize_v2':
        netG = Unet_resize_conv_v2(opt, skip)
    if len(gpu_ids) >= 0:
        netG.cuda(device=gpu_ids[0])
        netG = torch.nn.DataParallel(netG, gpu_ids)
    netG.apply(weights_init)
    return netG


##############################################################################
# Final network (+CERM+TPE): sid_unet_resize_v2
##############################################################################
class Unet_resize_conv_v2(nn.Module):
    def __init__(self, opt, skip):
        super(Unet_resize_conv_v2, self).__init__()

        self.opt = opt
        self.skip = skip
        p = 1
        
        # First stage
        if opt.tpe:
            dim = 64
            self.dgcn = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(4, dim, kernel_size=3, stride=1,bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(dim),
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1,bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(dim),
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1,bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(dim),
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, 3, kernel_size=3, stride=1,bias=True),
                nn.Sigmoid())
        if opt.self_attention:
            if not opt.concat_gray:
                self.conv1_1 = nn.Conv2d(3, 32, 3, padding=p)
            else:
                self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p)
            self.downsample_1 = nn.MaxPool2d(2)
            self.downsample_2 = nn.MaxPool2d(2)
            self.downsample_3 = nn.MaxPool2d(2)
            self.downsample_4 = nn.MaxPool2d(2)
        else:
            self.conv1_1 = nn.Conv2d(3, 32, 3, padding=p)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn1_2 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn2_2 = nn.BatchNorm2d(64)
        self.max_pool2 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn3_2 = nn.BatchNorm2d(128)
        self.max_pool3 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn4_2 = nn.BatchNorm2d(256)
        self.max_pool4 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn5_2 = nn.BatchNorm2d(512)

        self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn6_1 = nn.BatchNorm2d(256)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn6_2 = nn.BatchNorm2d(256)

        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn7_1 = nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn7_2 = nn.BatchNorm2d(128)

        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn8_1 = nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn8_2 = nn.BatchNorm2d(64)

        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn9_1 = nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv10 = nn.Conv2d(32, 3, 1)
        if self.opt.tanh:
            self.tanh = nn.Tanh()
            
    def forward(self, input, gray):
        flag = 0
        # if image too large, then downsample it first, then upsample it back
        if input.size()[3] > 2200:
            input = F.interpolate(input, scale_factor=0.5, mode='bilinear')
            gray = F.interpolate(gray, scale_factor=0.5, mode='bilinear')
            flag = 1
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        gray, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gray)
        if self.opt.tpe:
            # if tpe is activated, then first adopt DGCN
            A_map = self.dgcn(torch.cat((input, gray), 1))
            stage1 = torch.pow(input+1, A_map) - 1
            input = stage1.clone()
        if self.opt.self_attention:
            gray_2 = self.downsample_1(gray)
            gray_3 = self.downsample_2(gray_2)
            gray_4 = self.downsample_3(gray_3)
            gray_5 = self.downsample_4(gray_4)
        if self.opt.use_norm == 1:
            if self.opt.self_attention:
                if self.opt.concat_gray:
                    x = self.bn1_1(self.LReLU1_1(self.conv1_1(torch.cat((input, gray), 1))))
                else:
                    x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
            else:
                x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
            conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
            x = self.max_pool1(conv1)

            x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
            conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
            x = self.max_pool2(conv2)

            x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
            conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
            x = self.max_pool3(conv3)

            x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))
            conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))
            x = self.max_pool4(conv4)

            x = self.bn5_1(self.LReLU5_1(self.conv5_1(x)))
            x = x*gray_5 if self.opt.self_attention else x
            conv5 = self.bn5_2(self.LReLU5_2(self.conv5_2(x)))
            
            conv5 = F.upsample(conv5, scale_factor=2, mode='bilinear')
            conv4 = conv4*gray_4 if self.opt.self_attention else conv4
            up6 = torch.cat([self.deconv5(conv5), conv4], 1)
            x = self.bn6_1(self.LReLU6_1(self.conv6_1(up6)))
            conv6 = self.bn6_2(self.LReLU6_2(self.conv6_2(x)))

            conv6 = F.upsample(conv6, scale_factor=2, mode='bilinear')
            conv3 = conv3*gray_3 if self.opt.self_attention else conv3
            up7 = torch.cat([self.deconv6(conv6), conv3], 1)
            x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))
            conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))

            conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')
            conv2 = conv2*gray_2 if self.opt.self_attention else conv2
            up8 = torch.cat([self.deconv7(conv7), conv2], 1)
            x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
            conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))

            conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
            conv1 = conv1*gray if self.opt.self_attention else conv1
            up9 = torch.cat([self.deconv8(conv8), conv1], 1)
            x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
            conv9 = self.LReLU9_2(self.conv9_2(x))

            latent = self.conv10(conv9)

            if self.opt.times_residual:
                latent = latent*gray

            if self.opt.tanh:
                latent = self.tanh(latent)
            if self.skip:
                if self.opt.linear_add:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent))/(torch.max(latent)-torch.min(latent))
                    input = (input - torch.min(input))/(torch.max(input) - torch.min(input))
                    output = latent + input*self.opt.skip
                    output = output*2 - 1
                else:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent))/(torch.max(latent)-torch.min(latent))
                    output = latent + input*self.opt.skip
            else:
                output = latent

            if self.opt.linear:
                output = output/torch.max(torch.abs(output))
            
                
        elif self.opt.use_norm == 0:
            if self.opt.self_attention:
                x = self.LReLU1_1(self.conv1_1(torch.cat((input, gray), 1)))
            else:
                x = self.LReLU1_1(self.conv1_1(input))
            conv1 = self.LReLU1_2(self.conv1_2(x))
            x = self.max_pool1(conv1)

            x = self.LReLU2_1(self.conv2_1(x))
            conv2 = self.LReLU2_2(self.conv2_2(x))
            x = self.max_pool2(conv2)

            x = self.LReLU3_1(self.conv3_1(x))
            conv3 = self.LReLU3_2(self.conv3_2(x))
            x = self.max_pool3(conv3)

            x = self.LReLU4_1(self.conv4_1(x))
            conv4 = self.LReLU4_2(self.conv4_2(x))
            x = self.max_pool4(conv4)

            x = self.LReLU5_1(self.conv5_1(x))
            x = x*gray_5 if self.opt.self_attention else x
            conv5 = self.LReLU5_2(self.conv5_2(x))

            conv5 = F.upsample(conv5, scale_factor=2, mode='bilinear')
            conv4 = conv4*gray_4 if self.opt.self_attention else conv4
            up6 = torch.cat([self.deconv5(conv5), conv4], 1)
            x = self.LReLU6_1(self.conv6_1(up6))
            conv6 = self.LReLU6_2(self.conv6_2(x))

            conv6 = F.upsample(conv6, scale_factor=2, mode='bilinear')
            conv3 = conv3*gray_3 if self.opt.self_attention else conv3
            up7 = torch.cat([self.deconv6(conv6), conv3], 1)
            x = self.LReLU7_1(self.conv7_1(up7))
            conv7 = self.LReLU7_2(self.conv7_2(x))

            conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')
            conv2 = conv2*gray_2 if self.opt.self_attention else conv2
            up8 = torch.cat([self.deconv7(conv7), conv2], 1)
            x = self.LReLU8_1(self.conv8_1(up8))
            conv8 = self.LReLU8_2(self.conv8_2(x))

            conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
            conv1 = conv1*gray if self.opt.self_attention else conv1
            up9 = torch.cat([self.deconv8(conv8), conv1], 1)
            x = self.LReLU9_1(self.conv9_1(up9))
            conv9 = self.LReLU9_2(self.conv9_2(x))

            latent = self.conv10(conv9)
            
            if self.opt.times_residual:
                latent = latent*gray

            if self.opt.tanh:
                latent = self.tanh(latent)
            if self.skip:
                if self.opt.linear_add:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent))/(torch.max(latent)-torch.min(latent))
                    input = (input - torch.min(input))/(torch.max(input) - torch.min(input))
                    output = latent + input*self.opt.skip
                    output = output*2 - 1
                else:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent))/(torch.max(latent)-torch.min(latent))
                    output = latent + input*self.opt.skip
            else:
                output = latent

            if self.opt.linear:
                output = output/torch.max(torch.abs(output))
        
        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        gray = pad_tensor_back(gray, pad_left, pad_right, pad_top, pad_bottom)
        if self.opt.tpe:
            stage1 = pad_tensor_back(stage1, pad_left, pad_right, pad_top, pad_bottom)
        if flag == 1:
            output = F.upsample(output, scale_factor=2, mode='bilinear')
            gray = F.upsample(gray, scale_factor=2, mode='bilinear')
            latent = F.upsample(latent, scale_factor=2, mode='bilinear')
        if self.skip:
            if self.opt.tpe:
                return output, latent, stage1, gray
            else:
                return output, latent
        else:
            return output
        

##############################################################################
# Pre-Denoising Module (PDM)
##############################################################################
class PDM(nn.Module):
    def __init__(self, num=64):
        super(PDM, self).__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, 3, 1, 0),
            nn.ReLU(), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 3, 3, 1, 0),
        )
        self._weight_init()
        
    def _weight_init(self):
        # load pretrained weights
        checkpoint = torch.load('pretrained_weights/pdm.pth')
        self.load_state_dict(checkpoint)

    def forward(self, input):
        return torch.sigmoid(self.net(input))
    
    