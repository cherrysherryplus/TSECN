import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X, opt):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        if opt.vgg_choose != "no_maxpool":
            h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        relu4_1 = h
        h = F.relu(self.conv4_2(h), inplace=True)
        relu4_2 = h
        conv4_3 = self.conv4_3(h)
        h = F.relu(conv4_3, inplace=True)
        relu4_3 = h

        if opt.vgg_choose != "no_maxpool":
            if opt.vgg_maxpooling:
                h = F.max_pool2d(h, kernel_size=2, stride=2)
        
        conv5_1 = self.conv5_1(h)
        relu5_1 = F.relu(conv5_1, inplace=True)
        relu5_2 = F.relu(self.conv5_2(relu5_1), inplace=True)
        conv5_3 = self.conv5_3(relu5_2) 
        h = F.relu(conv5_3, inplace=True)
        relu5_3 = h
        if opt.vgg_choose == "conv4_3":
            return conv4_3
        elif opt.vgg_choose == "relu4_2":
            return relu4_2
        elif opt.vgg_choose == "relu4_1":
            return relu4_1
        elif opt.vgg_choose == "relu4_3":
            return relu4_3
        elif opt.vgg_choose == "conv5_3":
            return conv5_3
        elif opt.vgg_choose == "conv5_1":
            return conv5_1
        elif opt.vgg_choose == "relu5_1":
            return relu5_1
        elif opt.vgg_choose == "relu5_2":
            return relu5_2
        elif opt.vgg_choose == "relu5_3" or "maxpool":
            return relu5_3

def vgg_preprocess(batch, opt):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    if opt.vgg_mean:
        mean = tensortype(batch.data.size())
        mean[:, 0, :, :] = 103.939
        mean[:, 1, :, :] = 116.779
        mean[:, 2, :, :] = 123.680
        batch = batch.sub(Variable(mean)) # subtract mean
    return batch

class PerceptualLoss(nn.Module):
    def __init__(self, opt):
        super(PerceptualLoss, self).__init__()
        self.opt = opt
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img, self.opt)
        target_vgg = vgg_preprocess(target, self.opt)
        img_fea = vgg(img_vgg, self.opt)
        target_fea = vgg(target_vgg, self.opt)
        if self.opt.no_vgg_instance:
            return torch.mean((img_fea - target_fea) ** 2)
        else:
            # NOTE 1108 原始的是平方
            # return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)
            return torch.mean(torch.abs((self.instancenorm(img_fea) - self.instancenorm(target_fea))))

def load_vgg16(model_dir, gpu_ids):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    vgg = Vgg16()
    # vgg.cuda()
    vgg.cuda(device=gpu_ids[0])
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    vgg = torch.nn.DataParallel(vgg, gpu_ids)
    return vgg