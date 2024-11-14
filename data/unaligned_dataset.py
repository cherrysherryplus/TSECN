import torch
from torch import nn
import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, store_dataset
import random
from PIL import Image
import PIL
from pdb import set_trace as st
from tools.vis_i3c import gradient

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
            input = padding(input).data
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.shape[2], input.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:,:, pad_top: height - pad_bottom, pad_left: width - pad_right]

# # Proposed 原始的！！！
def i3c(img):
    # NOTE 1110: remove "/2"? to align with EnlightenGAN
    r,g,b = (img[0]+1)/2,(img[1]+1)/2,(img[2]+1)/2
    # r,g,b = attn_map(r),attn_map(g),attn_map(b)
    r_weight = torch.mean(r)/(torch.mean(r)+torch.mean(g)+torch.mean(b))
    g_weight = torch.mean(g)/(torch.mean(r)+torch.mean(g)+torch.mean(b))
    b_weight = torch.mean(b)/(torch.mean(r)+torch.mean(g)+torch.mean(b))
    new_img = 1-((r_weight)*r +(g_weight)*g + (b_weight)*b)
    return torch.unsqueeze(new_img, 0)

# # Gradient + Proposed 正在跑的 GradAdd
# def i3c(img):
#     # proposed
#     r,g,b = (img[0]+1)/2,(img[1]+1)/2,(img[2]+1)/2
#     # r,g,b = attn_map(r),attn_map(g),attn_map(b)
#     r_weight = torch.mean(r)/(torch.mean(r)+torch.mean(g)+torch.mean(b))
#     g_weight = torch.mean(g)/(torch.mean(r)+torch.mean(g)+torch.mean(b))
#     b_weight = torch.mean(b)/(torch.mean(r)+torch.mean(g)+torch.mean(b))
#     A_gray_mean = 1-((r_weight)*r +(g_weight)*g + (b_weight)*b)
#     # gradient
#     A_gray_gradient = gradient(img).squeeze_(0)
#     new_img = A_gray_mean * 0.85 + A_gray_gradient * 0.15
#     return torch.unsqueeze(new_img, 0)


# # (1+Gradient)*proposed 正在跑的 GradMultiply
# def i3c(img):
#     # proposed
#     r,g,b = (img[0]+1)/2,(img[1]+1)/2,(img[2]+1)/2
#     # r,g,b = attn_map(r),attn_map(g),attn_map(b)
#     r_weight = torch.mean(r)/(torch.mean(r)+torch.mean(g)+torch.mean(b))
#     g_weight = torch.mean(g)/(torch.mean(r)+torch.mean(g)+torch.mean(b))
#     b_weight = torch.mean(b)/(torch.mean(r)+torch.mean(g)+torch.mean(b))
#     # gradient
#     A_gray_gradient = gradient(img).squeeze_(0)
#     A_gray_mean = 1-((r_weight)*r +(g_weight)*g + (b_weight)*b) * (1 - A_gray_gradient)
#     return torch.unsqueeze(A_gray_mean, 0)
    

# 未更新，还是原始的i3c，因为i3c_again不咋用所以就没更新了
def i3c_batch(images):
    # 假设images是一个形状为(batch_size, 3, height, width)的张量，其中：
    # batch_size是批次中图像的数量，3是颜色通道数（RGB），height和width是图像的高度和宽度。
    
    # 将图像数据从[-1,1]范围标准化到[0,1]范围
    images = (images + 1) / 2
    
    # 分离RGB通道
    r, g, b = torch.chunk(images, 3, dim=1)
    
    # 计算每个通道的平均值
    r_mean = torch.mean(r, dim=(2, 3), keepdim=True)
    g_mean = torch.mean(g, dim=(2, 3), keepdim=True)
    b_mean = torch.mean(b, dim=(2, 3), keepdim=True)
    
    # 计算每个通道的权重
    r_weight = r_mean / (r_mean + g_mean + b_mean)
    g_weight = g_mean / (r_mean + g_mean + b_mean)
    b_weight = b_mean / (r_mean + g_mean + b_mean)
    
    # 计算新的图像值
    new_images = 1 - ((r_weight * r) + (g_weight * g) + (b_weight * b))
    
    return new_images

class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        # self.A_paths = make_dataset(self.dir_A)
        # self.B_paths = make_dataset(self.dir_B)
        self.A_imgs, self.A_paths = store_dataset(self.dir_A)
        self.B_imgs, self.B_paths = store_dataset(self.dir_B)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_img = self.A_imgs[index % self.A_size]
        B_img = self.B_imgs[index % self.B_size]
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        
        if self.opt.resize_or_crop == 'no':
            if self.opt.i3c:
                A_gray = i3c(A_img)
            else:
                r,g,b = A_img[0]+1, A_img[1]+1, A_img[2]+1
                A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
                A_gray = torch.unsqueeze(A_gray, 0)
            input_img = A_img
            # A_gray = (1./A_gray)/255.
        else:
            w = A_img.size(2)
            h = A_img.size(1)
            
            # A_gray = (1./A_gray)/255.
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(2, idx)
                B_img = B_img.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(1, idx)
                B_img = B_img.index_select(1, idx)
            if self.opt.vary == 1 and (not self.opt.no_flip) and random.random() < 0.5:
                times = random.randint(self.opt.low_times,self.opt.high_times)/100.
                input_img = (A_img+1)/2./times
                input_img = input_img*2-1
            else:
                input_img = A_img
            if self.opt.lighten:
                B_img = (B_img + 1)/2.
                B_img = (B_img - torch.min(B_img))/(torch.max(B_img) - torch.min(B_img))
                B_img = B_img*2. -1
            # from pdb import set_trace as st
            if self.opt.i3c:
                A_gray = i3c(input_img)
            else:
                r,g,b = input_img[0]+1, input_img[1]+1, input_img[2]+1
                A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
                A_gray = torch.unsqueeze(A_gray, 0)
        return {'A': A_img, 'B': B_img, 'A_gray': A_gray, 'input_img': input_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'


