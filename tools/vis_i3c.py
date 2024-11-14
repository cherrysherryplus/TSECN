import sys
# 需要在TSECN根目录下运行脚本 python tools/vis_i3c.py ...
sys.path.append('./')

from PIL import Image
import os
import shutil
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from util.visualizer import save_image
from util.util import atten2im


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# 归一化
def normalize(x):
    min_val = x.min()
    max_val = x.max()
    if max_val > min_val:
        return (x - min_val) / (max_val - min_val)
    else:
        return x  # 如果所有值相同，返回原值


def gaussian_filter(input_img, kernel_size=5, sigma=1.0):
    # 创建高斯核
    gauss = torch.exp(-torch.linspace(-sigma, sigma, kernel_size)**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    # 进行二维卷积
    filtered_img = F.conv2d(input_img.unsqueeze(0), gauss.view(1, 1, kernel_size, 1), padding=(kernel_size // 2, 0))
    filtered_img = F.conv2d(filtered_img, gauss.view(1, 1, 1, kernel_size), padding=(0, kernel_size // 2))
    return filtered_img.squeeze(0)


def i3c(img):
    # NOTE 1110: remove "/2"? to align with EnlightenGAN
    r,g,b = (img[0]+1)/2,(img[1]+1)/2,(img[2]+1)/2
    # r,g,b = attn_map(r),attn_map(g),attn_map(b)
    r_weight = torch.mean(r)/(torch.mean(r)+torch.mean(g)+torch.mean(b))
    g_weight = torch.mean(g)/(torch.mean(r)+torch.mean(g)+torch.mean(b))
    b_weight = torch.mean(b)/(torch.mean(r)+torch.mean(g)+torch.mean(b))
    new_img = 1-((r_weight)*r +(g_weight)*g + (b_weight)*b)
    return torch.unsqueeze(new_img, 0)


def i3c_wMaximum(img):
    # proposed
    r,g,b = (img[0]+1)/2,(img[1]+1)/2,(img[2]+1)/2
    # r,g,b = attn_map(r),attn_map(g),attn_map(b)
    r_weight = torch.mean(r)/(torch.mean(r)+torch.mean(g)+torch.mean(b))
    g_weight = torch.mean(g)/(torch.mean(r)+torch.mean(g)+torch.mean(b))
    b_weight = torch.mean(b)/(torch.mean(r)+torch.mean(g)+torch.mean(b))
    # gradient
    A_gray_gradient = gradient(img).squeeze_(0)
    A_gray_mean = 1-((r_weight)*r +(g_weight)*g + (b_weight)*b) * (1 - A_gray_gradient)
    return torch.unsqueeze(A_gray_mean, 0)


def original(img):
    r,g,b = img[0]+1, img[1]+1, img[2]+1
    A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
    return torch.unsqueeze(A_gray, 0)


def maximum(img):
    img = (img+1) / 2
    A_gray, _ = torch.max(img, dim=0)
    A_gray = 1-A_gray
    return torch.unsqueeze(A_gray, 0)


def gradient(input_img):
    img = maximum(input_img)
    height = img.size(1)
    width = img.size(2)
    gradient_h = (img[:,2:,:]-img[:,:height-2,:]).abs()
    gradient_w = (img[:, :, 2:] - img[:, :, :width-2]).abs()
    gradient_h = F.pad(gradient_h, [0, 0, 1, 1], 'replicate')
    gradient_w = F.pad(gradient_w, [1, 1, 0, 0], 'replicate')
    gradient2_h = (img[:,4:,:]-img[:,:height-4,:]).abs()
    gradient2_w = (img[:, :, 4:] - img[:, :, :width-4]).abs()
    gradient2_h = F.pad(gradient2_h, [0, 0, 2, 2], 'replicate')
    gradient2_w = F.pad(gradient2_w, [2, 2, 0, 0], 'replicate')
    # return gradient_h*gradient2_h, gradient_w*gradient2_w
    # 合并梯度
    # combined_gradient_h = gradient_h * gradient2_h
    # combined_gradient_w = gradient_w * gradient2_w
    # combined_gradient = torch.sqrt(combined_gradient_h**2 + combined_gradient_w**2)
    combined_gradient = torch.sqrt(gradient_h**2 + gradient_w**2)
    smoothed_combined_gradient = gaussian_filter(combined_gradient)
    return normalize(smoothed_combined_gradient)


def atten2im_v2(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor[0]
    image_tensor = torch.cat((image_tensor, image_tensor, image_tensor), 0)
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = image_numpy/(image_numpy.max()/255.0)
    return image_numpy.astype(imtype)


if __name__ == '__main__':
    imagedir = 'datasets/test/LOL-v1-real/testA'
    for mode in ['i3c','i3c_wMaximum','original','maximum','gradient']:
        save_dir = 'tools/vis_results/%s' % mode
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        for imagefile in os.listdir(imagedir)[:10]:
            imagepath = os.path.join(imagedir, imagefile)
            save_imagepath = os.path.join(save_dir, imagefile)
            try:
                image = Image.open(imagepath).convert('RGB')
            except Exception as e:
                image = None
                print(e)
            if image:
                func = globals().get(mode)
                imagetensor = func(transform(image)).unsqueeze_(0)
                i3c_image = atten2im_v2(imagetensor)
                print('Processing %s, save at %s' % (imagefile, save_imagepath))
                save_image(i3c_image, save_imagepath)
            