import torch
from torchvision.transforms import ToTensor, Normalize, Compose
import PIL.Image as Image
import numpy as np


def color_loss(x): 
    x = (x+1) / 2.0
    mean_rgb = torch.mean(x,[2,3],keepdim=True)
    mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
    Drg = torch.pow(mr-mg,2)
    Drb = torch.pow(mr-mb,2)
    Dgb = torch.pow(mb-mg,2)
    k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5).mean()
    return k

def angle(a, b):
    vector = torch.mul(a, b)
    up     = torch.sum(vector)
    down   = torch.sqrt(torch.sum(torch.square(a))) * torch.sqrt(torch.sum(torch.square(b)))
    theta  = torch.acos(up/down) # 弧度制
    return theta

def color_loss_pair(out_image, gt_image): # 颜色损失  希望增强前后图片的颜色一致性 (b,c,h,w)
    loss = torch.mean(angle(out_image[:,0,:,:],gt_image[:,0,:,:]) + 
                      angle(out_image[:,1,:,:],gt_image[:,1,:,:]) +
                      angle(out_image[:,2,:,:],gt_image[:,2,:,:]))
    return loss

def color_loss_pair_patched(out_image, gt_image, patch_size=16):
    # 使用unfold将图像分割成16x16的小块，unfold会返回一个新的张量，其中每个小块都是连续的
    unfolded_x = out_image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    unfolded_y = gt_image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    loss = torch.mean(angle(unfolded_x[:,:,:,0,:],unfolded_y[:,:,:,0,:]) + 
                      angle(unfolded_x[:,:,:,1,:],unfolded_y[:,:,:,1,:]) +
                      angle(unfolded_x[:,:,:,2,:],unfolded_y[:,:,:,2,:]))
    return loss

if __name__ == '__main__':
    normalize = Compose([ToTensor(), Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    image1 = Image.open('datasets/test/LOL-v1-real/testA/1.png').convert('RGB')
    image2 = Image.open('datasets/test/LOL-v1-real/testB/1.png').convert('RGB')
    # image2 = Image.open('ablation/tsecn/LOL-v1-real_test_200/images/1_fake_B.png').convert('RGB')
    
    tensor1 = normalize(image1).unsqueeze_(0).cuda()
    tensor1 = (tensor1+1) / 2.0
    print(color_loss(tensor1))

    tensor2= normalize(image2).unsqueeze_(0).cuda()
    tensor2 = (tensor2+1) / 2.0
    print(color_loss(tensor2))
    
    print(color_loss_pair(tensor2, tensor1))