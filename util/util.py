import numpy as np
import os
import torch


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)

def atten2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor[0]
    image_tensor = torch.cat((image_tensor, image_tensor, image_tensor), 0)
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = image_numpy/(image_numpy.max()/255.0)
    return image_numpy.astype(imtype)

def latent2im(image_tensor, imtype=np.uint8):
    # image_tensor = (image_tensor - torch.min(image_tensor))/(torch.max(image_tensor)-torch.min(image_tensor))
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)

def max2im(image_1, image_2, imtype=np.uint8):
    image_1 = image_1[0].cpu().float().numpy()
    image_2 = image_2[0].cpu().float().numpy()
    image_1 = (np.transpose(image_1, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_2 = (np.transpose(image_2, (1, 2, 0))) * 255.0
    output = np.maximum(image_1, image_2)
    output = np.maximum(output, 0)
    output = np.minimum(output, 255)
    return output.astype(imtype)

def variable2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].data.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
        
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
