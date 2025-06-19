import glob
import csv
import os
import time
from collections import OrderedDict
import numpy as np
import torch
import cv2
from natsort import natsort
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import pyiqa

# pytorch_msssim.ssim(pred_img, real_img, data_range = data_range)?
class Measure():
    def __init__(self, net='alex', use_gpu=False):
        self.device = 'cuda' if use_gpu else 'cpu'
        self.model = lpips.LPIPS(net=net)
        self.model.to(self.device)

    def measure(self, imgA, imgB):
        return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim, self.lpips]]

    def lpips(self, imgA, imgB, model=None):
        tA = t(imgA).to(self.device)
        tB = t(imgB).to(self.device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB):
        
        score, diff = ssim(imgA, imgB, full=True, channel_axis=2)
        return score

    def psnr(self, imgA, imgB):
        psnr_val = psnr(imgA, imgB)
        return psnr_val


def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def format_result(psnr, ssim, lpips):
    return f'{psnr:0.2f}, {ssim:0.3f}, {lpips:0.3f}'

def format_result_rf(niqe, brisque, ilniqe):
    return f'{niqe:0.2f}, {brisque:0.3f}, {ilniqe:0.3f}'

def measure_dirs(dirA, dirB, use_gpu, verbose=False, type='png'):
    if verbose:
        vprint = lambda x: print(x)
    else:
        vprint = lambda x: None


    t_init = time.time()

    # .{type}
    paths_A = fiFindByWildcard(os.path.join(dirA, f'*'))
    # NOTE 1104: Only extract images end with '_fakeB'
    paths_B = fiFindByWildcard(os.path.join(dirB, f'*'))
    print(paths_A, paths_B)
    # st()

    vprint("Comparing: ")
    vprint(dirA)
    vprint(dirB)

    measure = Measure(use_gpu=use_gpu)

    results = []
    # st()
    for pathA, pathB in zip(paths_A, paths_B):
        result = OrderedDict()

        t = time.time()
        result['psnr'], result['ssim'], result['lpips'] = measure.measure(imread(pathA), imread(pathB))
        d = time.time() - t
        vprint(f"{pathA.split('/')[-1]}, {pathB.split('/')[-1]}, {format_result(**result)}, {d:0.1f}")

        results.append(result)

    psnr = np.mean([result['psnr'] for result in results])
    ssim = np.mean([result['ssim'] for result in results])
    lpips = np.mean([result['lpips'] for result in results])

    vprint(f"Final Result: {format_result(psnr, ssim, lpips)}, {time.time() - t_init:0.1f}s")
    return psnr, ssim, lpips


def measure_dirs_rf(dirB, use_gpu, verbose=False, type='png'):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    niqe = pyiqa.create_metric('niqe', device=device)
    brisque = pyiqa.create_metric('brisque', device=device)
    ilniqe = pyiqa.create_metric('ilniqe', device=device)
    if verbose:
        vprint = lambda x: print(x)
    else:
        vprint = lambda x: None
    t_init = time.time()
    # NOTE 1104: Only extract images end with '_fakeB'
    paths_B = fiFindByWildcard(os.path.join(dirB, f'*.{type}'))

    results = []
    # st()
    for pathB in paths_B:
        result = OrderedDict()

        t = time.time()
        result['niqe'], result['brisque'], result['ilniqe'] = niqe(pathB).item(), brisque(pathB).item(), ilniqe(pathB).item()
        d = time.time() - t
        vprint(f"{pathB.split('/')[-1]}, {format_result_rf(**result)}, {d:0.1f}")
        results.append(result)

    mean_niqe = np.mean([result['niqe'] for result in results])
    mean_brisque = np.mean([result['brisque'] for result in results])
    mean_ilniqe = np.mean([result['ilniqe'] for result in results])

    vprint(f"Final Result: {format_result_rf(mean_niqe, mean_ilniqe, mean_brisque)}, {time.time() - t_init:0.1f}s")
    return mean_niqe, mean_brisque, mean_ilniqe


def append_eval_results(filename, psnr, ssim, lpips, which_epoch):
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["PSNR", "SSIM", "LPIPS", "Epoch"])

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([psnr, ssim, lpips, which_epoch])
        
        
def append_eval_results_rf(filename, niqe, brisque, ilniqe, which_epoch):
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["NIQE", "BRISQUE", "ILNIQE", "Epoch"])

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([niqe, brisque, ilniqe, which_epoch])