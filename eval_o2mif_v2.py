# O2MIF: find optimal exposure_level for each image and save the best
import os
import csv
from tqdm import tqdm
import shutil
import numpy as np
from copy import deepcopy
from pdb import set_trace as st
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import save_image
from util.eval import Measure, imread

import pyiqa
import torch


def append_results(datasetname, filename, imagename, **kwargs):
    paired = True
    if 'LOL' not in datasetname:
        paired = False
        header = ["ImageName", "BestExposure", "NIQE", "BRISQUE", "ILNIQE"]
        best_exposure, niqe, ilniqe, brisque = kwargs['best_exposure'], kwargs['niqe'], kwargs['ilniqe'], kwargs['brisque']
    else:
        header = ["ImageName", "BestExposure", "PSNR", "SSIM", "LPIPS"]
        best_exposure, psnr, ssim, lpips = kwargs['best_exposure'], kwargs['psnr'], kwargs['ssim'], kwargs['lpips']
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not paired:
            writer.writerow([imagename, best_exposure, niqe, brisque, ilniqe])
        else:
            writer.writerow([imagename, best_exposure, psnr, ssim, lpips])
            

test_options = TestOptions()
test_options.parser.add_argument('--which_dataset', type=str, default='LOL-v1-real', help='which dataset to test?')
test_options.parser.add_argument('--step', default=0.5, type=float)
test_options.parser.add_argument('--type', default='png')
test_options.parser.add_argument('--use_gpu', action='store_true', default=True)
opt = test_options.parse()

opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.result_root = './ablation'

opt_clone = deepcopy(opt)
opt_clone.dataroot = os.path.join(opt.dataroot, opt.which_dataset)
data_loader = CreateDataLoader(opt_clone)
dataset = data_loader.load_data()
model = create_model(opt_clone)

# test
o2mif_result_dir = 'o2mif_result_dir'
paired = True if 'LOL' in opt.which_dataset else False

# metrics
if not paired:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    niqe = pyiqa.create_metric('niqe', device=device)
    ilniqe = pyiqa.create_metric('ilniqe', device=device)
    brisque = pyiqa.create_metric('brisque', device=device)
    sum_res = {'niqe':0., 'ilniqe':0., 'brisque':0.}
else:
    measure = Measure(use_gpu=opt.use_gpu)
    reference_dir = os.path.join(opt_clone.dataroot, 'testB')
    sum_res = {'psnr':0., 'ssim':0., 'lpips':0.}

print(len(dataset))
for i, data in tqdm(enumerate(dataset), desc="Outer loop", position=0):
    model.set_input(data)
    img_path = model.get_image_paths()[0]
    name = os.path.basename(img_path)
    base_name = os.path.splitext(name)[0]
    dir_name = os.path.join(o2mif_result_dir, opt.name, opt.which_dataset, base_name)
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name, exist_ok=True)
    best_save_image_path = ''
    if not paired:
        best_avg = np.Inf
        best_res = {'best_exposure':0., 'niqe':np.inf, 'ilniqe':np.inf, 'brisque':np.inf}
    else:
        best_avg = 0.
        best_res = {'best_exposure':0., 'psnr':0., 'ssim':0., 'lpips':np.inf}
    # save result
    csv_filename = os.path.join(o2mif_result_dir, opt.name, opt.which_dataset, 'best_exposures.csv')
    for exposure_level in tqdm(np.arange(0.05, 2.05, opt.step), desc="Inner loop", leave=False, position=1):
        result = {}
        visuals = model.predict(exposure_level=exposure_level)
        # st()
        save_image_path = os.path.join(dir_name, f'{base_name}_{exposure_level:.4f}.{opt_clone.type}')
        save_image(visuals['fake_B'], save_image_path)
        if not paired:
            result['niqe'] = niqe(save_image_path).item()
            result['ilniqe'] = ilniqe(save_image_path).item()
            result['brisque'] = brisque(save_image_path).item()
            average = (result['niqe']+result['ilniqe']+result['brisque']) / 3.0
            if average < best_avg:
                best_avg = average
                best_save_image_path = save_image_path
                best_res.update(result, best_exposure=exposure_level)
        else:
            reference_img = os.path.join(reference_dir, f'{base_name}.{opt_clone.type}')
            result['psnr'], result['ssim'], result['lpips'] = measure.measure(imread(reference_img), imread(save_image_path))
            average = (result['psnr']+result['ssim']*10-result['lpips']*20) / 3.0
            if average > best_avg:
                best_avg = average
                best_save_image_path = save_image_path
                best_res.update(result, best_exposure=exposure_level)
    # st()
    append_results(opt.which_dataset, csv_filename, best_save_image_path, **best_res)
    if not paired:
        sum_res['niqe'] += best_res['niqe']
        sum_res['ilniqe'] += best_res['ilniqe']
        sum_res['brisque'] += best_res['brisque']
    else:
        sum_res['psnr'] += best_res['psnr']
        sum_res['ssim'] += best_res['ssim']
        sum_res['lpips'] += best_res['lpips']

############################################
# 计算csv_filename的平均值，并写入到最后一行
############################################
if not paired:
    sum_res['niqe'] /= len(dataset)
    sum_res['ilniqe'] /= len(dataset)
    sum_res['brisque'] /= len(dataset)
else:
    sum_res['psnr'] /= len(dataset)
    sum_res['ssim'] /= len(dataset)
    sum_res['lpips'] /= len(dataset)
sum_res['best_exposure'] = -1
append_results(opt.which_dataset, csv_filename, 'AVERAGE', **sum_res)
