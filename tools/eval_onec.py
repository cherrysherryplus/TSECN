# ONEC: find optimal exposure_level for each image and save the best
import sys
sys.path.append('.')

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
from util.util import str2bool

import pyiqa
import torch


def append_results(datasetname, filename, imagename, **kwargs):
    paired = True
    if 'LOL' not in datasetname and 'LSRW' not in datasetname:
        paired = False
        header = ["ImageName", "Exposure", "NIQE", "BRISQUE", "ILNIQE"]
        exposure, niqe, ilniqe, brisque = kwargs['exposure'], kwargs['niqe'], kwargs['ilniqe'], kwargs['brisque']
    else:
        header = ["ImageName", "Exposure", "PSNR", "SSIM", "LPIPS"]
        exposure, psnr, ssim, lpips = kwargs['exposure'], kwargs['psnr'], kwargs['ssim'], kwargs['lpips']
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not paired:
            writer.writerow([imagename, exposure, niqe, brisque, ilniqe])
        else:
            writer.writerow([imagename, exposure, psnr, ssim, lpips])
            

test_options = TestOptions()
test_options.parser.add_argument('--which_dataset', type=str, default='LOL-v1-real', help='which dataset to test?')
test_options.parser.add_argument('--start', default=0.0, type=float)
test_options.parser.add_argument('--end', default=2.05, type=float)
test_options.parser.add_argument('--step', default=0.05, type=float)
test_options.parser.add_argument('--type', default='png')
test_options.parser.add_argument('--use_gpu', action='store_true', default=True)
# SAVE DIR NAME
test_options.parser.add_argument('--save_dir_name', type=str, default='', help='set save dir name, dont use name as dir name')
# USE PDM
test_options.parser.add_argument('--use_pdm', type=str2bool, default=False, help='whether use PDM')
opt = test_options.parse()

opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

# NOTE whether use pdm
# opt.use_pdm = False

if opt.use_pdm:
    from models.networks import PDM

opt_clone = deepcopy(opt)
opt_clone.dataroot = os.path.join(opt.dataroot, opt.which_dataset)
data_loader = CreateDataLoader(opt_clone)
dataset = data_loader.load_data()
model = create_model(opt_clone)

# test
onec_result_dir = 'results_onec'
paired = True if 'LOL' in opt.which_dataset or 'LSRW' in opt.which_dataset else False
target_type = 'png' if 'LOL' in opt.which_dataset else 'jpg'

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

if opt.use_pdm:
    pdm = PDM().cuda()
    pdm.eval()

for i, data in tqdm(enumerate(dataset), desc="Outer loop", position=0):
    if opt.use_pdm:
        with torch.no_grad():
            input_A = data['A'].cuda()
            input_A_gray = data['A_gray'].cuda()
            denorm_input_A = (input_A+1.0) / 2.0
            projected_input_A = pdm(denorm_input_A)
            data.update(A = projected_input_A*2.0 - 1)
    model.set_input(data)
    img_path = model.get_image_paths()[0]
    name = os.path.basename(img_path)
    base_name = os.path.splitext(name)[0]
    dir_name = os.path.join(onec_result_dir, opt.save_dir_name if opt.save_dir_name else opt.name, opt.which_dataset, base_name)
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name, exist_ok=True)
    # save result
    csv_filename = os.path.join(onec_result_dir, opt.save_dir_name if opt.save_dir_name else opt.name, opt.which_dataset, 'metrics.csv')
    for exposure_level in tqdm(np.arange(opt.start, opt.end, opt.step), desc="Inner loop", leave=False, position=1):
        save_image_path = os.path.join(dir_name, f'{base_name}_{exposure_level:.4f}.{opt_clone.type}')
        save_image_path_stage1 = os.path.join(dir_name, f'{base_name}_stage1.{opt_clone.type}')
        result = {'exposure': f'{exposure_level:.4f}'}
        # st()
        if not os.path.exists(save_image_path):
            visuals = model.predict(exposure_level=exposure_level)
            save_image(visuals['fake_B'], save_image_path)
            if opt.tpe:
                save_image(visuals['stage1'], save_image_path_stage1)
        if not paired:
            result['niqe'] = niqe(save_image_path).item()
            result['ilniqe'] = ilniqe(save_image_path).item()
            result['brisque'] = brisque(save_image_path).item()
        else:
            # st()
            reference_img = os.path.join(reference_dir, f'{base_name}.{target_type}')
            result['psnr'], result['ssim'], result['lpips'] = measure.measure(imread(reference_img), imread(save_image_path))
        # st()
        append_results(opt.which_dataset, csv_filename, save_image_path, **result)