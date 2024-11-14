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


import pyiqa
import torch


def append_results(filename, imagename, best_exposure, niqe, ilniqe, brisque):
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ImageName", "BestExposure", "NIQE", "BRISQUE", "ILNIQE"])

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([imagename, best_exposure, niqe, brisque, ilniqe])


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

# metrics
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
niqe = pyiqa.create_metric('niqe', device=device)
ilniqe = pyiqa.create_metric('ilniqe', device=device)
brisque = pyiqa.create_metric('brisque', device=device)

# test
o2mif_result_dir = 'o2mif_result_dir'
sum_res = {'niqe':0., 'ilniqe':0., 'brisque':0.}
print(len(dataset))
for i, data in tqdm(enumerate(dataset), desc="Outer loop", position=0):
    model.set_input(data)
    img_path = model.get_image_paths()[0]
    name = os.path.basename(img_path)
    base_name = os.path.splitext(name)[0]
    dir_name = os.path.join(o2mif_result_dir, opt.which_dataset, base_name)
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name, exist_ok=True)
    best_exposure_level = 0.0
    best_avg = np.Inf
    best_save_image_path = ''
    best_res = {'niqe':np.inf, 'ilniqe':np.inf, 'brisque':np.inf}
    # save result
    csv_filename = os.path.join(o2mif_result_dir, opt.which_dataset, 'best_exposures.csv')
    for exposure_level in tqdm(np.arange(0, 2.0, opt.step), desc="Inner loop", leave=False, position=1):
        result = {}
        visuals = model.predict(exposure_level=exposure_level)
        # st()
        save_image_path = os.path.join(dir_name, f'{base_name}_{exposure_level:.4f}.{opt_clone.type}')
        save_image(visuals['fake_B'], save_image_path)
        result['niqe'] = niqe(save_image_path).item()
        result['ilniqe'] = ilniqe(save_image_path).item()
        result['brisque'] = brisque(save_image_path).item()
        average = (result['niqe']+result['ilniqe']+result['brisque']) / 3.0
        if average < best_avg:
            best_avg = average
            best_exposure_level = exposure_level
            best_save_image_path = save_image_path
            best_res.update(result)
    # st()
    append_results(csv_filename, best_save_image_path, best_exposure_level, **best_res)
    sum_res['niqe'] += best_res['niqe']
    sum_res['ilniqe'] += best_res['ilniqe']
    sum_res['brisque'] += best_res['brisque']

############################################
# 计算csv_filename的平均值，并写入到最后一行
############################################
sum_res['niqe'] /= len(dataset)
sum_res['ilniqe'] /= len(dataset)
sum_res['brisque'] /= len(dataset)
append_results(csv_filename, 'AVERAGE', -1, **sum_res)
