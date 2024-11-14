import os
import time
from copy import deepcopy
from pdb import set_trace as st

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
from util.eval import measure_dirs, measure_dirs_rf, append_eval_results, append_eval_results_rf


test_options = TestOptions()
test_options.parser.add_argument('--which_dataset', type=str, default='LOL-v1-real', help='which dataset to test?')
# for evaluation (eval.py)
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
# opt_clone_dict = vars(opt_clone)  # Convert Namespace to dictionary
# opt_clone_dict.pop('which_dataset')
data_loader = CreateDataLoader(opt_clone)
dataset = data_loader.load_data()
model = create_model(opt_clone)
visualizer = Visualizer(opt_clone)
# create website
dir_name = '%s_%s_%s' % (opt_clone.which_dataset, opt_clone.phase, opt_clone.which_epoch)
web_dir = os.path.join(opt.result_root,
                       opt_clone.name,
                       dir_name)
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' % (opt_clone.name, opt_clone.phase, opt_clone.which_epoch))
# test
print(len(dataset))
for i, data in enumerate(dataset):
    model.set_input(data)
    visuals = model.predict(exposure_level=1.0)
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()

## evaluation
dirA = os.path.join(opt_clone.dataroot, 'testB')
dirB = os.path.join(opt_clone.result_root, opt_clone.name, dir_name, 'images')
print(dirA, dirB, end='\n')
## save
csv_filename = os.path.join(opt.result_root, f"{opt_clone.name}_{opt_clone.which_dataset}_epoch_results.csv")


if len(dirA) > 0 and len(dirB) > 0:
    if 'LOL' in opt_clone.which_dataset:
        psnr, ssim, lpips = measure_dirs(dirA, dirB, use_gpu=opt_clone.use_gpu, verbose=True, type=opt_clone.type)
        append_eval_results(csv_filename, psnr, ssim, lpips, opt.which_epoch)
    else:
        niqe, brisque, ilniqe = measure_dirs_rf(dirB, use_gpu=opt_clone.use_gpu, verbose=True, type=opt_clone.type)
        append_eval_results_rf(csv_filename, niqe, brisque, ilniqe, opt.which_epoch)