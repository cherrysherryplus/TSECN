import sys
sys.path.append('.')

import torch
from models import networks
from models.networks import PDM
from options.test_options import TestOptions
from util.fps_benchmark import FPSBenchmark


def input_constructor(input_res):
    h, w = input_res[-2:]
    batch = {'input':torch.ones(()).new_empty((1, *input_res), dtype=torch.float32, device='cuda:0'),
             'gray':torch.ones(()).new_empty((1, 1, h, w), dtype=torch.float32, device='cuda:0')}
    return batch


test_options = TestOptions()
test_options.parser.add_argument('--use_gpu', action='store_true', default=True)
opt = test_options.parse_simple()


# default configures for network
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
    
# baseline+CERM+TPE
opt.tpe = True
opt.cerm = True
opt.use_pdm = True
netG_A = networks.Unet_resize_conv_v2(opt, skip=True)

if opt.use_pdm:
    from models.networks import PDM
    pdm = PDM(num=64)

fps_benmark = FPSBenchmark(model=netG_A,
                           input_size=(3, 256, 256),
                           input_constructor=input_constructor,
                           preprocessor=pdm if opt.use_pdm else torch.nn.Identity(),
                           device="cuda:0",
                           repeat_num=5)

fps_benmark.repeat_measure_inference_speed()
