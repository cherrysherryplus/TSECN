import sys
sys.path.append('.')

import torch
from ptflops import get_model_complexity_info

from options.test_options import TestOptions
from models import networks


test_options = TestOptions()
test_options.parser.add_argument('--use_gpu', action='store_true', default=True)
opt = test_options.parse_simple()

# default configures for network
opt.tpe = True
opt.cerm = True
opt.use_pdm = True

opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip


if opt.use_pdm:
    from models.networks import PDM
    pdm = PDM(num=64).cuda()


def input_constructor(input_res):
    h, w = input_res[-2:]
    batch = {'input':torch.ones(()).new_empty((1, *input_res), dtype=torch.float32, device='cuda:0'),
             'gray':torch.ones(()).new_empty((1, 1, h, w), dtype=torch.float32, device='cuda:0')}
    return batch


with torch.cuda.device(0):
    netG_A = networks.Unet_resize_conv_v2(opt, skip=True).cuda()
    macs, params = get_model_complexity_info(netG_A, (3, 256, 256), as_strings=True, backend='pytorch',
                                            print_per_layer_stat=False, verbose=True, input_constructor=input_constructor, output_precision=4)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    if opt.use_pdm:
        macs_pdm, params_pdm = get_model_complexity_info(pdm, (3, 256, 256), as_strings=True, backend='pytorch',
                                                           print_per_layer_stat=False, verbose=True, output_precision=4)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs_pdm))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params_pdm))


param_dgcn = networks.print_network(netG_A.dgcn)  # 
param_pdm = networks.print_network(pdm)          #
param_gan = networks.print_network(netG_A)          #8714726
print(param_dgcn, param_pdm, param_gan)
# # 0.021819603573800212
# # 192646 8829033 2.18%
print( param_dgcn+param_pdm, param_gan+param_pdm, (param_dgcn+param_pdm) / (param_gan+param_pdm) )

    