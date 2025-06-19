import os
import torch
import pyiqa
import glob
import sys

device = torch.device("cuda")
niqe = pyiqa.create_metric('niqe', device=device)
ilniqe = pyiqa.create_metric('ilniqe', device=device)
brisque = pyiqa.create_metric('brisque', device=device)


def metrics(im_dir):
    avg_niqe = 0
    avg_ilniqe = 0
    avg_brisque = 0
    n = 0

    for item in sorted(glob.glob(im_dir)):
        if item.endswith('.csv'):
            continue
        n += 1
        niqe_val = niqe(item).item()
        ilniqe_val = ilniqe(item).item()
        brisque_val = brisque(item).item()
    
        avg_niqe += niqe_val
        avg_ilniqe += ilniqe_val
        avg_brisque += brisque_val

    avg_niqe = avg_niqe / n
    avg_ilniqe = avg_ilniqe / n
    avg_brisque = avg_brisque / n
    return avg_niqe, avg_ilniqe, avg_brisque


if __name__ == '__main__':
    method, dataset = sys.argv[-2],sys.argv[-1]
    im_dir = f'results/{method}/{dataset}/*'

    avg_niqe, avg_ilniqe, avg_brisque = metrics(im_dir)
    out = open(f'results/{method}_eval_metrics_results.txt', 'a')
    print("==={:s}===".format(dataset), file=out)
    print("===> Avg.NIQE: {:.4f} ".format(avg_niqe), file=out)
    print("===> Avg.ILNIQE: {:.4f} ".format(avg_ilniqe), file=out)
    print("===> Avg.BRISQUE: {:.4f} ".format(avg_brisque), file=out)
    out.close()
