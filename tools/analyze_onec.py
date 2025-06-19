import os
import os.path as osp
import pandas as pd
import numpy as np
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, default='baseline_wCERM_wTPE', help='onec results dir')
parser.add_argument('--which_dataset', type=str, default='LOL-v1-real', help='which dataset to test?')
parser.add_argument('--save_dir_name', type=str, default='', help='set save dir name, dont use name as dir name')
parser.add_argument('--wPSNR', type=float, default=1.0, help='weight')
parser.add_argument('--wSSIM', type=float, default=30.0, help='weight')
parser.add_argument('--wLPIPS', type=float, default=20.0, help='weight')
parser.add_argument('--wNIQE', type=float, default=5.0, help='weight')
parser.add_argument('--wBRISQUE', type=float, default=2.0, help='weight')
parser.add_argument('--wILNIQE', type=float, default=1.0, help='weight')
parser.add_argument('--wExposure', type=float, default=5.0, help='weight')
opt = parser.parse_args()

save_root = 'results'
onec_root = 'results_onec'
if 'LOL' in opt.which_dataset or 'LSRW' in opt.which_dataset:
    paired = True
else:
    paired = False

onec_dir = osp.join(onec_root, opt.name)
metrics_csv = osp.join(onec_dir, opt.which_dataset, 'metrics.csv')
save_dir_name = opt.name if not opt.save_dir_name else opt.save_dir_name
save_dir = osp.join(save_root, save_dir_name, opt.which_dataset)
csv_file = osp.join(save_dir, 'statistics.csv')
if osp.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)

df = pd.read_csv(metrics_csv, sep=',')
df = df[df['Exposure'] >= 0.0]

df['Group_ImageName'] = df['ImageName'].apply(lambda x: x.split('/')[-2])
grouped_df = df.groupby('Group_ImageName')
if paired:
    df['Sort_Key'] = df['PSNR'] * opt.wPSNR + df['SSIM'] * opt.wSSIM - df['LPIPS'] * opt.wLPIPS - abs(df['Exposure']-1.0) * opt.wExposure
    idx = grouped_df['Sort_Key'].idxmax()
    columns = ['PSNR', 'SSIM', 'LPIPS']
else:
    df['Sort_Key'] = df['NIQE'] * opt.wNIQE + df['BRISQUE'] * opt.wBRISQUE + df['ILNIQE'] * opt.wILNIQE + abs(df['Exposure']-1.0) * opt.wExposure
    idx = grouped_df['Sort_Key'].idxmin()
    columns = ['NIQE', 'BRISQUE', 'ILNIQE']
    
best_images = df.loc[idx]
avgs = best_images.loc[:, columns].abs().agg('mean').reset_index()

# copy images to save_dir
for best_image_path in best_images['ImageName']:
    target_path = osp.join(save_dir, osp.basename(best_image_path))
    print(f'cp from {best_image_path} to {target_path}')
    shutil.copy(best_image_path, target_path)
    
# save statistics to csv file
print(avgs)
avgs.to_csv(csv_file, ',')