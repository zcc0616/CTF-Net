import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from net.ctfnet import Net
from utils.tdataloader import test_dataset

import time
import datetime
from numpy import mean

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--pth_path', type=str, default='./checkpoints/CTF-Net/Net_epoch_best.pth')

for _data_name in ['CAMO','CHAMELEON','COD10K','NC4K']:
    data_path = '/home/zcc/data/COD/COD 10K/TestDataset/{}/'.format(_data_name)
    save_path = './results/CTF-Net/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = Net()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path+'edge/', exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_list = []
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        start_each = time.time()
        _, _, _, res, e = model(image)
        time_each = time.time() - start_each
        time_list.append(time_each)

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imwrite(save_path+name, (res*255).astype(np.uint8))
        # e = F.upsample(e, size=gt.shape, mode='bilinear', align_corners=True)
        # e = e.data.cpu().numpy().squeeze()
        # e = (e - e.min()) / (e.max() - e.min() + 1e-8)
        # imageio.imwrite(save_path+'edge/'+name, (e*255).astype(np.uint8))
    print("{}'s average Time Is : {:.3f} s".format(_data_name, mean(time_list)))
    print("{}'s average Time Is : {:.1f} fps".format(_data_name, 1 / mean(time_list)))

