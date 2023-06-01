import os
from numpy.core.fromnumeric import mean

import torch
import numpy as np
import argparse
from models.SUFT import *
# from src.models.CA_uncertainty import CANet_U

from data.nyu_dataloader import *
from data.rgbdd_dataloader import *
from utils import calc_rmse, rgbdd_calc_rmse

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils

import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import logging
from datetime import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=8, help='scale factor')
parser.add_argument('--lr',  default='0.0001', type=float, help='learning rate')
parser.add_argument('--result',  default='experiment', help='learning rate')
parser.add_argument('--epoch',  default=200, type=int, help='max epoch')
parser.add_argument('--device',  default="1", type=str, help='which gpu use')
parser.add_argument("--decay_iterations", type=list, default=[5e4, 1e5], help="steps to start lr decay")
parser.add_argument("--num_feats", type=int, default=32, help="channel number of the middle hidden layer")
parser.add_argument("--gamma", type=float, default=0.1, help="decay rate of learning rate")
parser.add_argument("--root_dir", type=str, default='/data/SRData/NYU_v2', help="root dir of dataset")
parser.add_argument("--batchsize", type=int, default=1, help="batchsize of training dataloader")

opt = parser.parse_args()
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

s = datetime.now().strftime('%Y%m%d%H%M%S')
dataset_name = opt.root_dir.split('/')[-1]
result_root = '%s/%s-lr_%s-s_%s-%s-b_%s'%(opt.result, s, opt.lr, opt.scale, dataset_name, opt.batchsize)
if not os.path.exists(result_root): 
    os.mkdir(result_root)

logging.basicConfig(filename='%s/train.log'%result_root,format='%(asctime)s %(message)s', level=logging.INFO)
logging.info(opt)

net = SUFT_network(num_feats=opt.num_feats, kernel_size=3, scale=opt.scale).cuda()
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=opt.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.decay_iterations, gamma=opt.gamma)
net.train()

data_transform = transforms.Compose([transforms.ToTensor()])
up = nn.Upsample(scale_factor=opt.scale, mode='bicubic')

if dataset_name == 'NYU_v2':
    test_minmax = np.load('%s/test_minmax.npy'%opt.root_dir)
    train_dataset = NYU_v2_datset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform, train=True)
    test_dataset = NYU_v2_datset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform, train=False)
if dataset_name == 'RGB-D-D':
    train_dataset = NYU_v2_datset(root_dir='/data/SRData/NYU_v2', scale=opt.scale, transform=data_transform, train=True)
    test_dataset = RGBDD_Dataset(root_dir=opt.root_dir, scale=opt.scale, downsample='bicubic', transform=data_transform, train=False)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=8)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)


max_epoch = opt.epoch
num_train = len(train_dataloader)
best_rmse = 10.0
best_epoch = 0
for epoch in range(max_epoch):
    # ---------
    # Training
    # ---------
    net.train()
    running_loss = 0.0

    for idx, data in enumerate(train_dataloader):
        batches_done = num_train * epoch + idx
        optimizer.zero_grad()
        guidance, lr, gt = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda()

        out = net((guidance, lr))

        loss = criterion(out, gt)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.data.item()
        
    logging.info('epoch:%d iteration:%d running_loss:%.10f' % (epoch + 1, batches_done + 1, running_loss/num_train))
            

    # -----------
    # Validating
    # -----------
    if epoch % 2 == 0:
        with torch.no_grad():

            net.eval()
            if dataset_name == 'NYU_v2':
                rmse = np.zeros(449)
            if dataset_name == 'RGB-D-D':
                rmse = np.zeros(405)
    
            for idx, data in enumerate(test_dataloader):
                if dataset_name == 'NYU_v2':
                    guidance, lr, gt = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda()
                    out = net((guidance, lr))
                    minmax = test_minmax[:,idx]
                    minmax = torch.from_numpy(minmax).cuda()
                    rmse[idx] = calc_rmse(gt[0,0], out[0,0], minmax)
                if dataset_name == 'RGB-D-D':
                    guidance, lr, gt, max, min = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda(), data['max'].cuda(), data['min'].cuda()
                    out = net((guidance, lr))
                    minmax = [max, min]
                    rmse[idx] = rgbdd_calc_rmse(gt[0,0], out[0,0], minmax)
            
            if epoch % 100 == 0:
                lr = up(lr)
                img_grid = torch.cat((lr, out, gt), -1)
                utils.save_image(img_grid, "%s/test%d.png" % (result_root,batches_done) , nrow=1, normalize=False)
                utils.save_image(guidance, "%s/test%d_RGB.png" % (result_root,batches_done) , nrow=1, normalize=False)
        
            r_mean = rmse.mean()
            if r_mean < best_rmse:
                best_rmse = r_mean
                best_epoch = epoch
                torch.save(net.state_dict(), "%s/best_model.pth"%result_root)
            logging.info('---------------------------------------------------------------------------------------------------------------------------')
            logging.info('epoch:%d lr:%f-------mean_rmse:%f (BEST: %f @epoch%d)'%(epoch+1, scheduler.get_last_lr()[0], r_mean, best_rmse, best_epoch))
            logging.info('---------------------------------------------------------------------------------------------------------------------------')
        
