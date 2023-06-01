import argparse
import os

from utils import *
import numpy as np
import torchvision.transforms as transforms
from torchvision import utils
from torch import Tensor
from PIL import Image
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.SUFT import *
from data.nyu_dataloader import *
from data.rgbdd_dataloader import *
from data.middlebury_dataloader import Middlebury_dataset

import torch.nn as nn
import torch.nn.functional as F
import torch
# os.environ['CUDA_VISIBLE_DEVICE']="4"

net = SUFT_network(num_feats=32, kernel_size=3, scale=4)
net.load_state_dict(torch.load("experiment/20220529122823-lr_0.0001-s_4-NYU_v2-b_4/best_model.pth", map_location='cuda:0'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

data_transform = transforms.Compose([transforms.ToTensor()])
dataset = NYU_v2_datset(root_dir='/data/SRData/NYU_v2', scale=4, transform=data_transform, train=False)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
data_num = len(dataloader)

rmse = np.zeros(449)
mad = 0.0
test_minmax = np.load('/data/SRData/NYU_v2/test_minmax.npy')
with torch.no_grad():
    net.eval()
    for idx, data in enumerate(dataloader):
        if idx == 1:
            guidance, lr, gt = data['guidance'].to(device), data['lr'].to(device), data['gt'].to(device)
            utils.save_image(gt, "./test/gt.png" , nrow=1, normalize=False)
            out = net((guidance, lr))
            minmax = test_minmax[:,idx]
            minmax = torch.from_numpy(minmax).to(device)
            rmse[idx] = calc_rmse(gt[0,0], out[0,0], minmax=minmax)
    print(rmse.mean())



