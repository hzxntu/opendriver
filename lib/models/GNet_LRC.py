import argparse
import os
import shutil
import time, math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from torch.autograd.variable import Variable
import torch.nn.functional as F
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet as efn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PYRModule(nn.Module):
    def __init__(self,inplanes,downsample=None):
        super(PYRModule, self).__init__()
        
        self.features = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        
        self.ca = ChannelAttention(inplanes)

    def forward(self, x):
        #residual =x
        x = self.ca(x) * x
        #x += residual
        x = self.features(x)
        return x


class GNet_LRC(nn.Module):

    def __init__(self,cls_num):
        super(GNet_LRC, self).__init__()
        #self.eyeModel = resnet34()
        #self.faceModel = U_Net()
        self.eyeModel = efn.from_pretrained('efficientnet-b0')
        
        self.planes_num=1280
        self.cls_num=cls_num
        
        self.feature_p_l=PYRModule(self.planes_num)
        self.feature_y_l=PYRModule(self.planes_num)
        self.feature_p_r=PYRModule(self.planes_num)
        self.feature_y_r=PYRModule(self.planes_num)
        
        self.idx_tensor = torch.FloatTensor(torch.range(0,self.cls_num-1)).cuda()

        self.fc_b_p_l = nn.Sequential(
            nn.Linear(self.planes_num, self.cls_num),
            )
        
        self.fc_b_y_l = nn.Sequential(
            nn.Linear(self.planes_num, self.cls_num),
            )
        self.fc_b_p_r = nn.Sequential(
            nn.Linear(self.planes_num, self.cls_num),
            )
        
        self.fc_b_y_r = nn.Sequential(
            nn.Linear(self.planes_num, self.cls_num),
            )
        
        self.max_pool_p_l=nn.MaxPool1d(3)
        self.max_pool_y_l=nn.MaxPool1d(3)
        self.max_pool_p_r=nn.MaxPool1d(3)
        self.max_pool_y_r=nn.MaxPool1d(3)
        
        
        self.softmax=nn.Softmax(dim=2).cuda()
        
        
    def forward(self, eyes_l, eyes_r):

        xEye_L = self.eyeModel.extract_features(eyes_l)
        xEye_R = self.eyeModel.extract_features(eyes_r)
        
        x_p_l=self.feature_p_l(xEye_L)
        x_y_l=self.feature_y_l(xEye_L)
        
        x_p_r=self.feature_p_r(xEye_R)
        x_y_r=self.feature_y_r(xEye_R)
        
        x_p_l=torch.flatten(x_p_l, 1)
        x_y_l=torch.flatten(x_y_l, 1)
        x_p_r=torch.flatten(x_p_r, 1)
        x_y_r=torch.flatten(x_y_r, 1)
        
        x_p_l_f=torch.unsqueeze(x_p_l,1)
        x_y_l_f=torch.unsqueeze(x_y_l,1)
        x_p_r_f=torch.unsqueeze(x_p_r,1)
        x_y_r_f=torch.unsqueeze(x_y_r,1)
        
        x_p_l_b=self.fc_b_p_l(x_p_l)
        x_y_l_b=self.fc_b_y_l(x_y_l)
        x_p_r_b=self.fc_b_p_r(x_p_r)
        x_y_r_b=self.fc_b_y_r(x_y_r)
        
        x_p_l_b=torch.unsqueeze(x_p_l_b,1)
        x_y_l_b=torch.unsqueeze(x_y_l_b,1)
        x_p_r_b=torch.unsqueeze(x_p_r_b,1)
        x_y_r_b=torch.unsqueeze(x_y_r_b,1)
        
        x_p_l_mp=self.max_pool_p_l(x_p_l_b)
        x_y_l_mp=self.max_pool_y_l(x_y_l_b)
        x_p_r_mp=self.max_pool_p_r(x_p_r_b)
        x_y_r_mp=self.max_pool_y_r(x_y_r_b)
        
        x_p_l_pre=self.softmax(x_p_l_b)
        x_y_l_pre=self.softmax(x_y_l_b)
        x_p_r_pre=self.softmax(x_p_r_b)
        x_y_r_pre=self.softmax(x_y_r_b)
        
        x_pl=torch.sum(x_p_l_pre * self.idx_tensor, 2)
        x_yl=torch.sum(x_y_l_pre * self.idx_tensor, 2)
        x_pr=torch.sum(x_p_r_pre * self.idx_tensor, 2)
        x_yr=torch.sum(x_y_r_pre * self.idx_tensor, 2)
        
        return torch.cat([x_pl,x_yl],1),torch.cat([x_pr,x_yr],1),torch.cat([x_p_l_b,x_y_l_b],1),torch.cat([x_p_r_b,x_y_r_b],1),\
               torch.cat([x_p_l_mp,x_y_l_mp],1),torch.cat([x_p_r_mp,x_y_r_mp],1),torch.cat([x_p_l_f,x_y_l_f],1),torch.cat([x_p_r_f,x_y_r_f],1)