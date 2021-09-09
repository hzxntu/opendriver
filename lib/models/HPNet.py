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
import torch.utils.model_zoo as model_zoo
from torch.autograd.variable import Variable
from .Resnets import *
import torch.nn.functional as F
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet as efn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
                               
        #self.fc = nn.Sequential(nn.Linear(in_planes, in_planes // 16, bias=False),
        #                        nn.ReLU(inplace=True),
        #                        nn.Linear(in_planes // 16, in_planes, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #b, c, _, _ = x.size()
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
        
        self.ca = ChannelAttention(inplanes)
        
        self.features = nn.Sequential(
                                      nn.AdaptiveAvgPool2d((1, 1)),
                                      )

    def forward(self, x):
        #residual =x
        x = self.ca(x) * x
        #x += residual
        x = self.features(x)
        return x



class HPNet(nn.Module):

    def __init__(self):
        super(HPNet, self).__init__()
        self.faceModel = efn.from_pretrained('efficientnet-b4')
        
        self.planes_num=1792#2304#2048#1536#1408#1280#1792
        self.cls_num=66
        
        self.feature_1 = PYRModule(self.planes_num)
        self.feature_2 = PYRModule(self.planes_num)
        self.feature_3 = PYRModule(self.planes_num)
        
                                              
        self.idx_tensor = torch.FloatTensor(torch.range(0,self.cls_num-1)*1).cuda()
        
        self.fc_b_1 = nn.Sequential(
            nn.Linear(self.planes_num, self.cls_num),
            )
        self.fc_b_2 = nn.Sequential(
            nn.Linear(self.planes_num, self.cls_num),
            )
        self.fc_b_3 = nn.Sequential(
            nn.Linear(self.planes_num, self.cls_num),
            )
        self.max_pool_1=nn.MaxPool1d(3)
        self.max_pool_2=nn.MaxPool1d(3)
        self.max_pool_3=nn.MaxPool1d(3)
        
        self.softmax=nn.Softmax(dim=2).cuda()
        self.sigmoid=nn.Sigmoid().cuda()
        
        
    def forward(self, faces):

        xFace = self.faceModel.extract_features(faces)
        
        
        x_p = self.feature_1(xFace)
        x_y = self.feature_2(xFace)
        x_r = self.feature_3(xFace)
        
        x_p = torch.flatten(x_p, 1)
        x_y = torch.flatten(x_y, 1)
        x_r = torch.flatten(x_r, 1)
        
        x_p_feat=torch.unsqueeze(x_p,1)
        x_y_feat=torch.unsqueeze(x_y,1)
        x_r_feat=torch.unsqueeze(x_r,1)
        
        x_feat=torch.cat([x_p_feat,x_y_feat,x_r_feat],1)
        
        x_p_b=self.fc_b_1(x_p)
        x_y_b=self.fc_b_2(x_y)
        x_r_b=self.fc_b_3(x_r)
        
        x_p_b=torch.unsqueeze(x_p_b,1)
        x_y_b=torch.unsqueeze(x_y_b,1)
        x_r_b=torch.unsqueeze(x_r_b,1)
        
        x_p_b_mp=self.max_pool_1(x_p_b)
        x_y_b_mp=self.max_pool_2(x_y_b)
        x_r_b_mp=self.max_pool_3(x_r_b)
        
        x_p_pre=self.softmax(x_p_b)
        x_y_pre=self.softmax(x_y_b)
        x_r_pre=self.softmax(x_r_b)
        
        x_p=torch.sum(x_p_pre * self.idx_tensor, 2) 
        x_y=torch.sum(x_y_pre * self.idx_tensor, 2) 
        x_r=torch.sum(x_r_pre * self.idx_tensor, 2)
        

        return torch.cat([x_p,x_y,x_r],1),torch.cat([x_p_b,x_y_b,x_r_b],1),torch.cat([x_p_b_mp,x_y_b_mp,x_r_b_mp],1),x_feat