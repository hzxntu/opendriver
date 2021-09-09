# ------------------------------------------------------------------------------
# Written by Zhonxu Hu (zhongxu.hu@ntu.edu.sg)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
from torch.autograd import Variable
from .centerloss import CenterLoss


class PYRMSELoss_Cent(nn.Module):
    def __init__(self,cls_num,f_dim):
        super(PYRMSELoss_Cent, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.criterion_m = nn.CrossEntropyLoss()
        self.weights=[1.0,1.0,1.0]
        self.center_loss_p = CenterLoss(num_classes=cls_num, feat_dim=f_dim, use_gpu=True)
        self.center_loss_y = CenterLoss(num_classes=cls_num, feat_dim=f_dim, use_gpu=True)
        self.center_loss_r = CenterLoss(num_classes=cls_num, feat_dim=f_dim, use_gpu=True)
        self.max_pool=nn.MaxPool1d(3,ceil_mode=True)

    def forward(self,output,target,output_m,target_m,output_m_mp,target_m_mp,features,target_m_mp_mp):
        loss=0
        num_joints = output.size(1)
        
        for idx in range(num_joints):
            loss += self.weights[idx]*0.5* self.criterion(output[:,idx],target[:,idx])
        
        
        for idx in range(num_joints):
            loss += self.weights[idx]*1.0* self.criterion_m(output_m[:,idx,:], target_m[:,idx])
    
        
        for idx in range(num_joints):
            loss += self.weights[idx]*2.0* self.criterion_m(output_m_mp[:,idx,:], target_m_mp[:,idx])
        
        
        loss+= 0.01 * self.center_loss_p(features[:,0,:],target_m[:,0])
        loss+= 0.01 * self.center_loss_y(features[:,1,:],target_m[:,1])
        loss+= 0.01 * self.center_loss_r(features[:,2,:],target_m[:,2])
        
        return loss

class PYRMSELoss_LRC(nn.Module):
    """docstring for PYRMSELoss"""
    def __init__(self,cls_num,f_dim):
        super(PYRMSELoss_LRC, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.weights=[1.0,1.0,1.0]
        self.criterion_m = nn.CrossEntropyLoss()
        self.center_loss_p_l = CenterLoss(num_classes=cls_num, feat_dim=f_dim, use_gpu=True)
        self.center_loss_y_l = CenterLoss(num_classes=cls_num, feat_dim=f_dim, use_gpu=True)
        self.center_loss_p_r = CenterLoss(num_classes=cls_num, feat_dim=f_dim, use_gpu=True)
        self.center_loss_y_r = CenterLoss(num_classes=cls_num, feat_dim=f_dim, use_gpu=True)

    def forward(self,output_l,target_l,output_r,target_r,out_l_b,g_l_b,out_r_b,g_r_b,out_l_mp,g_l_mp,out_r_mp,g_r_mp,out_l_f,out_r_f):
        loss=0
        num_joints = output_l.size(1)
        
        for idx in range(num_joints):
            loss+=self.weights[idx]*0.5*self.criterion(output_l[:,idx],target_l[:,idx])
            loss+=self.weights[idx]*0.5*self.criterion(output_r[:,idx],target_r[:,idx])
        
        for idx in range(num_joints):
            loss += self.weights[idx]*1.0* self.criterion_m(out_l_b[:,idx,:], g_l_b[:,idx])
            loss += self.weights[idx]*1.0* self.criterion_m(out_r_b[:,idx,:], g_r_b[:,idx])
        
        for idx in range(num_joints):
            loss += self.weights[idx]*2.0* self.criterion_m(out_l_mp[:,idx,:], g_l_mp[:,idx])
            loss += self.weights[idx]*2.0* self.criterion_m(out_r_mp[:,idx,:], g_r_mp[:,idx])
        
        loss+= 0.01 * self.center_loss_p_l(out_l_f[:,0,:],g_l_b[:,0])
        loss+= 0.01 * self.center_loss_y_l(out_l_f[:,1,:],g_l_b[:,1])
        loss+= 0.01 * self.center_loss_p_r(out_r_f[:,0,:],g_r_b[:,0])
        loss+= 0.01 * self.center_loss_y_r(out_r_f[:,1,:],g_r_b[:,1])
            
        return loss
        
