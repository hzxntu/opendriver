import torch.utils.data as data
import scipy.io as sio
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import re
import struct
import math
import random
import pandas as pd
import pathlib
import scipy.io
import csv


def convert_gaze(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    pitch = np.arcsin(-y) * 180 / np.pi 
    yaw = np.arctan2(-x, -z) * 180 / np.pi 
    return np.array([pitch, yaw]).astype(np.float32)

def readcsv(file_path):

     csvfile=open(file_path,'r')
     reader = csv.reader(csvfile)
     rows= [row for row in reader]
     data=np.array(rows)
     data=data[:,:9]
     data=data.astype(np.float)
     
     return data



class UTMULTIVIEW(data.Dataset):
    def __init__(self,split = 'train', cali_num=0,imSize=(224,224),aug=False):

        self.imSize = imSize
        self.utmv_root='./data/ut-multiview'
        self.train_dir= list(range(32,50))#+list(range(32,50))
        self.val_dir = range(0,16)
        
        self.cali=False
        self.cali_num=cali_num

        print('Loading UT-Multiview dataset...')

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.transformEye = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
        ])
        

        if split == 'train':
            self.indices=self._get_db(True)
        else:
            self.indices=self._get_db(False)
        
        print('Loaded UT-Multiview dataset split "%s" with %d records...' % (split, len(self.indices)))

    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)

        return im
        

    def __getitem__(self, index):
        sample = self.indices[index]
        
        
        imgPath_left = sample['img_left']
        raw_img_left = self.loadImage(imgPath_left)
        
        imgPath_right = sample['img_right']
        raw_img_right = self.loadImage(imgPath_right)
        
        imEye_L = self.transformEye(raw_img_left)
        imEye_R = self.transformEye(raw_img_right)
        
        gaze_l=sample['gaze_l']
        gaze_r=sample['gaze_r']
        
        
        gaze_l=np.clip(gaze_l,-54,53.99)
        gaze_r=np.clip(gaze_r,-54,53.99)

        bins = np.array(range(-54, 55, 1))
        g_l_b = np.digitize([gaze_l[0], gaze_l[1]], bins) - 1
        g_r_b = np.digitize([gaze_r[0], gaze_r[1]], bins) - 1
        g_l_b = torch.LongTensor(g_l_b)
        g_r_b = torch.LongTensor(g_r_b)
        
        bins_mp = np.array(range(-54, 57, 3))
        g_l_mp = np.digitize([gaze_l[0], gaze_l[1]], bins_mp) - 1
        g_r_mp = np.digitize([gaze_r[0], gaze_r[1]], bins_mp) - 1
        g_l_mp = torch.LongTensor(g_l_mp)
        g_r_mp = torch.LongTensor(g_r_mp)
            
        gaze_l=(gaze_l+54)/1
        gaze_r=(gaze_r+54)/1
        gaze_l = torch.FloatTensor(gaze_l)
        gaze_r = torch.FloatTensor(gaze_r)

        return imEye_L,imEye_R,gaze_l,gaze_r,g_l_b,g_r_b,g_l_mp,g_r_mp
    
    def __len__(self):
        return len(self.indices)
        
    def _get_db(self,is_train):
        if is_train:
            # use ground truth bbox
            gt_db = self._load_samples(self.utmv_root,self.train_dir,True)
        else:
            # use bbox from detection
            gt_db = self._load_samples(self.utmv_root,self.val_dir,False)
        return gt_db
        
    def _load_samples(self,root_path,dir_ls,is_train):
        # get images and groundtruths(position of center of head, pitch,yaw,roll,rotation matrix)
        gt_db=[]
        
        for idx in dir_ls:
           if is_train and self.cali:
              selected=random.sample(range(160),self.cali_num)
           else:
              selected=range(160)
           
           for i in range(160):
               if i not in selected:
                  continue
               csv_path_left='%s/s%02d/%s/%03d_%s.csv'%(root_path,idx,'test',i,'left')
               csv_path_right='%s/s%02d/%s/%03d_%s.csv'%(root_path,idx,'test',i,'right')
               gts_left=readcsv(csv_path_left)
               gts_right=readcsv(csv_path_right)
               for j in range(gts_left.shape[0]):
                   img_path_left='%s/s%02d/%s/%03d_%s/%08d.bmp'%(root_path,idx,'test',i,'left',j)
                   img_path_right='%s/s%02d/%s/%03d_%s/%08d.bmp'%(root_path,idx,'test',i,'right',j)
                   gt_v_l=gts_left[j,:3]
                   gt_v_r=gts_right[j,:3]
                   gt_l=convert_gaze(gt_v_l)
                   gt_r=convert_gaze(gt_v_r)
                   
                   gt_db.append({
                                'img_left': img_path_left,
                                'img_right': img_path_right,
                                'gaze_l':gt_l,
                                'gaze_r':gt_r,
                                })
        return gt_db

