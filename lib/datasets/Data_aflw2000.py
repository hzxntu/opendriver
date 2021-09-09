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

class AFLW(data.Dataset):
    def __init__(self, split = 'test', imSize=(224,224),aug=False):

        self.imSize = imSize
        self.aflw_root= './data/'

        print('Loading AFLW2000 dataset...')

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transformFace = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            self.normalize,
        ])
        
        self.testidx=['AFLW2000']
        
        self.indices=self._get_db()
        
        self.aug=aug
            
        print('Loaded AFLW2000 dataset split "%s" with %d records...' % (split, len(self.indices)))

    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)

        return im

    def __getitem__(self, index):
        sample = self.indices[index]

        imgPath=sample['img']
        raw_img = self.loadImage(imgPath)
        
        pt2d_x=sample['pt2d_x']
        pt2d_y=sample['pt2d_y']
        PYR=sample['pose_para']
        
        img_w = raw_img.size[0]
        img_h = raw_img.size[1]
        
		    # Crop the face loosely
        x_min = int(min(pt2d_x))
        x_max = int(max(pt2d_x))
        y_min = int(min(pt2d_y))
        y_max = int(max(pt2d_y))
		
        h = y_max-y_min
        w = x_max-x_min
            
        #bias=[0,0,0,0]
        #pt2d_x_org=sample['pt2d_x_org']
        #pt2d_y_org=sample['pt2d_y_org']

        #if pt2d_x_org[12]<0:
        #   bias[0]=0.05
        #if pt2d_x_org[16]<0:
        #   bias[1]=0.1
        
        ad=0.6
        
        x_min = max(int(x_min - (ad+bias[0]) * w), 0)
        x_max = min(int(x_max + (ad+bias[1]) * w), img_w - 1)
        y_min = max(int(y_min - ad * h), 0)
        y_max = min(int(y_max + (ad+bias[2]) * h), img_h - 1)
        
        
        imFace = raw_img.crop((x_min,y_min,x_max,y_max))
        
        #imFace.save('%s'%(imgPath.split('/')[-1]))
        
        imFace = self.transformFace(imFace)
        
        PYR=np.clip(PYR,-99,99)

        bins = np.array(range(-99, 102, 3))
        hp_m = np.digitize([PYR[0], PYR[1], PYR[2]], bins) - 1
        hp_m = torch.LongTensor(hp_m)
        
        bins_mp = np.array(range(-99, 108, 9))
        hp_m_mp = np.digitize([PYR[0], PYR[1], PYR[2]], bins_mp) - 1
        hp_m_mp = torch.LongTensor(hp_m_mp)
        
        
        hp=(PYR+99)/3
        hp = torch.FloatTensor(hp)

        return imFace,hp,hp_m,hp_m_mp
    
    def __len__(self):
        return len(self.indices)
        
    def _get_db(self):
        
        gt_db = self._load_samples(self.testidx,self.aflw_root)
        
        return gt_db
        
    def _load_samples(self,idx_list,root_path):
        # get images and groundtruths(position of center of head, pitch,yaw,roll,rotation matrix)
        gt_db=[]

        for idx in idx_list:
            image_files,mat_files=self.samples_from_idx(idx,root_path)
            
            imagefile_path=os.path.join(root_path,idx)
            matfile_path=os.path.join(root_path,idx)
            
            e_idx=len(image_files)
            for i in range(e_idx):

                pose_para,pt2d_x,pt2d_y,pt2d_x_org,pt2d_y_org=self._read_mat_file(os.path.join(matfile_path,mat_files[i]))
                
                pitch = pose_para[0] * 180 / np.pi
                yaw = pose_para[1] * 180 / np.pi
                roll = pose_para[2] * 180 / np.pi
                
                if if max(abs(pitch),abs(yaw),abs(roll))>99:
                   continue
                
                pose_para=np.array([pitch,yaw,roll])
                
                gt_db.append({
                    'img': os.path.join(imagefile_path,image_files[i]),
                    'pose_para':pose_para,
                    'pt2d_x':pt2d_x,
                    'pt2d_y':pt2d_y,
                    'pt2d_x_org':pt2d_x_org,
                    'pt2d_y_org':pt2d_y_org,
                    })
                
        return gt_db
    def _read_mat_file(self,matfile):
        mat_contents = sio.loadmat(matfile)
        pose_para = mat_contents['Pose_Para'][0]
        pt2d = mat_contents['pt2d']
		
        pt2d_x = pt2d[0,:]
        pt2d_y = pt2d[1,:]
        
        pt2d_x_org=pt2d[0,:]
        pt2d_y_org=pt2d[1,:]

        # remove negative value in AFLW2000
        pt2d_idx = pt2d_x>0.0
        pt2d_idy= pt2d_y>0.0

        pt2d_id = pt2d_idx
        if sum(pt2d_idx) > sum(pt2d_idy):
           pt2d_id = pt2d_idy
		
        pt2d_x = pt2d_x[pt2d_id]
        pt2d_y = pt2d_y[pt2d_id]
        
        return pose_para,pt2d_x,pt2d_y,pt2d_x_org,pt2d_y_org
    def samples_from_idx(self,idx,root_path):

        imagefile_path=os.path.join(root_path,idx)

        file_names=os.listdir(imagefile_path)

        image_files=[]
        mat_files=[]

        for file_name in file_names:
            if os.path.splitext(file_name)[1]==('.jpg'):


                image_files.append(file_name)

                mat_file=os.path.splitext(file_name)[0]+'.mat'
                mat_files.append(mat_file)

        return image_files,mat_files
