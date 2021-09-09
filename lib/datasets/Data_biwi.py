import torch.utils.data as data
import scipy.io as sio
from PIL import Image,ImageOps,ImageEnhance
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import re
import struct
import math
import random
import cv2

class BIWI(data.Dataset):
    def __init__(self, split = 'train', imSize=(224,224),aug=False):

        self.dataPath = './data/biwi'
        self.imSize = imSize
        self.gt_root='./data/biwi_org/biwi/kinect_head_pose_db/hpdb'
        
        self.aug=False

        print('Loading BIWI dataset...')

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.transformFace = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            self.normalize,
        ])
        
        #self.trainidx=list(range(1,25,3))+list(range(2,25,3))
        self.testidx=list(range(1,25))

        if split == 'train':
            self.indices=self._get_db(True)
            self.aug=True
        else:
            self.indices=self._get_db(False)
            
        print('Loaded BIWI dataset split "%s" with %d records...' % (split, len(self.indices)))

    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)

        return im

        
    def _read_pyr_bin(self,filepath):

        binfile = open(filepath, 'rb')       
        size = os.path.getsize(filepath) 
        PosPYR=[]
        for i in range(int(size/4)):
            data = binfile.read(4)         
            num = struct.unpack('f', data)
            PosPYR.append(num[0])
        binfile.close()

        return PosPYR
    def _read_pyr_txt(self,filepath,calibration):
			  # Load pose in degrees
  			pose_annot = open(filepath, 'r')
  			R = []
  			for line in pose_annot:
  				line = line.strip('\n').split(' ')
  				L = []
  				if line[0] != '':
  					for nb in line:
  						if nb == '':
  							continue
  						L.append(float(nb))
  					R.append(L)
  
  			R = np.array(R)
  			T = R[3,:]
  			R = R[:3,:]
  			pose_annot.close()
        
        #original
  			R = np.transpose(R)
  			roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
  			yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
  			pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi
        
  			return np.array([pitch,yaw,roll])
    
    def _readCalibrationFile(self,calibration_file):
        """
        Reads the calibration parameters
        """
        cal  = {}
        fh = open(calibration_file, 'r')
        # Read the [intrinsics] section
        vals = []
        for i in range(3):
            vals.append([float(val) for val in fh.readline().strip().split(' ')])
        cal['intrinsics'] = np.array(vals).reshape(3,3)
        
        # Read the [intrinsics] section
        fh.readline().strip()
        vals = []
        vals.append([float(val) for val in fh.readline().strip().split(' ')])
        cal['dist'] = np.array(vals).reshape(4,1)
        
        # Read the [R] section
        fh.readline().strip()
        vals = []
        for i in range(3):
            vals.append([float(val) for val in fh.readline().strip().split(' ')])
        cal['R'] = np.array(vals).reshape(3,3)
        
         # Read the [T] section
        fh.readline().strip()
        vals = []
        vals.append([float(val) for val in fh.readline().strip().split(' ')])
        cal['T'] = np.array(vals).reshape(3,1)
    
        # Read the [resolution] section
        fh.readline().strip()
        cal['size'] = [int(val) for val in fh.readline().strip().split(' ')]
        cal['size'] = cal['size'][0], cal['size'][1]    
        
        fh.close()
        return cal

    def __getitem__(self, index):
        sample = self.indices[index]

        imgPath=sample['img']
        raw_img = self.loadImage(imgPath)
        
        kpsPath=sample['face']
        kps=np.load(kpsPath)
        
        hp=self._read_pyr_txt(sample['gt'],sample['cali'])
        PYR=np.clip(hp,-81,81)
        
        ad=0.3
        
        x_min=kps[0]
        x_max=kps[0]+kps[2]
        y_min=kps[1]
        y_max=kps[1]+kps[3]
        
        h = y_max-y_min
        w = x_max-x_min
                
        x_min = max(int(x_min - (ad+0.05) * w), 0)
        x_max = min(int(x_max + (ad) * w), 640 - 1)
        y_min = max(int(y_min - (ad-0.15) * h), 0)
        y_max = min(int(y_max + (ad-0.15) * h), 480 - 1)
        
        bb=np.array([x_min,y_min,x_max,y_max],dtype=np.int32)
        
        imFace=raw_img.crop((bb[0],bb[1],bb[2],bb[3]))
        
        imFace = self.transformFace(imFace)
        
        bins = np.array(range(-81, 84, 3))
        hp_m = np.digitize([PYR[0], PYR[1], PYR[2]], bins) - 1
        hp_m = torch.LongTensor(hp_m)
        
        bins_mp = np.array(range(-81, 90, 9))
        hp_m_mp = np.digitize([PYR[0], PYR[1], PYR[2]], bins_mp) - 1
        hp_m_mp = torch.LongTensor(hp_m_mp)
            
        hp=(PYR+81)/3
        hp = torch.FloatTensor(hp)
        
        return imFace,hp,hp_m,hp_m_mp
    
    def __len__(self):
        return len(self.indices)
        
    def _get_db(self,is_train):
        if is_train:
            # use ground truth bbox
            gt_db = self._load_samples(self.trainidx)
        else:
            # use bbox from detection
            gt_db = self._load_samples(self.testidx)
        return gt_db
        
    def _load_samples(self,idx_list):
        # get images and groundtruths(position of center of head, pitch,yaw,roll,rotation matrix)
        gt_db=[]

        for idx in idx_list:
            
            file_names=os.listdir('%s/%02d_face'%(self.dataPath,idx))
            
            calibration = self._readCalibrationFile('%s/%02d/rgb.cal'%(self.gt_root,idx))
               
            for file_name in file_names:

                if os.path.splitext(file_name)[1]==('.npy'):
                  
                  str_split=os.path.splitext(file_name)[0].split('_')
                  #pyr_file=str_split[0]+'_'+str_split[1]+'_pose.bin'
                  pyr_file=str_split[0]+'_'+str_split[1]+'_pose.txt'
                  rawimg_file=str_split[0]+'_'+str_split[1]+'_rgb.png'
                
                  #img_name=gazedata[0].split('.')[0]
                  face_path=os.path.join(self.dataPath,'%02d_face'%idx,'%s'%file_name)
                  rawimg_path=os.path.join(self.gt_root,'%02d'%idx,rawimg_file)
                  gt_path=os.path.join(self.gt_root,'%02d'%idx,pyr_file)
                  

                  if os.path.exists(gt_path) and os.path.exists(face_path):
                        
                        gt_db.append({
                            'gt': gt_path,
                            'face':face_path,
                            'img':rawimg_path,
                            'cali': calibration,
                            })
                
        return gt_db
