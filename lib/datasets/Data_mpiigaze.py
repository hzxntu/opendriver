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
from ..utils.heatmap import gen_heatmap
import random
import pandas as pd
import pathlib
import scipy.io
import cv2


def convert_gaze(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    pitch = np.arcsin(-y) * 180 / np.pi 
    yaw = np.arctan2(-x, -z) * 180 / np.pi 
    return np.array([pitch, yaw]).astype(np.float32)

class MPIIGAZE(data.Dataset):
    def __init__(self, split = 'train', train_ls=range(14), test_ls=[14],imSize=(224,224), cali_num=0,side='left'):

        self.imSize = imSize
        self.mpii_root='./data/mpiigaze/MPIIGaze/Data/Normalized/'
        self.mpii_img_root='./data/mpiigaze/MPIIGaze/Data/Normalized_imgs/'
        self.eval_root= './data/mpiigaze/MPIIGaze/Evaluation Subset/sample list for eye image/'
        
        self.side=side
        self.cali=True
        self.cali_num=cali_num

        print('Loading MPIIGaze dataset...')

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.transformEye = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        ])
        
        self.trainidx=train_ls
        self.testidx=test_ls
        
        print('test dataset:', self.testidx)


        if split == 'train':
            self.indices=self._get_db(True)
        elif split == 'test':
            self.indices=self._get_db(False)
        elif split=='refine':
            self.indices=self._get_db_refine()
        
            
        print('Loaded MPIIGaze dataset split "%s" with %d records...' % (split, len(self.indices)))

    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)

        return im
        

    def __getitem__(self, index):
        sample = self.indices[index]
        
        if sample['side']=='left':
           lr_typo='l'
        else:
           lr_typo='r'
        
        smplPath_L='%s/%s/%s/%s_%s.npy'%( self.mpii_img_root,sample['person'],sample['day'],sample['img'].split('.')[0],'l')
        smplPath_R='%s/%s/%s/%s_%s.npy'%( self.mpii_img_root,sample['person'],sample['day'],sample['img'].split('.')[0],'r')
        
        imEye_L=np.load(smplPath_L)
        imEye_R=np.load(smplPath_R)
        imEye_R=np.flip(imEye_R,1)
        gaze_l=sample['gaze_l']
        gaze_r=sample['gaze_r']
        
        imEye_L = self.transformEye(imEye_L)
        imEye_R = self.transformEye(imEye_R)
        
        
        gaze_l=np.clip(gaze_l,-30,29.9)
        gaze_r=np.clip(gaze_r,-30,29.9)

        bins = np.array(range(-30, 31, 1))
        g_l_b = np.digitize([gaze_l[0], gaze_l[1]], bins) - 1
        g_r_b = np.digitize([gaze_r[0], gaze_r[1]], bins) - 1
        g_l_b = torch.LongTensor(g_l_b)
        g_r_b = torch.LongTensor(g_r_b)
        
        bins_mp = np.array(range(-30, 33, 3))
        g_l_mp = np.digitize([gaze_l[0], gaze_l[1]], bins_mp) - 1
        g_r_mp = np.digitize([gaze_r[0], gaze_r[1]], bins_mp) - 1
        g_l_mp = torch.LongTensor(g_l_mp)
        g_r_mp = torch.LongTensor(g_r_mp)
            
        gaze_l=(gaze_l+30)/1
        gaze_r=(gaze_r+30)/1
        gaze_l = torch.FloatTensor(gaze_l)
        gaze_r = torch.FloatTensor(gaze_r)

        return imEye_L,imEye_R,gaze_l,gaze_r,g_l_b,g_r_b,g_l_mp,g_r_mp
    
    def __len__(self):
        return len(self.indices)
        
    def _get_db(self,is_train):
        if is_train:
            # use ground truth bbox
            gt_db = self._load_samples(self.trainidx,True)
        else:
            # use bbox from detection
            gt_db = self._load_samples(self.testidx,False)
        return gt_db
    
        
    def _load_samples(self,dataidx,is_train):
        gt_db_train=[]
        gt_db_val=[]
        gt_db=[]
        
        for idx in dataidx:
             
           filenames = dict()
           left_gazes = dict()
           right_gazes = dict()
           masks_left=dict()
           masks_right=dict()
           
           person_id = 'p%02d'%idx
           person_dir = self.mpii_root + person_id
           person_dir=pathlib.Path(person_dir)
           
           df = self._get_eval_info(person_id, self.eval_root)
           
          
           for path in sorted(person_dir.glob('*')):
               mat_data = scipy.io.loadmat(path.as_posix(),
                                  struct_as_record=False,
                                  squeeze_me=True)
               data = mat_data['data']
               day = path.stem
               filenames[day] = mat_data['filenames']
               left_gazes[day] = data.left.gaze
               right_gazes[day] = data.right.gaze
               masks_left[day]=np.zeros([len(filenames[day])])
               masks_right[day]=np.zeros([len(filenames[day])])
               
               if not isinstance(filenames[day], np.ndarray):
                  left_gazes[day] = np.array([left_gazes[day]])
                  right_gazes[day] = np.array([right_gazes[day]])
                  filenames[day] = np.array([filenames[day]])
               

           for _, row in df.iterrows():
              day = row.day
              index = np.where(filenames[day] == row.filename)[0][0]
              if row.side == 'left':
                  masks_left[day][index]=1
              else:
                  masks_right[day][index]=1
           
           if is_train and self.cali:
              selected=random.sample(range(1500),self.cali_num)
           else:
              selected=range(1500)
           
           count=0
           
           for path in sorted(person_dir.glob('*')):
               day = path.stem
               
               for i in range(len(filenames[day])):
                   if self.side=='left':
                       if masks_left[day][i]==1:  
                           #random select
                           if count not in selected:
                              count+=1
                              continue
                           count+=1
                           
                           if np.max(convert_gaze(left_gazes[day][i]))>30 or np.min(convert_gaze(left_gazes[day][i]))<-30 :
                              continue
                           if np.max(convert_gaze(right_gazes[day][i]))>30 or np.min(convert_gaze(right_gazes[day][i]))<-30 :
                              continue
                           gt_db.append({
                                'img': filenames[day][i],
                                'person':person_id,
                                'day': day,
                                'side':'left',
                                'gaze_l':convert_gaze(left_gazes[day][i]),
                                'gaze_r':convert_gaze(right_gazes[day][i]),
                                })
                   
                   else:
                       if masks_right[day][i]==1:
                           
                           #random select
                           if count not in selected:
                              count+=1
                              continue
                           count+=1
                           
                           if np.max(convert_gaze(right_gazes[day][i]))>30 or np.min(convert_gaze(right_gazes[day][i]))<-30 :
                              continue
                           if np.max(convert_gaze(left_gazes[day][i]))>30 or np.min(convert_gaze(left_gazes[day][i]))<-30 :
                              continue
                           gt_db.append({
                                'img': filenames[day][i],
                                'person':person_id,
                                'day': day,
                                'side':'right',
                                #'gaze':convert_gaze(right_gazes[day][i]) * np.array([1, -1]),
                                'gaze_l':convert_gaze(left_gazes[day][i]),
                                'gaze_r':convert_gaze(right_gazes[day][i]),
                                })
                   
                
        return gt_db #gt_db_train,gt_db_val

    def _get_db_refine(self):
        gt_db_train=[]
        gt_db_val=[]
        gt_db=[]
        
        
        dicts={}
        for p in range(60):
           for y in range(60):
               dicts['%02d%02d'%(p,y)]=[]

        for idx in self.testidx:
           
           filenames = dict()
           left_gazes = dict()
           right_gazes = dict()
           masks_left=dict()
           masks_right=dict()
           
           person_id = 'p%02d'%idx
           person_dir = self.mpii_root + person_id
           person_dir=pathlib.Path(person_dir)
           
           df = self._get_eval_info(person_id, self.eval_root)
           
          
           for path in sorted(person_dir.glob('*')):
               mat_data = scipy.io.loadmat(path.as_posix(),
                                  struct_as_record=False,
                                  squeeze_me=True)
               data = mat_data['data']
               day = path.stem
               filenames[day] = mat_data['filenames']
               left_gazes[day] = data.left.gaze
               right_gazes[day] = data.right.gaze
               masks_left[day]=np.zeros([len(filenames[day])])
               masks_right[day]=np.zeros([len(filenames[day])])
               
               if not isinstance(filenames[day], np.ndarray):
                  left_gazes[day] = np.array([left_gazes[day]])
                  right_gazes[day] = np.array([right_gazes[day]])
                  filenames[day] = np.array([filenames[day]])
               

           for _, row in df.iterrows():
              day = row.day
              index = np.where(filenames[day] == row.filename)[0][0]
              if row.side == 'left':
                  masks_left[day][index]=1
              else:
                  masks_right[day][index]=1


           for path in sorted(person_dir.glob('*')):
               day = path.stem
               
               d_num=0
               for i in range(len(filenames[day])):
               
                   if masks_right[day][i]!=1:
                       if np.max(convert_gaze(right_gazes[day][i]))>30 or np.min(convert_gaze(right_gazes[day][i]))<-30 :
                          continue
                       
                       gt_db.append({
                            'img': filenames[day][i],
                            'person':person_id,
                            'day': day,
                            'side':'right',
                            'gaze':convert_gaze(right_gazes[day][i]) * np.array([1, -1]),
                            })
                       d_num+=1
                       if d_num>2:
                          break
                       
                
        return gt_db
    
    def _get_eval_info(self,person_id, eval_dir):
        eval_path = eval_dir + '%s.txt'%person_id
        df = pd.read_csv(eval_path,
                         delimiter=' ',
                         header=None,
                         names=['path', 'side'])
        df['day'] = df.path.apply(lambda path: path.split('/')[0])
        df['filename'] = df.path.apply(lambda path: path.split('/')[1])
        df = df.drop(['path'], axis=1)
        return df
