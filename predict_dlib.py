import math, shutil, os, time, argparse
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import dlib
from imutils import face_utils

from head_gaze_model.lib.models import HPNet,HGNet_G

from math import cos, sin
from PIL import Image
import cv2

from drawgaze import *
from detect_blinks_functions import *
#from detect_yawn_functions import *

from HRNet_Human_Pose_Estimation.lib.config import update_config,cfg
import HRNet_Human_Pose_Estimation.lib.models.pose_hrnet as pose_hrnet
#from HRNet_Human_Pose_Estimation.lib.core.inference import get_max_preds
from HRNet_Human_Pose_Estimation.lib.utils.transforms import *

#from mtcnn.mtcnn import MTCNN


DATA_PATH='./datas/xy/'

CHECKPOINTS_PATH = './output/head'

HP_CHECKPOINTS_PATH='./head_gaze_model/output/head/'
GL_CHECKPOINTS_PATH='./head_gaze_model/output/gaze/utmultiview'
GR_CHECKPOINTS_PATH='./head_gaze_model/output/gaze/utmultiview'
BP_CHECKPOINTS_PATH='./HRNet_Human_Pose_Estimation/output/mpii/pose_hrnet/w32_256x256_adam_lr1e-3'

HP_CHECKPOINTS_FILE='best_checkpoint_437_biwi_b4.pth.tar'
GL_CHECKPOINTS_FILE='best_checkpoint_232_gl_ut.pth.tar'
GR_CHECKPOINTS_FILE='best_checkpoint_231_gr_ut.pth.tar'
BP_CHECKPOINTS_FILE='model_best.pth'

EYE_AR_THRESH = 0.23  #"threshold to determine closed eyes" default=0.27
EYE_AR_CONSEC_FRAMES = 2 #"the number of consecutive frames the eye must be below the threshold" default=2

MOUTH_AR_THRESH = 0.70  #"threshold to determine closed eyes" default=0.27
MOUTH_AR_CONSEC_FRAMES = 3 #"the number of consecutive frames the eye must be below the threshold" default=2

def loadImage(path):
    try:
        im = Image.open(path).convert('RGB')
    except OSError:
        raise RuntimeError('Could not read image: ' + path)
        #im = Image.new("RGB", self.imSize, "white")

    return im

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='./HRNet_Human_Pose_Estimation/experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    args = parser.parse_args()
    return args

def main():
    global args, best_prec1, weight_decay, momentum
    
    args = parse_args()
    update_config(cfg, args)
    
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    cudnn.benchmark = True 
    
    COUNTER = 0
    TOTAL = 0
    
    COUNTER_MOUTH = 0
    TOTAL_MOUTH = 0
    
    hp_model = HPNet()
    gl_model = HGNet_G()
    gr_model = HGNet_G()
    bp_model = eval('pose_hrnet'+'.get_pose_net')(cfg, is_train=False)
    
    
    hp_model = torch.nn.DataParallel(hp_model).cuda()
    gl_model = torch.nn.DataParallel(gl_model).cuda()
    gr_model = torch.nn.DataParallel(gr_model).cuda()
    bp_model = torch.nn.DataParallel(bp_model).cuda()
    imSize=(224,224)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    transformFace = transforms.Compose([
            transforms.Resize(imSize),
            transforms.ToTensor(),
            normalize,
            transforms.Lambda(lambda x: torch.unsqueeze(x, 0)),
            ])
    transformEye = transforms.Compose([
            transforms.Resize(imSize),
            transforms.ToTensor(),
            #normalize,
            transforms.Lambda(lambda x: torch.unsqueeze(x, 0)),
            ])
    
    ##load the trained models
    hp_saved = load_checkpoint(HP_CHECKPOINTS_PATH,HP_CHECKPOINTS_FILE)
    if hp_saved:
        print('Loading head pose checkpoint with loss %.5f ...' % (hp_saved['best_prec1']))
        state = hp_saved['state_dict']
        try:
            hp_model.module.load_state_dict(state,strict=True)
        except:
            hp_model.load_state_dict(state,strict=True)
    else:
        print('Warning: Could not read head pose checkpoint!')
    
    gl_saved = load_checkpoint(GL_CHECKPOINTS_PATH,GL_CHECKPOINTS_FILE)
    if gl_saved:
        print('Loading gaze left checkpoint with loss %.5f ...' % (gl_saved['best_prec1']))
        state = gl_saved['state_dict']
        try:
            gl_model.module.load_state_dict(state,strict=True)
        except:
            gl_model.load_state_dict(state,strict=True)
    else:
        print('Warning: Could not read gaze left checkpoint!')
    
    gr_saved = load_checkpoint(GR_CHECKPOINTS_PATH,GR_CHECKPOINTS_FILE)
    if gr_saved:
        print('Loading gaze right checkpoint with loss %.5f ...' % (gr_saved['best_prec1']))
        state = gr_saved['state_dict']
        try:
            gr_model.module.load_state_dict(state,strict=True)
        except:
            gr_model.load_state_dict(state,strict=True)
    else:
        print('Warning: Could not read gaze right checkpoint!')
    
    print("[INFO] loading facial landmark predictor...")
    dlib_detector = dlib.get_frontal_face_detector()
    dlib_predictor = dlib.shape_predictor("./eye-blink-detection/shape_predictor_68_face_landmarks.dat")
    ##################################################################
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    keypoints={}
        
    for idx in range(0, 1):

        # detect face 
        #img_path='%s/frame%05d.jpg'%(DATA_PATH,idx)
        img_path='%s/162505508.png'%(DATA_PATH)
        
        image=cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        raw_img=loadImage(img_path)
        
        #result=detector.detect_faces(image)
        result=dlib_detector(image,0)
        
        # Result is an array with all the bounding boxes detected.
        shape = dlib_predictor(image, result[0])
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        bounding_box = [result[0].left(),result[0].top(),result[0].width(),result[0].height()]
        keypoints['left_eye']=np.mean(leftEye,axis=0)
        keypoints['right_eye']=np.mean(rightEye,axis=0)
        
        ad=0.3
        
        #loosely crop the face image
        x_min=bounding_box[0]
        x_max=bounding_box[0]+bounding_box[2]
        y_min=bounding_box[1]
        y_max=bounding_box[1]+bounding_box[3]
        
        h = y_max-y_min
        w = x_max-x_min
                
        x_min = max(int(x_min - ad * w), 0)
        x_max = min(int(x_max + ad * w), 640 - 1)
        y_min = max(int(y_min - (ad-0.15) * h), 0)
        y_max = min(int(y_max + (ad-0.15) * h), 360 - 1)
        bb=np.array([x_min,y_min,x_max,y_max],dtype=np.int32)
        
        
        imFace=raw_img.crop((bb[0],bb[1],bb[2],bb[3]))
        
        eye_w=int(w/6)
        eye_h=int(w/6*0.6)
        imEyeR=raw_img.crop((keypoints['left_eye'][0]-eye_w,keypoints['left_eye'][1]-eye_h,keypoints['left_eye'][0]+eye_w,keypoints['left_eye'][1]+eye_h))
        imEyeL=raw_img.crop((keypoints['right_eye'][0]-eye_w,keypoints['right_eye'][1]-eye_h,keypoints['right_eye'][0]+eye_w,keypoints['right_eye'][1]+eye_h))
        
        
        imFace_blink=imFace.copy()
        img_head=np.asarray(imFace.copy())
        img_gaze=np.asarray(imFace.copy())
        img_gaze=cv.cvtColor(img_gaze, cv.COLOR_RGB2BGR)
        #imFace_blink=imFace_blink.convert('BGR')
        
        imFace = transformFace(imFace)
        imEyeL = transformEye(imEyeL)
        imEyeR = transformEye(imEyeR)
        
        
        #detection
        with torch.no_grad():
            hp_output,hp_output_m,_,_= hp_model(imFace)
            gl_output,gl_output_m,_= gl_model(imEyeL)
            gr_output,gr_output_m,_= gr_model(imEyeR)
            bp_outputs = bp_model(imBody)
        
        ear,imFace_blink,mar=detect_ear(imFace_blink,dlib_detector,dlib_predictor)
        imgb_h,imgb_w=imFace_blink.shape[0],imFace_blink.shape[1]
        
        imEye_area=imFace_blink[int(imgb_h*0/9):int(imgb_h*8/9),int(imgb_w*2/9):int(imgb_w*7/9),:]
        
        if ear < EYE_AR_THRESH:
        	 COUNTER += 1
    		# otherwise, the eye aspect ratio is not below the blink
    		# threshold
        else:
    			# if the eyes were closed for a sufficient number of
    			# then increment the total number of blinks
        	if COUNTER >= EYE_AR_CONSEC_FRAMES:
        		TOTAL += 1
    			# reset the eye frame counter
        	COUNTER = 0
        
        if mar < MOUTH_AR_THRESH:
        	 COUNTER_MOUTH += 1
    		# otherwise, the eye aspect ratio is not below the blink
    		# threshold
        else:
    			# if the eyes were closed for a sufficient number of
    			# then increment the total number of blinks
        	if COUNTER_MOUTH >= MOUTH_AR_CONSEC_FRAMES:
        		TOTAL_MOUTH += 1
    			# reset the eye frame counter
        	COUNTER_MOUTH = 0
                
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #calculate head pose
        hp_pred=torch.argmax(hp_output_m,dim=2)
        hp_pred=hp_pred*3-99+1.5
        hp_output=(hp_output)*3-99
        
        hp_output=hp_output.cpu()
        [pitch,yaw,roll]=hp_output[0]
        
        print([pitch,yaw,roll])
        
        print(img_head.size)
        img_head=draw_axis(img_head,pitch,yaw,roll)
        
        #calculate gaze left
        gl_pred=torch.argmax(gl_output_m,dim=2)
        gl_pred=gl_pred*1-48+0.5
        gl_output=(gl_output)*1-48
        gl_output=(gl_output+gl_pred)/2
        
        #calculate gaze right
        #gr_pred=softmax(gr_output_m)
        gr_pred=torch.argmax(gr_output_m,dim=2)
        gr_pred=gr_pred*1-48+0.5
        gr_output=(gr_output)*1-48
        gr_output=(gr_output+gr_pred)/2
        
        g_output=(gl_output+gr_output)/2
        gl_output=g_output
        gr_output=g_output
        print(g_output)
        
        gl_output=gl_output.cpu()
        pitchyaw_l=gl_output[0]*np.pi/180
        img_gaze=draw_gaze(img_gaze,[keypoints['left_eye'][0]-bb[0],keypoints['left_eye'][1]-bb[1]],pitchyaw_l)
        
        
        gr_output=gr_output.cpu()
        pitchyaw_r=gr_output[0]*np.pi/180
        img_gaze=draw_gaze(img_gaze,[keypoints['right_eye'][0]-bb[0],keypoints['right_eye'][1]-bb[1]],pitchyaw_r)
        
        
        for k in range(shape.shape[0]):
            cv2.circle(image, (int(shape[k,0]),int(shape[k,1])), 2, (255,255,255), 1)

        image[0:0+imEye_area.shape[0],0:0+imEye_area.shape[1],:]=imEye_area
        
        img_head=img_head[int(imgb_h*0/9):int(imgb_h*8/9),int(imgb_w*2/9):int(imgb_w*7/9),:]
        image[120:120+img_head.shape[0],0:0+img_head.shape[1],:]=img_head
        
        img_gaze = imutils.resize(img_gaze, height=135)
        img_gaze=img_gaze[int(imgb_h*0/9):int(imgb_h*8/9),int(imgb_w*2/9):int(imgb_w*7/9),:]
        image[240:240+img_gaze.shape[0],0:0+img_gaze.shape[1],:]=img_gaze
        
        cv2.imwrite('./vis/%06d.jpg'%idx,image)
        
        print(idx, hp_output)


def draw_axis(img, pitch,yaw, roll, tdx=None, tdy=None, size = 30):
    print(yaw,roll,pitch)
    pitch = pitch * np.pi / 180
    yaw = (yaw * np.pi / 180)
    roll = roll * np.pi / 180
    
    img = imutils.resize(img, height=135)
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    
    
    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img


def angle_to_vector(pitch,yaw):
    pitch=pitch*np.pi/180
    yaw=yaw*np.pi/180
    normalized_gaze_vector = -np.array([
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            np.cos(pitch) * np.cos(yaw)
        ])
    return normalized_gaze_vector


def load_checkpoint(checkpoints_path,filename='best_checkpoint_437_biwi_b4.pth.tar'):
    filename = os.path.join(checkpoints_path, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state



if __name__ == "__main__":
    
    main()
    print('DONE')
