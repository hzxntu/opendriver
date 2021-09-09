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

from lib.datasets import MPIIGAZE
from lib.models import GNet_LRC
from lib.utils.loss import PYRMSELoss_LRC

from PIL import Image
import cv2

from scipy import stats

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def convert_to_unit_vector(angles):
    pitches = angles[:, 0]*np.pi/180
    yaws = angles[:, 1]*np.pi/180
    x = -torch.cos(pitches) * torch.sin(yaws)
    y = -torch.sin(pitches)
    z = -torch.cos(pitches) * torch.cos(yaws)
    norm = torch.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z

def compute_angle_error(predictions,labels):
    pred_x, pred_y, pred_z = convert_to_unit_vector(predictions)
    label_x, label_y, label_z = convert_to_unit_vector(labels)
    angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
    return torch.acos(angles) * 180 / np.pi


parser = argparse.ArgumentParser(description='opendriver-pytorch-Trainer.')
parser.add_argument('--sink', type=str2bool, nargs='?', const=False, default=False, help="Just sink and terminate.")
parser.add_argument('--reset', type=str2bool, nargs='?', const=False, default=True, help="Start from scratch (do not load).")
args = parser.parse_args()

# Change there flags to control what happens.
doLoad = not args.reset # Load checkpoint at the beginning
doTest = args.sink # Only run test, no training

workers = 32
epochs = 40
batch_size=16

base_lr = 0.0001
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
prec1 = 0
best_prec1 = 7.0
lr = base_lr

count_test = 0
count = 0

CHECKPOINTS_PATH = './output/gaze/mpii_lrc_loop_l'

def main():
    global args, best_prec1, weight_decay, momentum
    

    os.environ['CUDA_VISIBLE_DEVICES']='0'
    cudnn.benchmark = True 
    
    model = GNet_LRC(cls_num=60)
    optimizer = torch.optim.Adam(model.parameters(), lr,
                                #momentum=momentum,
                                weight_decay=weight_decay)

    
    criterion = PYRMSELoss_LRC(cls_num=60,f_dim=1280).cuda()
    
    model = torch.nn.DataParallel(model).cuda()

    epoch = 0
    if doLoad:
        saved = load_checkpoint()
        if saved:
            print('Loading checkpoint for epoch %05d with loss %.5f ...' % (saved['epoch'], saved['best_prec1']))
            state = saved['state_dict']
            try:
                model.module.load_state_dict(state,strict=True)
            except:
                model.load_state_dict(state,strict=True)
            
            epoch = saved['epoch']
            best_prec1 = saved['best_prec1']
        else:
            print('Warning: Could not read checkpoint!')
    
    dataTrain = MPIIGAZE(split='train', train_ls=range(14),test_ls=[14],side='left')
    dataVal = MPIIGAZE(split='test', train_ls=range(14),test_ls=[14],side='left')
    
   
    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    
    # Quick test
    if doTest:
        validate(val_loader, model, criterion, epoch)
        return


    for epoch in range(0, epoch):
        adjust_learning_rate(optimizer, epoch)
        
    for epoch in range(epoch, epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)
        

        # remember best prec@1 and save checkpoint
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def train(train_loader, model, criterion,optimizer, epoch):
    global count
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (imFace_L,imFace_R,gaze_l,gaze_r,g_l_b,g_r_b,g_l_mp,g_r_mp) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)
        imFace_L = imFace_L.cuda()
        imFace_R = imFace_R.cuda()
        gaze_l = gaze_l.cuda()
        gaze_r = gaze_r.cuda()
        g_l_b=g_l_b.cuda()
        g_r_b=g_r_b.cuda()
        g_l_mp=g_l_mp.cuda()
        g_r_mp=g_r_mp.cuda()
        
        # compute output
        output_l,output_r,out_l_b,out_r_b,out_l_mp,out_r_mp,out_l_f,out_r_f= model(imFace_L,imFace_R)
        
        loss = criterion(output_l,gaze_l,output_r,gaze_r,out_l_b,g_l_b,out_r_b,g_r_b,out_l_mp,g_l_mp,out_r_mp,g_r_mp,out_l_f,out_r_f)
        losses.update(loss.data.item(), imFace_L.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        count=count+1
        
        if i%100==0: 
            print('Epoch (train): [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses))

def validate(val_loader, model, criterion, epoch):
    global count_test
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()
    lossesMAE = AverageMeter()
    gt_p=AverageMeter()
    gt_y=AverageMeter()
    pred_p=AverageMeter()
    pred_y=AverageMeter()
    
    m_feat_l=AverageMeter()
    m_feat_r=AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    #idx_tensor = torch.arange(66).cuda()
    softmax=nn.Softmax(dim=2).cuda()
    
    for i, (imFace_L,imFace_R,gaze_l,gaze_r,g_l_b,g_r_b,g_l_mp,g_r_mp) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        imFace_L = imFace_L.cuda()
        imFace_R = imFace_R.cuda()
        gaze_l = gaze_l.cuda()
        gaze_r = gaze_r.cuda()
        g_l_b=g_l_b.cuda()
        g_r_b=g_r_b.cuda()
        g_l_mp=g_l_mp.cuda()
        g_r_mp=g_r_mp.cuda()

        # compute output
        with torch.no_grad():
            output_l,output_r,out_l_b,out_r_b,out_l_mp,out_r_mp,out_l_f,out_r_f= model(imFace_L,imFace_R)

        loss = criterion(output_l,gaze_l,output_r,gaze_r,out_l_b,g_l_b,out_r_b,g_r_b,out_l_mp,g_l_mp,out_r_mp,g_r_mp,out_l_f,out_r_f)
        
        #calculate the prediction
        pred_l=softmax(out_l_b)
        pred_l=torch.argmax(pred_l,dim=2)
        pred_l=pred_l*1+0.5
        
        pred_r=softmax(out_r_b)
        pred_r=torch.argmax(pred_r,dim=2)
        pred_r=pred_r*1+0.5
        
        output_l=(output_l+pred_l)/2
        output_r=(output_r+pred_r)/2
        
        #print(diff_w)
        output=(output_r+output_l)/2
        #output=(1-diff_w)*output_r+diff_w*output_l
        output=(output-30)*1
        hp=(gaze_l-30)*1
        
        gt_p.update(torch.mean(hp[:,0]),hp.size(0))
        gt_y.update(torch.mean(hp[:,1]),hp.size(0))
        pred_p.update(torch.mean(output[:,0]),output.size(0))
        pred_y.update(torch.mean(output[:,1]),output.size(0))
        #output=(output+pred)/2
        
        
        #calculate the error
        lossLin=compute_angle_error(output,hp)
        lossLin[torch.isnan(lossLin)] = 3
        lossLin=torch.mean(lossLin)
        
        lossmae= output- hp
        lossmae = torch.mul(lossmae,lossmae)
        lossmae = torch.mean(torch.sqrt(lossmae))
        lossesMAE.update(lossmae.item(), output.size(0))
        
        
        losses.update(loss.data.item(), output.size(0))
        lossesLin.update(lossLin.item(), output.size(0))
     
        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i%10==0:

            print('Epoch (val): [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                        epoch, i, len(val_loader), batch_time=batch_time,
                       loss=losses,lossLin=lossesLin))
            #print('MAE: ', lossesMAE.avg)
    print(lossesLin.avg)
    print(gt_p.avg-pred_p.avg)
    print(gt_y.avg-pred_y.avg)
    return lossesLin.avg


def load_checkpoint(filename='best_checkpoint.pth.tar'):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.isdir(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH, 0o777)
    bestFilename = os.path.join(CHECKPOINTS_PATH, 'best_' + filename)
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestFilename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


if __name__ == "__main__":
    
    main()
    print('DONE')
