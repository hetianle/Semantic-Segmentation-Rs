# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import cv2
from matplotlib.pyplot import axes, axis
from skimage.io import imread,imsave
from skimage.color import label2rgb #,lab2rgb

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
import sys

from lib.utils.utils import AverageMeter
from lib.utils.utils import get_confusion_matrix
from lib.utils.utils import adjust_learning_rate
from lib.utils.utils import get_world_size, get_rank


def put_palette(seg):
        # if palette is not None:
        palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                   [255, 0, 255], [255, 255, 0], [0, 255, 255], [100, 100, 0],
                   [100, 0, 100], [0, 100, 100]]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # color_seg = color_seg[..., ::-1]
        return color_seg


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp

def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
         trainloader, optimizer, model, writer_dict, device):
    
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()

    for i_iter, results in enumerate(trainloader):
        images= results['img']
        labels = results['gt']
        images = images.to(device)
        labels = labels.long().to(device)
        losses, _ = model(images, labels)
        loss = losses.mean()

        reduced_loss = reduce_tensor(loss)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)
        # print(f"RANK {rank}")
        if i_iter % config.PRINT_FREQ == 0 and rank == 0:
            print_loss = ave_loss.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters, 
                      batch_time.average(), lr, print_loss)
            logging.info(msg)
            
            writer.add_scalar('train_loss', print_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


def testval(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=True, num_classes=8,ignore_label=255):
    model.eval()
    confusion_matrix = np.zeros(
        (num_classes, num_classes))
    with torch.no_grad():
        for index, results in enumerate(tqdm(testloader)):
            image= results['img']
            label = results['gt']
            name = results['meta']['name'][0]
            size = label.size()
            _, pred = model(image, label,return_loss=False)
            pred = F.upsample(input=pred, size=(
                        size[-2], size[-1]), mode='bilinear')

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')
            pred = F.softmax(pred,dim=1)
            pred = torch.argmax(pred,dim=1)
            pred = pred.cpu().detach().numpy()
            pred = np.squeeze(pred)

            label = label.cpu().detach().numpy()
            label = np.squeeze(label)

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                num_classes,
                ignore_label)

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                imsave(os.path.join(sv_path,name+'.png'),np.uint8(pred))

                sv_path_color = os.path.join(sv_dir,'test_val_results_color')
                if not os.path.exists(sv_path_color):
                    os.mkdir(sv_path_color)
                imsave(os.path.join(sv_path,name+'.png'),pred)

                pred_color = put_palette(pred)
                imsave(os.path.join(sv_path_color,name+'.jpg'), pred_color,check_contrast=False)
                
            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, results in enumerate(tqdm(testloader)):
            image= results['img']
            name = results['meta']['name'][0]
            
            _, pred = model(image, None,return_loss=False)
            
            if pred.shape[-2] != image.shape[-2] or pred.shape[-1] != image.shape[-1]:
                pred = F.upsample(pred, image.shape[2:], mode='bilinear')

            if sv_pred:
                pred = F.softmax(pred,1)
                pred = torch.argmax(pred,1)
                pred = pred.cpu().detach().numpy()
                pred = np.squeeze(pred)
                pred = np.uint8(pred)
                
                
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)

                imsave(os.path.join(sv_path,name+'.png'),pred)
                sv_path_color = os.path.join(sv_dir,'test_results_color')
                if not os.path.exists(sv_path_color):
                    os.makedirs(sv_path_color)
                pred_color = put_palette(pred)
                imsave(os.path.join(sv_path_color,name+'.jpg'), pred_color,check_contrast=False)
                    
