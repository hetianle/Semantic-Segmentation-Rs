
from ast import arg
import sys
import os

from sklearn.utils import shuffle
script_file = os.path.realpath(__file__)
script_dir = os.path.dirname(script_file)
sys.path.insert(0, f'{script_dir}/../')
import argparse
import os
import pprint
import shutil


import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist

from tensorboardX import SummaryWriter

from lib.models import UNet
from lib.datasets.base_dataset import BaseDataset
from lib.datasets.pipelines.transforms import LoadingFile,RandomCrop

from lib.core.criterion import CrossEntropy, OhemCrossEntropy
from lib.core.function import train
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, get_rank

from yacs.config import CfgNode

import yaml

from mmcv.cnn.utils import revert_sync_batchnorm


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='/mnt/lustrenew2/hetianle/IGSNRR/light_project/Semantic-Segmentation-Rs/experiments/loveda/seg_unet_2048x2048_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml',
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--launcher",default=None)

    args = parser.parse_args()
   

    return args

def main():
    args = parse_args()
    config = CfgNode.load_cfg(open(args.cfg))

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    if args.local_rank == 0:
        logger.info(pprint.pformat(args))
        logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    

    device = torch.device('cuda:{}'.format(args.local_rank))

    # build model

    model = UNet(3,8)

    if args.launcher == 'pytorch':
        distributed = True
    else: 
        distributed = False


    if args.local_rank == 0:
        # provide the summary of model
        dump_input = torch.rand(
            (1, 3, 512 , 512)
            )
        logger.info(get_model_summary(model.to(device), dump_input.to(device)))
        logger.info(f"Distribted : {distributed}")

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        gpus =  range(world_size)
    else:
        model = revert_sync_batchnorm(model)
        gpus = [0]

    # prepare data

    if isinstance(config.DATASET,(list,tuple)):
        if args.local_rank==0:
            print(config.DATASET)


        datasetlist = [BaseDataset(
            img_dir= ds["IMG_DIR"],
            label_dir= ds["LABEL_DIR"],
            pipelines=  ds["PIPELINE"]) for ds in config.DATASET]

        train_dataset = torch.utils.data.dataset.ConcatDataset(datasetlist)

    else:
        train_dataset = BaseDataset(
            img_dir= config.DATASET.IMG_DIR,
            label_dir= config.DATASET.LABEL_DIR,
            pipelines=  config.DATASET.PIPELINE
        )


    if args.local_rank == 0:
        logger.info(f"Loading {len(train_dataset)} samples.")

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if distributed:
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
            num_workers=config.TRAIN.WORKERS_PER_GPU,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler)
    else:
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
            num_workers=config.TRAIN.WORKERS_PER_GPU * len(gpus),
            pin_memory=True,
            drop_last=True,
            shuffle= True,
            sampler=train_sampler)


    criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,weight=torch.tensor([0,2,4,1,2,1,1,1]).float().cuda())

    model = FullModel(model, criterion)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        model = nn.parallel.DataParallel(
            model,device_ids=[args.local_rank] , output_device=args.local_rank
        )

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD([{'params':
                                  filter(lambda p: p.requires_grad,
                                         model.parameters()),
                                  'lr': config.TRAIN.LR}],
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = np.int(train_dataset.__len__() / 
                        config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    best_mIoU = 0
    last_epoch = 0

    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, 
                        map_location=lambda storage, loc: storage)
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH 
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    # extra_iters = config.TRAIN.EXTRA_EPOCH * epoch_iters
    
    for epoch in range(last_epoch, end_epoch):
        
        train(config, epoch, config.TRAIN.END_EPOCH, 
                epoch_iters, config.TRAIN.LR, num_iters,
                trainloader, optimizer, model, writer_dict,
                device)

        if args.local_rank == 0 and (epoch+1) % 10 == 0:
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + f'epoch_{epoch+1}_checkpoint.pth.tar'))
            torch.save({
                'epoch': epoch+1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, f'epoch_{epoch+1}_checkpoint.pth.tar'))

            if epoch == end_epoch - 1:
                torch.save(model.module.state_dict(),
                       os.path.join(final_output_dir, 'final_state.pth'))

                writer_dict['writer'].close()
                end = timeit.default_timer()
                logger.info('Hours: %d' % np.int((end-start)/3600))
                logger.info('Done')


if __name__ == '__main__':
    main()
    # nn.SyncBatchNorm.convert_sync_batchnorm()