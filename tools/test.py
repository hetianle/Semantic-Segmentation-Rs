import sys
import argparse
import os
import pprint
import shutil
import sys

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
from tensorboardX import SummaryWriter

from lib.models import UNet
from lib.datasets.base_dataset import BaseDataset
from lib.datasets.test_dataset import TestDataset
from lib.datasets.pipelines.transforms import LoadingFile, RandomCrop,NotmaLization

from lib.core.criterion import CrossEntropy, OhemCrossEntropy
from lib.core.function import train, validate
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, get_rank
from lib.core.function import test,testval

from yacs.config import CfgNode




def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--checkpoint', help='checkpoint file', default='')
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--img-suf', default='.png')
    parser.add_argument('--label-dir', default=None)
    parser.add_argument('--label-suf', default='.png')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config = CfgNode.load_cfg(open(args.cfg))

    logger, final_output_dir, _ = create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    # model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)
    model = UNet(3, 8)
    model.cuda()


    criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL)
    model = FullModel(model, criterion)

    

    logger.info('=> loading model from {}'.format(args.checkpoint))
    pretrained_dict = torch.load(args.checkpoint)
    model.load_state_dict(pretrained_dict['state_dict'])

    model = nn.DataParallel(model, device_ids=[0]).cuda()

    # prepare data
    if args.label_dir:
        dataset = BaseDataset(args.img_dir,
                                label_dir=args.label_dir,
                                img_suffix=args.img_suf,
                                label_suffix=args.label_suf,
                                pipelines=[{"name":"LoadingFile"},
                                            {"name":"NotmaLization"},
                                            ])
    else:
        dataset = TestDataset(args.img_dir,
                                label_dir=args.label_dir,
                                img_suffix=args.img_suf,
                                label_suffix=args.label_suf,
                                pipelines=[{"name":"LoadingFile"},
                                            {"name":"NotmaLization"},
                                            ])

    

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=1,
                                             pin_memory=True)

    start = timeit.default_timer()

    if args.label_dir:
        mean_IoU, IoU_array, pixel_acc, mean_acc = testval(
            config, dataset, dataloader, model,sv_dir=final_output_dir )

        msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
            Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, pixel_acc,
                                                    mean_acc)
        logging.info(msg)
        logging.info(IoU_array)
    else:
        test(config, dataset, dataloader, model, sv_dir=final_output_dir)

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int((end - start) / 60))
    logger.info('Done')


if __name__ == '__main__':
    main()
