
import os
import cv2
import numpy as np
import random
import torch
from torch.nn import functional as F
from torch.utils import data
from glob import glob
import os
import logging

from .pipelines.transforms import LoadingFile,RandomCrop,NotmaLization

class BaseDataset(data.Dataset):
    def __init__(self, 
                img_dir,
                label_dir=None,
                img_suffix=".png",
                label_suffix=".png", 
                pipelines = []
                 ):
        self.logger = logging.getLogger()
        
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_suffix = img_suffix
        self.label_suffix = label_suffix

        self.files = self.get_file_list()
        
        self.pipelines = []
        for operation in pipelines:
            op_name = operation['name']
            if 'args' in operation:
                args = operation['args']
            else:
                args = {}
            OP = eval(op_name)##  eval("LoadingFile")
            self.pipelines.append(OP(**args))
            # self.logger.info(f"Building Pipeline {operation}")

    def get_file_list(self):
        img_pres = [os.path.splitext(os.path.basename(fn))[0] for fn in glob(os.path.join(self.img_dir,f"*{self.img_suffix}"))]
        label_pres = [os.path.splitext(os.path.basename(fn))[0] for fn in glob(os.path.join(self.label_dir,f"*{self.label_suffix}"))]
        pres = set(img_pres) & set(label_pres)

        files = []
        for pre in pres:
            img_filename = os.path.join(self.img_dir , pre+self.img_suffix)
            label_filename = os.path.join(self.label_dir , pre+self.label_suffix)
            files.append({'img_filename':img_filename, 'label_filename':label_filename ,'name':pre})
        return files
        

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        file_item = self.files[index]
        results = {'meta':file_item}

        for op in self.pipelines:
            results = op(results)
        return results ## tensor img tensor label


