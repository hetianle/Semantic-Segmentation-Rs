import sys
import cv2
import random
import numpy as np


class LoadingFile(object):
    def __init__(self,order='rgb'):
        self.order = order
        
    def __call__(self,results):
        img_filename = results['meta']['img_filename']
        label_filename = results['meta']['label_filename']
        img = cv2.imread(img_filename,cv2.IMREAD_COLOR)
        if self.order == 'rgb':
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if label_filename!=None:
            label = cv2.imread(label_filename,cv2.IMREAD_GRAYSCALE)
        else:
            label = None

        
        results['img'] = img
        results['gt'] = label
        return results


class RandomCrop(object):
    def __init__(self,crop_size=(512,512)):
        self.crop_size = crop_size
    
    def __call__(self,results):
        img = results['img']
        gt = results['gt']

        H, W = img.shape[:2]

        # image = self.pad_image(image, h, w, self.crop_size,
        #                         (0.0, 0.0, 0.0))
        # label = self.pad_image(label, h, w, self.crop_size,
                                # (self.ignore_label,))
        
        # new_h, new_w = label.shape
        h = random.randint(0, H - self.crop_size[0])
        w = random.randint(0, W - self.crop_size[1])


        img = img[h:h+self.crop_size[0], w:w+self.crop_size[1]]
        gt = gt[h:h+self.crop_size[0], w:w+self.crop_size[1]]

        results['img'] = img
        results['gt'] = gt
        
        return results


class NotmaLization(object):
    def __init__(self,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225], channel_first=True):
        self.mean = mean
        self.std = std
        self.channel_first = channel_first
    
    def __call__(self, results):
        image = results['img']
        label = results['gt']
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        
        if self.channel_first:
            image = image.transpose((2, 0, 1))
        
        results['img'] = image
        
        if isinstance(label,np.ndarray):
            label = np.array(label).astype('int32')
            results['gt'] = label
        else:
            del results['gt']
            del results['meta']['label_filename']
       
        return results
        
