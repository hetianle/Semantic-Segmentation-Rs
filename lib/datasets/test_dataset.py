
from .base_dataset import BaseDataset
import os
from glob import glob


class TestDataset(BaseDataset):
    
    def get_file_list(self):
        img_pres = [os.path.splitext(os.path.basename(fn))[0] for fn in glob(os.path.join(self.img_dir,f"*{self.img_suffix}"))]
        files = []
        for pre in img_pres:
            img_filename = os.path.join(self.img_dir , pre+self.img_suffix)
            files.append({'img_filename':img_filename, 'label_filename':None ,'name':pre})
        return files