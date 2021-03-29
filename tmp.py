import os
import torch
import yaml
import numpy as np
from PIL import Image
import torchvision.utils as vutils

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
test_img = np.asarray(pil_loader('./atom_data/test/000001.png'))
crop_test = test_img[110:366, 110:366,:]
crop_test =Image.fromarray(crop_test)
#vutils.save_image(crop_test, './tmp/sample.png', padding=0, normalize=True)
crop_test.save('./tmp/sample.png')