from skimage import io

from utils.tools import get_config, random_bbox, mask_image, is_image_file, default_loader, normalize, get_model_list
import numpy as np
from utils.tools import tif_loader, transfer2tensor

refImage = tif_loader('./tmp/refCrop.tif')
originRef = io.imread('./tmp/refCrop.tif')
refScale = refImage[...,0]*originRef.max()
refScale = np.asarray(refScale, dtype=np.uint32)
io.imsave('./tmp/debug.tif', refScale)
#io.imsave('./tmp/debug_ref.tif', )
