
import numpy as np
from skimage import io

predPath = './tmp/pred_atom_10.tif'
predImage = io.imread(predPath)
predImage = np.array(predImage)

atomPath = './tmp/atom_10.tif'
atomImage = io.imread(atomPath)
atomImage = np.array(atomImage)
atomMax = atomImage.max()
atomMin = atomImage.min()
atomImage = (atomImage - atomMin)/(atomMax - atomMin) * 2**32


refPath = './tmp/ref_10.tif'
refImage = io.imread(refPath)
refImage = np.array(refImage)[110:366,110:366]
refMax = refImage.max()
refMin = refImage.min()
refImage = (refImage - refMin)/(refMax - refMin) * 2**32
print(predImage.max(), predImage.min())
print('Atom max min', atomImage.max(), atomImage.min())

#io.imsave('./tmp/refCrop.tif', refImage)
predOD = atomImage[110:366,110:366] - predImage#[110:366,110:366]
predRef = refImage - predImage
predOD = np.asarray(predOD, np.uint32)
predRef = np.asarray(predRef, np.uint32)
io.imsave('./tmp/predOD_10.tif', predOD)
io.imsave('./tmp/predRef_10.tif',predRef)

#print(predImage.shape)
#print(atomImage.shape)