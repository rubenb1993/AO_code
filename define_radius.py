import sys
import os
import time
if "C:\Program Files\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Program Files\Micro-Manager-1.4")
import MMCorePy
import PIL.Image
import numpy as np
#from matplotlib import rc
import Hartmann as Hm
import displacement_matrix as Dm
import mirror_control as mc
import matplotlib.pyplot as plt
from matplotlib import cm
import edac40
import matplotlib.ticker as ticker

px_size_int = 5.2e-6
#x0 = 545
#y0 = 490


#### Gather interferogram 
impath_int = os.path.abspath("dm_int.tif")
image_int = np.asarray(PIL.Image.open(impath_int)).astype(float)
[ny,nx] = image_int.shape
x = np.linspace(1, nx, nx)
y = np.linspace(1, ny, ny)
xx, yy = np.meshgrid(x, y)
centre = np.zeros(2)
image_int[image_int<6] = 0
image_int[image_int > 6] = 255
norm_photons = 1.0/np.sum(image_int)
centre[0] = norm_photons * np.sum(image_int * xx)
centre[1] = norm_photons * np.sum(image_int * yy)

x0 = centre[0]
y0 = centre[1]
print nx-x0, ny-y0

size = 375
mask = [np.sqrt((xx-x0)**2 + (yy-y0)**2) > size]
image_int *= np.squeeze(mask)
print size*px_size_int
#image_int[image_int < 10] = 0
##image_int[:, :530] = 0
##image_int[:, 600:] = 0
##image_int[135:870, :] = 0
plt.imshow(image_int, cmap = 'bone')
plt.scatter(centre[0], centre[1])
plt.show()
#a = np.where(image_int > 10.0)
#print(np.array(a).shape[1])
#ind = np.triu_indices(np.array(a).shape[1], k=1) # indices of upper triangular matrix
#dist_vec = np.sqrt((a[0][ind[0]] - a[0][ind[1]])**2 + (a[1][ind[0]] - a[1][ind[1]])**2)
#plt.hist(dist_vec[dist_vec > 200])
#plt.show()
#print np.max(dist_vec), np.max(dist_vec) * px_size_int
