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
impath_mirror = os.path.abspath("20170105_interferograms\image_ref_mirror.tif")
image_mirror = np.asarray(PIL.Image.open(impath_mirror)).astype(float)
impath_int = os.path.abspath("dm_int.tif")
image_int = np.asarray(PIL.Image.open(impath_int)).astype(float)


[ny,nx] = image_int.shape
x = np.linspace(1, nx, nx)
y = np.linspace(1, ny, ny)
xx, yy = np.meshgrid(x, y)
centre = np.zeros(2)
#image_mirror[0:100, :] = 0
##image_mirror[image_mirror<8] = 0
##image_mirror[image_mirror > 6] = 255
##x_pos_zero, y_pos_zero = Hm.zero_positions(image_mirror)
##centre = Hm.centroid_centre(x_pos_zero, y_pos_zero)
##plt.imshow(image_mirror, cmap = 'bone')
##plt.scatter(centre[0], centre[1])
##plt.show()
##
##
##
##x0 = centre[0]
##y0 = centre[1]
##print x0, y0
x0 = 550
y0 = 484
size = 340
mask = [np.sqrt((xx-x0)**2 + (yy-y0)**2) <= size]
image_int *= np.squeeze(mask)
#image_mirror *= np.squeeze(mask)
img_int_sq = image_int[y0 - size: y0 + size,x0 - size: x0 + size]
#img_mirror_sq = image_mirror[centre[1] - 1.2*size: centre[1] + 1.2*size, centre[0] - 1.2*size: centre[0] + 1.2*size]
#print size*px_size_int
image_int[image_int < 10] = 0
##image_int[:, :530] = 0
##image_int[:, 600:] = 0
##image_int[135:870, :] = 0
f, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(img_int_sq, cmap = 'bone')
ax1.scatter(432, 432)
ax1.scatter(432 - 550 + x0, -484 + 432 + y0, color = 'r')
#ax2.imshow(img_mirror_sq, cmap = 'bone')
ax2.scatter(432, 432)
ax2.scatter(432 - 550 + x0, -484 + 432 + y0, color = 'r')
plt.show()
#a = np.where(image_int > 10.0)
#print(np.array(a).shape[1])
#ind = np.triu_indices(np.array(a).shape[1], k=1) # indices of upper triangular matrix
#dist_vec = np.sqrt((a[0][ind[0]] - a[0][ind[1]])**2 + (a[1][ind[0]] - a[1][ind[1]])**2)
#plt.hist(dist_vec[dist_vec > 200])
#plt.show()
#print np.max(dist_vec), np.max(dist_vec) * px_size_int
