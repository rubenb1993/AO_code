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

#### Gather interferogram 
impath_int = os.path.abspath("interference_for_checking_radius.tif")
image_int = np.asarray(PIL.Image.open(impath_int)).astype(float)
image_int[image_int < 10] = 0
image_int[:, :530] = 0
image_int[:, 600:] = 0
image_int[135:870, :] = 0
plt.imshow(image_int, cmap = 'bone')
plt.show()
a = np.where(image_int > 10.0)
ind = np.triu_indices(np.array(a).shape[1], k=1) # indices of upper triangular matrix
dist_vec = np.sqrt((a[0][ind[0]] - a[0][ind[1]])**2 + (a[1][ind[0]] - a[1][ind[1]])**2)
plt.hist(dist_vec[dist_vec > 200])
plt.show()
print np.max(dist_vec)
