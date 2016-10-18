import sys
import os
import time
if "C:\Program Files\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Program Files\Micro-Manager-1.4")
import mirror_control as mc
import edac40
import MMCorePy
import PIL.Image
import Hartmann as Hm
import Zernike as Zn
import numpy as np
import PIL.Image
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

# Test figures
impath_zero = os.path.abspath("dm_sh.tif")
impath_dist = os.path.abspath("ref_sh.tif")
zero_image = np.asarray(PIL.Image.open(impath_zero)).astype(float)
dist_image = np.asarray(PIL.Image.open(impath_dist)).astype(float)

[ny,nx] = zero_image.shape
px_size_sh = 5.2e-6     # width of pixels
px_size_int = 5.2e-6
f_sh = 17.6e-3            # focal length
r_int_px = 370
r_sh_m = r_int_px * px_size_int
r_sh_px = r_sh_m / px_size_sh
x = np.linspace(1, nx, nx)
y = np.linspace(1, ny, ny)
xx, yy = np.meshgrid(x, y)


x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image)
x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, dist_image, xx, yy)
centre = Hm.centroid_centre(x_pos_flat, y_pos_flat)

x_pos_norm = ((x_pos_flat - centre[0]))/r_sh_px
y_pos_norm = ((y_pos_flat - centre[1]))/r_sh_px

fix, ax = plt.subplots(figsize = plt.figaspect(1.))
curAx = plt.gca()
ax.scatter(x_pos_norm, y_pos_norm)
len_box = 70/r_sh_px
half_len_box = 35/r_sh_px
circle1 = plt.Circle([0,0] , 1, color = 'k', fill=False)
for i in range(len(x_pos_flat)):
    curAx.add_patch(Rectangle((x_pos_norm[i] - half_len_box, y_pos_norm[i] - half_len_box), len_box, len_box, alpha=1, facecolor='none'))
ax.add_artist(circle1)
plt.show()
