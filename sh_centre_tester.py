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
import LSQ_method as LSQ
import matplotlib.pyplot as plt
from matplotlib import cm
import edac40
import matplotlib.ticker as ticker

sh, int_cam = mc.set_up_cameras()
global mirror
mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here
px_size_sh = 5.2e-6     # width of pixels 
r_sh_px = 375
r_sm_m = r_sh_px * px_size_sh
u_dm = np.zeros(19)
mc.set_displacement(u_dm, mirror)
time.sleep(0.05)
sh.snapImage()
image_zeros = sh.getImage().astype(float)
[ny,nx] = image_zeros.shape
x = np.linspace(1, nx, nx)
y = np.linspace(1, ny, ny)
xx, yy = np.meshgrid(x, y)
dm_zeros = Hm.zero_positions(image_zeros)
sh.snapImage()
image_zeros = sh.getImage().astype(float)
points = Hm.centroid_positions(dm_zeros[0][:], dm_zeros[1][:], image_zeros, xx, yy)
weighted_avg = np.zeros(2)
noiseless_zero = image_zeros
noiseless_zero[noiseless_zero < 4] = 0
weight = 1.0/np.sum(noiseless_zero)
weighted_avg = np.sum(noiseless_zero*xx)*weight, np.sum(noiseless_zero*yy)*weight



centre_from_peaks = np.zeros(2)
centre_from_peaks[0] = np.sum(points[0][:]) / len(dm_zeros[0][:])
centre_from_peaks[1] = np.sum(points[1][:]) / len(dm_zeros[1][:])
#x_pos_norm = ((points[0][:] - centres[0]))/r_sh_px
#y_pos_norm = ((points[1][:] - centres[1]))/r_sh_px


image_zeros[image_zeros < 4] = 0
image_zeros[image_zeros > 4] = 255
centres2 = Hm.centroid_centre(points[0][:], points[1][:])

### look at points well within cirlce
inside = np.where(np.sqrt((points[0][:]-centre_from_peaks[0])**2 + (points[1][:]-centre_from_peaks[1])**2) <= r_sh_px+35)


outside = np.where(np.sqrt((points[0][:]-centre_from_peaks[0])**2 + (points[1][:]-centre_from_peaks[1])**2) > r_sh_px+35)
points_filtered = np.array(points)[:,np.array(inside)]
points_filtered_inverted = np.array(points)[:, np.array(outside)]

fig, ax = plt.subplots(figsize = plt.figaspect(1.))
plt.scatter(points_filtered[0][:], points_filtered[1][:])
plt.scatter(points_filtered_inverted[0][:], points_filtered_inverted[1][:], color = 'r')
#plt.scatter(weighted_avg[0], weighted_avg[1], color = 'r')
plt.scatter(centres2[0], centres2[1], color = 'g')
plt.scatter(centre_from_peaks[0], centre_from_peaks[1], color = 'k')
circle1 = plt.Circle(centres2, r_sh_px, color = 'g', fill=False)
#circle2 = plt.Circle(weighted_avg, r_sh_px, color = 'r', fill = False)
circle3 = plt.Circle(centre_from_peaks, r_sh_px, color = 'k', fill = False)
ax = plt.gca()
ax.add_artist(circle1)
#ax.add_artist(circle2)
ax.add_artist(circle3)
plt.show()
