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
import Zernike as Zn
import edac40
import matplotlib.pyplot as plt
import edac40
import mirror_control as mc
import LSQ_method as LSQ

mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here
sh, int_cam = mc.set_up_cameras()


u_dm = np.zeros(19)
mc.set_displacement(u_dm, mirror)
time.sleep(2)
raw_input('block DM')
int_cam.snapImage()
PIL.Image.fromarray(int_cam.getImage().astype("float")).save("flat_mirror.tif")
sh.snapImage()
image_control = sh.getImage().astype(float)
raw_input('block REF')
int_cam.snapImage()
PIL.Image.fromarray(int_cam.getImage().astype("float")).save("def_mirror.tif")
sh.snapImage()
zero_image = sh.getImage().astype(float)
raw_input('block none')
int_cam.snapImage()
PIL.Image.fromarray(int_cam.getImage().astype("float")).save("interferogram.tif")


## Given paramters for centroid gathering and displacing
px_size_sh = 5.2e-6     # width of pixels
px_size_int = 5.2e-6
f_sh = 17.6e-3            # focal length
r_int_px = 340
r_sh_px = 370
r_sh_m = r_sh_px * px_size_int
j_max= 20          # maximum fringe order
wavelength = 632e-9 #[m]
box_len = 35.0 #half width of subaperture box in px

u_dm, x_pos_norm_f, y_pos_norm_f, x_pos_zero_f, y_pos_zero_f, V2D = mc.flat_wavefront(u_dm, zero_image, image_control, r_sh_px, r_int_px, sh, mirror, show_accepted_spots = False)

a = np.zeros(j_max)
ind = np.array([0])
a[2] = 0.75 * wavelength

V2D_inv = np.linalg.pinv(V2D)
G = LSQ.matrix_avg_gradient(x_pos_norm_f, y_pos_norm_f, j_max, r_sh_px)
v_abb = (f_sh/(r_sh_m * px_size_sh)) * np.dot(V2D_inv, np.dot(G, a))
u_dm -= v_abb
mc.set_displacement(u_dm, mirror)
raw_input("remove piece of paper")
time.sleep(0.2)
int_cam.snapImage()
PIL.Image.fromarray(int_cam.getImage().astype("float")).save("interferogram_75_defocus.tif")
