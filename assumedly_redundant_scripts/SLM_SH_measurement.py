import numpy as np
import matplotlib.pyplot as plt
import Hartmann as Hm
import PIL.Image as pil
import mirror_control as mc
import LSQ_method as LSQ
import Zernike as Zn
import scipy.optimize as opt

def rms_piston(piston, *args):
    """function for rms value to be minimized. args should contain j_max, a_filt, N, Z_mat, orig, mask in that order"""
    if args:
        j_max, a_filt, N, Z_mat, orig, mask, fliplr = args
    else:
        print("you should include the right arguments!")
        return
    orig = np.ma.array(orig, mask = mask)
    inter = np.ma.array(Zn.int_for_comp(j_max, a_filt, N, piston, Z_mat, fliplr), mask = mask)
    rms = np.sqrt(np.sum((inter - orig)**2)/N**2)
    return rms


sh_folder = "SLM_codes_matlab/20170330_defocus/"
flat_slm_sh = np.asarray(pil.open(sh_folder + "zero_pos_dm.tif"), dtype = 'float')
flat_mir_sh = np.asarray(pil.open(sh_folder + "image_ref_mirror.tif"), dtype = 'float')
def_slm_sh = np.asarray(pil.open(sh_folder + "dist_image.tif"), dtype = 'float')
inter_meas = np.copy(np.asarray(pil.open(sh_folder + "interferogram_0.tif"), dtype = 'float'))

#### determine spotsize
f_sh = 6.7e-3
pitch_sh = 150.0e-6
px_size_sh = 4.65e-6
wavelength = 632.9e-9
r_sh_m = 1.8e-3
r_sh_px = int(r_sh_m/px_size_sh)
box_len = 30/2
j_max = 30
j = np.arange(2, j_max) #start at 2, end at j_max-1
power_mat = Zn.Zernike_power_mat(j_max+2)


N = 2*r_sh_px
print(N)
xi, yi = np.linspace(-1, 1, N), np.linspace(-1, 1, N)
xi, yi = np.meshgrid(xi, yi)
mask = [np.sqrt((xi) ** 2 + (yi) ** 2) >= 1]
#power_mat = Zn.Zernike_power_mat(j_max+2)
j_range = np.arange(2, j_max+2)
Z_mat = Zn.Zernike_xy(xi, yi, power_mat, j_range)

print("zero position")
x_pos_zero, y_pos_zero = Hm.zero_positions(np.copy(flat_slm_sh), spotsize = box_len)
print("amount of spots: " + str(len(x_pos_zero)))
##ui = np.triu_indices(len(x_pos_zero), k = 1)
##dist = np.sqrt((x_pos_zero[ui[0]] - x_pos_zero[ui[1]])**2 + (y_pos_zero[ui[0]] - y_pos_zero[ui[1]])**2)
##f, ax = plt.subplots(1,1)
##ax.hist(dist, range = [50, 75])
##f, ax = plt.subplots(1,1)
##ax.imshow(flat_slm_sh, origin = 'lower', cmap = 'gray')
##ax.scatter(x_pos_zero, y_pos_zero)
##plt.show()
[ny, nx] = flat_mir_sh.shape
i, j = np.arange(0, nx, 1), np.arange(ny, 0, -1)
ii, jj = np.meshgrid(i, j)
print("flat positions")
x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, np.copy(flat_mir_sh), ii, jj, spot_size = box_len) 
x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_zero, y_pos_zero, np.copy(def_slm_sh), ii, jj, spot_size = box_len)
#slope_x, slope_y = Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat, y_pos_flat, px_size_sh, f_sh, r_sh_m, wavelength) 
centre = Hm.centroid_centre(x_pos_flat, y_pos_flat)
x_pos_norm = ((x_pos_flat - centre[0]))/float(r_sh_px)
y_pos_norm = ((y_pos_flat - centre[1]))/float(r_sh_px)
inside = np.where(np.sqrt(x_pos_norm**2 + y_pos_norm**2) <= (1 + (float(box_len)/r_sh_px))) #35 is the half the width of the pixel box aroudn a centroid and r_sh_px is the scaling factor
x_pos_zero_f, y_pos_zero_f, x_pos_flat_f, y_pos_flat_f, x_pos_norm_f, y_pos_norm_f, x_pos_dist_f, y_pos_dist_f = mc.filter_positions(inside, x_pos_zero, y_pos_zero, x_pos_flat, y_pos_flat, x_pos_norm, y_pos_norm, x_pos_dist, y_pos_dist)
print("Geometry matrix")
G = LSQ.matrix_avg_gradient(x_pos_norm_f, y_pos_norm_f, j_max, r_sh_px, power_mat)
f, ax = plt.subplots(1,1)
ax.scatter(x_pos_flat, y_pos_flat)
Circle1 = plt.Circle((centre[0], centre[1]), r_sh_px, fill = False)
ax.add_artist(Circle1)
plt.show()
s = np.hstack(Hm.centroid2slope(x_pos_dist_f, y_pos_dist_f, x_pos_flat_f, y_pos_flat_f, px_size_sh, f_sh, r_sh_m, wavelength))
plt.hist(s)
plt.show()
print("solving for a")
a = np.linalg.lstsq(G,s)[0]
print(a)
##inter_meas = np.ma.array(inter_meas[116:116+638, 208:208+638], mask = mask)
##inter_meas /= inter_meas.max()
##
##piston, rms = opt.fmin(rms_piston, 0, args = (j_max, a, N, Z_mat, inter_meas, mask, False), full_output = True)[:2]
##
##
##inter = np.ma.array(Zn.int_for_comp(j_max, a, N, piston, Z_mat, fliplr = False), mask = mask)
##a_added = np.zeros(j_max)
##a_added[2] = 8

##f, ax = plt.subplots(nrows = 1, ncols = 4, figsize = (4.98, 4.98/4.4), gridspec_kw={'width_ratios':[4, 4, 4, 0.4]})
##ax[0].imshow(inter_meas, cmap = 'gray', origin = 'lower')
##interf= Zn.imshow_interferogram(j_max, a, N, ax = ax[1], piston = piston, fliplr = False)
##Zn.imshow_interferogram(j_max, a_added, N, ax = ax[2], piston = piston, fliplr = False)
##titles = ['Measured', 'LSQ', 'SLM']
##for ii in range(3):
##    ax[ii].set(adjustable = 'box-forced', aspect = 'equal')
##    ax[ii].get_xaxis().set_ticks([])
##    ax[ii].get_yaxis().set_ticks([])
##    ax[ii].set_title(titles[ii], fontsize = 9)
##
##cbar = plt.colorbar(interf, cax = ax[-1], ticks = [0.0, 0.25, 0.5, 0.75, 1.0])
##cbar.ax.tick_params(labelsize=7)
##cbar.ax.set_ylabel("Normalized Intensity", fontsize = 8)
##f.savefig(sh_folder + "rudimentary_measurement.png", bbox_inches = 'tight', dpi = 300)
##plt.show()
