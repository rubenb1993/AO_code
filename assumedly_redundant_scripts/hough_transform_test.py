# -*- coding: utf-8 -*-
import sys
import os
import time

if "C:\Program Files\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Program Files\Micro-Manager-1.4")
if "H:\Desktop\numba" not in sys.path:
    sys.path.append("H:\Desktop\\numba")
##import MMCorePy
import PIL.Image
import numpy as np
import Hartmann as Hm
import displacement_matrix as Dm
#import LSQ_method as LSQ
import mirror_control as mc
import matplotlib.pyplot as plt
import edac40
import math
import phase_extraction as PE
import peakdetect
import matplotlib.ticker as ticker
import phase_unwrapping_test as pw
import detect_peaks as dp
import scipy.fftpack as fftp
import itertools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc

# Define font for figures
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

#from numba import jit


# Set up ranges
def make_jit_arrays(img):
    width, height = img.shape
    diag_len = int(math.ceil(math.sqrt(width * width + height * height)))  # diagonal length of image
    rhos = np.linspace(-diag_len, diag_len, 2.0 * diag_len)  # x-axis for plot with hough transform
    thetas = np.linspace(0, np.pi, 360)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    Acc = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    y_id, x_id = np.nonzero(img)
    return cos_t, sin_t, x_id, y_id, diag_len, Acc, num_thetas, rhos, thetas

#@jit
def hough_jit(img, cos_t, sin_t, x_id, y_id, diag_len, Acc, num_thetas):
    # binning
    for i in range(len(x_id)):
        x = x_id[i]
        y = y_id[i]
        for j in range(num_thetas):
            rho = round(x * cos_t[j] + y * sin_t[j]) + diag_len
            Acc[rho, j] += 1
    return Acc

def hough_numpy(img, x, y):
    width, height = img.shape
    diag_len = np.ceil(np.sqrt(width * width + height * height)).astype(int)  # diagonal length of image
    rhos = np.linspace(-diag_len, diag_len, 2.0 * diag_len)  # x-axis for plot with hough transform
    
    # pre-compute angles
    thetas = np.linspace(0, 0.6 * np.pi, 200)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    
    Acc = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    y_id, x_id = np.nonzero(img)
    y_tr, x_tr = y[y_id], x[x_id]
    cos_tile, sin_tile = np.tile(cos_t, (len(x_id), 1)), np.tile(sin_t, (len(x_id), 1))
    x_tr_tile, y_tr_tile = np.tile(x_tr, (len(thetas), 1)).T, np.tile(y_tr, (len(thetas), 1)).T
    rho = np.round(x_tr_tile * cos_tile - y_tr_tile * sin_tile) + diag_len  # precompute rho
    rho = rho.astype(int)
    # binning more efficiently
    for j, i in itertools.product(range(len(x_id)), range(num_thetas)):
        Acc[rho[j, i], i] += 1

    return Acc, rhos, thetas, diag_len

def max_finding(Acc, rhos, thetas, lookahead = 30, delta = 30):
    idmax = np.argmax(Acc)
    rho_max_index, theta_max_index = np.unravel_index(idmax, Acc.shape)
    peak_indices = dp.detect_peaks(Acc[:, theta_max_index], mph = 110, mpd = lookahead)
    s_max = np.max(rhos[peak_indices])
    theta_max = thetas[theta_max_index]
    return theta_max, s_max, theta_max_index, peak_indices

def set_subplot_interferograms(*args, **kwargs):
    if 'i' in kwargs:
        i = kwargs['i']
    for arg in args:
        arg.get_xaxis().set_ticks([])
        arg.get_yaxis().set_ticks([])
        #arg.axis('off')
        arg.set_title(titles[i], fontsize = 9, loc = 'left')

def make_colorbar(f, image):
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.3, 0.05, 0.4])
    cbar = f.colorbar(image, cax = cbar_ax)
    tick_locator = ticker.MaxNLocator(nbins = 5)
    cbar.locator = tick_locator
    cbar.ax.yaxis.set_major_locator(ticker.AutoLocator())
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize = 6)
    cbar.set_label('Intensity level', size = 7)

def weight_thicken(indices, weight_matrix, border = 10):
    indices = np.vstack(indices) #make an array from the tuple
    where_inside = np.where((indices[0] > border) & (indices[1] > border) & (indices[0] < weight_matrix.shape[0]-border) & (indices[1] < weight_matrix.shape[0]-border) )
    coords = np.squeeze(indices[:, np.vstack(where_inside)])
    weight_matrix[coords[0], coords[1]] = 0
    weight_matrix[coords[0] -1, coords[1]] = 0
    weight_matrix[coords[0], coords[1] -1] = 0
    weight_matrix[coords[0] +1, coords[1]] = 0
    weight_matrix[coords[0], coords[1] +1] = 0
    return weight_matrix

## Given paramters for centroid gathering and displacing
px_size_sh = 5.2e-6     # width of pixels
px_size_int = 5.2e-6
f_sh = 17.6e-3            # focal length
r_int_px = 375
r_sh_px = 410
r_sh_m = r_sh_px * px_size_int
j_max= 20          # maximum fringe order
wavelength = 632e-9 #[m]
box_len = 35.0 #half width of subaperture box in px



## centre and radius of interferogam. Done by eye, with the help of define_radius.py
x0 = 550
y0 = 484
radius = int(340)

constants = np.stack((px_size_sh, px_size_int, f_sh, r_int_px, r_sh_px, r_sh_m, j_max, wavelength, box_len, x0, y0, radius))

## image making parameters
dpi_num = 600
int_im_size = (4.98, 3.07)
int_im_size_23 = (0.66 * 4.98, 3.07)
int_im_size_13 = (0.33 * 4.98, 3.07)

new_yn = raw_input("do you want to take new images? y/n")
choise = raw_input("save figures y/n")
folder_name = "20161130_five_inter_test/"
if new_yn == 'y':
    global mirror
    mirror = edac40.OKOMirror("169.254.69.72") # Enter real IP in here
    sh, int_cam = mc.set_up_cameras()

    actuators = 19
    u_dm = np.zeros(actuators)
    mc.set_displacement(u_dm, mirror)
    time.sleep(0.2)

    raw_input("Did you calibrate the reference mirror? Block DM")
    sh.snapImage()
    image_ref_mirror = sh.getImage().astype(float)
    PIL.Image.fromarray(image_ref_mirror).save(folder_name + "image_ref_mirror.tif")


    raw_input("block reference mirror!")
    sh.snapImage()
    zero_pos_dm = sh.getImage().astype(float)
    PIL.Image.fromarray(zero_pos_dm).save(folder_name + "zero_pos_dm.tif")

    ### make flat wavefront
    u_dm, x_pos_norm_f, y_pos_norm_f, x_pos_zero_f, y_pos_zero_f, V2D = mc.flat_wavefront(u_dm, zero_pos_dm, image_ref_mirror, r_sh_px, r_int_px, sh, mirror, show_accepted_spots = True)

    ##
##    raw_input('remove black paper!')
##    int_cam.snapImage()
##    flat_wf = int_cam.getImage().astype('float')
##    flat_wf = np.flipud(flat_wf)
##    flat_wf = np.fliplr(flat_wf)


    G = LSQ.matrix_avg_gradient(x_pos_norm_f, y_pos_norm_f, j_max, r_sh_px)
    a = np.zeros(j_max)

    ind = np.array([0])
    #a[ind] = 0.15 * wavelength
    a[ind] = 3 * wavelength

    u_dm_flat = u_dm
    #V2D_inv = np.linalg.pinv(V2D)
    v_abb = (f_sh/(r_sh_m * px_size_sh)) * np.linalg.lstsq(V2D, np.dot(G, a))[0]#np.dot(V2D_inv, np.dot(G, a))
    u_dm -= v_abb

    if np.any(np.abs(u_dm) > 1.0):
        print("maximum deflection of mirror reached")
        print(u_dm)
        
    mc.set_displacement(u_dm, mirror)

    raw_input("keep it covered!")
    sh.snapImage()
    dist_image = sh.getImage().astype('float')
    PIL.Image.fromarray(dist_image).save(folder_name + "dist_image.tif")

    raw_input("remove piece of paper")
    time.sleep(0.2)
    int_cam.snapImage()
    image_i0 = int_cam.getImage().astype(float)
    PIL.Image.fromarray(image_i0).save(folder_name + "interferogram_0.tif")

    raw_input("tip and tilt 1")
    time.sleep(1)
    int_cam.snapImage()
    image_i1 = int_cam.getImage().astype(float)
    PIL.Image.fromarray(image_i1).save(folder_name + "interferogram_1.tif")

    raw_input("tip and tilt 2")
    time.sleep(1)
    int_cam.snapImage()
    image_i2 = int_cam.getImage().astype(float)
    PIL.Image.fromarray(image_i2).save(folder_name + "interferogram_2.tif")

    raw_input("tip and tilt 3")
    time.sleep(1)
    int_cam.snapImage()
    image_i3 = int_cam.getImage().astype(float)
    PIL.Image.fromarray(image_i3).save(folder_name + "interferogram_3.tif")

    raw_input("tip and tilt 4")
    time.sleep(1)
    int_cam.snapImage()
    image_i4 = int_cam.getImage().astype(float)
    PIL.Image.fromarray(image_i4).save(folder_name + "interferogram_4.tif")
else:
##    image_ref_mirror = np.array(PIL.Image.open(folder_name + "image_ref_mirror.tif"))
##    zero_pos_dm = np.array(PIL.Image.open(folder_name + "zero_image.tif"))
##    dist_image = np.array(PIL.Image.open(folder_name + "zero_image.tif"))
    image_i0 = np.array(PIL.Image.open(folder_name + "interferogram_0.tif"))
    image_i1 = np.array(PIL.Image.open(folder_name + "interferogram_1.tif"))
    image_i2 = np.array(PIL.Image.open(folder_name + "interferogram_2.tif"))
    image_i3 = np.array(PIL.Image.open(folder_name + "interferogram_3.tif"))
    image_i4 = np.array(PIL.Image.open(folder_name + "interferogram_4.tif"))

interferogram_stack = np.dstack((image_i0, image_i1, image_i2, image_i3, image_i4))
org_phase, delta_i = PE.phase_extraction(constants)


##
##Id1 = image_i1 - image_i0
##Id2 = image_i2 - image_i0
##Id3 = image_i3 - image_i0
##Id4 = image_i4 - image_i0
##Id_tot = np.dstack((Id1, Id2, Id3, Id4))
##
ny, nx = image_i0.shape
x, y = np.linspace(-1.0 * radius, 1.0 * radius, 2*radius), np.linspace(-1.0 * radius, 1.0 * radius, 2*radius)
xx, yy = np.meshgrid(x, y)
##ss = np.sqrt(xx**2 + yy**2)
##
##Id_int = np.zeros((2*radius, 2*radius, Id_tot.shape[-1]))
##Id_zeros = np.zeros(Id_int.shape, dtype = float)
##Im_i0 = np.zeros((2*radius, 2*radius, 3))
##Im_i0[..., 0] = image_i0[y0-radius:y0+radius, x0-radius:x0+radius]
##Im_i0[..., 1] = image_i1[y0-radius:y0+radius, x0-radius:x0+radius]
##Im_i0[..., 2] = image_i2[y0-radius:y0+radius, x0-radius:x0+radius]
####
#####get region of interest, and the zeros in those regions
##Id_int = Id_tot[y0-radius:y0+radius, x0-radius:x0+radius, :]
##zeros_i = np.abs(Id_int) <= 1
##Id_zeros[zeros_i] = 1
##
mask = [np.sqrt((xx) ** 2 + (yy) ** 2) >= radius]
#mask_tile = np.tile(mask, (Id_zeros.shape[-1],1,1)).T
##Id_zeros_mask = np.ma.array(Id_zeros, mask=mask_tile)
##
##
##### make Hough transform of all points
###Initialize constants
##width, height = Id_int[...,0].shape
##num_thetas = 200
##diag_len = np.ceil(np.sqrt(width * width + height * height)).astype(int)
##Acc = np.zeros((2 * diag_len, num_thetas, 4), dtype=np.uint64)
##Lambda = np.zeros(4)
##ti = np.zeros((2,4))
##sigma = np.zeros(4)
##x_shape = list(xx.shape)
##x_shape.append(4)
##tau_i = np.zeros(x_shape)
##delta_i = np.zeros(tau_i.shape)
##Id_hat = np.zeros(Id_int.shape)
##theta_max = np.zeros(4)
##sin_diff = np.zeros(tau_i.shape)
##s_max = np.zeros(4)
##
##print("making hough transforms... crunch crunch")
##
##for jj in range(4):
##    Acc[...,jj], rhos, thetas, diag_len = hough_numpy(Id_zeros_mask[..., jj], x, y)
##    print("Hough transform " + str(jj+1) + " done!")
##    theta_max[jj], s_max[jj], theta_max_index, peak_indices = max_finding(Acc[...,jj], rhos, thetas)
##    ## uncomment to check if peaks align with found peaks visually
####    f2, axarr2 = plt.subplots(2,1)
####    axarr2[0].imshow(Acc[...,jj].T, cmap = 'bone')
####    axarr2[0].scatter(peak_indices, np.tile(theta_max_index, len(peak_indices)))
####    axarr2[1].plot(rhos, Acc[:, theta_max_index, jj])
####    dtct = axarr2[1].scatter(rhos[peak_indices], Acc[peak_indices, theta_max_index, jj], c = 'r')
####    plt.show()
##    Lambda[jj] = np.sum(np.diff(rhos[peak_indices]), dtype=float) / (len(peak_indices)-1.0)
##
#### make tile lists to compute 3d matrices in a numpy way
##ti_tile = list(Id_hat[...,0].shape)
##ti_tile.append(1)
##xx_tile = list(Id_hat[0,0,:].shape)
##xx_tile.append(1)
##xx_tile.append(1)
##
#### Compute Id_hat in a numpy way
##ti = (2 * np.pi/ Lambda) * np.array([np.cos(theta_max), np.sin(theta_max)])
##sigma = 2 * np.pi * s_max / Lambda
##tau_i = -np.tile(ti[0, :], ti_tile)  * np.tile(xx, xx_tile).transpose(1,2,0) + np.tile(ti[1, :], ti_tile) * np.tile(yy, xx_tile).transpose(1,2,0)
##delta_i = (tau_i + sigma)/2.0
##sin_diff = np.sin(delta_i)
##Id_hat = Id_int/(-2.0 * sin_diff)

##f, axarr = plt.subplots(2,4, figsize = int_im_size)
titles = [r'a)', r'b)', r'c)', r'd)', r'e)', r'f)', r'g)', r'h)', r'i)', r'j)', r'k)', r'l)', r'm)', r'n)']

##for jj in range(4):
##    axarr[0,jj].imshow(Id_zeros_mask[..., jj], cmap = 'bone_r')
##    im = axarr[1,jj].imshow(np.ma.array(Id_hat[...,jj], mask = mask), vmin = -70, vmax = 70, cmap = 'bone')
##    set_subplot_interferograms(axarr[0,jj], i = jj)
##    set_subplot_interferograms(axarr[1,jj], i = jj)
##
##make_colorbar(f, im)
##choise = raw_input('save_figure? y/n')
##if choise == 'y':
##    f.savefig('20161205_images_for_discussion/zeros_n_id_hat.png', bbox_inches='tight', dpi = dpi_num)
##    
##print("theta = " + str(theta_max/np.pi) + "lambda = " + str(Lambda) + "s = " + str(s_max))

## constants for unwrapping
x_pcg, y_pcg = np.linspace(-1, 1, 2*radius), np.linspace(-1, 1, 2*radius)
N = len(x_pcg)
i, j = np.linspace(0, N-1, N), np.linspace(0, N-1, N)
xx_alg, yy_alg = np.meshgrid(x_pcg, y_pcg)
ii, jj = np.meshgrid(i, j)
##
#### allocate memory for phase extraction
nmbr_inter = interferogram_stack.shape[-1] #number of interferogram differences
Un_sol = np.triu_indices(nmbr_inter, 1) #indices of unique phase retrievals. upper triangular indices, s.t. 12 and 21 dont get counted twice
shape_unwrp = org_phase.shape
#shape_unwrp.append(len(Un_sol[0])) #square by amount of solutions
##Unwr_mat = np.zeros(shape_unwrp)
##angfact = np.zeros(shape_unwrp)
##atany = np.zeros(shape_unwrp)
##atanx = np.zeros(shape_unwrp)
##org_phase = np.zeros(shape_unwrp)
##org_phase_plot = np.zeros(shape_unwrp)
org_unwr = np.zeros(shape_unwrp)
##
###phase extraction
##angfact = delta_i[..., Un_sol[1]] - delta_i[..., Un_sol[0]]
##atany = Id_hat[..., Un_sol[0]]
##atanx = (Id_hat[..., Un_sol[1]] - np.cos(angfact) * Id_hat[..., Un_sol[0]]) / np.sin(angfact) ## sin needs to be added here for arctan2 to know the correct sign of y and x

f, axarr = plt.subplots(1, org_phase.shape[-1], figsize = int_im_size)
for k in range(org_phase.shape[-1]):
    ##org_phase[..., k] = np.arctan2(atany[..., k], atanx[..., k])
    org_unwr[...,k] = pw.unwrap_phase_dct(org_phase[..., k], xx_alg, yy_alg, ii, jj, N, N)
    org_unwr[...,k] -= delta_i[..., Un_sol[0][k]]

## remove piston
mean_unwr = np.mean(org_unwr, axis = (0,1))
org_unwr -= mean_unwr

## plot everything
for k in range(org_phase.shape[-1]):
    im = axarr[k].imshow(np.ma.array(org_unwr[...,k], mask = mask), vmin = -10, vmax = 15)
    set_subplot_interferograms(axarr[k], i = k)
make_colorbar(f, im)
if choise == 'y':
    f.savefig('20161205_images_for_discussion/dct_unwrapped_additional_interferograms.png', bbox_inches='tight', dpi = dpi_num)


org_med = np.median(org_phase[..., :3], axis =2)
unwr_med = np.median(org_unwr, axis = 2) #take median to try to remove noisy signal
f, axarr = plt.subplots(1,2, figsize = int_im_size)
axarr[0].imshow(np.ma.array(Image_i0[..., 0], mask = mask), cmap = 'bone', origin = 'lower')
im = axarr[1].imshow(np.ma.array(unwr_med, mask = mask), cmap = 'jet', origin = 'lower')
set_subplot_interferograms(axarr[0], i = 0)
set_subplot_interferograms(axarr[1], i = 1)
make_colorbar(f, im)
if choise == 'y':
    f.savefig('20161205_images_for_discussion/median_interferogram.png', bbox_inches='tight', dpi = dpi_num)

#plt.show()


### Set up weighted unwrapping algorithm
sin_d1 = np.sin(delta_i[...,0])
sin_d2 = np.sin(delta_i[...,1])
sin_d2_d1 = np.sin(delta_i[...,1] - delta_i[...,0])

weight = np.ones(delta_i[...,0].shape)
weight = weight_thicken(np.where(np.abs(sin_d1) <= 2e-2), weight, 30)
weight = weight_thicken(np.where(np.abs(sin_d2) <= 2e-2), weight, 30)
weight = weight_thicken(np.where(np.abs(sin_d2_d1) <= 2e-2), weight, 30)

matrix_yn = raw_input('pcg unwrapping? y/n')
if matrix_yn == 'y':
    print("making matrices")
    A, c, A_weighted, c_weighted = pw.least_squares_matrix(org_phase[..., 0], weight)
    print("solving!")
    phi = pw.pcg_algorithm(A_weighted, c_weighted, N, 1e-12, xx_alg, yy_alg, ii, jj)
    f, axarr = plt.subplots(1,3, figsize = int_im_size)
    #axarr[0].imshow(np.ma.array(Im_i0[..., 0], mask = mask), cmap = 'bone')
    axarr[0].imshow(np.ma.array(org_unwr[..., 0], mask = mask), cmap = 'jet', vmin = -20, vmax = 20)
    axarr[1].imshow(weight, cmap = 'bone')
    im = axarr[2].imshow(np.ma.array(phi.reshape((N,N)) - delta_i[..., 0], mask = mask), cmap = 'jet', vmin = -20, vmax = 10)
    for i in range(3):
        set_subplot_interferograms(axarr[i], i = i)
    make_colorbar(f, im)
    if choise == 'y':
        f.savefig('20161205_images_for_discussion/weighted_unwrapping_gone_wrong.png', bbox_inches='tight', dpi = dpi_num)


### comparison


#plt.show()

##unwr_phase = np.zeros(phase_shape)
##for i in range(3):
##    unwr_phase[..., i] = pw.unwrap_phase_dct(org_phase[..., i], xx_alg, yy_alg, ii, jj, N, N)
##    unwr_phase[..., i] -= delta_i[..., 1]
##unwr_phase[..., -1] = np.median(unwr_phase[..., :3], axis = 2)
##    
##
##cmap = plt.get_cmap('jet')
##lev = np.linspace(np.min(np.ma.array(unwr_phase[..., -1], mask = mask)), np.max(np.ma.array(unwr_phase[..., -1], mask = mask)))
##norm1 = cm.BoundaryNorm(lev, 256)
##
##
##
##f = plt.figure(figsize = (4.98,3.07))
##axarr = f.add_subplot(151)
##axarr.imshow(Im_i0[..., 0], cmap = 'bone', vmin = np.min(Im_i0[...,0]), vmax = np.max(Im_i0[...,0]), origin = 'lower')
##for i in range(4):
##    axarr = f.add_subplot("15" + str(i+2), aspect = 'equal')
##    three_im = axarr.contourf(xx_alg, yy_alg, np.ma.array(unwr_phase[..., i], mask = mask), linewidth = 0, cmap = cmap, norm = norm1)
##f.subplots_adjust(right=0.8)
##cbar_ax = f.add_axes([0.85, 0.3, 0.05, 0.4])
##cbar = f.colorbar(three_im, cax = cbar_ax)
##tick_locator = ticker.MaxNLocator(nbins = 5)
##cbar.locator = tick_locator
##cbar.ax.yaxis.set_major_locator(ticker.AutoLocator())
##cbar.update_ticks()
##cbar.ax.tick_params(labelsize = 6)
##cbar.set_label(r'$\varphi_0$', size = 7)
##plt.show()
##yn = raw_input("Plot? y/n")
##if (yn == 'y'):
##    mask = [np.sqrt((xx) ** 2 + (yy) ** 2) >= radius]
##    titles = [r'a)', r'b)', r'c)']
##    f, axarr = plt.subplots(1,5, figsize=int_im_size, frameon = False)
##    for i in range(5):
##        im1 = axarr[i].imshow(np.ma.array(Im_i0[..., i], mask = mask), cmap = 'bone', vmin = 0, vmax = 170)
##        set_subplot_interferograms(axarr[i])
##    make_colorbar(f, im1)
##    f2, axarr2 = plt.subplots(1,4, figsize = int_im_size_23)
##    f3, axarr3 = plt.subplots(1,4, figsize = int_im_size_23)
##    f4, axarr4 = plt.subplots(4, 1, figsize = int_im_size)
##    f5, axarr5 = plt.subplots(1,4, figsize = int_im_size_23)
##    f6, axarr6 = plt.subplots(1,1, figsize = int_im_size_13)
##    for i in range(4):
##        #differences
##        im2 = axarr2[i].imshow(np.ma.array(Id_int[..., i], mask = mask), cmap = 'bone', vmin = -170, vmax = 170)
##        set_subplot_interferograms(axarr2[i])
##        #zeros
##        im3 = axarr3[i].imshow(np.ma.array(Id_zeros[..., i], mask = mask), cmap = 'bone_r')
##        set_subplot_interferograms(axarr3[i])
##        ## Hough transform
##        no_ticks_x = np.arange(0,5)
##        no_ticks_y = np.arange(0,5,2)
##        tick_size_x, tick_size_y = (np.array(Acc[...,i].shape) -1)/4.0 #first x than y because it is already transposed
##        x_ticks_orig, y_ticks_orig = np.rint(tick_size_x * no_ticks_x).astype(int), np.rint(tick_size_y * no_ticks_y).astype(int)
##        x_ticks_new = np.round(rhos[x_ticks_orig]).astype(int)
##        y_ticks_new = [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']
##        x_ticks_new[2] = 0.0
##        im4 = axarr4[i].imshow(Acc[...,i].T, interpolation='none', cmap = 'bone_r', origin = 'lower')
##        axarr4[i].set_xticks(x_ticks_orig)
##        axarr4[i].set_xticklabels(x_ticks_new, fontsize = 9)
##        axarr4[i].set_yticks(y_ticks_orig)
##        axarr4[i].set_yticklabels(y_ticks_new, fontsize = 9)
##        axarr4[i].set_title(titles[i], fontsize = 9, loc = 'left')
##        ## sinuses
##        im5 = axarr5[i].imshow(np.ma.array(Id_hat[..., i], mask = mask), cmap = 'bone', vmin = -170, vmax = 170)
##        set_subplot_interferograms(axarr5[i])
##    im6 = axarr6.imshow(np.ma.array(org_phase, mask = mask), cmap = 'bone', vmin = -np.pi, vmax = np.pi)
##    axarr6.get_xaxis().set_ticks([])
##    axarr6.get_yaxis().set_ticks([])
##    axarr6.axis('off')
##    make_colorbar(f2, im2)
##    make_colorbar(f5, im5)
##    ## subplot 3 special colorbar
##    f3.subplots_adjust(right = 0.8)
##    cbar_ax = f3.add_axes([0.85, 0.3, 0.05, 0.4])
##    cbar = f3.colorbar(im3, cax = cbar_ax, ticks = [0, 1])
##    cbar.ax.tick_params(labelsize = 6)
##    cbar.set_label('Intensity level', size = 7)
##    
##    ## subplot 6 special colorbar
##    f6.subplots_adjust(right=0.8)
##    cbar_ax = f6.add_axes([0.85, 0.3, 0.05, 0.4])
##    cbar = f6.colorbar(im6, cax = cbar_ax, ticks = [-np.pi, 0, np.pi])
##    #tick_locator = ticker.MaxNLocator(nbins = 5)
##    #cbar.locator = tick_locator
##    cbar.ax.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
##    #cbar.update_ticks()
##    cbar.ax.tick_params(labelsize = 6)
##    cbar.set_label('Intensity level', size = 7)
##
##
##
##    axarr4[1].set_xlabel(r'$\rho$')
##    axarr4[1].set_ylabel(r'$\theta$')
##    f.savefig('20161130_five_inter_test/original_interferograms.png', bbox_inches='tight', dpi = dpi_num)
##    f2.savefig('20161130_five_inter_test/difference_interferograms.png', bbox_inches = 'tight', dpi = dpi_num)
##    f3.savefig('20161130_five_inter_test/zeros_diff_interferograms.png', bbox_inches = 'tight', dpi = dpi_num)
##    f4.savefig('20161130_five_inter_test/hough_accumulator.png', bbox_inches = 'tight', dpi = dpi_num)
##    f5.savefig('20161130_five_inter_test/I_hat_interferograms.png', bbox_inches = 'tight', dpi = dpi_num)
##    f6.savefig('20161130_five_inter_test/recovered_phase.png', bbox_inches = 'tight', dpi = dpi_num)
##    plt.show() 


deltax = (x[1] - x[0])/640.0 #px
delta_f = 1/(len(x) * deltax)
fx = np.arange(-0.5/deltax, -.5/deltax, delta_f)
fy = fx
f, axarr = plt.subplots(3,2, figsize = (4.98, 2 * 3.07))
f_trans_phase = fftp.fftshift(fftp.fft2(org_unwr[...,0]))
lp_filt = [np.sqrt(xx**2 + yy**2) <= 50]
f_phase_filt = np.squeeze(lp_filt * f_trans_phase)
f_trans_weight = fftp.fftshift(fftp.fft2(weight))
axarr[0,0].imshow(np.ma.array(org_unwr[...,0], mask = mask), cmap = 'jet', vmin = -10, vmax = 15)
axarr[0,0].set_title(r'$\varphi$', fontsize = 7)
set_subplot_interferograms(axarr[0,0], i =0)
axarr[0,1].imshow(np.abs(f_trans_phase), vmin = 0, vmax = 1000, extent = [-0.5/deltax, 0.5/deltax, 0.5/deltax, -0.5/deltax])
axarr[0,1].set_xlabel('fx [a.u.]', fontsize = 7)
axarr[0,1].set_ylabel('fy [a.u.]', fontsize = 7)
axarr[0,1].set_title(r'$|\mathcal{F}(\varphi)|$', fontsize = 7)

axarr[1,0].imshow(weight, cmap = 'bone_r')
set_subplot_interferograms(axarr[1,0], i = 1)
axarr[1,1].imshow(np.abs(f_trans_weight), vmax = 1000, extent = [-0.5/deltax, 0.5/deltax, 0.5/deltax, -0.5/deltax])
axarr[1,1].set_xlabel('fx [a.u.]', fontsize = 7)
axarr[1,1].set_ylabel('fy [a.u.]', fontsize = 7)

axarr[1,0].imshow(weight, cmap = 'bone_r')
set_subplot_interferograms(axarr[1,0], i = 1)
axarr[2,1].imshow(np.abs(f_phase_filt), vmax = 1000, extent = [-0.5/deltax, 0.5/deltax, 0.5/deltax, -0.5/deltax])
axarr[2,1].set_xlabel('fx [a.u.]', fontsize = 7)
axarr[2,1].set_ylabel('fy [a.u.]', fontsize = 7)

filtered_phase = fftp.ifft2(fftp.ifftshift(f_phase_filt))
axarr[2,0].imshow(np.ma.array(np.real(filtered_phase), mask = mask), cmap = 'jet', vmin = -10, vmax = 15)
set_subplot_interferograms(axarr[2,0], i = 2)

for i in range(3):
    axarr[i,1].tick_params(axis = 'both', labelsize = 6)

if choise == 'y':
    f.savefig('20161205_images_for_discussion/fourier_filtering.png', bbox_inches='tight', dpi = dpi_num)

f, ax = plt.subplots(3,4,figsize = (4.98, 1.5*3.07))
ax[0,0].imshow(np.ma.array(org_phase[...,0], mask = mask), cmap = 'bone', vmin = -np.pi, vmax = np.pi)
ax[0,0].set_ylabel('wrapped phase', fontsize = 7)
unwrapfilt = pw.unwrap_phase_dct(org_phase[..., 0], xx_alg, yy_alg, ii, jj, N, N)
ax[2,0].imshow(np.ma.array(unwrapfilt, mask = mask), cmap = 'jet', vmin = -10, vmax = 15)
set_subplot_interferograms(ax[2,0], i = 8)
#dx_org, dy_org = pw.delta_x(org_phase[...,0]), pw.delta_y(org_phase[...,0])
qual_map_org = pw.phase_derivative_var_map(org_phase[...,0], 3)
ax[1,0].imshow(np.ma.array(qual_map_org, mask = mask), cmap = 'bone_r')
ax[0,0].set_title('unfiltered', fontsize = 7)
set_subplot_interferograms(ax[0,0], i = 0)
set_subplot_interferograms(ax[1,0], i = 4)
for i in range(3):
    filter_phase = pw.filter_wrapped_phase(org_phase[...,0], 3 + 2*i)
    #dx_filt, dy_filt = pw.delta_x(filter_phase), pw.delta_y(filter_phase)
    qual_map_filt = pw.phase_derivative_var_map(filter_phase, 3)
    im = ax[0,i+1].imshow(np.ma.array(filter_phase, mask = mask), cmap = 'bone', vmin = -np.pi, vmax = np.pi)
    ax[0,i+1].set_title(str(3+2*i) + ' x ' + str(3+2*i), fontsize = 7)
    k = i+1
    ax[1, i+1].imshow(np.ma.array(qual_map_filt, mask = mask), cmap = 'bone_r', vmin = 0, vmax = np.max(qual_map_org))
    set_subplot_interferograms(ax[0,i+1], i = k)
    set_subplot_interferograms(ax[1, i+1], i = k+4)
    unwrapfilt = pw.unwrap_phase_dct(filter_phase, xx_alg, yy_alg, ii, jj, N, N)
    ax[2, i+1].imshow(np.ma.array(unwrapfilt, mask = mask), cmap = 'jet', vmin = -10, vmax= 15)
    set_subplot_interferograms(ax[2, i+1], i = k + 8)
    #ax[2].imshow(pw.filter_wrapped_phase(org_phase[...,0], 11)vmin = -np.pi, vmax = np.pi)
    #ax[3].imshow(pw.filter_wrapped_phase(org_phase[...,0], 13)vmin = -np.pi, vmax = np.pi)
#make_colorbar(f, im)
ax[1,0].set_ylabel('quality map (3x3)', fontsize = 7)
ax[2,0].set_ylabel('DCT unwrapped phase', fontsize = 7)
if choise == 'y':
    f.savefig('20161205_images_for_discussion/ghiglia_filter_applied.png', bbox_inches='tight', dpi = dpi_num)

plt.show()
##
##### very crude processing
##f_filtered_phase = f_trans_phase
##f_sin_d1 = fftp.fftshift(fftp.fft2(sin_d1))
##f_sin_d2 = fftp.fftshift(fftp.fft2(sin_d2))
##f_sin_d1_d2 = fftp.fftshift(fftp.fft2(sin_d1_d2))
##f_sin_weight = fftp.fftshift(fftp.fft2(Id_zeros_all))
##f, axarr = plt.subplots(1,3)
##axarr[0].imshow(np.abs(f_filtered_phase), vmin = 0, vmax = 2000)
##axarr[1].imshow(np.abs(f_sin_weight), vmin = 0, vmax = 2000)
####sin_d1_filter = np.where(np.abs(f_sin_d1) > 100)
####sin_d2_filter = np.where(np.abs(f_sin_d2) > 100)
####sin_d1_d2_filter = np.where(np.abs(f_sin_d1_d2) > 100)
##sin_weight_filter = np.where(np.abs(f_sin_weight) > 100)
##f_filtered_phase[sin_weight_filter] = 0
###f_filtered_phase[sin_d2_filter] = 0
###f_filtered_phase[sin_d1_d2_filter] = 0
####f_filtered_phase[:50, :] = 0
####f_filtered_phase[-50:-1, :] = 0
####f_filtered_phase[:, :50] = 0
####f_filtered_phase[:, -50:-1] = 0
##low_pass_filter = np.where(np.sqrt(xx**2 + yy**2) > 100)
##f_filtered_phase[low_pass_filter] = 0
###f_filtered_phase[filter_high] = 0
##axarr[2].imshow(np.abs(f_filtered_phase), vmin = 0, vmax = 2000, origin = 'lower')
##fx = np.arange(-radius, radius)
####index_x = np.arange(0,2*radius)
####index_y_1 = index_x * ti[1,0]/ti[0,0]
####index_y_2 = index_x * ti[1,1]/ti[0,1]
####index_y_3 = index_x * (ti[1,0] - ti[1,1])/(ti[0,0] - ti[0,1])
##fy_1 = fx * -ti[1,0]/ti[0,0] + radius
##fy_2 = fx * -ti[1,1]/ti[0,1] + radius
##fy_3 = fx * -(ti[1,0] - ti[1,1])/(ti[0,0] - ti[0,1]) + radius
##fx_index = fx + radius
##axarr[0].scatter(fx_index,fy_1)
##axarr[0].scatter(fx_index, fy_2)
##axarr[0].scatter(fx_index, fy_3)

### transform back
##filtered_phase = fftp.ifft2(fftp.ifftshift(f_filtered_phase))
##f, axarr = plt.subplots(1,2, figsize = (4.98*0.66, 3.07))
##axarr[0].imshow(np.abs(f_filtered_phase), vmin=0, vmax = 1000)
##axarr[1].imshow(np.real(filtered_phase))

#f, axarr = plt.subplots(3,4)
##f, axarr = plt.subplots(2,3, figsize=(9.31,5.91))
##axarr[0, 0].imshow(np.ma.array(Im_i0[...,2], mask = mask), cmap = 'bone')
##axarr[0, 0].set_title('Original 2nd shifted', fontsize = 9)
##axarr[0, 1].imshow(np.ma.array(np.cos(org_phase), mask = mask), cmap = 'bone')
##axarr[0, 1].set_title('recovered interferogram', fontsize = 9)
##axarr[0, 2].imshow(np.ma.array(org_phase, mask = mask), cmap = 'bone')
##axarr[0, 2].set_title('recovered (wrapped) phase', fontsize = 9)
##axarr[1, 0].imshow(np.ma.array(Id_zeros_all, mask = mask), cmap = 'bone')
##axarr[1, 0].set_title('mask', fontsize = 9)
##axarr[1, 1].imshow(np.ma.array(org_phase, mask = complete_mask), cmap = 'bone')
##axarr[1, 1].set_title('masked phase', fontsize = 9)
##axarr[1, 2].imshow(np.cos(np.ma.array(org_phase, mask = complete_mask)), cmap = 'bone')
##axarr[1, 2].set_title('int masked phase', fontsize = 9)
##for i in range(6):
##    j = np.unravel_index(i, axarr.shape)
##    axarr[j].get_xaxis().set_ticks([])
##    axarr[j].get_yaxis().set_ticks([])

##plt.show()#block=False)

#raw_input('savefig?')
#plt.savefig('wrong_phase_interferogram.png', bbox_inches='tight')
