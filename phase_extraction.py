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

def phase_extraction(constants, take_new_img = False, folder_name = "20161130_five_inter_test/"):
    px_size_sh, px_size_int, f_sh, r_int_px, r_sh_px, r_sh_m, j_max, wavelength, box_len, x0, y0, radius = constants
    x0, y0, radius = int(x0), int(y0), int(radius)
    if take_new_img == True:
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
##        image_ref_mirror = np.array(PIL.Image.open(folder_name + "image_ref_mirror.tif"))
##        zero_pos_dm = np.array(PIL.Image.open(folder_name + "zero_image.tif"))
##        dist_image = np.array(PIL.Image.open(folder_name + "zero_image.tif"))
        image_ref_mirror = np.zeros(2)
        zero_pos_dm = np.zeros(2)
        dist_image = np.zeros(2)

        
        image_i0 = np.array(PIL.Image.open(folder_name + "interferogram_0.tif"))
        image_i1 = np.array(PIL.Image.open(folder_name + "interferogram_1.tif"))
        image_i2 = np.array(PIL.Image.open(folder_name + "interferogram_2.tif"))
        image_i3 = np.array(PIL.Image.open(folder_name + "interferogram_3.tif"))
        image_i4 = np.array(PIL.Image.open(folder_name + "interferogram_4.tif"))

    sh_spots = np.dstack((image_ref_mirror, zero_pos_dm, dist_image))        
    interferograms = np.dstack((image_i0, image_i1, image_i2, image_i3, image_i4))
    indices_id = np.arange(1, interferograms.shape[-1])
    id_shape = list(interferograms.shape[0:2])
    id_shape.append(interferograms.shape[-1] -1)
    Id_tot = np.zeros(id_shape)
    int_0_tile = np.tile(interferograms[...,0], (interferograms.shape[-1] -1, 1, 1)).transpose(1,2,0)
    assert(np.allclose(interferograms[...,0], int_0_tile[...,1]))

    Id_tot = interferograms[..., indices_id] - int_0_tile

    ny, nx = interferograms[...,0].shape
    x, y = np.linspace(-1.0 * radius, 1.0 * radius, 2*radius), np.linspace(-1.0 * radius, 1.0 * radius, 2*radius)
    xx, yy = np.meshgrid(x, y)
    ss = np.sqrt(xx**2 + yy**2)

    Id_int = np.zeros((2*radius, 2*radius, Id_tot.shape[-1]))
    Id_zeros = np.zeros(Id_int.shape, dtype = float)

    Id_int = Id_tot[y0-radius:y0+radius, x0-radius:x0+radius, :]
    zeros_i = np.abs(Id_int) <= 1
    Id_zeros[zeros_i] = 1

    mask = [np.sqrt((xx) ** 2 + (yy) ** 2) >= radius]
    mask_tile = np.tile(mask, (Id_zeros.shape[-1],1,1)).T
    Id_zeros_mask = np.ma.array(Id_zeros, mask=mask_tile)

    ### make Hough transform of all points
    #Initialize constants
    width, height = Id_int[...,0].shape
    num_thetas = 200
    diag_len = np.ceil(np.sqrt(width * width + height * height)).astype(int)
    Acc = np.zeros((2 * diag_len, num_thetas, 4), dtype=np.uint64)
    Lambda = np.zeros(4)
    ti = np.zeros((2,4))
    sigma = np.zeros(4)
    x_shape = list(xx.shape)
    x_shape.append(4)
    tau_i = np.zeros(x_shape)
    delta_i = np.zeros(tau_i.shape)
    Id_hat = np.zeros(Id_int.shape)
    theta_max = np.zeros(4)
    sin_diff = np.zeros(tau_i.shape)
    s_max = np.zeros(4)

    print("making hough transforms... crunch crunch")
    for jj in range(4):
        Acc[...,jj], rhos, thetas, diag_len = hough_numpy(Id_zeros_mask[..., jj], x, y)
        print("Hough transform " + str(jj+1) + " done!")
        theta_max[jj], s_max[jj], theta_max_index, peak_indices = max_finding(Acc[...,jj], rhos, thetas)
        ## uncomment to check if peaks align with found peaks visually
    ##    f2, axarr2 = plt.subplots(2,1)
    ##    axarr2[0].imshow(Acc[...,jj].T, cmap = 'bone')
    ##    axarr2[0].scatter(peak_indices, np.tile(theta_max_index, len(peak_indices)))
    ##    axarr2[1].plot(rhos, Acc[:, theta_max_index, jj])
    ##    dtct = axarr2[1].scatter(rhos[peak_indices], Acc[peak_indices, theta_max_index, jj], c = 'r')
    ##    plt.show()
        Lambda[jj] = np.sum(np.diff(rhos[peak_indices]), dtype=float) / (len(peak_indices)-1.0)

    ## make tile lists to compute 3d matrices in a numpy way
    ti_tile = list(Id_hat[...,0].shape)
    ti_tile.append(1)
    xx_tile = list(Id_hat[0,0,:].shape)
    xx_tile.append(1)
    xx_tile.append(1)

    ## Compute Id_hat in a numpy way
    ti = (2 * np.pi/ Lambda) * np.array([np.cos(theta_max), np.sin(theta_max)])
    sigma = 2 * np.pi * s_max / Lambda
    tau_i = -np.tile(ti[0, :], ti_tile)  * np.tile(xx, xx_tile).transpose(1,2,0) + np.tile(ti[1, :], ti_tile) * np.tile(yy, xx_tile).transpose(1,2,0)
    delta_i = (tau_i + sigma)/2.0
    sin_diff = np.sin(delta_i)
    Id_hat = Id_int/(-2.0 * sin_diff)

    nmbr_inter = Id_zeros.shape[-1] #number of interferogram differences
    Un_sol = np.triu_indices(nmbr_inter, 1) #indices of unique phase retrievals. upper triangular indices, s.t. 12 and 21 dont get counted twice
    shape_unwrp = list(Id_zeros.shape[:2])
    shape_unwrp.append(len(Un_sol[0])) #square by amount of solutions
    Unwr_mat = np.zeros(shape_unwrp)
    angfact = np.zeros(shape_unwrp)
    atany = np.zeros(shape_unwrp)
    atanx = np.zeros(shape_unwrp)
    org_phase = np.zeros(shape_unwrp)
    org_phase_plot = np.zeros(shape_unwrp)
    org_unwr = np.zeros(shape_unwrp)

    #phase extraction
    angfact = delta_i[..., Un_sol[1]] - delta_i[..., Un_sol[0]]
    atany = Id_hat[..., Un_sol[0]]
    atanx = (Id_hat[..., Un_sol[1]] - np.cos(angfact) * Id_hat[..., Un_sol[0]]) / np.sin(angfact) ## sin needs to be added here for arctan2 to know the correct sign of y and x

    for k in range(len(Un_sol[0])):
        org_phase[..., k] = np.arctan2(atany[..., k], atanx[..., k])

    return org_phase, delta_i, sh_spots, image_i0
##    
##newy_n = raw_input("Do you want to make new interferograms? y/n")
##if newy_n == "y":
##    ### set up cameras and mirror
##    sh, int_cam = mc.set_up_cameras()
##    global mirror
##    mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here
##    ### images for making wavefront flat
##    actuators = 19
##    u_dm = np.zeros(actuators)
##    mc.set_displacement(u_dm, mirror)
##    time.sleep(0.2)
##
##    raw_input("Did you calibrate the reference mirror? Block DM")
##    sh.snapImage()
##    image_control = sh.getImage().astype(float)
##
##    raw_input("block reference mirror!")
##    sh.snapImage()
##    zero_image = sh.getImage().astype(float)
##
##    ### make actual wf flat
##    u_dm, x_pos_norm_f, y_pos_norm_f, x_pos_zero_f, y_pos_zero_f, V2D = mc.flat_wavefront(u_dm, zero_image, image_control, r_sh_px, r_int_px, sh, mirror, show_accepted_spots = False)
##
##    ### Choose abberations
##    a = np.zeros(j_max)
##    ind = np.array([0])
##    #a[3] = 0.1 * wavelength
##    #a[2] = 0.5 * wavelength
##    #a[5] = -0.3 * wavelength
##    a[3] = 1 * wavelength
##
##    #V2D_inv = np.linalg.pinv(V2D)
##    G = LSQ.matrix_avg_gradient(x_pos_norm_f, y_pos_norm_f, j_max, r_sh_px)
##    v_abb = (f_sh/(r_sh_m * px_size_sh)) * np.linalg.lstsq(V2D, np.dot(G, a))[0]
##    u_dm -= v_abb
##    mc.set_displacement(u_dm, mirror)
##
##    raw_input("remove piece of paper")
##    time.sleep(0.2)
##    int_cam.snapImage()
##    image_i0 = int_cam.getImage().astype(float)
##    PIL.Image.fromarray(image_i0).save("20161130_five_inter_test/interferogram_0.tif")
##
##    raw_input("tip and tilt 1")
##    time.sleep(1)
##    int_cam.snapImage()
##    image_i1 = int_cam.getImage().astype(float)
##    PIL.Image.fromarray(image_i1).save("20161130_five_inter_test/interferogram_1.tif")
##
##    raw_input("tip and tilt 2")
##    time.sleep(1)
##    int_cam.snapImage()
##    image_i2 = int_cam.getImage().astype(float)
##    PIL.Image.fromarray(image_i2).save("20161130_five_inter_test/interferogram_2.tif")
##
##    raw_input("tip and tilt 3")
##    time.sleep(1)
##    int_cam.snapImage()
##    image_i3 = int_cam.getImage().astype(float)
##    PIL.Image.fromarray(image_i3).save("20161130_five_inter_test/interferogram_3.tif")
##
##    raw_input("tip and tilt 4")
##    time.sleep(1)
##    int_cam.snapImage()
##    image_i4 = int_cam.getImage().astype(float)
##    PIL.Image.fromarray(image_i4).save("20161130_five_inter_test/interferogram_4.tif")
##
##else:
##    #impath_i0 = os.path.abspath("20161125_interferograms_for_theory/interferogram_0.tif")
##    image_i0 = np.array(PIL.Image.open("20161130_five_inter_test/interferogram_0.tif"))
##    image_i1 = np.array(PIL.Image.open("20161130_five_inter_test/interferogram_1.tif"))
##    image_i2 = np.array(PIL.Image.open("20161130_five_inter_test/interferogram_2.tif"))
##    image_i3 = np.array(PIL.Image.open("20161130_five_inter_test/interferogram_3.tif"))
##    image_i4 = np.array(PIL.Image.open("20161130_five_inter_test/interferogram_4.tif"))
