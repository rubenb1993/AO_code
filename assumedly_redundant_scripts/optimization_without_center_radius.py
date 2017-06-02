import sys
##import os
import time
if "C:\Program Files\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Program Files\Micro-Manager-1.4")
import MMCorePy
import PIL.Image
import numpy as np
from matplotlib import rc
import Hartmann as Hm
import displacement_matrix as Dm
import Zernike as Zn
import mirror_control as mc
import LSQ_method as LSQ
import matplotlib.pyplot as plt
import janssen
import phase_extraction as PE
import phase_unwrapping_test as pw
import scipy.optimize as opt
import scipy.ndimage as ndimage
import json #to write editable txt files

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

def rms_lsq(variables, *args):
    """calculate the rms value of the LSQ method with variables:
    variables[0] = piston,
    variables[1] = centre_x,
    variables[2] = centre_y,
    variables[3] = radius of circle on sh sensor[px],
    Z_mat is used to calculate the Zernike polynomial on a grid, to compare with original interferogram.
    arguments should be in order"""
    if args:
       wavelength, j_max, f_sh, px_size_sh, dist_image, image_control, y_pos_flat_f, x_pos_flat_f, xx, yy, orig, mask, N, Z_mat, power_mat, box_len = args
    else:
        print("put the arguments!")
        return
    r_sh_m = px_size_sh * variables[3]

    x_pos_norm = (x_pos_flat_f - variables[1])/variables[3]
    y_pos_norm = (y_pos_flat_f - variables[2])/variables[3]

    inside = np.sqrt(x_pos_norm ** 2 + y_pos_norm**2) <= (1+ (box_len/variables[3]))
    x_pos_norm, y_pos_norm, x_pos_flat_f, y_pos_flat_f = mc.filter_positions(inside, x_pos_norm, y_pos_norm, x_pos_flat_f, y_pos_flat_f)

    G = LSQ.matrix_avg_gradient(x_pos_norm, y_pos_norm, j_max, variables[3], power_mat)
    
    x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat_f, y_pos_flat_f, dist_image, xx, yy)

    s = np.hstack(Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat_f, y_pos_flat_f, px_size_sh, f_sh, r_sh_m, wavelength))
    a = np.linalg.lstsq(G,s)[0]

    orig = np.ma.array(orig, mask = mask)
    inter = np.ma.array(Zn.int_for_comp(j_max, a, N, variables[0], Z_mat, fliplr = False), mask = mask)
    rms = np.sqrt(np.sum((inter - orig)**2)/N**2)
    return rms

def rms_lsq_wh_xy(variables, *args):
    """calculate the rms value of the LSQ method with variables:
    variables[0] = piston,
    variables[1] = radius of circle on sh sensor[px],
    Z_mat is used to calculate the Zernike polynomial on a grid, to compare with original interferogram.
    arguments should be in order"""
    if args:
       wavelength, j_max, f_sh, px_size_sh, dist_image, image_control, y_pos_flat_f, x_pos_flat_f, xx, yy, orig, mask, N, Z_mat, power_mat, box_len, centre = args
    else:
        print("put the arguments!")
        return
    r_sh_m = px_size_sh * variables[1]

    x_pos_norm = (x_pos_flat_f - centre[0])/variables[1]
    y_pos_norm = (y_pos_flat_f - centre[1])/variables[1]

    ### not implemented due to constraints
    #inside = np.sqrt(x_pos_norm ** 2 + y_pos_norm**2) <= (1+ (box_len/variables[1]))
    #x_pos_norm, y_pos_norm, x_pos_flat_f, y_pos_flat_f = mc.filter_positions(inside, x_pos_norm, y_pos_norm, x_pos_flat_f, y_pos_flat_f)

    G = LSQ.matrix_avg_gradient(x_pos_norm, y_pos_norm, j_max, variables[1], power_mat)
    
    x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat_f, y_pos_flat_f, dist_image, xx, yy)

    s = np.hstack(Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat_f, y_pos_flat_f, px_size_sh, f_sh, r_sh_m, wavelength))
    a = np.linalg.lstsq(G,s)[0]

    orig = np.ma.array(orig, mask = mask)
    inter = np.ma.array(Zn.int_for_comp(j_max, a, N, variables[0], Z_mat, fliplr = False), mask = mask)
    rms = np.sqrt(np.sum((inter - orig)**2)/N**2)
    return rms

def rms_janss(variables, *args):
    """calculate the rms value of the Janssen method with variables:
    variables[0] = piston,
    variables[1] = radius of circle on sh sensor[px],
    Z_mat is used to calculate the Zernike polynomial on a grid, to compare with original interferogram.
    arguments should be in order"""
    if args:
       x_pos_zero, y_pos_zero, image_control, dist_image, px_size_sh, f_sh, j_max, wavelength, xx, yy, N, orig, mask, Z_mat, box_len, centre = args
    else:
        print("put the arguments!")
        return
    x_pos_flat_f, y_pos_flat_f = Hm.centroid_positions(x_pos_zero, y_pos_zero, image_control, xx, yy)

    r_sh_m = px_size_sh * variables[1]

    x_pos_norm = (x_pos_flat_f - centre[0])/variables[1]
    y_pos_norm = (y_pos_flat_f - centre[1])/variables[1]

    ### check if all is within circle
    ### not implemented due to constraints
    #inside = np.sqrt(x_pos_norm ** 2 + y_pos_norm**2) <= (1+ (box_len/variables[1]))
    #x_pos_norm, y_pos_norm, x_pos_flat_f, y_pos_flat_f = mc.filter_positions(inside, x_pos_norm, y_pos_norm, x_pos_flat_f, y_pos_flat_f)

    ## calculate slopes
    x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat_f, y_pos_flat_f, dist_image, xx, yy)
    dWdx, dWdy = Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat_f, y_pos_flat_f, px_size_sh, f_sh, r_sh_m, wavelength)

    ## make zernike matrix
    kmax = np.power(np.ceil(np.sqrt(j_max)),2) #estimation of maximum fringe number
    n, m = Zn.Zernike_j_2_nm(np.array(range(1, int(kmax)+1))) #find n and m pairs for maximum fringe number
    Kmax = np.max(Zn.Zernike_nm_2_j(n+1, np.abs(m)+1)) #find highest order of j for which beta is needed
    Z_mat_complex = janssen.avg_complex_zernike(x_pos_norm, y_pos_norm, Kmax, variables[1])

    #Invert and solve for beta
    dW_plus = dWdx + 1j * dWdy
    dW_min = dWdx - 1j * dWdy
    beta_plus = np.linalg.lstsq(Z_mat_complex, dW_plus)[0]
    beta_min = np.linalg.lstsq(Z_mat_complex, dW_min)[0]

    kmax = int(kmax)
    a = np.zeros(kmax, dtype = np.complex_)
    a_check = np.zeros(j_max, dtype = np.complex_)
    #a_avg = np.zeros(j_max, dtype = np.complex_)
    for jj in range(2, kmax+1):
        n, m = Zn.Zernike_j_2_nm(jj)
        index1 = int(Zn.Zernike_nm_2_j(n - 1.0, m + 1.0) - 1)
        index2 = int(Zn.Zernike_nm_2_j(n - 1.0, m - 1.0) - 1)
        index3 = int(Zn.Zernike_nm_2_j(n + 1.0, m + 1.0) - 1)
        index4 = int(Zn.Zernike_nm_2_j(n + 1.0, m - 1.0) - 1)
        fact1 = 1.0 / ( 2 * n * ( 1 + (n != abs(m))))
        fact2 = 1.0 / (2 * (n+2) * ( 1 + (((n+2) != abs(m)))))
        if m + 1.0 > n - 1.0:
            a[jj-1] = fact1 * (beta_min[index2]) - fact2 * (beta_plus[index3] + beta_min[index4])
        elif np.abs(m - 1.0) > np.abs(n - 1.0):
            a[jj-1] = fact1 * (beta_plus[index1]) - fact2 * (beta_plus[index3] + beta_min[index4])
        else:
            a[jj-1] = fact1 * (beta_plus[index1] + beta_min[index2]) - fact2 * (beta_plus[index3] + beta_min[index4])

    for jj in range(2, j_max+2):
        n, m = Zn.Zernike_j_2_nm(jj)
        if m > 0:
            j_min = int(Zn.Zernike_nm_2_j(n, -m))
            a_check[jj-2] = (1.0/np.sqrt(2*n+2))*(a[jj-1] + a[j_min-1])
        elif m < 0:
            j_plus = int(Zn.Zernike_nm_2_j(n, np.abs(m)))
            a_check[jj-2] = (1.0/np.sqrt(2*n+2)) * (a[j_plus - 1] - a[jj-1]) * 1j
        else:
            a_check[jj-2] = (1.0/np.sqrt(n+1)) * a[jj-1]
    a_janss = np.real(a_check)

    orig = np.ma.array(orig, mask = mask)
    inter = np.ma.array(Zn.int_for_comp(j_max, a_janss, N, variables[0], Z_mat, fliplr = False), mask = mask)
    rms = np.sqrt(np.sum((inter - orig)**2)/N**2)
    return rms

# Define font for figures
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size' : 7})
rc('text', usetex=False)

## Given paramters for centroid gathering
px_size_sh = 5.2e-6     # width of pixels
px_size_int = 5.2e-6
f_sh = 17.6e-3            # focal length
r_int_px = 280
r_sh_px = 340
r_sh_m = r_sh_px * px_size_int
j_max= 30         # maximum fringe order
wavelength = 632e-9 #[m]
box_len = 35.0 #half width of subaperture box in px
gold = (1 + np.sqrt(5))/2

## plot making parameters
dpi_num = 300
int_im_size = (4.98, 3.07)
int_im_size_23 = (0.66 * 4.98, 3.07)
int_im_size_13 = (0.33 * 4.98, 3.07)

## centre and radius of interferogam. Done by eye, with the help of define_radius.py
x0 = 550
y0 = 484
radius = int(r_int_px)

## pack everything neatly in 1 vector against clutter
constants = np.stack((px_size_sh, px_size_int, f_sh, r_int_px, r_sh_px, r_sh_m, j_max, wavelength, box_len, x0, y0, radius))

x_pcg, y_pcg = np.linspace(-1, 1, 2*radius), np.linspace(-1, 1, 2*radius)
N = len(x_pcg)
i, j = np.linspace(0, N-1, N), np.linspace(0, N-1, N)
xx_alg, yy_alg = np.meshgrid(x_pcg, y_pcg)
ii, jj = np.meshgrid(i, j)

xy_inside = np.where(np.sqrt(xx_alg**2 + yy_alg**2) <= 1)
x_in, y_in = xx_alg[xy_inside], yy_alg[xy_inside]


j = np.arange(2, j_max) #start at 2, end at j_max-1
power_mat = Zn.Zernike_power_mat(j_max+2)
Zernike_3d = Zn.Zernike_xy(x_in, y_in, power_mat, j) ## create a 3d matrix with values of zernike polynomials at x_in and y_in 
Zernike_2d = np.zeros((len(x_in), j_max)) ## flatten to perform least squares fit
for i in range(len(j)):
    Zernike_2d[:, i] = Zernike_3d[...,i].flatten()

folder_names = ["20170126_single_actuators/1/", "20170126_single_actuators/3/"]#, "20170126_single_actuators/6/", "20170126_single_actuators/8/", "20170125_spherical/", "20170125_astigmatism/", "20170125_trifoil/", "20170125_coma/", "20170216_desired_vs_created/"]
#folder_names = ["20170125_coma/"]

save_string = ["actuators_1", "actuators_3", "actuators_6", "actuators_8", "spherical", "astigmatism", "trifoil", "coma", "defocus"]
save_fold = "20170214_post_processing/int_zn/"


for i in range(len(folder_names)):
    folder_name = folder_names[i]
    fold_name = folder_name
    inter_0 = np.array(PIL.Image.open(folder_name + "interferogram_0.tif"))
    org_phase = np.load(folder_name + "org_phase.npy")
    butter_unwr = np.load(folder_name + "filtered_phase.npy")

    mask = [np.sqrt((xx_alg) ** 2 + (yy_alg) ** 2) >= 1]
    mask_tile = np.tile(mask, (org_phase.shape[-1],1,1)).T
    butter_mask = np.ma.array(butter_unwr, mask = mask_tile)
    mean_butt = butter_mask.mean(axis=(0,1))
    butter_unwr -= mean_butt


    but_med = np.median(butter_unwr, axis = 2)

    but_med_flat = but_med[xy_inside]

    a_butt = np.linalg.lstsq(Zernike_2d, but_med_flat)[0]
    a_inter= a_butt
    #a_inter *= wavelength
    [ny_i, nx_i] = inter_0.shape
    flipped_y0 = y0
    flipped_x0 = nx_i - x0
    #flipped_x0 = x0
    orig = np.fliplr(np.array(PIL.Image.open(folder_name + "orig_scale.tif")))#np.ma.array(inter_0[flipped_y0-radius:flipped_y0+radius, flipped_x0-radius:flipped_x0+radius], mask = mask)
    
    xi, yi = np.linspace(-1, 1, N), np.linspace(-1, 1, N)
    xi, yi = np.meshgrid(xi, yi)
    j_range = np.arange(2, j_max+2)
    Z_mat = Zn.Zernike_xy(xi, yi, power_mat, j_range)
    #orig /= orig.max()

    flipint = False

    piston, inter_rms = opt.fmin(rms_piston, 0, args = (j_max, a_butt, N, Z_mat, orig, mask, flipint), full_output = True)[:2]

    image_ref_mirror = np.array(PIL.Image.open(folder_name + "image_ref_mirror.tif"))
    zero_pos_dm = np.array(PIL.Image.open(folder_name + "zero_pos_dm.tif"))
    dist_image = np.array(PIL.Image.open(folder_name + "dist_image.tif"))
    flat_wf = np.array(PIL.Image.open(folder_name + "flat_wf.tif"))
    sh_spots = np.dstack((image_ref_mirror, zero_pos_dm, dist_image)) 

    zero_image = sh_spots[..., 1]
    zero_image_zeros = np.copy(sh_spots[..., 1])
    dist_image = sh_spots[..., 2]
    image_control = sh_spots[..., 0]
    [ny,nx] = zero_image.shape
    x = np.linspace(1, nx, nx)
    y = np.linspace(1, ny, ny)
    xx, yy = np.meshgrid(x, y)

    x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image_zeros)
    x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, image_control, xx, yy)
    x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat, y_pos_flat, dist_image, xx, yy)
    centre = Hm.centroid_centre(x_pos_flat, y_pos_flat)
    x_pos_norm = ((x_pos_flat - centre[0]))/r_sh_px
    y_pos_norm = ((y_pos_flat - centre[1]))/r_sh_px
    inside = np.where(np.sqrt(x_pos_norm**2 + y_pos_norm**2) <= (1 + (box_len/r_sh_px))) #35 is the half the width of the pixel box aroudn a centroid and r_sh_px is the scaling factor
    x_pos_zero_f, y_pos_zero_f, x_pos_flat_f, y_pos_flat_f, x_pos_norm_f, y_pos_norm_f, x_pos_dist_f, y_pos_dist_f = mc.filter_positions(inside, x_pos_zero, y_pos_zero, x_pos_flat, y_pos_flat, x_pos_norm, y_pos_norm, x_pos_dist, y_pos_dist)
    G = LSQ.matrix_avg_gradient(x_pos_norm_f, y_pos_norm_f, j_max, r_sh_px, power_mat)
    s = np.hstack(Hm.centroid2slope(x_pos_dist_f, y_pos_dist_f, x_pos_flat_f, y_pos_flat_f, px_size_sh, f_sh, r_sh_m, wavelength))
##    a_lsq_opt = np.linalg.lstsq(G, s)[0]
##    a_janss_opt = janssen.coeff_optimum(x_pos_flat_f, y_pos_flat_f, x_pos_norm_f, y_pos_norm_f, xx, yy, dist_image, image_control, px_size_sh, f_sh, r_sh_m, wavelength, j_max)

    lsq_args = (wavelength, j_max, f_sh, px_size_sh, dist_image, image_control, y_pos_flat, x_pos_flat, xx, yy, orig, mask, N, Z_mat, power_mat, box_len, centre)
    janss_args = (x_pos_zero, y_pos_zero, image_control, dist_image, px_size_sh, f_sh, j_max, wavelength, xx, yy, N, orig, mask, Z_mat, box_len, centre)
    
    

    #### ran a constrained minimization scheme for finding a radius which minimizes rms error. All further calculations will be done with that. average radius between optimum for janssen and SH
##    print("optimizing SH")
##    minradius = 0.85 * np.max(np.sqrt((x_pos_flat_f - centre[0])**2 + (y_pos_flat_f - centre[1])**2))
##    print("mindradius = " + str(minradius))
##    bounds_min = ((None, None), (minradius, None))
##    initial_guess = [0, 320.0]
##
##
##    bf_lsq = time.time()
##    optresult_lsq = opt.minimize(rms_lsq_wh_xy, initial_guess, args = lsq_args, method = 'L-BFGS-B', bounds = bounds_min, options = {'maxiter': 100, 'disp':True})
##    aft_lsq = time.time()
##    print("1000 iterations LSQ took " + str(aft_lsq-bf_lsq) +" s")
##
##    bf_janss = time.time()
##    optresult_janss = opt.minimize(rms_janss, optresult_lsq.x, args = janss_args, method = 'L-BFGS-B', bounds = bounds_min, options = {'maxiter': 100, 'disp':True})
##    aft_janss = time.time()
##    print("All iterations Janssen took " + str(aft_janss - bf_janss) + " s")
##    vars_lsq = optresult_lsq.x
##    vars_janss = optresult_janss.x
##    print("radius lsq = " + str(vars_lsq[1]) + ", radius janss = " + str(vars_janss[1]))
##    r_sh_px = (vars_lsq[1] + vars_janss[1])/2.0
    ## uncomment for radius determined by Janssen method
    r_sh_px = np.load("20170214_post_processing/int_zn/determined_radius.npy").item()
    print("optimizing SH methods")

    ## find coefficients according to optimum
    r_sh_lsq = r_sh_px
    x_pos_norm_lsq = (x_pos_flat - centre[0])/r_sh_lsq
    y_pos_norm_lsq = (y_pos_flat - centre[1])/r_sh_lsq
    inside_lsq = np.where(np.sqrt(x_pos_norm_lsq**2 + y_pos_norm_lsq**2) <= (1 + (box_len/r_sh_lsq)))
    x_pos_flat_f, y_pos_flat_f = mc.filter_positions(inside_lsq, x_pos_flat, y_pos_flat)
    x_pos_norm_lsq, y_pos_norm_lsq = mc.filter_positions(inside_lsq, x_pos_norm_lsq, y_pos_norm_lsq)
    G = LSQ.matrix_avg_gradient(x_pos_norm_lsq, y_pos_norm_lsq, j_max, r_sh_lsq, power_mat)
    r_sh_px_lsq_opt = r_sh_lsq
    r_sh_m_lsq_opt = px_size_sh * r_sh_lsq
    x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat_f, y_pos_flat_f, dist_image, xx, yy)
    #x_pos_dist, y_pos_dist = mc.filter_positions(inside_lsq, x_pos_dist, y_pos_dist, x_pos_flat_f, y_pos_flat_f)
    s = np.hstack(Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat_f, y_pos_flat_f, px_size_sh, f_sh, r_sh_m_lsq_opt, wavelength))
    a_lsq_opt = np.linalg.lstsq(G, s)[0]

    pist_args_lsq = (j_max, a_lsq_opt, N, Z_mat, orig, mask, False)
    pist_lsq, lsq_rms = opt.fmin(rms_piston, 0, pist_args_lsq, full_output = True)[:2]
    vars_lsq = pist_lsq

    ### find coefficients janss according to optimum
    r_sh_janss = r_sh_px
    x_pos_norm_janss = (x_pos_flat - centre[0])/r_sh_janss
    y_pos_norm_janss = (y_pos_flat - centre[1])/r_sh_janss
    inside_janss = np.where(np.sqrt(x_pos_norm_janss**2 + y_pos_norm_janss**2) <= (1 + (box_len/r_sh_janss)))
    x_pos_norm_janss, y_pos_norm_janss, x_pos_flat_f, y_pos_flat_f = mc.filter_positions(inside_janss, x_pos_norm_janss, y_pos_norm_janss, x_pos_flat, y_pos_flat)
    r_sh_m_janss = px_size_sh * r_sh_janss
    a_janss_opt = janssen.coeff_optimum(x_pos_flat_f, y_pos_flat_f, x_pos_norm_janss, y_pos_norm_janss, xx, yy, dist_image, image_control, px_size_sh, f_sh, r_sh_m_janss, wavelength, j_max)

    pist_args_janss = (j_max, a_janss_opt, N, Z_mat, orig, mask, False)
    pist_janss, janss_rms = opt.fmin(rms_piston, 0, pist_args_janss, full_output = True)[:2]
    vars_janss = pist_janss
    
    #pist_lsq = vars_lsq[0]#opt.fmin(rms_piston, 0, args = (j_max, a_lsq, N, Z_mat, orig, mask, fliplsq))
    #pist_janss = vars_janss[0]#opt.fmin(rms_piston, 0, args = (j_max, np.real(a_janss), N, Z_mat, orig, mask, flipjanss))

    #lsq_rms = optresult_lsq.fun
    #janss_rms = optresult_janss.fun

    ### plot results
    rms_mat = np.array([inter_rms, lsq_rms, janss_rms])
    f, axes = plt.subplots(nrows = 1, ncols = 5, figsize = (4.98, 4.98/4.4), gridspec_kw={'width_ratios':[4, 4, 4, 4, 0.4]})#, 'height_ratios':[1,1,1,1,1]})
    titles = [r'Original', r'Interferogram', r'LSQ', r'Janssen']
    interf = axes[0].imshow(np.ma.array(orig, mask = mask), vmin = 0, vmax = 1, cmap = 'bone', origin = 'lower', interpolation = 'none')
    Zn.imshow_interferogram(j_max, a_inter, N, piston = piston, ax = axes[1])
    Zn.imshow_interferogram(j_max, a_lsq_opt, N, piston = pist_lsq, ax = axes[2], fliplr = False)
    Zn.imshow_interferogram(j_max, a_janss_opt, N, piston = pist_janss, ax = axes[3], fliplr = False)
    ##axes[0].set_title('original')
    ##axes[1].set_title('from int')
    ##axes[2].set_title('from lsq')
    ##axes[3].set_title('from janss')
    for ii in range(4):
        axes[ii].set(adjustable = 'box-forced', aspect = 'equal')
        axes[ii].get_xaxis().set_ticks([])
        axes[ii].get_yaxis().set_ticks([])
        axes[ii].set_title(titles[ii], fontsize = 9)
        if ii == 0:
            axes[ii].text(N/2, -N/6, r"$\varepsilon~=~$", fontsize = 7, ha = 'center')
        else:
            axes[ii].text(N/2, -N/6, "%.4f"%rms_mat[ii-1], fontsize = 7, ha = 'center')
    cbar = plt.colorbar(interf, cax = axes[-1], ticks = [0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.set_ylabel("Normalized Intensity", fontsize = 8)
    f.savefig(fold_name+'methods_compared_avg_radius.png', bbox_inches = 'tight', dpi = dpi_num)
    f.savefig(save_fold + "interferograms_"+save_string[i]+".png", bbox_inches = 'tight', dpi = dpi_num)

    f2, ax2 = plt.subplots(nrows = 1, ncols = 1, figsize = (4.98, 4.98/4.4))
    ax2.plot((max(j_range)+1)*[0], 'k-', linewidth = 0.5)
    ##msv = np.argmax(np.abs(a_janss_opt)) #most_significant_value
    ##sign_janss = np.sign(a_janss_opt[msv])
    ##sign_int = np.sign(a_inter[msv])
    ##a_inter *= (sign_janss * sign_int) ## make signs equal of janss and interferogram. if signs are equal, will result in 1, else -1.

#### full plot    
##    a_x = np.arange(2, j_max+2)
##    ax2.plot(a_x, a_inter, 'sk', label = 'Interferogram')
##    ax2.plot(a_x, a_janss_opt, 'og', label = 'Janssen')
##    ax2.plot(a_x, a_lsq_opt, '2b', label = 'LSQ')
##    ax2.set_xlim([0,31.5])
##    min_a = np.min([np.min(a_inter), np.min(a_janss_opt), np.min(a_lsq_opt)])
##    max_a = np.max([np.max(a_inter), np.max(a_janss_opt), np.max(a_lsq_opt)])
##    b_gold = max_a - min_a / (2*gold)
##    ax2.set_ylim([min_a - b_gold, max_a + b_gold])
##    ax2.set_ylabel(r'$a_j$')
##    ax2.set_xlabel(r'$j$')

    #### high order
    a_x = np.arange(2, j_max+2)
    ax2.plot(a_x, a_inter, 'sk', label = 'Interferogram')
    ax2.plot(a_x, a_janss_opt, 'og', label = 'Janssen')
    ax2.plot(a_x, a_lsq_opt, '2b', label = 'LSQ')
    ax2.set_xlim([14.5,31.5])
    ax2.set_ylim([-0.3, 0.5])
    min_a = np.min([np.min(a_inter), np.min(a_janss_opt), np.min(a_lsq_opt)])
    max_a = np.max([np.max(a_inter), np.max(a_janss_opt), np.max(a_lsq_opt)])
    b_gold = max_a - min_a / (2*gold)
    #ax2.set_ylim([min_a - b_gold, max_a + b_gold])
    ax2.set_ylabel(r'$a_j$')
    ax2.set_xlabel(r'$j$')

##    ax2.legend(prop={'size':7}, loc = 'upper right')
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax2.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.15), ncol = 3, fontsize = 6)
    f2.savefig(fold_name+'zernike_coeff_compared_avg_radius_zoom.png', bbox_inches = 'tight', dpi = dpi_num)
    f2.savefig(save_fold + "zn_coeff_"+save_string[i]+"_zoom.png", bbox_inches = 'tight', dpi = dpi_num)

    rms_dict = {'rms_inter':inter_rms,'rms_lsq':lsq_rms, 'rms_janss':janss_rms}
    vars_dict = {'vars_lsq':vars_lsq, 'vars_janss':vars_janss, 'pist_inter':piston}
    coeff_dict = {'coeff_inter':a_inter, 'coeff_lsq':a_lsq_opt, 'coeff_janss':a_janss_opt}
    np.save(folder_name + 'vars_dictionary_avg_radius.npy', vars_dict)
    np.save(folder_name + 'coeff_dictionary_avg_radius.npy', coeff_dict)
    json.dump(rms_dict, open(folder_name + "rms_dict_avg_radius.txt", 'w'))
    print(" rms interferogram: " + str(inter_rms) + ",\n rms LSQ: " + str(lsq_rms) + ",\n rms Janssen: " + str(janss_rms))

plt.show()
