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
import phase_extraction_slm as PE
import phase_unwrapping_test as pw
import scipy.optimize as opt
import scipy.ndimage as ndimage
import matplotlib.gridspec as gridspec
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

def rms_phi_lsq(variables, *args):
    """Calculate the 2-norm distance from the intended phase to the measured phase,
    variables[0] = x0
    variables[1] = y0
    variables[2] = radius
    Z_mat is used to calculate Zernike polynomials on a grid, to compare with original phase, 3rd dimension should be length j_max"""
    if args:
        wavelength, j_max, f_sh, px_size_sh, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist, xx, yy, mask, N, Z_mat, power_mat, box_len, phi_ref = args
    else:
        print("argument error!")
        return
    r_sh_m = px_size_sh * variables[2]

    x_pos_norm = (x_pos_flat - variables[0])/variables[2]
    y_pos_norm = (y_pos_flat - variables[1])/variables[2]

    inside = np.sqrt(x_pos_norm ** 2 + y_pos_norm**2) <= (1+ (box_len/variables[2]))
    x_pos_norm_it, y_pos_norm_it, x_pos_flat_it, y_pos_flat_it, x_pos_dist_it, y_pos_dist_it = mc.filter_positions(inside, x_pos_norm, y_pos_norm, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist)

    G = LSQ.matrix_avg_gradient(x_pos_norm_it, y_pos_norm_it, j_max, variables[1], power_mat, box_len)
    
    s = np.hstack(Hm.centroid2slope(x_pos_dist_it, y_pos_dist_it, x_pos_flat_it, y_pos_flat_it, px_size_sh, f_sh, r_sh_m, wavelength))
    a = np.linalg.lstsq(G,s)[0]

    phi_lsq = np.ma.array(np.dot(Z_mat, a), mask = mask)
    phi_diff = np.ma.array(phi_ref - phi_lsq, mask = mask)
    rms = np.sqrt((phi_diff**2).sum())/N
    return rms

def rms_lsq_wh_xy(variables, *args):
    """calculate the rms value of the LSQ method with variables:
    variables[0] = piston,
    variables[1] = radius of circle on sh sensor[px],
    Z_mat is used to calculate the Zernike polynomial on a grid, to compare with original interferogram.
    arguments should be in order"""
    if args:
       wavelength, j_max, f_sh, px_size_sh, y_pos_flat, x_pos_flat, x_pos_dist, y_pos_dist, xx, yy, orig, mask, N, Z_mat, power_mat, box_len, centre = args
    else:
        print("put the arguments!")
        return
    r_sh_m = px_size_sh * variables[1]

    x_pos_norm = (x_pos_flat - centre[0])/variables[1]
    y_pos_norm = (y_pos_flat - centre[1])/variables[1]

    ### not implemented due to constraints
    inside = np.sqrt(x_pos_norm ** 2 + y_pos_norm**2) <= (1+ (box_len/variables[1]))
    x_pos_norm_it, y_pos_norm_it, x_pos_flat_it, y_pos_flat_it, x_pos_dist_it, y_pos_dist_it = mc.filter_positions(inside, x_pos_norm, y_pos_norm, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist)

    G = LSQ.matrix_avg_gradient(x_pos_norm_it, y_pos_norm_it, j_max, variables[1], power_mat, box_len)
    
    #x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat_f, y_pos_flat_f, dist_image, xx, yy, spot_size = box_len)

    s = np.hstack(Hm.centroid2slope(x_pos_dist_it, y_pos_dist_it, x_pos_flat_it, y_pos_flat_it, px_size_sh, f_sh, r_sh_m, wavelength))
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
       x_pos_zero, y_pos_zero, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist, px_size_sh, f_sh, j_max, wavelength, xx, yy, N, orig, mask, Z_mat, box_len, centre = args
    else:
        print("put the arguments!")
        return
    #x_pos_flat_f, y_pos_flat_f = Hm.centroid_positions(x_pos_zero, y_pos_zero, image_control, xx, yy, spot_size = box_len)

    r_sh_m = px_size_sh * variables[1]

    x_pos_norm = (x_pos_flat - centre[0])/variables[1]
    y_pos_norm = (y_pos_flat - centre[1])/variables[1]

    ### check if all is within circle
    ### not implemented due to constraints
    inside = np.sqrt(x_pos_norm ** 2 + y_pos_norm**2) <= (1+ (box_len/variables[1]))
    x_pos_norm_it, y_pos_norm_it, x_pos_flat_it, y_pos_flat_it, x_pos_dist_it, y_pos_dist_it = mc.filter_positions(inside, x_pos_norm, y_pos_norm, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist)

    ## calculate slopes
    #x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat_f, y_pos_flat_f, dist_image, xx, yy, spot_size = box_len)
    dWdx, dWdy = Hm.centroid2slope(x_pos_dist_it, y_pos_dist_it, x_pos_flat_it, y_pos_flat_it, px_size_sh, f_sh, r_sh_m, wavelength)

    ## make zernike matrix
    kmax = np.power(np.ceil(np.sqrt(j_max+1)),2) #estimation of maximum fringe number
    n, m = Zn.Zernike_j_2_nm(np.array(range(1, int(kmax)+1))) #find n and m pairs for maximum fringe number
    Kmax = np.max(Zn.Zernike_nm_2_j(n+1, np.abs(m)+1)) #find highest order of j for which beta is needed
    Z_mat_complex = janssen.avg_complex_zernike(x_pos_norm_it, y_pos_norm_it, Kmax, variables[1], spot_size = box_len)

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

def rms_phi_janssen(variables, *args):
    """Calculate the 2-norm distance from the intended phase to the measured phase using Janssen's method,
    variables[0] = x0
    variables[1] = y0
    variables[2] = radius
    Z_mat is used to calculate Zernike polynomials on a grid, to compare with original phase, 3rd dimension should be length j_max"""

    if args:
       x_pos_zero, y_pos_zero, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist, px_size_sh, f_sh, j_max, wavelength, xx, yy, N, mask, Z_mat, box_len, phi_ref = args
    else:
        print("put the arguments!")
        return
    r_sh_m = px_size_sh * variables[2]

    x_pos_norm = (x_pos_flat - variables[0])/variables[2]
    y_pos_norm = (y_pos_flat - variables[1])/variables[2]

    ### check if all is within circle
    ### not implemented due to constraints
    inside = np.sqrt(x_pos_norm ** 2 + y_pos_norm**2) <= (1+ (box_len/variables[2]))
    x_pos_norm_it, y_pos_norm_it, x_pos_flat_it, y_pos_flat_it, x_pos_dist_it, y_pos_dist_it = mc.filter_positions(inside, x_pos_norm, y_pos_norm, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist)

    ## calculate slopes
    #x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat_f, y_pos_flat_f, dist_image, xx, yy, spot_size = box_len)
    dWdx, dWdy = Hm.centroid2slope(x_pos_dist_it, y_pos_dist_it, x_pos_flat_it, y_pos_flat_it, px_size_sh, f_sh, r_sh_m, wavelength)

    ## make zernike matrix
    kmax = np.power(np.ceil(np.sqrt(j_max+1)),2) #estimation of maximum fringe number
    n, m = Zn.Zernike_j_2_nm(np.array(range(1, int(kmax)+1))) #find n and m pairs for maximum fringe number
    Kmax = np.max(Zn.Zernike_nm_2_j(n+1, np.abs(m)+1)) #find highest order of j for which beta is needed
    Z_mat_complex = janssen.avg_complex_zernike(x_pos_norm_it, y_pos_norm_it, Kmax, variables[2], spot_size = box_len)

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

    phi_janss = np.ma.array(np.dot(Z_mat, a_janss), mask = mask)
    phi_diff = np.ma.array(phi_ref - phi_janss, mask = mask)
    rms = np.sqrt((phi_diff**2).sum())/N
    return rms
    
# Define font for figures
##rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size' : 7})
rc('text', usetex=False)

## Given paramters for centroid gathering
px_size_sh = 4.65e-6     # width of pixels
px_size_int = 3.45e-6
f_sh = 14.2e-3            # focal length
r_int_px = 600/2
r_sh_m = 2.048e-3
r_sh_px = int(r_sh_m/px_size_sh)
j_maxes = np.insert(np.rint(np.logspace(1.1, 2.1, num = 15)),0, 47)#np.rint(np.logspace(0.3, 1.7, num = 10))#
wavelength = 632e-9 #[m]
box_len = 30 #half width of subaperture box in px
gold = (1 + np.sqrt(5))/2
pitch_sh = 150.0e-6

## plot making parameters
dpi_num = 300
int_im_size = (4.98, 3.07)
int_im_size_23 = (0.66 * 4.98, 3.07)
int_im_size_13 = (0.33 * 4.98, 3.07)

## centre and radius of interferogam. Done by eye, with the help of define_radius.py
### x0, y0 of 20170404 measurements: 464, 216, r_int_px =600
### x0, y0 of 20170405 measurements: 434, 254, r_int_px = 600
### x0, y0 of 20170412 measurements: 446, 224
### test x0, y0 12-4-17 468,240, 570 r_int_px
### x0, y0 of 20170413 measurements: 460, 222 r_int_px = 600
### x0, y0 of 20170420 measurements: 434,250,586, can be improved by 433, 246, 596
### x0, y0 of 20170424 measurements: 424, 242, 600
### x0, y0 of 20170504 measurements: 346, 246, 590 or maybe 332, 236, 600
x0 = 332 + r_int_px
y0 = 236 + r_int_px
radius = int(r_int_px)



x_pcg, y_pcg = np.linspace(-1, 1, 2*radius), np.linspace(-1, 1, 2*radius)
N_int = len(x_pcg)
i, j = np.linspace(0, N_int-1, N_int), np.linspace(0, N_int-1, N_int)
#i, j = np.arange(0, nx, 1), np.arange(0, ny, 1)
xx_alg, yy_alg = np.meshgrid(x_pcg, y_pcg)
ii, jj = np.meshgrid(i, j)

xy_inside = np.where(np.sqrt(xx_alg**2 + yy_alg**2) <= 1)
x_in, y_in = xx_alg[xy_inside], yy_alg[xy_inside]

folder_extensions = ["5_1/", "5_5/", "6_2/", "6_4/", "6_6/", "3_zerns_1/", "3_zerns_2/", "3_zerns_3/", "sub_zerns_1/", "sub_zerns_2/", "sub_zerns_3/", "sub_zerns_4/"]
folder_names = ["SLM_codes_matlab/20170504_" + folder_extensions[i] for i in range(len(folder_extensions))]
#folder_names = ["SLM_codes_matlab/20170504_6_4/"]#, "SLM_codes_matlab/20170424_make_flat_opt_1/"]

#save_string = ["coma", "defocus", "astigmatism", "spherical"]
#save_string = ["5_coma", "5_trifoil", "5_pentafoil", "mix_defocus", "mix_trifoil", "mix_spherical", "all_spherical", "all_coma"]#["x_coma", "y_coma", "mix", "x_coma_high_order", "asymetric_coma", "asymetric_astigmatism"]
#save_string = ["coma", "trefoil", "spherical", "quadrafoil", "pentafoil"]
#save_string = [save_string[i] + "_int_correction_6" for i in range(len(save_string))]
save_string = ["5_1", "5_5", "6_2", "6_4", "6_6", "3_zerns_1", "3_zerns_2", "3_zerns_3", "sub_zerns_1", "sub_zerns_2", "sub_zerns_3", "sub_zerns_4"]
#save_string = ["all_spherical"]

save_fold = "SLM_codes_matlab/reconstructions/phase_n_int/"
radius_fold = "SLM_codes_matlab/reconstructions/"
min_height = 70
look_ahead = 50
show_hough = False
show_id_hat = False
## make mask and find mean within mask

hist_yes = np.where(xx_alg**2 + yy_alg**2 <= 1)
hist_mask = np.zeros(xx_alg.shape)
hist_mask[hist_yes] = 1
cut_off = 0.06 #cutoff percentage

rms_lsq_vec = np.zeros(len(j_maxes))
rms_janss_vec = np.zeros(len(j_maxes))
rms_inter_vec = np.zeros(len(j_maxes))

phase_rms_vec_lsq = np.zeros(len(j_maxes))
phase_rms_vec_janss = np.zeros(len(j_maxes))

xi, yi = np.linspace(-1, 1, N_int), np.linspace(-1, 1, N_int)
xi, yi = np.meshgrid(xi, yi)
power_mat_30 = Zn.Zernike_power_mat(32)
power_mat_50 = Zn.Zernike_power_mat(52)
Z_mat_30 = Zn.Zernike_xy(xi, yi, power_mat_30, np.arange(2, 32))
Z_mat_50 = Zn.Zernike_xy(xi, yi, power_mat_50, np.arange(2, 52))

for ite in range(len(j_maxes)):
    j_max = int(j_maxes[ite])
    print("Current maximum Zernike order: " + str(j_max))
    j = np.arange(2, j_max) #start at 2, end at j_max-1
    power_mat = Zn.Zernike_power_mat(j_max+2)
    Zernike_3d = Zn.Zernike_xy(x_in, y_in, power_mat, j) ## create a 3d matrix with values of zernike polynomials at x_in and y_in 
    Zernike_2d = np.zeros((len(x_in), j_max)) ## flatten to perform least squares fit
    for i in range(len(j)):
        Zernike_2d[:, i] = Zernike_3d[...,i].flatten()

    ## pack everything neatly in 1 vector against clutter
    constants = np.stack((px_size_sh, px_size_int, f_sh, r_int_px, r_sh_px, r_sh_m, j_max, wavelength, box_len, x0, y0, radius))

    xy_inside = np.where(np.sqrt(xx_alg**2 + yy_alg**2) <= 1)
    x_in, y_in = xx_alg[xy_inside], yy_alg[xy_inside]

    j = np.arange(2, j_max+2) #start at 2, end at j_max-1
    power_mat = Zn.Zernike_power_mat(j_max+2)
    Zernike_3d = Zn.Zernike_xy(x_in, y_in, power_mat, j) ## create a 3d matrix with values of zernike polynomials at x_in and y_in 
    Zernike_2d = np.zeros((len(x_in), j_max)) ## flatten to perform least squares fit
    for i in range(len(j)):
        Zernike_2d[:, i] = Zernike_3d[...,i].flatten()

    Z_mat = Zn.Zernike_xy(xi, yi, power_mat, j)

##    with open("SLM_codes_matlab/20170331_x_coma_high_order/optimized_radius.txt") as data:
##        r_sh_px = json.load(data)

    for i in range(len(folder_names)):
        folder_name = folder_names[i]
        fold_name = folder_name
        mask = [np.sqrt((xx_alg) ** 2 + (yy_alg) ** 2) >= 1]
        
        if ite == 0:
            inter_0 = np.array(PIL.Image.open(folder_name + "interferogram_0.tif"))
            org_phase, delta_i, sh_spots, image_i0, flat_wf, flip_bool = PE.phase_extraction(constants, folder_name = folder_name, show_id_hat = show_id_hat, show_hough_peaks = show_hough, min_height = min_height, look_ahead = look_ahead, flip_ask = False)
            mask_tile = np.tile(mask, (org_phase.shape[-1],1,1)).T
            ##
            print(flip_bool)
            Un_sol = np.triu_indices(delta_i.shape[-1], k = 1) ## delta i has the shape of the amount of difference interferograms, while org phase has all possible combinations
            org_unwr = np.zeros(org_phase.shape)
            np.save(folder_name + "org_phase.npy", org_phase)

            print("filtering phase")
            k_phi = 17 ## filtering window size
            f0 = 15
            n = 2
            butter_phase = np.zeros(org_phase.shape)
            butter_unwr = np.zeros(butter_phase.shape)
            [ny, nx] = butter_phase.shape[:2]
            res = [2**kji for kji in range(15)]
            nx_pad = np.where( res > np.tile(nx, len(res)))
            nx_pad = res[nx_pad[0][0]]
            dif_x = (nx_pad - int(nx))/2
            for k in range(org_phase.shape[-1]):
                org_pad = np.lib.pad(org_phase[..., k], dif_x,'reflect')
                butter_pad = pw.butter_filter(org_pad, n, f0)
                butter_phase[..., k] = butter_pad[dif_x: nx_pad - dif_x, dif_x: nx_pad - dif_x]
                butter_unwr[..., k] = pw.unwrap_phase_dct(butter_phase[..., k], xx_alg, yy_alg, ii, jj, N_int, N_int)
                butter_unwr[..., k] -= delta_i[..., Un_sol[0][k]]    

            np.save(folder_name + "filtered_phase.npy", butter_unwr)
            np.save(folder_name + "delta_i.npy", delta_i)

            butter_mask = np.ma.array(butter_unwr, mask = mask_tile)
            mean_butt = butter_mask.mean(axis=(0,1))
            butter_unwr -= mean_butt
            
            ## smoothing interferogram due to median
            but_med = np.median(butter_unwr, axis = 2)
            but_med *= -1.0
            
            if flip_bool:
                but_med = np.fliplr(but_med)
                
            but_med_flat = but_med[xy_inside]
            np.save(folder_name + "but_med_flat.npy", but_med_flat)
        else:
            but_med_flat = np.load(folder_name + "but_med_flat.npy")
            inter_0 = np.array(PIL.Image.open(folder_name + "interferogram_0.tif"))

        a_butt = np.linalg.lstsq(Zernike_2d, but_med_flat)[0]
        a_inter= np.copy(a_butt)
##        a_avg = np.load(save_fold + "average_to_6.npy")
##        a_inter[:len(a_avg)] -= a_avg
        print(a_inter)

        [ny_i, nx_i] = inter_0.shape
        flipped_y0 = y0

        flipped_x0 = x0
        orig = np.ma.array(inter_0[flipped_y0-radius:flipped_y0+radius, flipped_x0-radius:flipped_x0+radius], mask = mask, dtype = 'float')#np.array(PIL.Image.open(folder_name + "interferogram_0.tif"))[#


        ### histogram normalization
        ###----
        # with
        ###---
        bins = np.linspace(0, orig.max(), 50)
        intensities, bin_edges = np.histogram(orig, bins, weights = hist_mask)
        tot = np.sum(orig)
        avgs = np.cumsum(np.diff(bin_edges)) - np.diff(bin_edges)[0]/2
        percentile = 1 - np.cumsum(intensities*avgs/tot)
        normalization_intensity = avgs[np.where(percentile < cut_off)[0][0]]
        orig /= normalization_intensity
        orig[orig>1.0] = 1.0
        ###---
        # without
        ###---
        orig /= orig.max()
        orig = np.flipud(np.fliplr(orig))
        ### ---
        # Actually! Orig == iterferogram!
        ### ---
        flipint = False
        piston, inter_rms = opt.fmin(rms_piston, 0, args = (j_max, a_inter, N_int, Z_mat, orig, mask, flipint), full_output = True)[:2]
        orig_plot = np.copy(orig)
        orig = np.ma.array(Zn.int_for_comp(j_max, a_inter, N_int, piston, Z_mat, False), mask = mask)
        #inter_rms = 0.0


        image_ref_mirror = np.array(PIL.Image.open(folder_name + "image_ref_mirror.tif"), dtype = 'float')
        zero_pos_dm = np.array(PIL.Image.open(folder_name + "zero_pos_dm.tif"), dtype = 'float')
        dist_image = np.array(PIL.Image.open(folder_name + "dist_image.tif"), dtype = 'float')
        flat_wf = np.array(PIL.Image.open(folder_name + "flat_wf.tif"), dtype = 'float')
        sh_spots = np.dstack((image_ref_mirror, zero_pos_dm, dist_image)) 

        zero_image = sh_spots[..., 1]
        zero_image_zeros = np.copy(zero_pos_dm)
        dist_image = sh_spots[..., 2]
        image_control = sh_spots[..., 0]
        [ny,nx] = zero_pos_dm.shape
        x = np.arange(0, nx, 1)
        y = np.arange(ny, 0, -1)
        xx, yy = np.meshgrid(x, y)

        print("gathering zeros")
        x_pos_zero, y_pos_zero = Hm.zero_positions(np.copy(zero_pos_dm), spotsize = box_len)
        print("amount of spots: " + str(len(x_pos_zero)))
        x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, np.copy(image_ref_mirror), xx, yy, spot_size = box_len)
        centre = Hm.centroid_centre(x_pos_flat, y_pos_flat)
        x_pos_norm = ((x_pos_flat - centre[0]))/float(r_sh_px)
        y_pos_norm = ((y_pos_flat - centre[1]))/float(r_sh_px)
        inside = np.where(np.sqrt(x_pos_norm**2 + y_pos_norm**2) <= (1 + (float(box_len)/r_sh_px))) #35 is the half the width of the pixel box aroudn a centroid and r_sh_px is the scaling factor
        x_pos_zero_f, y_pos_zero_f, x_pos_flat_f, y_pos_flat_f, x_pos_norm_f, y_pos_norm_f = mc.filter_positions(inside, x_pos_zero, y_pos_zero, x_pos_flat, y_pos_flat, x_pos_norm, y_pos_norm)
        x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_zero, y_pos_zero, np.copy(dist_image), xx, yy, spot_size = box_len)
        x_pos_dist_f, y_pos_dist_f = mc.filter_positions(inside, x_pos_dist, y_pos_dist)

    ##    print("calculating Janssen")
    ##    a_janss_opt = janssen.coeff_from_dist(x_pos_flat_f, y_pos_flat_f, x_pos_dist_f, y_pos_dist_f, x_pos_norm_f, y_pos_norm_f, px_size_sh, f_sh, r_sh_m, wavelength, j_max, r_sh_px, box_len)
    ##
    ##    print("calculating geometry matrix")
    ##    G = LSQ.matrix_avg_gradient(x_pos_norm_f, y_pos_norm_f, j_max, r_sh_px, power_mat, box_len)
        s = np.hstack(Hm.centroid2slope(x_pos_dist_f, y_pos_dist_f, x_pos_flat_f, y_pos_flat_f, px_size_sh, f_sh, r_sh_m, wavelength))
    ##
    ##    print("calculating LSQ")
    ##    a_lsq_opt = np.linalg.lstsq(G, s)[0]
    ##    
        lsq_args = (wavelength, j_max, f_sh, px_size_sh, y_pos_flat, x_pos_flat, x_pos_dist, y_pos_dist, xx, yy, orig, mask, N_int, Z_mat, power_mat, box_len, centre)
        janss_args = (x_pos_zero, y_pos_zero, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist, px_size_sh, f_sh, j_max, wavelength, xx, yy, N_int, orig, mask, Z_mat, box_len, centre)

        a_ref = np.load(folder_name + "reference_slm_vector.npy")
        try:
            if a_ref.shape[0] == 30:
                phi_ref = np.ma.array(np.dot(Z_mat_30, a_ref), mask = mask)
            elif a_ref.shape[0] == 50:
                phi_ref = np.ma.array(np.dot(Z_mat_50, a_ref), mask = mask)
            else:
                raise ValueError('Z_mat_ref and a_ref are not the same size')
        except ValueError as err:
            print(err.args)

        lsq_args_phi = wavelength, j_max, f_sh, px_size_sh, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist, xx, yy, mask, N_int, Z_mat, power_mat, box_len, phi_ref
        janss_args_phi = x_pos_zero, y_pos_zero, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist, px_size_sh, f_sh, j_max, wavelength, xx, yy, N_int, mask, Z_mat, box_len, phi_ref

        #### ran a constrained minimization scheme for finding a radius which minimizes rms error. All further calculations will be done with that. average radius between optimum for janssen and SH
        if ite == 0:
            print("optimizing SH")
            minradius = 0.85 * np.max(np.sqrt((x_pos_flat_f - centre[0])**2 + (y_pos_flat_f - centre[1])**2))
            maxradius = 1.15/0.85 * minradius
            print("mindradius = " + str(minradius))
            bounds_min = ((nx/2 - 50, nx/2 + 50), (ny/2 - 50, ny/2 + 50), (minradius, maxradius))
            initial_guess = [0, float(r_sh_px)]
            initial_guess_phi = [nx/2, ny/2, float(r_sh_px)] #changed from centre[0], centre[1] under the assumption that it is aligned in the middle using the cross and thorlabs software (which was done)

            bf_lsq = time.time()
            optresult_lsq = opt.minimize(rms_phi_lsq, initial_guess_phi, args = lsq_args_phi, method = 'L-BFGS-B', bounds = bounds_min, options = {'maxiter': 1000, 'disp':True})
            aft_lsq = time.time()
            print("All iterations LSQ took " + str(aft_lsq-bf_lsq) +" s")
            vars_lsq = optresult_lsq.x
            #vars_lsq = np.array([ 636.21492519,  523.25652217,  439.32821153])
            
            bf_janss = time.time()
            optresult_janss = opt.minimize(rms_phi_janssen, vars_lsq, args = janss_args_phi, method = 'L-BFGS-B', bounds = bounds_min, options = {'maxiter': 1000, 'disp':True})
            aft_janss = time.time()
            print("All iterations Janssen took " + str(aft_janss - bf_janss) + " s")
            
            vars_janss = optresult_janss.x
            print("radius lsq = " + str(vars_lsq[1]) + ", radius janss = " + str(vars_janss[1]))
            r_sh_px = (vars_lsq[2] + vars_janss[2])/2.0#vars_lsq[2] + vars_janss[2])/2.0
            r_sh_px_lsq = vars_lsq[2]
            r_sh_px_janss = vars_janss[2]

            np.save(folder_name + "optimized_lsq_vars.npy", vars_lsq)
            np.save(folder_name + "optimized_janss_vars.npy", vars_janss)
            
            json.dump(r_sh_px, open(folder_name + "optimized_radius_04_24.txt", 'w'))
            json.dump(r_sh_px_lsq, open(folder_name + "optimized_radius_lsq_04_24.txt", 'w'))
            json.dump(r_sh_px_janss, open(folder_name + "optimized_radius_janss_04_24.txt", 'w'))
        else:
            try:
                vars_lsq = np.load(folder_name + "optimized_lsq_vars.npy")
                vars_janss = np.load(folder_name + "optimized_janss_vars.npy")
    ##            with open(folder_name + "optimized_radius_04_24.txt") as data:
    ##                r_sh_px = json.load(data)
                with open(folder_name + "optimized_radius_lsq_04_24.txt") as data:
                    r_sh_px_lsq = json.load(data)
                with open(folder_name + "optimized_radius_janss_04_24.txt") as data:
                    r_sh_px_janss = json.load(data)
            except: #purely to test if the whole routine operates as it should
                vars_lsq = [centre[0], centre[1], r_sh_px]
                vars_janss = [centre[0], centre[1], r_sh_px]
                r_sh_px_lsq = r_sh_px
                r_sh_px_janss = r_sh_px
                np.save(folder_name + "vars_lsq_exception.npy", vars_lsq)
                
            
        ## uncomment for radius determined by Janssen method

        #r_sh_px = #np.load("20170214_post_processing/int_zn/determined_radius.npy").item()
        print("optimizing piston methods")
        ## find coefficients according to optimum
        r_sh_lsq = vars_lsq[2]
        x_pos_norm_lsq = (x_pos_flat - vars_lsq[0])/r_sh_lsq
        y_pos_norm_lsq = (y_pos_flat - vars_lsq[1])/r_sh_lsq
        inside_lsq = np.where(np.sqrt(x_pos_norm_lsq**2 + y_pos_norm_lsq**2) <= (1 + (box_len/r_sh_lsq)))
        x_pos_flat_lsq, y_pos_flat_lsq, x_pos_dist_lsq, y_pos_dist_lsq = mc.filter_positions(inside_lsq, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist)
        x_pos_norm_lsq, y_pos_norm_lsq = mc.filter_positions(inside_lsq, x_pos_norm_lsq, y_pos_norm_lsq)

        G = LSQ.matrix_avg_gradient(x_pos_norm_lsq, y_pos_norm_lsq, j_max, r_sh_lsq, power_mat, box_len)
        r_sh_px_lsq_opt = r_sh_lsq
        r_sh_m_lsq_opt = px_size_sh * r_sh_lsq
        #x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat_f, y_pos_flat_f, dist_image, xx, yy)
        #x_pos_dist, y_pos_dist = mc.filter_positions(inside_lsq, x_pos_dist, y_pos_dist, x_pos_flat_f, y_pos_flat_f)
        #s = np.hstack(Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat_f, y_pos_flat_f, px_size_sh, f_sh, r_sh_m_lsq_opt, wavelength))
        s = np.hstack(Hm.centroid2slope(x_pos_dist_lsq, y_pos_dist_lsq, x_pos_flat_lsq, y_pos_flat_lsq, px_size_sh, f_sh, r_sh_m, wavelength))

        a_lsq_opt = np.linalg.lstsq(G, s)[0]
        pist_args_lsq = (j_max, a_lsq_opt, N_int, Z_mat, orig, mask, False)
        pist_lsq, lsq_rms = opt.fmin(rms_piston, 0, pist_args_lsq, full_output = True)[:2]
        #vars_lsq = pist_lsq

        ### find coefficients janss according to optimum
        r_sh_janss = vars_janss[2]
        x_pos_norm_janss = (x_pos_flat - vars_janss[0])/r_sh_janss
        y_pos_norm_janss = (y_pos_flat - vars_janss[1])/r_sh_janss
        inside_janss = np.where(np.sqrt(x_pos_norm_janss**2 + y_pos_norm_janss**2) <= (1 + (box_len/r_sh_janss)))
        x_pos_norm_janss, y_pos_norm_janss, x_pos_flat_janss, y_pos_flat_janss, x_pos_dist_janss, y_pos_dist_janss = mc.filter_positions(inside_janss, x_pos_norm_janss, y_pos_norm_janss, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist)
        r_sh_m_janss = px_size_sh * r_sh_janss
        a_janss_opt = janssen.coeff_from_dist(x_pos_flat_janss, y_pos_flat_janss, x_pos_dist_janss, y_pos_dist_janss, x_pos_norm_janss, y_pos_norm_janss, px_size_sh, f_sh, r_sh_janss*px_size_sh, wavelength, j_max, r_sh_janss, box_len)

        pist_args_janss = (j_max, a_janss_opt, N_int, Z_mat, orig, mask, False)
        pist_janss, janss_rms = opt.fmin(rms_piston, 0, pist_args_janss, full_output = True)[:2]
        #vars_janss = pist_janss
        
        #pist_lsq = vars_lsq[0]#opt.fmin(rms_piston, 0, args = (j_max, a_lsq, N, Z_mat, orig, mask, fliplsq))
        #pist_janss = vars_janss[0]#opt.fmin(rms_piston, 0, args = (j_max, np.real(a_janss), N, Z_mat, orig, mask, flipjanss))

        #lsq_rms = optresult_lsq.fun
        #janss_rms = optresult_janss.fun

        try:
            a_ref_args = (j_max, a_ref, N_int, Z_mat_30, orig, mask, False)
            a_ref_pist, ref_rms = opt.fmin(rms_piston, 0, a_ref_args, full_output = True)[:2]
        except:
            a_ref_args = (j_max, a_ref, N_int, Z_mat_50, orig, mask, False)
            a_ref_pist, ref_rms = opt.fmin(rms_piston, 0, a_ref_args, full_output = True)[:2]
        phi_lsq = np.ma.array(np.dot(Z_mat, a_lsq_opt), mask = mask)
        phi_int = np.ma.array(np.dot(Z_mat, a_inter), mask = mask)
        phi_janss = np.ma.array(np.dot(Z_mat, a_janss_opt), mask = mask)
        phi_diff_int, phi_diff_lsq, phi_diff_janss = np.ma.array(phi_ref - phi_int, mask = mask), np.ma.array(phi_ref - phi_lsq, mask = mask), np.ma.array(phi_ref - phi_janss, mask = mask)
        eps_vec = np.stack((np.sqrt((phi_diff_int**2).sum())/N_int, np.sqrt((phi_diff_lsq**2).sum())/N_int, np.sqrt((phi_diff_janss**2).sum())/N_int))

        phase_rms_vec_lsq[ite] = eps_vec[1]
        phase_rms_vec_janss[ite] = eps_vec[2]

        titles = [r'Intended', r'Interferogram', r'LSQ', r'Janssen']
        f_phi, ax_phi = plt.subplots(2,5, figsize = (4.98, 4.98/2.2), gridspec_kw = {'width_ratios':[4, 4, 4, 4, 0.4]})

        vmin_phi, vmax_phi = np.min([phi_ref.min(), phi_lsq.min(), phi_int.min(), phi_janss.min()]), np.max((phi_ref.max(), phi_lsq.max(), phi_int.max(), phi_janss.max()))
        vmin_phi = np.pi * (np.sign(vmin_phi) * (np.abs(vmin_phi)//(np.pi)) + np.sign(vmin_phi))
        vmax_phi = np.pi * (np.sign(vmax_phi) * (np.abs(vmax_phi)//(np.pi)) + np.sign(vmax_phi))
        vmin_diff, vmax_diff = np.min([phi_diff_int.min(), phi_diff_lsq.min(), phi_diff_janss.min()]), np.max([phi_diff_int.max(), phi_diff_lsq.max(), phi_diff_janss.max()])
        vmin_diff = np.pi * (np.sign(vmin_diff) * (np.abs(vmin_diff)//(np.pi)) + np.sign(vmin_diff))
        vmax_diff = np.pi * (np.sign(vmax_diff) * (np.abs(vmax_diff)//(np.pi)) + np.sign(vmax_diff))
        

        phi_0 = ax_phi[0,0].imshow(np.ma.array(phi_ref, mask = mask), vmin = vmin_phi, vmax = vmax_phi, cmap = 'jet', origin = 'lower', interpolation = 'none')
        ax_phi[0, 1].imshow(np.ma.array(phi_int, mask = mask), vmin = vmin_phi, vmax = vmax_phi, cmap = 'jet', origin = 'lower', interpolation = 'none')
        ax_phi[0, 2].imshow(np.ma.array(phi_lsq, mask = mask), vmin = vmin_phi, vmax = vmax_phi, cmap = 'jet', origin = 'lower', interpolation = 'none')
        ax_phi[0, 3].imshow(np.ma.array(phi_janss, mask = mask), vmin = vmin_phi, vmax = vmax_phi, cmap = 'jet', origin = 'lower', interpolation = 'none')

        phi_diff = ax_phi[1,1].imshow(np.ma.array(phi_diff_int, mask = mask), vmin = vmin_diff, vmax = vmax_diff, cmap = 'jet', origin = 'lower', interpolation = 'none')
        ax_phi[1,2].imshow(np.ma.array(phi_diff_lsq, mask = mask), vmin = vmin_diff, vmax = vmax_diff, cmap = 'jet', origin = 'lower', interpolation = 'none')
        ax_phi[1,3].imshow(np.ma.array(phi_diff_janss, mask = mask), vmin = vmin_diff, vmax = vmax_diff, cmap = 'jet', origin = 'lower', interpolation = 'none')
        

        for ii_it in range(4):
            ax_phi[0,ii_it].set(adjustable = 'box-forced', aspect = 'equal')
            ax_phi[0,ii_it].get_xaxis().set_ticks([])
            ax_phi[0,ii_it].get_yaxis().set_ticks([])
            ax_phi[1,ii_it].set(adjustable = 'box-forced', aspect = 'equal')
            ax_phi[1,ii_it].get_xaxis().set_ticks([])
            ax_phi[1,ii_it].get_yaxis().set_ticks([])

            ax_phi[0,ii_it].set_title(titles[ii_it], fontsize = 9)
            if ii_it == 0:
                ax_phi[0, ii_it].text(N_int/2, -N_int/6, r"$\varepsilon~=~$", fontsize = 7, ha = 'center')
            else:
                ax_phi[0, ii_it].text(N_int/2, -N_int/6, "%.4f"%eps_vec[ii_it-1], fontsize = 7, ha = 'center')

        
        
        Ticks_phase = np.pi * np.linspace(np.floor_divide(vmin_phi, np.pi), np.floor_divide(vmax_phi, np.pi), 5)
        Ticks_diff = np.pi * np.linspace(np.floor_divide(vmin_diff, np.pi), np.floor_divide(vmax_diff, np.pi), 5)
        cbar = plt.colorbar(phi_0, cax = ax_phi[0, -1], ticks = Ticks_phase)
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.set_ylabel("Phase", fontsize = 8)
        cbar.ax.set_yticklabels(["%.2f"%(Ticks_phase[tick_it]/(np.pi)) + r'$\pi$' for tick_it in range(len(Ticks_phase))])

        cbar2 = plt.colorbar(phi_diff, cax = ax_phi[1, -1], ticks = Ticks_diff)
        cbar2.ax.tick_params(labelsize = 7)
        cbar2.ax.set_ylabel("Phase difference", fontsize = 8)
        cbar2.ax.set_yticklabels(["%.2f"%(Ticks_diff[tick_it]/(np.pi)) + r'$\pi$' for tick_it in range(len(Ticks_diff))])

        f_phi.savefig(fold_name + "phase_" + save_string[i] + "_j_" + str(j_max) + ".png", bbox_inches = 'tight', dpi = dpi_num)
        f_phi.savefig(save_fold + "phase_" + save_string[i] + "_j_" + str(j_max) + ".png", bbox_inches = 'tight', dpi = dpi_num)
        plt.close(f_phi)
        
        ### plot results
        titles_int = [r'Measured', r'Intended', r'Interferogram', r'LSQ', r'Janssen', r'']
        rms_lsq_vec[ite] = lsq_rms
        rms_janss_vec[ite] = janss_rms
        rms_inter_vec[ite] = inter_rms
        rms_mat = np.array([ref_rms, inter_rms, lsq_rms, janss_rms])
        f, axes = plt.subplots(nrows = 1, ncols = 6, figsize = (4.98, 4.98/4.4), gridspec_kw={'width_ratios':[4, 4, 4, 4, 4, 0.4]})#, 'height_ratios':[4,4]})
        interf = axes[0].imshow(np.ma.array(orig_plot, mask = mask), vmin = 0, vmax = 1, cmap = 'bone', origin = 'lower', interpolation = 'none')
        try:
            Zn.imshow_interferogram(j_max, a_ref, N_int, piston = a_ref_pist, ax = axes[1], Z_mat = Z_mat_30, power_mat = power_mat_30)
        except:
            Zn.imshow_interferogram(j_max, a_ref, N_int, piston = a_ref_pist, ax = axes[1], Z_mat = Z_mat_50, power_mat = power_mat_50)            
        Zn.imshow_interferogram(j_max, a_inter, N_int, piston = piston, ax = axes[2], Z_mat = Z_mat, power_mat = power_mat)
        Zn.imshow_interferogram(j_max, a_lsq_opt, N_int, piston = pist_lsq, ax = axes[3], fliplr = False, Z_mat = Z_mat, power_mat = power_mat)
        Zn.imshow_interferogram(j_max, a_janss_opt, N_int, piston = pist_janss, ax = axes[4], fliplr = False, Z_mat = Z_mat, power_mat = power_mat)
        for ii_it in range(5):
            axes[ii_it].set(adjustable = 'box-forced', aspect = 'equal')
            axes[ii_it].get_xaxis().set_ticks([])
            axes[ii_it].get_yaxis().set_ticks([])
            axes[ii_it].set_title(titles_int[ii_it], fontsize = 9)
            if ii_it == 0:
                axes[ii_it].text(N_int/2, -N_int/6, r"$\varepsilon~=~$", fontsize = 7, ha = 'center')
            else:
                axes[ii_it].text(N_int/2, -N_int/6, "%.4f"%rms_mat[ii_it-1], fontsize = 7, ha = 'center')
        cbar = plt.colorbar(interf, cax = axes[-1], ticks = [0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.set_ylabel("Normalized Intensity", fontsize = 8)
        f.savefig(fold_name + "interferograms_"+save_string[i]+"_j_" + str(j_max) + ".png", bbox_inches = 'tight', dpi = dpi_num)
        f.savefig(save_fold + "interferograms_"+save_string[i]+"_j_" + str(j_max) + ".png", bbox_inches = 'tight', dpi = dpi_num)
        plt.close(f)

        if a_ref.shape[0] == 30:
            f2, ax2 = plt.subplots(nrows = 1, ncols = 1, figsize = (4.98, 4.98/4.4))
            ax2.plot(32*[0], 'k-', linewidth = 0.5)

            x_max_plot = len(a_ref)
            a_x = np.arange(2, j_max+2)
            a_x_ref = np.arange(2, x_max_plot+2)
            ax2.plot(a_x, a_inter, 'sk', label = 'Interferogram')
            ax2.plot(a_x, a_janss_opt, 'og', label = 'Janssen')
            ax2.plot(a_x, a_lsq_opt, '2b', label = 'LSQ')
            ax2.plot(a_x_ref, a_ref, 'rx', label = 'Intended', markersize = 1)
            ax2.set_xlim([0,31.5])
            min_a = np.min([np.min(a_inter), np.min(a_janss_opt), np.min(a_lsq_opt)])
            max_a = np.max([np.max(a_inter), np.max(a_janss_opt), np.max(a_lsq_opt)])
            b_gold = max_a - min_a / (2*gold)
            ax2.set_ylim([min_a - b_gold, max_a + b_gold])
            ax2.set_ylabel(r'$a_j$')
            ax2.set_xlabel(r'$j$')
            ax2.legend(prop={'size':7}, loc = 'upper right')
            box = ax2.get_position()
            ax2.set_position([box.x0, box.y0, box.width, box.height * 0.8])
            ax2.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.15), ncol = 4, fontsize = 6)
        else:
            f2, ax2 = plt.subplots(nrows = 2, ncols = 1, sharey = True, figsize = (4.98, 4.98/2.2))
            N_half = 25
            ax2[0].plot((N_half+2)*[0], 'k-', linewidth = 0.5)

            x_max_plot = len(a_ref)
            
            a_x = np.arange(2, j_max+2)
            a_x_ref = np.arange(2, x_max_plot+2)
            ax2[0].plot(a_x[:N_half], a_inter[:N_half], 'sk', label = 'Interferogram')
            ax2[0].plot(a_x[:N_half], a_janss_opt[:N_half], 'og', label = 'Janssen')
            ax2[0].plot(a_x[:N_half], a_lsq_opt[:N_half], '2b', label = 'LSQ')
            ax2[0].plot(a_x_ref[:N_half], a_ref[:N_half], 'rx', label = 'Intended', markersize = 1)
            ax2[0].set_xlim([0, N_half+1.5])

            ax2[1].plot((2*N_half + 4)*[0], 'k-', linewidth = 0.5)
            ax2[1].plot(a_x[N_half:], a_inter[N_half:], 'sk', label = 'Interferogram')
            ax2[1].plot(a_x[N_half:], a_janss_opt[N_half:], 'og', label = 'Janssen')
            ax2[1].plot(a_x[N_half:], a_lsq_opt[N_half:], '2b', label = 'LSQ')
            ax2[1].plot(a_x_ref[N_half:], a_ref[N_half:], 'rx', label = 'Intended', markersize = 1)
            ax2[1].set_xlim([N_half, 2*N_half + 1.5])


            min_a = np.min([np.min(a_inter), np.min(a_janss_opt), np.min(a_lsq_opt)])
            max_a = np.max([np.max(a_inter), np.max(a_janss_opt), np.max(a_lsq_opt)])
            b_gold = max_a - min_a / (2*gold)
            ax2[0].set_ylim([min_a - b_gold, max_a + b_gold])
            ax2[1].set_ylim([min_a - b_gold, max_a + b_gold])
            ax2[1].set_ylabel(r'$a_j$')
            ax2[1].set_xlabel(r'$j$')

            ax2[0].legend(prop={'size':7}, loc = 'upper right')
            box = ax2[0].get_position()
            ax2[0].set_position([box.x0, box.y0, box.width, box.height * 0.8])
            ax2[0].legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.15), ncol = 4, fontsize = 6)

            box = ax2[1].get_position()
            ax2[1].set_position([box.x0, box.y0, box.width, box.height * 0.8])

        
        f2.savefig(fold_name+'zn_coeff_' + save_string[i] + '_j_' + str(j_max) + '.png', bbox_inches = 'tight', dpi = dpi_num)
        f2.savefig(save_fold + "zn_coeff_"+save_string[i]+"_j_" + str(j_max) + ".png", bbox_inches = 'tight', dpi = dpi_num)
        plt.close(f2)

        rms_dict = {'rms_inter':inter_rms,'rms_lsq':lsq_rms, 'rms_janss':janss_rms}
        vars_dict = {'vars_lsq':vars_lsq, 'vars_janss':vars_janss, 'pist_inter':piston}
        coeff_dict = {'coeff_inter':a_inter, 'coeff_lsq':a_lsq_opt, 'coeff_janss':a_janss_opt}
        np.save(folder_name + 'vars_dictionary_j_' + save_string[i]+ "_" + str(j_max) + '.npy', vars_dict)
        np.save(folder_name + 'coeff_dictionary_j_' + save_string[i] + "_" + str(j_max)+ '.npy', coeff_dict)
        np.save(folder_name + 'uncorrected_zn_coeff_inter_' + save_string[i] + "_" + str(j_max) + '.npy', a_butt)
        json.dump(rms_dict, open(folder_name + "rms_dict_j_" + save_string[i] + "_" + str(j_max) + ".txt", 'w'))
        print(" rms interferogram: " + str(inter_rms) + ",\n rms LSQ: " + str(lsq_rms) + ",\n rms Janssen: " + str(janss_rms))

        f3, ax3 = plt.subplots(1,1)
        x_plot = np.linspace(-1, 1, N_int)
        ax3.plot(x_plot, orig_plot[N_int/2, :], 'g', label = 'original')
        ax3.plot(x_plot, Zn.int_for_comp(j_max, a_inter, N_int, piston, Z_mat)[N_int/2, :], 'r', label = 'Interferogram')
        #ax3.plot(x_plot, Zn.int_for_comp(j_max, a_ref, N_int, piston, Z_mat)[N_int/2, :], 'b', label = 'Intended')
        ax3.plot(x_plot, Zn.int_for_comp(j_max, a_lsq_opt, N_int, pist_lsq, Z_mat)[N_int/2, :], 'r--', label = 'LSQ')
        ax3.plot(x_plot, Zn.int_for_comp(j_max, a_janss_opt, N_int, pist_janss, Z_mat)[N_int/2, :], 'r-.', label = 'Janssen')
        ax3.set_xlabel('x')
        ax3.set_ylabel('Normalized Intensity')
        ax3.legend(prop={'size':7}, loc = 'upper right')
        box = ax3.get_position()
        ax3.set_position([box.x0, box.y0, box.width, box.height * 0.8])
        ax3.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.15), ncol = 5, fontsize = 6)
        f3.savefig(save_fold + "x_cutthrough_intensity" + save_string[i]+"_j_" + str(j_max) +".png", bbox_inches = 'tight', dpi = dpi_num)
        plt.close(f3)

        np.save(folder_name + save_string[i] + "_lsq_rms_var_j.npy", rms_lsq_vec)
        np.save(folder_name + save_string[i] + "_janss_rms_var_j.npy", rms_janss_vec)
        np.save(folder_name + save_string[i] + "_inter_rms_var_j.npy", rms_inter_vec)
        np.save(folder_name + save_string[i] + "_phase_rms_lsq.npy", phase_rms_vec_lsq)
        np.save(folder_name + save_string[i] + "_phase_rms_janss.npy", phase_rms_vec_janss)
        np.save(folder_name + save_string[i] + "_j_vector.npy", j_maxes)
plt.show()
