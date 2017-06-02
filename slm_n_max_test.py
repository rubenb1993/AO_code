"""
Calculates the Zernike coefficients in three ways, using the SH sensor (least-squares and Janssen's)
and using an interferogram method. Based on these coefficients, different things are calculated,
such as the measured phase, the discrepancy between the measured and intended phase, interferograms
are simulated (but not saved atm), and plots are made of the zernike coefficients.

Important: This script will save big arrays to your memory, up to 1GB per array (for 12 zernike orders).
This is done to speed up the code. Make sure you have enough space on your hard disk.

Inputs:
    folder names: list containing the string with paths to all the folders containing the measurements
    a_ref_ordering: string containing the ordering of the reference vector. This can be Fringe or Brug
    save_string: list of strings containing how additional information will be saved. Used to differentiate different measurements in the same save folder
    x0, y0, radius: centre positions and radius of the interferogram. This changes per day, but not per measurement
    save_fold: string with the folder where additional plots are saved (in order to copy and paste them easily to a folder where LaTeX can use them, for instance).
    make_hough_transforms: Boolean value if hough transforms should be made. If False, it takes saved data from the folder. 
    optimize_center_radius: Boolean value if center and radius should be optimized. This can take several hours depending on the maximum zernike order
        the first n in n_max is taken as the value for which the center and radius will be optimized. This depends on the amount of spots on your SH sensor
    old_vars_taken: Should be FALSE. Was from before optimzed_(n_max) was added to the saving string for optmized center and radius
    show_hough: Boolean to show individual hough transforms to see if the parameters are chosen well (for visual inspection)
    show_id_hat: Boolean to show the recovered wrapped phase Id_hat, again for visual inspection
    only_use_lsq_vars: Boolean to use the center and radius for Janssen's calculations as well. For now purely to test
    min_height: minimum height of the peaks in the Hough transform. THIS IS VERY DEPENDENT ON THE MEASUREMENT
    look_ahead: minimum pixels between peaks in the Hough transform. Again very dependent on the measurement and the amount of tip/tilt added.
    f0: Filter cut-off frequency to filter the phase recovered by Soloviev&Vdovin's method. Not in per meter!!!
    n: Filter order. 
    all parameters under "given parameters". These are estimations of radii and known constants from the SH sensor.


    In the specified folder there should be:
        image_ref_mirror: a tif image of the shack-hartmann pattern of the reference mirror
        zero_pos_dm: a tif image of the shack-hartmann pattern of the unabberated position of the deformable optical element
        dist_image: a tif image of the shack-hartmann pattern of the abberated position of the deformable optical element
        flat_wf: a tif image of the interferogram of the unaberrated wavefront
        interferogram_i: tif images of the interferogram of the aberrated wavefront. 0 should be without arbitrary tip/tilt. 1 through n should be with arbitrary tip and tilt
        reference_slm_vector: a npy array of a 1d vector containing the intended zernike coefficients of the aberrated wavefront. Tipically n_max = 10 for the SLM setup.


    Output:
        power_mat_xx: a matrix containing the powers of x and y as described in H van Brug's paper for zernike polynomials in cartesian coordinates.
        Z_mat_30/50: a 3D array containing the values of zernike polynomials at positions xi, yi. First 2 dimensions are for x and y, third is the polynomial order j (either fringe or brug)
            xi and yi are sampled in the same way the pixels on the interferogram camera are. So if the radius of the interferogram is 300 pixels, the size would be [300, 300, j] (j the amount of zernikes).
            30 and 50 are for when fringe ordering was used, and they contain up to the highest used order in either the first 30 or 50 zernikes.
        org_phase.npy: a 3D array containing the zonal reconstruction of the phase from the interferogram method. first two dimensions are x and y, third is different measurements. third dimension
            is as long as there are unique combinations of pairs of tip/tilt interferograms
        filtered_phase.npy: The phase from org_pahse, but filtered using a butterworth filter 
        delta_i: Half of the measured tip/tilt from Soloviev&Vdovin's method.
        but_med_flat: A 1d array containing the filtered and median of the filtered phase, where the values inside the radius are flattened to a 1d array.
        Zernike_2d: a 2d array made from Z_mat. The xy coordinates are flattened to the first dimension, so that the second dimension is the amount of zernike coefficients. Flattened to be able to
            solve: zernike_2d * coefficients = but_med_flat
        optimized_*_vars_lsq/janss: npy arrays containing 1d length 3 vectors containing the x, y and r positions optimized with respect to the intended phase. In pixels.

        lsq_rms_var_j.npy: RMS value with respect to INTERFEROGRAM. as long as n_max
        same for janssen and inter
        phase_rms_lsq.npy: RMS value with respect to inteded PHASE. as long as n_max
        same for janss and inter
        n_vector: the n_maxes that are calculated for. Useful when plotting rms and knowing in which order you did the calculations.

        rms_dict: a dictionary item saved as npy array (use np.load(dict).item() ). Is RMS value with RESPECT TO INTERFEROGRAM. DEPRECATED.
        vars_dict: contain x0, y0 and radius for the shack-hartmann sensor, or piston  for the interferogram. Not necessarily used, doesn't change per iteration. Legacy (I think)
        coeff_dict: dictionary item saved as npy array (use np.load(dict).item() ). Contains the vectors of zernike coefficients that are calculated for a certain n_max. use:
            coeff_inter' for interferogram coefficeints, 'coeff_lsq' for LSQ, 'coeff_janss' for Janssen.
        text-files: dumped for your viewing pleasure, are not used, mainly to check. Even contains old RMS values (with respect to interferogram).

        Images:
            Phi: showsintended vs measured phase for all n_max calculated.
            zn: shows all zernike coefficients calculated 
"""

import sys
##import os
import time
##if "C:\Program Files\Micro-Manager-1.4" not in sys.path:
##    sys.path.append("C:\Program Files\Micro-Manager-1.4")
##import MMCorePy
import PIL.Image
import os
import numpy as np
from matplotlib import rc
import Hartmann as Hm
##import displacement_matrix as Dm
import Zernike as Zn
##import mirror_control as mc
import filter_inside as fi
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
       wavelength, j, f_sh, px_size_sh, dist_image, image_control, y_pos_flat_f, x_pos_flat_f, xx, yy, orig, mask, N, Z_mat, power_mat, box_len = args
    else:
        print("put the arguments!")
        return
    r_sh_m = px_size_sh * variables[3]

    x_pos_norm = (x_pos_flat_f - variables[1])/variables[3]
    y_pos_norm = (y_pos_flat_f - variables[2])/variables[3]

    inside = np.sqrt(x_pos_norm ** 2 + y_pos_norm**2) <= (1+ (box_len/variables[3]))
    x_pos_norm, y_pos_norm, x_pos_flat_f, y_pos_flat_f = fi.filter_positions(inside, x_pos_norm, y_pos_norm, x_pos_flat_f, y_pos_flat_f)

    G = LSQ.matrix_avg_gradient(x_pos_norm, y_pos_norm, j, variables[3], power_mat)
    
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
        wavelength, j, f_sh, px_size_sh, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist, xx, yy, mask, N, Z_mat, power_mat, box_len, phi_ref = args
    else:
        print("argument error!")
        return
    r_sh_m = px_size_sh * variables[2]

    x_pos_norm = (x_pos_flat - variables[0])/variables[2]
    y_pos_norm = (y_pos_flat - variables[1])/variables[2]

    inside = np.sqrt(x_pos_norm ** 2 + y_pos_norm**2) <= (1+ (box_len/variables[2]))
    x_pos_norm_it, y_pos_norm_it, x_pos_flat_it, y_pos_flat_it, x_pos_dist_it, y_pos_dist_it = fi.filter_positions(inside, x_pos_norm, y_pos_norm, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist)

    G = LSQ.matrix_avg_gradient(x_pos_norm_it, y_pos_norm_it, j, variables[1], power_mat, box_len)
    
    s = np.hstack(Hm.centroid2slope(x_pos_dist_it, y_pos_dist_it, x_pos_flat_it, y_pos_flat_it, px_size_sh, f_sh, r_sh_m, wavelength))
    a = np.linalg.lstsq(G,s)[0]

    phi_lsq = np.ma.array(np.dot(Z_mat, a), mask = mask)
    phi_diff = np.ma.array(phi_ref - phi_lsq, mask = mask)
    rms = np.sqrt((phi_diff**2).sum())/N
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
    x_pos_norm_it, y_pos_norm_it, x_pos_flat_it, y_pos_flat_it, x_pos_dist_it, y_pos_dist_it = fi.filter_positions(inside, x_pos_norm, y_pos_norm, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist)

    a_janss = janssen.coeff_from_dist(x_pos_flat_it, y_pos_flat_it, x_pos_dist_it, y_pos_dist_it, x_pos_norm_it, y_pos_norm_it, px_size_sh, f_sh, r_sh_m, wavelength, n_max, r_sh_px, box_len, order)    

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
       x_pos_zero, y_pos_zero, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist, px_size_sh, f_sh, n_max, wavelength, xx, yy, N, mask, Z_mat, box_len, phi_ref, order = args
    else:
        print("put the arguments!")
        return
    r_sh_m = px_size_sh * variables[2]

    x_pos_norm = (x_pos_flat - variables[0])/variables[2]
    y_pos_norm = (y_pos_flat - variables[1])/variables[2]

    ### check if all is within circle
    inside = np.sqrt(x_pos_norm ** 2 + y_pos_norm**2) <= (1+ (box_len/variables[2]))
    x_pos_norm_it, y_pos_norm_it, x_pos_flat_it, y_pos_flat_it, x_pos_dist_it, y_pos_dist_it = fi.filter_positions(inside, x_pos_norm, y_pos_norm, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist)

    a_janss = janssen.coeff_from_dist(x_pos_flat_it, y_pos_flat_it, x_pos_dist_it, y_pos_dist_it, x_pos_norm_it, y_pos_norm_it, px_size_sh, f_sh, r_sh_m, wavelength, n_max, r_sh_px, box_len, order)

    phi_janss = np.ma.array(np.dot(Z_mat, a_janss), mask = mask)
    phi_diff = np.ma.array(phi_ref - phi_janss, mask = mask)
    rms = np.sqrt((phi_diff**2).sum())/N
    return rms

## Given paramters for centroid gathering
px_size_sh = 4.65e-6        #width of pixels on sh sensor in [m]
px_size_int = 3.45e-6       #width of the pixels of the interferogram camera [m]
f_sh = 14.2e-3              #effective focal length lenslets sh sensor
r_int_px = 600/2            #radius in px of the spot on the interferogram camera
N_int = 2*r_int_px          #The square around the spot on the interferogram camera is N_int x N_int big
r_sh_m = 2.048e-3           #initial guess of the radius in [m] of the spot on the shack-hartmann camera
r_sh_px = int(r_sh_m/px_size_sh) #initial guess of the radius in [px] of the spot on the shack-hartmann sensor
n_optimization = 10
n_maxes = np.insert(np.arange(2, 12), 0, n_optimization) #vector of the zernike orders that need to be fitted
wavelength = 632e-9         #wavelength of laser in [m]
box_len = 30                #half of the distance between two nearest neighbour spots on the sh camera 
gold = (1 + np.sqrt(5))/2   #golden ratio for plotting
f0 = 15     #Filter frequency, not in per meter
n = 2       #Filter order
#pitch_sh = 150.0e-6         #
order = 'Brug'              #single index ordering of desired plots, decided to be 'Brug' on 20170509 
    
#### 20170504 zernike measurements
##folder_extensions = ["sub_zerns_1/",  "sub_zerns_2/", "sub_zerns_3/", "sub_zerns_4/", "5_1/", "5_5/", "6_2/", "6_4/", "6_6/", "3_zerns_1/", "3_zerns_2/", "3_zerns_3/",]
##folder_names = ["SLM_codes_matlab/20170504_" + folder_extensions[i] for i in range(len(folder_extensions))]
##a_ref_ordering = 'fringe' #for these measurements, fringe ordering was still used
##save_string = ["sub_zerns_1", "sub_zerns_2", "sub_zerns_3", "sub_zerns_4", "5_1", "5_5", "6_2", "6_4", "6_6", "3_zerns_1", "3_zerns_2", "3_zerns_3", ]
##old_vars_string = [ "sub_zerns_2"]#"sub_zerns_1",
##save_string = [save_string[i] + "_optimized_" + str(n_optimization) for i in range(len(save_string))] #string to save different plots with
### x0, y0 of 20170504 measurements: 332, 236, 600
##x0 = 332 + r_int_px         #middle x-position in [px] of the spot on the interferogram camera. Measured by making a box around the spot using ImageJ
##y0 = 236 + r_int_px         #middle y-position in [px] of the spot on the interferogram camera. Measured by making a box around the spot using ImageJ
##radius = int(r_int_px)      #radius in px of the spot on the interferogram camera
##save_fold = "SLM_codes_matlab/reconstructions/20170504_measurement/"

###### 20170512 random surface measurements
##folder_names = ["SLM_codes_matlab/20170512_rand_surf_" + str(i)+"/" for i in range(4)]
##save_string = ["low", "medium", "high", "extreme"]
##save_string = [save_string[i] + "_optimized_" + str(n_optimization) for i in range(len(save_string))] #string to save different plots with
##a_ref_ordering = 'brug'
##### x0, y0 of 20170512 measurements: 344, 256, 600
##x0 = 344 + r_int_px
##y0 = 256 + r_int_px
##radius = int(r_int_px)
##save_fold = "SLM_codes_matlab/reconstructions/20170512_measurement/"

###### 20170515 zernike measurements
folder_extensions = ["365_22/", "544_04/", "544_08/","544_17/","544_33/", "365_03/", "365_06/", "365_11/"]
folder_names = ["SLM_codes_matlab/20170515_leica_" + folder_extensions[i] for i in range(len(folder_extensions))]
a_ref_ordering = 'brug' #for these measurements, fringe ordering was still used
save_string = ["365_22", "544_04", "544_08", "544_17", "544_33", "365_03", "365_06","365_11"]
##load_string = ["365_22", "544_04", "544_08", "544_17", "544_33", "365_03", "365_06", "365_11"]
save_string = [save_string[i] + "_optimized_" + str(n_optimization) for i in range(len(save_string))] #string to save different plots with
# x0, y0 of 20170515 measurements: 342, 254, 600
x0 = 342 + r_int_px         #middle x-position in [px] of the spot on the interferogram camera. Measured by making a box around the spot using ImageJ
y0 = 254 + r_int_px         #middle y-position in [px] of the spot on the interferogram camera. Measured by making a box around the spot using ImageJ
radius = int(r_int_px)      #radius in px of the spot on the interferogram camera
save_fold = "SLM_codes_matlab/reconstructions/20170515_measurement/"

#radius_fold = "SLM_codes_matlab/reconstructions/"

### minimum height and distance between peaks found using hough transform
min_height = 110
look_ahead = 30
make_hough_transforms = False
optimize_center_radius = True
old_vars_taken = False
show_hough = False
show_id_hat = True

only_use_lsq_vars = False
if only_use_lsq_vars:
    save_string = [save_string[i] + "_only_use_lsq" for i in range(len(save_string))]


# Define font for figures, use LaTeX to compile
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size' : 7})
rc('text', usetex=False)

## plot making parameters
dpi_num = 300                           #dots per inch of png image
int_im_size = (4.98, 3.07)              #size of images in inches
int_im_size_23 = (0.66 * 4.98, 3.07)    #size of images in inches
int_im_size_13 = (0.33 * 4.98, 3.07)    #size of images in inches

## centre and radius of interferogam. Done by eye, with the help of image_j
### x0, y0 of 20170404 measurements: 464, 216, r_int_px =600
### x0, y0 of 20170405 measurements: 434, 254, r_int_px = 600
### x0, y0 of 20170412 measurements: 446, 224
### test x0, y0 12-4-17 468,240, 570 r_int_px
### x0, y0 of 20170413 measurements: 460, 222 r_int_px = 600
### x0, y0 of 20170420 measurements: 434,250,586, can be improved by 433, 246, 596
### x0, y0 of 20170424 measurements: 424, 242, 600

### Create grid to evaluate zernikes on, the same size as the interferogram spot
xi, yi = np.linspace(-1, 1, N_int), np.linspace(-1, 1, N_int)
xi, yi = np.meshgrid(xi, yi)
i, j = np.linspace(0, N_int-1, N_int), np.linspace(0, N_int-1, N_int)
##xx_alg, yy_alg = np.meshgrid(x_pcg, y_pcg)
mask = [np.sqrt((xi) ** 2 + (yi) ** 2) >= 1]                    #mask to make it circular
ii, jj = np.meshgrid(i, j)                                              #used for phase unwrapping the interferogram signal
xy_inside = np.where(np.sqrt(xi**2 + yi**2) <= 1)
x_in, y_in = xi[xy_inside], yi[xy_inside]                       #flat vector containing x and y positions inside the circle

### constants for normalizing the intensity of the measured interferogram.
### It will normalize not using np.max(int) but the maximum of the dimmest 100*(1-cut_off)% of the interferogram
hist_yes = np.where(xi**2 + yi**2 <= 1)
hist_mask = np.zeros(xi.shape)
hist_mask[hist_yes] = 1
cut_off = 0.06 #cutoff percentage


### Allocate memmory
rms_lsq_vec = np.zeros((len(folder_names), len(n_maxes)))
rms_janss_vec = np.zeros((len(folder_names), len(n_maxes)))
rms_inter_vec = np.zeros((len(folder_names), len(n_maxes)))

phase_rms_vec_lsq = np.zeros((len(folder_names), len(n_maxes)))
phase_rms_vec_janss = np.zeros((len(folder_names), len(n_maxes)))
phase_rms_vec_inter = np.zeros((len(folder_names), len(n_maxes)))

if (a_ref_ordering == 'fringe' or a_ref_ordering == 'Fringe') and (order == 'brug' or order == 'Brug'):
    convert_a = True
    print("conversion is true, making Z matrices")
    try:
        print("found the SLM matrix!")
        Z_mat_30 = np.load("Z_mat_30.npy")
        Z_mat_50 = np.load("Z_mat_50.npy")
        power_mat_30 = np.load("power_mat_30.npy")
        power_mat_50 = np.load("power_mat_50.npy")
    except:
        j_30 = np.arange(2, 32)
        n_30, m_30 = Zn.Zernike_j_2_nm(j_30, ordering = 'fringe')
        j_brug_30 = Zn.Zernike_nm_2_j(n_30, m_30, ordering = 'brug')
        n_max_30 = np.max(n_30).astype(int)
        print("n_max 30 = " + str(n_max_30))
        j_brug_30 = Zn.max_n_to_j(n_max_30, order = 'brug')[str(n_max_30)]
        n_max_30 = np.max(n_30).astype(int)
        power_mat_30 = Zn.Zernike_power_mat(n_max_30, order = 'brug')
        Z_mat_30 = Zn.Zernike_xy(xi, yi, power_mat_30, j_brug_30)

        j_50 = np.arange(2, 52)
        n_50, m_50 = Zn.Zernike_j_2_nm(j_50, ordering = 'fringe')
        j_brug_50 = Zn.Zernike_nm_2_j(n_50, m_50, ordering = 'brug')
        n_max_50 = np.max(n_50).astype(int)
        print("n_max 50 = " + str(n_max_50))
        j_brug_50 = Zn.max_n_to_j(n_max_50, order = 'brug')[str(n_max_50)]
        n_max_50 = np.max(n_50).astype(int)
        power_mat_50 = Zn.Zernike_power_mat(n_max_50, order = 'brug')
        Z_mat_50 = Zn.Zernike_xy(xi, yi, power_mat_50, j_brug_50)
        
        np.save("Z_mat_30.npy", Z_mat_30)
        np.save("Z_mat_50.npy", Z_mat_50)
        np.save("power_mat_30.npy", power_mat_30)
        np.save("power_mat_50.npy", power_mat_50)

else:
    convert_a = False
    n_max_fit = 10
    j_ref = Zn.max_n_to_j(n_max_fit, order = order)[str(n_max_fit)]
    a_ref = np.load(folder_names[0] + "reference_slm_vector.npy")
    assert(len(j_ref) == len(a_ref))
    print("it is " + str(len(j_ref) == len(a_ref)) + " that a_ref is up to order " + str(n_max_fit))
    try:
        Z_mat_ref = np.load("Z_mat_ref_" + str(n_max_fit) + ".npy")
        power_mat_ref = np.load("power_mat_ref_" + str(n_max_fit) + ".npy")
    except:
        power_mat_ref = Zn.Zernike_power_mat(n_max_fit, order = order)
        Z_mat_ref = Zn.Zernike_xy(xi, yi, power_mat_ref, j_ref)
        np.save("Z_mat_ref_" + str(n_max_fit) + ".npy", Z_mat_ref)
        np.save("power_mat_ref_" + str(n_max_fit) + ".npy", power_mat_ref)

#### convert matrices to new ordering system. Used to be j_max, now is n_max
if make_hough_transforms:
    for fold in folder_names:
        ## pack everything neatly in 1 vector against clutter
        j_max = 50 ### fluff because the phase extraction here doesn't recquire j_max
        constants = np.stack((px_size_sh, px_size_int, f_sh, r_int_px, r_sh_px, r_sh_m, j_max, wavelength, box_len, x0, y0, radius))
        inter_0 = np.array(PIL.Image.open(fold + "interferogram_0.tif"))
        ### gather the wrapped phase using Soloviev & Vdovin 
        org_phase, delta_i, sh_spots, image_i0, flat_wf, flip_bool = PE.phase_extraction(constants, folder_name = fold, show_id_hat = show_id_hat, show_hough_peaks = show_hough, min_height = min_height, look_ahead = look_ahead, flip_ask = False)
        mask_tile = np.tile(mask, (org_phase.shape[-1],1,1)).T
        ##
        print(flip_bool)    #flip bool is an artifact that should be True
        Un_sol = np.triu_indices(delta_i.shape[-1], k = 1) ## delta i has the shape of the amount of difference interferograms, while org phase has all possible combinations
        org_unwr = np.zeros(org_phase.shape)
        np.save(fold + "org_phase.npy", org_phase)

        print("filtering phase")
        
        butter_phase = np.zeros(org_phase.shape)
        butter_unwr = np.zeros(butter_phase.shape)
        #find tnext power of 2, to pad fourier transform and to make it easier to calculate fourier transform
        [ny, nx] = butter_phase.shape[:2]
        res = [2**kji for kji in range(15)]
        nx_pad = np.where( res > np.tile(nx, len(res)))
        nx_pad = res[nx_pad[0][0]]
        dif_x = (nx_pad - int(nx))/2
        
        for k in range(org_phase.shape[-1]):
            org_pad = np.lib.pad(org_phase[..., k], dif_x,'reflect') #pad original signal
            butter_pad = pw.butter_filter(org_pad, n, f0) #filter
            butter_phase[..., k] = butter_pad[dif_x: nx_pad - dif_x, dif_x: nx_pad - dif_x]
            butter_unwr[..., k] = pw.unwrap_phase_dct(butter_phase[..., k], xi, yi, ii, jj, N_int, N_int) #unwrap phase
            butter_unwr[..., k] -= delta_i[..., Un_sol[0][k]]    #remove delta_i, see Soloviev & Vdovin
        np.save(fold + "filtered_phase.npy", butter_unwr)
        np.save(fold + "delta_i.npy", delta_i)

        #Remove the piston term from the phase
        butter_mask = np.ma.array(butter_unwr, mask = mask_tile)
        mean_butt = butter_mask.mean(axis=(0,1))
        butter_unwr -= mean_butt
        
        ## find "true" phase by taking the median of all recovered phases
        ## Done in order to conteract noise
        but_med = np.median(butter_unwr, axis = 2)
        but_med *= -1.0 ### the solution found was -phase (see Soloviev & Vdovin)
        if flip_bool:
            but_med = np.fliplr(but_med)

        #Make flat for fitting Zernike polynomials to it
        but_med_flat = but_med[xy_inside]
        np.save(fold + "but_med_flat.npy", but_med_flat)

for ite in range(len(n_maxes)):
    n_max = n_maxes[ite]
    j = Zn.max_n_to_j(n_max, order = order)[str(n_max)]
    j_janss = Zn.max_n_to_j(n_max-1, order = order)[str(n_max-1)]
    j_max = np.max(j)

    ## Check if the matrices have been pre-calculated
    ## These matrices change with radius, hence saved in a different folder
    try:
        power_mat = np.load("matrices_" + str(2*r_int_px) + "_radius/power_mat_" + str(n_max) + ".npy")
        Zernike_2d = np.load("matrices_" + str(2*r_int_px) + "_radius/Zernike_2d_" + str(n_max)+ ".npy")
        Z_mat = np.load("matrices_" + str(2*r_int_px) + "_radius/Z_mat_" + str(n_max)+ ".npy")
        Z_mat_janss = np.load("matrices_" + str(2*r_int_px) + "_radius/Z_mat_" + str(n_max - 1)+ ".npy")
    except:
        power_mat = Zn.Zernike_power_mat(n_max, order = order)
        Zernike_3d = Zn.Zernike_xy(x_in, y_in, power_mat, j) ## create a 3d matrix with values of zernike polynomials at x_in and y_in 
        Zernike_2d = np.zeros((len(x_in), len(j))) ## flatten to perform least squares fit
        for i in range(len(j)):
            Zernike_2d[:, i] = Zernike_3d[...,i].flatten()
        Z_mat = Zn.Zernike_xy(xi, yi, power_mat, j)
        Z_mat_janss = Zn.Zernike_xy(xi, yi, power_mat, j_janss)

        np.save("matrices_" + str(2*r_int_px) + "_radius/power_mat_" + str(n_max) + ".npy", power_mat)
        np.save("matrices_" + str(2*r_int_px) + "_radius/Zernike_2d_" + str(n_max) + ".npy", Zernike_2d)
        np.save("matrices_" + str(2*r_int_px) + "_radius/Z_mat_" + str(n_max) + ".npy", Z_mat)
        np.save("matrices_" + str(2*r_int_px) + "_radius/Z_mat_" + str(n_max - 1) + ".npy", Z_mat_janss)
        
    print("Current maximum Zernike order: " + str(n_max))
    
    for i in range(len(folder_names)):
        folder_name = folder_names[i]
        fold_name = folder_name
        
        but_med_flat = np.load(folder_name + "but_med_flat.npy")
        inter_0 = np.array(PIL.Image.open(folder_name + "interferogram_0.tif"))

        ## Fit Zernike coefficients
        a_butt = np.linalg.lstsq(Zernike_2d, but_med_flat)[0]
        a_inter= np.copy(a_butt)

        ## Cut out original measured interferogram. flipped_y0 is artefact
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
        # Actually! Orig == iterferogram! Replace cutout with the normalized one
        ### ---
        flipint = False
        piston, inter_rms = opt.fmin(rms_piston, 0, args = (j_max, a_inter, N_int, Z_mat, orig, mask, flipint), full_output = True)[:2]
        orig_plot = np.copy(orig)
        orig = np.ma.array(Zn.int_for_comp(j_max, a_inter, N_int, piston, Z_mat, False), mask = mask)
        #inter_rms = 0.0

        ### import necessary shack hartmann patterns 
        image_ref_mirror = np.array(PIL.Image.open(folder_name + "image_ref_mirror.tif"), dtype = 'float')
        zero_pos_dm = np.array(PIL.Image.open(folder_name + "zero_pos_dm.tif"), dtype = 'float')
        dist_image = np.array(PIL.Image.open(folder_name + "dist_image.tif"), dtype = 'float')
        flat_wf = np.array(PIL.Image.open(folder_name + "flat_wf.tif"), dtype = 'float')
        sh_spots = np.dstack((image_ref_mirror, zero_pos_dm, dist_image)) 

        #### create copies and new grids for analysis
        zero_image = sh_spots[..., 1]
        zero_image_zeros = np.copy(zero_pos_dm)
        dist_image = sh_spots[..., 2]
        image_control = sh_spots[..., 0]
        [ny,nx] = zero_pos_dm.shape
        x = np.arange(0, nx, 1)
        y = np.arange(ny, 0, -1)
        xx, yy = np.meshgrid(x, y)

        ### Find spot positions of the flat wavefront and the distorted wavefront
        print("gathering zeros")
        x_pos_zero, y_pos_zero = Hm.zero_positions(np.copy(zero_pos_dm), spotsize = box_len)
        print("amount of spots: " + str(len(x_pos_zero)))
        x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, np.copy(image_ref_mirror), xx, yy, spot_size = box_len)
        centre = Hm.centroid_centre(x_pos_flat, y_pos_flat)
        x_pos_norm = ((x_pos_flat - centre[0]))/float(r_sh_px)
        y_pos_norm = ((y_pos_flat - centre[1]))/float(r_sh_px)
        inside = np.where(np.sqrt(x_pos_norm**2 + y_pos_norm**2) <= (1 + (float(box_len)/r_sh_px))) #35 is the half the width of the pixel box aroudn a centroid and r_sh_px is the scaling factor
        x_pos_zero_f, y_pos_zero_f, x_pos_flat_f, y_pos_flat_f, x_pos_norm_f, y_pos_norm_f = fi.filter_positions(inside, x_pos_zero, y_pos_zero, x_pos_flat, y_pos_flat, x_pos_norm, y_pos_norm)
        x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_zero, y_pos_zero, np.copy(dist_image), xx, yy, spot_size = box_len)
        x_pos_dist_f, y_pos_dist_f = fi.filter_positions(inside, x_pos_dist, y_pos_dist)

        
        a_ref = np.load(folder_name + "reference_slm_vector.npy")
        if convert_a:
            a_ref = Zn.convert_fringe_2_brug(a_ref)
            try:
                if a_ref.shape[0] == Z_mat_30.shape[-1]:
                    phi_ref = np.ma.array(np.dot(Z_mat_30, a_ref), mask = mask)
                elif a_ref.shape[0] == Z_mat_50.shape[-1]:
                    phi_ref = np.ma.array(np.dot(Z_mat_50, a_ref), mask = mask)
                else:
                    raise ValueError('Z_mat_ref and a_ref are not the same size')
            except ValueError as err:
                print(err.args)
        else:
            try:
                if a_ref.shape[0] == Z_mat_ref.shape[-1]:
                    phi_ref = np.ma.array(np.dot(Z_mat_ref, a_ref), mask = mask)
                else:
                    raise ValueError('Z_mat_ref and a_ref are not the same size')
            except ValueError as err:
                print(err.args)

        ### bundle arguments for optimization of piston
        lsq_args = (wavelength, j, f_sh, px_size_sh, y_pos_flat, x_pos_flat, x_pos_dist, y_pos_dist, xx, yy, orig, mask, N_int, Z_mat, power_mat, box_len, centre)
        janss_args = (x_pos_zero, y_pos_zero, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist, px_size_sh, f_sh, n_max, wavelength, xx, yy, N_int, orig, mask, Z_mat_janss, box_len, centre)

        ### bundle arguments for optimization of centre, radius using the phase
        lsq_args_phi = wavelength, j, f_sh, px_size_sh, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist, xx, yy, mask, N_int, Z_mat, power_mat, box_len, phi_ref
        janss_args_phi = x_pos_zero, y_pos_zero, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist, px_size_sh, f_sh, n_max, wavelength, xx, yy, N_int, mask, Z_mat_janss, box_len, phi_ref, order

        #### ran a constrained minimization scheme for finding a radius which minimizes rms error. All further calculations will be done with that. average radius between optimum for janssen and SH
        if (optimize_center_radius and ite == 0):
            print("optimizing SH")
            minradius = 0.85 * np.max(np.sqrt((x_pos_flat_f - centre[0])**2 + (y_pos_flat_f - centre[1])**2))
            maxradius = 1.15/0.85 * minradius
            print("mindradius = " + str(minradius))
            bounds_min = ((nx/2 - 50, nx/2 + 50), (ny/2 - 50, ny/2 + 50), (minradius, maxradius))
            initial_guess = [0, float(r_sh_px)]
            initial_guess_phi = [nx/2, ny/2, float(r_sh_px)] #changed from centre[0], centre[1] under the assumption that it is aligned in the middle using the cross and thorlabs software (which was done)

            bf_janss = time.time()
            optresult_janss = opt.minimize(rms_phi_janssen, initial_guess_phi, args = janss_args_phi, method = 'L-BFGS-B', bounds = bounds_min, options = {'maxiter': 1000, 'disp':True})
            aft_janss = time.time()
            print("All iterations Janssen took " + str(aft_janss - bf_janss) + " s")
        
            bf_lsq = time.time()
            optresult_lsq = opt.minimize(rms_phi_lsq, initial_guess_phi, args = lsq_args_phi, method = 'L-BFGS-B', bounds = bounds_min, options = {'maxiter': 1000, 'disp':True})
            aft_lsq = time.time()
            print("All iterations LSQ took " + str(aft_lsq-bf_lsq) +" s")

            vars_lsq = optresult_lsq.x            
            vars_janss = optresult_janss.x
            print("radius lsq = " + str(vars_lsq[1]) + ", radius janss = " + str(vars_janss[1]))

##            r_sh_px = (vars_lsq[2] + vars_janss[2])/2.0#vars_lsq[2] + vars_janss[2])/2.0
            r_sh_px_lsq = vars_lsq[2]
            r_sh_px_janss = vars_janss[2]

            np.save(folder_name + "optimized_" + str(n_optimization) + "_lsq_vars.npy", vars_lsq)
            np.save(folder_name + "optimized_" + str(n_optimization) + "_janss_vars.npy", vars_janss)
            
            json.dump(r_sh_px, open(folder_name + "optimized_" + str(n_optimization) + "_radius_04_24.txt", 'w'))
            json.dump(r_sh_px_lsq, open(folder_name + "optimized_" + str(n_optimization) + "_radius_lsq_04_24.txt", 'w'))
            json.dump(r_sh_px_janss, open(folder_name + "optimized_" + str(n_optimization) + "_radius_janss_04_24.txt", 'w'))
        else:
            if old_vars_taken:
                var_dict = np.load(folder_name + "vars_dictionary_j_" + old_vars_string[i] + "_65.npy").item()
                vars_lsq = var_dict['vars_lsq']
                vars_janss = var_dict['vars_janss']
            else:
                try:
                    if only_use_lsq_vars:
                        vars_janss = np.load(folder_name + "optimized_" + str(n_optimization) + "_lsq_vars.npy")
                    else:
                        vars_janss = np.load(folder_name + "optimized_" + str(n_optimization) + "_janss_vars.npy")
                        
                    vars_lsq = np.load(folder_name + "optimized_" + str(n_optimization) + "_lsq_vars.npy") 
                    with open(folder_name + "optimized_" + str(n_optimization) + "_radius_lsq_04_24.txt") as data:
                        r_sh_px_lsq = json.load(data)
                    with open(folder_name + "optimized_" + str(n_optimization) + "_radius_janss_04_24.txt") as data:
                        r_sh_px_janss = json.load(data)
                except: #purely to test if the whole routine operates as it should
                    print("exception occured! Optimized radius was not found and estimated poorly!")
                    sys.exit(1) ## exit the programme showing the error status
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
        x_pos_flat_lsq, y_pos_flat_lsq, x_pos_dist_lsq, y_pos_dist_lsq = fi.filter_positions(inside_lsq, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist)
        x_pos_norm_lsq, y_pos_norm_lsq = fi.filter_positions(inside_lsq, x_pos_norm_lsq, y_pos_norm_lsq)

        G = LSQ.matrix_avg_gradient(x_pos_norm_lsq, y_pos_norm_lsq, j, r_sh_lsq, power_mat, box_len)
        r_sh_px_lsq_opt = r_sh_lsq
        r_sh_m_lsq_opt = px_size_sh * r_sh_lsq
        s = np.hstack(Hm.centroid2slope(x_pos_dist_lsq, y_pos_dist_lsq, x_pos_flat_lsq, y_pos_flat_lsq, px_size_sh, f_sh, r_sh_m, wavelength))
        a_lsq_opt = np.linalg.lstsq(G, s)[0]

        ### optimize piston w.r.t. original interferogram measured (and normalized)
        pist_args_lsq = (j_max, a_lsq_opt, N_int, Z_mat, orig, mask, False)
        pist_lsq, lsq_rms = opt.fmin(rms_piston, 0, pist_args_lsq, full_output = True)[:2]

        ### find coefficients janss according to optimum
        r_sh_janss = vars_janss[2]
        x_pos_norm_janss = (x_pos_flat - vars_janss[0])/r_sh_janss
        y_pos_norm_janss = (y_pos_flat - vars_janss[1])/r_sh_janss
        inside_janss = np.where(np.sqrt(x_pos_norm_janss**2 + y_pos_norm_janss**2) <= (1 + (box_len/r_sh_janss)))
        x_pos_norm_janss, y_pos_norm_janss, x_pos_flat_janss, y_pos_flat_janss, x_pos_dist_janss, y_pos_dist_janss = fi.filter_positions(inside_janss, x_pos_norm_janss, y_pos_norm_janss, x_pos_flat, y_pos_flat, x_pos_dist, y_pos_dist)
        r_sh_m_janss = px_size_sh * r_sh_janss
        a_janss_opt = janssen.coeff_from_dist(x_pos_flat_janss, y_pos_flat_janss, x_pos_dist_janss, y_pos_dist_janss, x_pos_norm_janss, y_pos_norm_janss, px_size_sh, f_sh, r_sh_janss*px_size_sh, wavelength, n_max, r_sh_janss, box_len, order)

        ### optimize piston w.r.t. original interferogram measured (and normalized)
        pist_args_janss = (j_max, a_janss_opt, N_int, Z_mat_janss, orig, mask, False)
        pist_janss, janss_rms = opt.fmin(rms_piston, 0, pist_args_janss, full_output = True)[:2]

        
        if convert_a: #if originally made with fringe ordering, it's either done with 30 or 50 coefficients. 
            try:
                a_ref_args = (j_max, a_ref, N_int, Z_mat_30, orig, mask, False)
                a_ref_pist, ref_rms = opt.fmin(rms_piston, 0, a_ref_args, full_output = True)[:2]
            except:
                a_ref_args = (j_max, a_ref, N_int, Z_mat_50, orig, mask, False)
                a_ref_pist, ref_rms = opt.fmin(rms_piston, 0, a_ref_args, full_output = True)[:2]
        else:
            a_ref_args = (j_max, a_ref, N_int, Z_mat_ref, orig, mask, False)
            a_ref_pist, ref_rms = opt.fmin(rms_piston, 0, a_ref_args, full_output = True)[:2]

        ### calculate RMS value between intended phase and measured
        phi_lsq = np.ma.array(np.dot(Z_mat, a_lsq_opt), mask = mask)
        phi_int = np.ma.array(np.dot(Z_mat, a_inter), mask = mask)
        phi_janss = np.ma.array(np.dot(Z_mat_janss, a_janss_opt), mask = mask)
        phi_diff_int, phi_diff_lsq, phi_diff_janss = np.ma.array(phi_ref - phi_int, mask = mask), np.ma.array(phi_ref - phi_lsq, mask = mask), np.ma.array(phi_ref - phi_janss, mask = mask)
        eps_vec = np.stack((np.sqrt((phi_diff_int**2).sum())/N_int, np.sqrt((phi_diff_lsq**2).sum())/N_int, np.sqrt((phi_diff_janss**2).sum())/N_int))

        phase_rms_vec_lsq[i, ite] = eps_vec[1]
        phase_rms_vec_janss[i, ite] = eps_vec[2]
        phase_rms_vec_inter[i, ite] = eps_vec[0]

        titles = [r'Intended', r'Interferogram', r'LSQ', r'Janssen']
        f_phi, ax_phi = plt.subplots(2,5, figsize = (4.98, 4.98/2.2), gridspec_kw = {'width_ratios':[4, 4, 4, 4, 0.4]})

        ### find nice values for minimum and maximum phase to plot
        vmin_phi, vmax_phi = np.min([phi_ref.min(), phi_lsq.min(), phi_int.min(), phi_janss.min()]), np.max((phi_ref.max(), phi_lsq.max(), phi_int.max(), phi_janss.max()))
        vmin_phi = np.pi * (np.sign(vmin_phi) * (np.abs(vmin_phi)//(np.pi)) + np.sign(vmin_phi))
        vmax_phi = np.pi * (np.sign(vmax_phi) * (np.abs(vmax_phi)//(np.pi)) + np.sign(vmax_phi))
        vmin_diff, vmax_diff = np.min([phi_diff_int.min(), phi_diff_lsq.min(), phi_diff_janss.min()]), np.max([phi_diff_int.max(), phi_diff_lsq.max(), phi_diff_janss.max()])
        vmin_diff = np.pi * (np.sign(vmin_diff) * (np.abs(vmin_diff)//(np.pi)) + np.sign(vmin_diff))
        vmax_diff = np.pi * (np.sign(vmax_diff) * (np.abs(vmax_diff)//(np.pi)) + np.sign(vmax_diff))
        
        ### plot intended and reconstructed phases, and the difference
        phi_0 = ax_phi[0,0].imshow(np.ma.array(phi_ref, mask = mask), vmin = vmin_phi, vmax = vmax_phi, cmap = 'jet', origin = 'lower', interpolation = 'none')
        ax_phi[0, 1].imshow(np.ma.array(phi_int, mask = mask), vmin = vmin_phi, vmax = vmax_phi, cmap = 'jet', origin = 'lower', interpolation = 'none')
        ax_phi[0, 2].imshow(np.ma.array(phi_lsq, mask = mask), vmin = vmin_phi, vmax = vmax_phi, cmap = 'jet', origin = 'lower', interpolation = 'none')
        ax_phi[0, 3].imshow(np.ma.array(phi_janss, mask = mask), vmin = vmin_phi, vmax = vmax_phi, cmap = 'jet', origin = 'lower', interpolation = 'none')

        phi_diff = ax_phi[1,1].imshow(np.ma.array(phi_diff_int, mask = mask), vmin = vmin_diff, vmax = vmax_diff, cmap = 'jet', origin = 'lower', interpolation = 'none')
        ax_phi[1,2].imshow(np.ma.array(phi_diff_lsq, mask = mask), vmin = vmin_diff, vmax = vmax_diff, cmap = 'jet', origin = 'lower', interpolation = 'none')
        ax_phi[1,3].imshow(np.ma.array(phi_diff_janss, mask = mask), vmin = vmin_diff, vmax = vmax_diff, cmap = 'jet', origin = 'lower', interpolation = 'none')

        ### make-up on the phase plots
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

        ### Make nice ticks for the phase plots
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
        
        ### do all the same for interferograms
##        titles_int = [r'Measured', r'Intended', r'Interferogram', r'LSQ', r'Janssen', r'']
##        rms_lsq_vec[i, ite] = lsq_rms
##        rms_janss_vec[i, ite] = janss_rms
##        rms_inter_vec[i, ite] = inter_rms
##        rms_mat = np.array([ref_rms, inter_rms, lsq_rms, janss_rms])
##        f, axes = plt.subplots(nrows = 1, ncols = 6, figsize = (4.98, 4.98/4.4), gridspec_kw={'width_ratios':[4, 4, 4, 4, 4, 0.4]})#, 'height_ratios':[4,4]})
##        interf = axes[0].imshow(np.ma.array(orig_plot, mask = mask), vmin = 0, vmax = 1, cmap = 'bone', origin = 'lower', interpolation = 'none')
##        if convert_a:
##            try:
##                Zn.imshow_interferogram(j_max, a_ref, N_int, piston = a_ref_pist, ax = axes[1], Z_mat = Z_mat_30, power_mat = power_mat_30)
##            except:
##                Zn.imshow_interferogram(j_max, a_ref, N_int, piston = a_ref_pist, ax = axes[1], Z_mat = Z_mat_50, power_mat = power_mat_50)   
##        else:
##            Zn.imshow_interferogram(j_max, a_ref, N_int, piston = a_ref_pist, ax = axes[1], Z_mat = Z_mat_ref, power_mat = power_mat_ref)
##
##                 
##        Zn.imshow_interferogram(j_max, a_inter, N_int, piston = piston, ax = axes[2], Z_mat = Z_mat, power_mat = power_mat)
##        Zn.imshow_interferogram(j_max, a_lsq_opt, N_int, piston = pist_lsq, ax = axes[3], fliplr = False, Z_mat = Z_mat, power_mat = power_mat)
##        Zn.imshow_interferogram(j_max, a_janss_opt, N_int, piston = pist_janss, ax = axes[4], fliplr = False, Z_mat = Z_mat_janss, power_mat = power_mat)
##        for ii_it in range(5):
##            axes[ii_it].set(adjustable = 'box-forced', aspect = 'equal')
##            axes[ii_it].get_xaxis().set_ticks([])
##            axes[ii_it].get_yaxis().set_ticks([])
##            axes[ii_it].set_title(titles_int[ii_it], fontsize = 9)
##            if ii_it == 0:
##                axes[ii_it].text(N_int/2, -N_int/6, r"$\varepsilon~=~$", fontsize = 7, ha = 'center')
##            else:
##                axes[ii_it].text(N_int/2, -N_int/6, "%.4f"%rms_mat[ii_it-1], fontsize = 7, ha = 'center')
##        cbar = plt.colorbar(interf, cax = axes[-1], ticks = [0.0, 0.25, 0.5, 0.75, 1.0])
##        cbar.ax.tick_params(labelsize=7)
##        cbar.ax.set_ylabel("Normalized Intensity", fontsize = 8)
##        f.savefig(fold_name + "interferograms_"+save_string[i]+"_j_" + str(j_max) + ".png", bbox_inches = 'tight', dpi = dpi_num)
##        f.savefig(save_fold + "interferograms_"+save_string[i]+"_j_" + str(j_max) + ".png", bbox_inches = 'tight', dpi = dpi_num)
##        plt.close(f)

        ### Either plot 30 or 50 coefficients
        try:
            if a_ref.shape[0] == power_mat_30.shape[-1]:
                f2, ax2 = plt.subplots(nrows = 1, ncols = 1, figsize = (4.98, 4.98/4.4))
                ax2.plot(32*[0], 'k-', linewidth = 0.5)

                x_max_plot = len(a_ref)
                a_x = np.copy(j)
                a_x_janss = np.copy(j_janss)
                a_x_ref = np.arange(2, x_max_plot+2)
                ax2.plot(a_x, a_inter, 'sk', label = 'Interferogram')
                ax2.plot(a_x_janss, a_janss_opt, 'og', label = 'Janssen')
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
                raise ValueError("Coefficient plot will be with 2 lines")
        except (ValueError, NameError):
            f2, ax2 = plt.subplots(nrows = 2, ncols = 1, sharey = True, figsize = (4.98, 4.98/2.2))
            N_half = 25
            ax2[0].plot((N_half+2)*[0], 'k-', linewidth = 0.5)

            x_max_plot = len(a_ref)
            
            a_x = np.copy(j)
            a_x_janss = np.copy(j_janss)
            a_x_ref = np.arange(2, x_max_plot+2)
            ax2[0].plot(a_x[:N_half], a_inter[:N_half], 'sk', label = 'Interferogram')
            ax2[0].plot(a_x_janss[:N_half], a_janss_opt[:N_half], 'og', label = 'Janssen')
            ax2[0].plot(a_x[:N_half], a_lsq_opt[:N_half], '2b', label = 'LSQ')
            ax2[0].plot(a_x_ref[:N_half], a_ref[:N_half], 'rx', label = 'Intended', markersize = 1)
            ax2[0].set_xlim([0, N_half+1.5])

            ax2[1].plot((2*N_half + 4)*[0], 'k-', linewidth = 0.5)
            ax2[1].plot(a_x[N_half:], a_inter[N_half:], 'sk', label = 'Interferogram')
            ax2[1].plot(a_x_janss[N_half:], a_janss_opt[N_half:], 'og', label = 'Janssen')
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

        ### Save all necessary information,
        """
        rms_dict containts rms values with respect to interferogram
        vars_dict contain x0, y0 and radius for the shack-hartmann sensor, or piston  for the interferogram
        coeff_dict contains the coefficients calculated for a certain amount of zernike orders fitted
        """
        rms_dict = {'rms_inter':inter_rms,'rms_lsq':lsq_rms, 'rms_janss':janss_rms}
        vars_dict = {'vars_lsq':vars_lsq, 'vars_janss':vars_janss, 'pist_inter':piston}
        coeff_dict = {'coeff_inter':a_inter, 'coeff_lsq':a_lsq_opt, 'coeff_janss':a_janss_opt}
        np.save(folder_name + 'vars_dictionary_j_' + save_string[i]+ "_" + str(j_max) + '.npy', vars_dict)
        np.save(folder_name + 'coeff_dictionary_j_' + save_string[i] + "_" + str(j_max)+ '.npy', coeff_dict)
        np.save(folder_name + 'uncorrected_zn_coeff_inter_' + save_string[i] + "_" + str(j_max) + '.npy', a_butt)
        json.dump(rms_dict, open(folder_name + "rms_dict_j_" + save_string[i] + "_" + str(j_max) + ".txt", 'w'))
        print(" rms interferogram: " + str(inter_rms) + ",\n rms LSQ: " + str(lsq_rms) + ",\n rms Janssen: " + str(janss_rms))

        ### information is saved every iteration so that in case of a crash, partial data is available
        np.save(folder_name + save_string[i] + "_lsq_rms_var_j.npy", rms_lsq_vec[i, :])
        np.save(folder_name + save_string[i] + "_janss_rms_var_j.npy", rms_janss_vec[i, :])
        np.save(folder_name + save_string[i] + "_inter_rms_var_j.npy", rms_inter_vec[i, :])
        np.save(folder_name + save_string[i] + "_phase_rms_lsq.npy", phase_rms_vec_lsq[i, :])
        np.save(folder_name + save_string[i] + "_phase_rms_janss.npy", phase_rms_vec_janss[i, :])
        np.save(folder_name + save_string[i] + "_phase_rms_inter.npy", phase_rms_vec_inter[i, :])
        np.save(folder_name + save_string[i] + "_n_vector.npy", n_maxes)
plt.show()
