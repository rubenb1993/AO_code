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
from mpl_toolkits.mplot3d import Axes3D
import janssen
import phase_extraction as PE
import phase_unwrapping_test as pw
import scipy.optimize as opt
from scipy import ndimage

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

def rms_filter(variables, *args):
    """variables are
    n, f0,
    args contain
    orig, mask, FX, FY, Zernike_2D, xy_inside, ft_med, nx, nx_pad, j_max, N, Z_mat"""
    n, f0 = variables
    orig, mask, FX, FY, Zernike_2D, xy_inside, ft_med, nx, nx_pad, j_max, N, Z_mat, piston = args
    flipint = False
    fliplr = False
    dif_x = (nx_pad - nx)/2
    orig = np.ma.array(orig, mask = mask)
    butt_filt = 1/(1 + ( np.sqrt(FX**2 + FY**2)/f0)**(2*n))
    shift = np.exp(-2*np.pi*1j*(FX+FY))
    org_filt= np.fft.ifftshift(np.fft.ifft2(butt_filt * ft_med)) / shift
    org_filt_unpad = org_filt[dif_x:nx_pad - dif_x,  dif_x : nx_pad - dif_x]
    a_filt = np.linalg.lstsq(Zernike_2d, np.real(org_filt_unpad[xy_inside]))[0]
    #piston = opt.fmin(rms_piston, 0, args = (j_max, a_filt, N, Z_mat, orig, mask, flipint))
    inter = np.ma.array(Zn.int_for_comp(j_max, a_filt, N, piston, Z_mat, fliplr), mask = mask)
    rms = np.sqrt(np.sum((inter - orig)**2)/N**2)
    return rms

def a_fit_filtered(variables, *args):
    n, f0 = variables
    orig, mask, FX, FY, Zernike_2D, xy_inside, ft_med, nx, nx_pad, j_max, N, Z_mat, piston = args
    flipint = False
    fliplr = False
    dif_x = (nx_pad - nx)/2
    orig = np.ma.array(orig, mask = mask)
    butt_filt = 1/(1 + ( np.sqrt(FX**2 + FY**2)/f0)**(2*n))
    shift = np.exp(-2*np.pi*1j*(FX+FY))
    org_filt= np.fft.ifftshift(np.fft.ifft2(butt_filt * ft_med)) / shift
    org_filt_unpad = org_filt[dif_x:nx_pad - dif_x,  dif_x : nx_pad - dif_x]
    a_filt = np.linalg.lstsq(Zernike_2d, np.real(org_filt_unpad[xy_inside]))[0]
    return a_filt

folder_name = "20161213_new_inters/"

x0 = 550
y0 = 484
radius = int(310)

image_ref_mirror = np.array(PIL.Image.open(folder_name + "image_ref_mirror.tif"))
zero_pos_dm = np.array(PIL.Image.open(folder_name + "zero_pos_dm.tif"))
dist_image = np.array(PIL.Image.open(folder_name + "dist_image.tif"))
flat_wf = np.array(PIL.Image.open(folder_name + "flat_wf.tif"))

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

show_hough_peaks = False
min_height = 50
look_ahead = 15
print("making hough transforms... crunch crunch")
for jj in range(4):
    Acc[...,jj], rhos, thetas, diag_len = PE.hough_numpy(Id_zeros_mask[..., jj], x, y)
    print("Hough transform " + str(jj+1) + " done!")
    theta_max[jj], s_max[jj], theta_max_index, peak_indices = PE.max_finding(Acc[...,jj], rhos, thetas, minheight = min_height, lookahead = look_ahead)
    ## uncomment to check if peaks align with found peaks visually
    if show_hough_peaks == True:
        f2, axarr2 = plt.subplots(2,1)
        axarr2[0].imshow(Acc[...,jj].T, cmap = 'bone')
        axarr2[0].scatter(peak_indices, np.tile(theta_max_index, len(peak_indices)))
        axarr2[1].plot(rhos, Acc[:, theta_max_index, jj])
        dtct = axarr2[1].scatter(rhos[peak_indices], Acc[peak_indices, theta_max_index, jj], c = 'r')
        plt.show()
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


f, axarr = plt.subplots(1,4, sharex = True)
med_Id_hat_3 = np.zeros(Id_hat.shape)
med_Id_hat_5 = np.zeros(Id_hat.shape)


x_pcg, y_pcg = np.linspace(-1, 1, 2*radius), np.linspace(-1, 1, 2*radius)
N = len(x_pcg)
i, j = np.linspace(0, N-1, N), np.linspace(0, N-1, N)
xx_alg, yy_alg = np.meshgrid(x_pcg, y_pcg)
ii, jj = np.meshgrid(i, j)

xy_inside = np.where(np.sqrt(xx_alg**2 + yy_alg**2) <= 1)
x_in, y_in = xx_alg[xy_inside], yy_alg[xy_inside]

j_max = 30
xi, yi = np.linspace(-1, 1, N), np.linspace(-1, 1, N)
xi, yi = np.meshgrid(xi, yi)

j = np.arange(2, j_max) #start at 2, end at j_max-1
power_mat = Zn.Zernike_power_mat(j_max+2)
Zernike_3d = Zn.Zernike_xy(x_in, y_in, power_mat, j) ## create a 3d matrix with values of zernike polynomials at x_in and y_in 
Zernike_2d = np.zeros((len(x_in), j_max)) ## flatten to perform least squares fit

for i in range(len(j)):
        Zernike_2d[:, i] = Zernike_3d[...,i].flatten()


Z_mat = Zn.Zernike_xy(xi, yi, power_mat, np.arange(2, j_max+2))

size_vec = np.arange(1, 23, 2)
mins = np.zeros(len(size_vec))
rms_vec = np.zeros(len(size_vec))
rms_vec_open = np.zeros(len(size_vec))
rms_vec_closing = np.zeros(len(size_vec))
rms_vec_med = np.zeros(len(size_vec))

orig = np.ma.array(image_i0[y0-radius:y0+radius, x0-radius:x0+radius], mask = mask)
orig /= np.max(orig)
flipint = False

filt_filt = raw_input("filter twice with large convolution?")
if filt_filt == 'y':
    k_phi = np.arange(1, 21, 2)
    k_I = np.arange(1, 21, 2)
    rms_mat = np.zeros((len(k_I), len(k_phi)))
    piston_mat = np.zeros((len(k_I), len(k_phi)))

    for i in range(len(j)):
        Zernike_2d[:, i] = Zernike_3d[...,i].flatten()


    for kkk in range(len(k_I)):
        for jjj in range(len(k_phi)):
            print(str(k_I[kkk]), str(k_phi[jjj]))
            Id_hat = np.load('id_hat_matrix.npy')
            for i in range(4):
                Id_hat[...,i] = ndimage.median_filter(Id_hat[...,i], k_I[kkk])
                

            nmbr_inter = Id_hat.shape[-1] #number of interferogram differences
            Un_sol = np.triu_indices(nmbr_inter, 1) #indices of unique phase retrievals. upper triangular indices, s.t. 12 and 21 dont get counted twice
            shape_unwrp = list(Id_hat.shape[:2])
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
                org_phase[..., k] = pw.filter_wrapped_phase(org_phase[..., k], k_phi[jjj])
                org_unwr[...,k] = pw.unwrap_phase_dct(org_phase[..., k], xx_alg, yy_alg, ii, jj, N, N)
                org_unwr[...,k] -= delta_i[..., Un_sol[0][k]]


            mask = [np.sqrt((xx_alg) ** 2 + (yy_alg) ** 2) >= 1]
            mask_tile = np.tile(mask, (org_phase.shape[-1],1,1)).T
            org_mask = np.ma.array(org_unwr, mask = mask_tile)
            mean_unwr = org_mask.mean(axis=(0,1))
            org_unwr -= mean_unwr
            org_med = np.median(org_unwr, axis = 2)
            org_med_flat = org_med[xy_inside]

            a_inter = np.linalg.lstsq(Zernike_2d, org_med_flat)[0]

            piston_mat[kkk, jjj], rms_mat[kkk, jjj] = opt.fmin(rms_piston, 0, args = (j_max, a_inter, N, Z_mat, orig, mask, flipint), full_output = True)[:2]
            #mins[kkk] = opt.fmin(rms_piston, 0, args = (j_max, a_inter, N, Z_mat, orig, mask, flipint))
            #rms_mat[kkk, jjj] = rms_piston(piston_mat[kkk, jjj], j_max, a_inter, N, Z_mat, orig, mask, flipint)


    ##f, ax = plt.subplots(1,2, sharey = True)
    ##ax[0].plot(size_vec, rms_vec_open, label = 'open')
    ##ax[0].plot(size_vec, rms_vec_closing, label = 'closing')
    ##ax[0].plot(size_vec, rms_vec_med, label = 'median')
    ##ax[0].plot(size_vec, rms_vec, label = 'phase filter')
    ##ax[0].set_xlabel(r'window size $k$')
    ##ax[0].set_ylabel(r'$\varepsilon$')
    ##ax[1].plot(size_vec_gauss, rms_vec_gauss, label = 'gaussian')
    ##ax[1].set_xlabel(r'$\sigma$')
    ##ax[1].set_ylabel(r'$\varepsilon$')
    k_II, k_pp = np.meshgrid(k_I, k_phi)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(k_II, k_pp, rms_mat, cmap = 'jet', rstride = 1, cstride = 1)
    ax.set_xlabel(r'$k_{\varphi}$')
    ax.set_ylabel(r'$k_{I}$')
    ax.set_zlabel(r'$\varepsilon$')
    plt.show()

filt_butt = raw_input("test butter filter?")
if filt_butt == 'y':
    k_phi = 19
    k_I = 5
    
    Id_hat = np.load('id_hat_matrix.npy')
    for i in range(4):
        Id_hat[...,i] = ndimage.median_filter(Id_hat[...,i], k_I)
        

    nmbr_inter = Id_hat.shape[-1] #number of interferogram differences
    Un_sol = np.triu_indices(nmbr_inter, 1) #indices of unique phase retrievals. upper triangular indices, s.t. 12 and 21 dont get counted twice
    shape_unwrp = list(Id_hat.shape[:2])
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
        org_phase[..., k] = pw.filter_wrapped_phase(org_phase[..., k], k_phi)
        org_unwr[...,k] = pw.unwrap_phase_dct(org_phase[..., k], xx_alg, yy_alg, ii, jj, N, N)
        org_unwr[...,k] -= delta_i[..., Un_sol[0][k]]


    mask = [np.sqrt((xx_alg) ** 2 + (yy_alg) ** 2) >= 1]
    mask_tile = np.tile(mask, (org_phase.shape[-1],1,1)).T
    org_mask = np.ma.array(org_unwr, mask = mask_tile)
    mean_unwr = org_mask.mean(axis=(0,1))
    org_unwr -= mean_unwr
    org_med = np.median(org_unwr, axis = 2)
    

    piston_seperate = np.zeros(org_mask.shape[-1])
    a_seperate = np.zeros((j_max, org_mask.shape[-1]))
    for kk in range(org_mask.shape[-1]):
        org_fit = org_unwr[..., kk]
        org_fit = org_fit[xy_inside]
        a_seperate[:, kk] = np.linalg.lstsq(Zernike_2d, org_fit)[0]
        piston_seperate[kk] = opt.fmin(rms_piston, 0, args = (j_max, a_seperate[:, kk], N, Z_mat, orig, mask, flipint))

    a_med = np.linalg.lstsq(Zernike_2d, org_med[xy_inside])[0]
    piston_med = opt.fmin(rms_piston, 0, args = (j_max, a_med, N, Z_mat, orig, mask, flipint))
    [ny, nx] = org_med.shape
    res = [2**i for i in range(15)]
    #find index power of 2 which is bigger than image
    nx_pad = np.where( res > np.tile(nx, len(res)))
    nx_pad = res[nx_pad[0][0]]
    dif_x = (nx_pad - int(nx))/2
    org_pad = np.lib.pad(org_med, dif_x, 'reflect')
    mask_pad = np.lib.pad(mask[0], dif_x, 'edge')
####    org_pad = np.zeros((nx_pad, nx_pad))
####    org_pad[dif_x:nx_pad - dif_x, dif_x:nx_pad - dif_x] = org_med
####    #top nd bottom
####    org_pad[:dif_x, dif_x:nx_pad-dif_x] = org_med[0, :] #broadcast magic
####    org_pad[nx_pad - dif_x:, dif_x:nx_pad - dif_x] = org_med[-1, :]
####    #left and right
####    org_pad[dif_x:nx_pad - dif_x, :dif_x] = np.tile(org_med[:, 0], (dif_x,1)).T #tiling magic
####    org_pad[dif_x:nx_pad - dif_x, nx_pad - dif_x:] = np.tile(org_med[:, -1], (dif_x, 1)).T
####    #corners
####    org_pad[:dif_x, :dif_x] = org_med[0,0]
####    org_pad[nx_pad - dif_x:, :dif_x] = org_med[-1, 0]
####    org_pad[:dif_x, nx_pad - dif_x:] = org_med[0,1]
####    org_pad[nx_pad - dif_x:, nx_pad -dif_x:] = org_med[-1, -1] 
    
    dx = 2.0/nx_pad
    dy = 2.0/nx_pad
    x_f = np.arange(-1, 1, dx)
    y_f = np.arange(-1, 1, dy)
    xx_f, yy_f = np.meshgrid(x_f, y_f)
    dfx = 0.5
    dfy = 0.5
    fx = np.arange(-0.5/dx, 0.5/dx, dfx)
    fy = np.arange(-0.5/dy, 0.5/dy, dfy)
    FX, FY = np.meshgrid(fx, fy)
    shift = np.exp(-2*np.pi*1j*(FX+FY))
    print(shift.shape)
    ft_med = shift *  np.fft.fftshift(np.fft.fft2(org_pad))
    org_med_filt = np.zeros((nx_pad, nx_pad, 10), dtype = np.complex_)
    f0 = np.linspace(50, 250, 6)
    n = 5
    f, ax = plt.subplots(1,1)
    for i in range(4):
        butt_filt = 1/(1 + ( np.sqrt(FX**2 + FY**2)/f0[i])**(2*n))
        org_med_filt[..., i] = np.fft.ifftshift(np.fft.ifft2(butt_filt * ft_med)) / shift
        
    
    ax.plot(org_med[nx/2, :])
    org_med_filt_crop = org_med_filt[dif_x:nx_pad - dif_x, dif_x : nx_pad - dif_x, :]
    for i in range(4):
        ax.plot(org_med_filt_crop[nx/2, :, i])

    ### test for optimal filter
    """variables are
    n, f0,
    args contain
    orig, mask, FX, FY, Zernike_2D, xy_inside, ft_med, nx, nx_pad, j_max, N, Z_mat"""
    dx_opt = 2.0/nx
    x_opt = np.arange(-1, 1, dx_opt)
    xx_opt, yy_opt = np.meshgrid(x_opt, x_opt)
    xy_inside = np.where(np.sqrt(xx_opt**2 +yy_opt**2) <= 1)
    x_in, y_in = xx_opt[xy_inside], yy_opt[xy_inside]
    Zernike_3d = Zn.Zernike_xy(x_in, y_in, power_mat, j) ## create a 3d matrix with values of zernike polynomials at x_in and y_in
    Zernike_2d = np.zeros((len(x_in), j_max)) ## flatten to perform least squares fit
    for i in range(len(j)):
        Zernike_2d[:, i] = Zernike_3d[...,i].flatten()

    a_phase_filt_dict = np.load('coeff_dictionary.npy').reshape(1,)[0]
    a_phase_filt_int = a_phase_filt_dict['coeff_inter']
    flipint = False
    piston = opt.fmin(rms_piston, 0, args = (j_max, a_phase_filt_int, N, Z_mat, orig, mask, flipint)) 
    varlist = (orig, mask, FX, FY, Zernike_2d, xy_inside, ft_med, nx, nx_pad, j_max, N, Z_mat, piston)

    filter_freqs = np.logspace(0, 3, 5)
    f, ax = plt.subplots(1,5)
    for i in range(5):
        filtered_a_fit = a_fit_filtered((5, filter_freqs[i]), orig, mask, FX, FY, Zernike_2d, xy_inside, ft_med, nx, nx_pad, j_max, N, Z_mat, piston)
        piston = opt.fmin(rms_piston, 0, args = (j_max, filtered_a_fit, N, Z_mat, orig, mask, flipint)) 
        Zn.plot_interferogram(j_max, filtered_a_fit, piston = piston, ax= ax[i])
        ax[i].set_title(r'$f_0 = $' + str(filter_freqs[i]))
    plt.show()    
