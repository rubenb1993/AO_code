import numpy as np
import PIL.Image
import janssen
import Zernike as Zn
import Hartmann as Hm
import PIL
import mirror_control as mc
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rms_janss(variables, *args):
    """calculate the rms value of the Janssen method with variables:
    variables[0] = piston,
    variables[1] = centre_x,
    variables[2] = centre_y,
    variables[3] = radius of circle on sh sensor[px],
    Z_mat is used to calculate the Zernike polynomial on a grid, to compare with original interferogram.
    arguments should be in order"""
    if args:
       x_pos_zero_f, y_pos_zero_f, image_control, dist_image, px_size_sh, f_sh, j_max, wavelength, xx, yy, N, orig, mask, Z_mat, box_len, integrate = args
    else:
        print("put the arguments!")
        return
    x_pos_flat_f, y_pos_flat_f = Hm.centroid_positions(x_pos_zero, y_pos_zero, image_control, xx, yy)

    r_sh_m = px_size_sh * variables[3]

    x_pos_norm = (x_pos_flat_f - variables[1])/variables[3]
    y_pos_norm = (y_pos_flat_f - variables[2])/variables[3]

    ### check if all is within circle
    inside = np.sqrt(x_pos_norm ** 2 + y_pos_norm**2) <= (1+ (box_len/variables[3]))
    x_pos_norm, y_pos_norm, x_pos_flat_f,y_pos_flat_f = mc.filter_positions(inside, x_pos_norm, y_pos_norm, x_pos_flat_f, y_pos_flat_f)

    ## calculate slopes
    x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat_f, y_pos_flat_f, dist_image, xx, yy)
    dWdx, dWdy = Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat_f, y_pos_flat_f, px_size_sh, f_sh, r_sh_m, wavelength)

    ## make zernike matrix
    kmax = np.power(np.ceil(np.sqrt(j_max)),2) #estimation of maximum fringe number
    n, m = Zn.Zernike_j_2_nm(np.array(range(1, int(kmax)+1))) #find n and m pairs for maximum fringe number
    Kmax = np.max(Zn.Zernike_nm_2_j(n+1, np.abs(m)+1)) #find highest order of j for which beta is needed
    if integrate == True:
        Z_mat_complex = janssen.avg_complex_zernike(x_pos_norm, y_pos_norm, Kmax, r_sh_px, spot_size = 35)
    else:    
        Z_mat_complex = Zn.complex_zernike(Kmax, x_pos_norm, y_pos_norm)

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
    inter = np.ma.array(Zn.int_for_comp(j_max, a_janss, N, variables[0], Z_mat, fliplr = True), mask = mask)
    rms = np.sqrt(np.sum((inter - orig)**2)/N**2)
    return rms

def rms_janss_int(variables, *args):
    """calculate the rms value of the Janssen method with variables:
    variables[0] = piston,
    variables[1] = centre_x,
    variables[2] = centre_y,
    variables[3] = radius of circle on sh sensor[px],
    Z_mat is used to calculate the Zernike polynomial on a grid, to compare with original interferogram.
    arguments should be in order"""
    if args:
       x_pos_zero_f, y_pos_zero_f, image_control, dist_image, px_size_sh, f_sh, j_max, wavelength, xx, yy, N, orig, mask, Z_mat, box_len = args
    else:
        print("put the arguments!")
        return
    x_pos_flat_f, y_pos_flat_f = Hm.centroid_positions(x_pos_zero, y_pos_zero, image_control, xx, yy)

    r_sh_m = px_size_sh * variables[3]

    x_pos_norm = (x_pos_flat_f - variables[1])/variables[3]
    y_pos_norm = (y_pos_flat_f - variables[2])/variables[3]

    ### check if all is within circle
    inside = np.sqrt(x_pos_norm ** 2 + y_pos_norm**2) <= (1+ (box_len/variables[3]))
    x_pos_norm, y_pos_norm, x_pos_flat_f,y_pos_flat_f = mc.filter_positions(inside, x_pos_norm, y_pos_norm, x_pos_flat_f, y_pos_flat_f)

    ## calculate slopes
    x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat_f, y_pos_flat_f, dist_image, xx, yy)
    dWdx, dWdy = Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat_f, y_pos_flat_f, px_size_sh, f_sh, r_sh_m, wavelength)

    ## make zernike matrix
    kmax = np.power(np.ceil(np.sqrt(j_max)),2) #estimation of maximum fringe number
    n, m = Zn.Zernike_j_2_nm(np.array(range(1, int(kmax)+1))) #find n and m pairs for maximum fringe number
    Kmax = np.max(Zn.Zernike_nm_2_j(n+1, np.abs(m)+1)) #find highest order of j for which beta is needed
    Z_mat_complex = janssen.avg_complex_zernike(x_pos_norm, y_pos_norm, Kmax, r_sh_px, spot_size = 35)

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
    inter = np.ma.array(Zn.int_for_comp(j_max, a_janss, N, variables[0], Z_mat, fliplr = True), mask = mask)
    rms = np.sqrt(np.sum((inter - orig)**2)/N**2)
    return rms

def a_from_vars_int(variables, *args):
    if args:
       x_pos_zero_f, y_pos_zero_f, image_control, dist_image, px_size_sh, f_sh, j_max, wavelength, xx, yy, N, orig, mask, Z_mat, box_len, integrate = args
    else:
        print("put the arguments!")
        return
    x_pos_flat_f, y_pos_flat_f = Hm.centroid_positions(x_pos_zero, y_pos_zero, image_control, xx, yy)

    r_sh_m = px_size_sh * variables[3]

    x_pos_norm = (x_pos_flat_f - variables[1])/variables[3]
    y_pos_norm = (y_pos_flat_f - variables[2])/variables[3]

    ### check if all is within circle
    inside = np.sqrt(x_pos_norm ** 2 + y_pos_norm**2) <= (1+ (box_len/variables[3]))
    x_pos_norm, y_pos_norm, x_pos_flat_f,y_pos_flat_f = mc.filter_positions(inside, x_pos_norm, y_pos_norm, x_pos_flat_f, y_pos_flat_f)

    ## calculate slopes
    x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat_f, y_pos_flat_f, dist_image, xx, yy)
    dWdx, dWdy = Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat_f, y_pos_flat_f, px_size_sh, f_sh, r_sh_m, wavelength)

    ## make zernike matrix
    kmax = np.power(np.ceil(np.sqrt(j_max)),2) #estimation of maximum fringe number
    n, m = Zn.Zernike_j_2_nm(np.array(range(1, int(kmax)+1))) #find n and m pairs for maximum fringe number
    Kmax = np.max(Zn.Zernike_nm_2_j(n+1, np.abs(m)+1)) #find highest order of j for which beta is needed
    if integrate == True:
        Z_mat_complex = janssen.avg_complex_zernike(x_pos_norm, y_pos_norm, Kmax, r_sh_px, spot_size = 35)
    else:
        Z_mat_complex = Zn.complex_zernike(Kmax, x_pos_norm, y_pos_norm)
        
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

    return a_janss

## plot making parameters
dpi_num = 600
int_im_size = (4.98, 3.07)
int_im_size_23 = (0.66 * 4.98, 3.07)
int_im_size_13 = (0.33 * 4.98, 3.07)
fold_name = "20161213_new_inters/"
folder_name = fold_name
## Given paramters for centroid gathering
px_size_sh = 5.2e-6     # width of pixels
px_size_int = 5.2e-6
f_sh = 17.6e-3            # focal length
r_int_px = 310
r_sh_px = 320
r_sh_m = r_sh_px * px_size_int
j_max= 30         # maximum fringe order
wavelength = 632e-9 #[m]
box_len = 35.0 #half width of subaperture box in px
radius = int(310)
N = 2*radius

image_i0 = np.array(PIL.Image.open(folder_name + "interferogram_0.tif"))
inter_0 = image_i0
image_ref_mirror = np.array(PIL.Image.open(folder_name + "image_ref_mirror.tif"))
zero_pos_dm = np.array(PIL.Image.open(folder_name + "zero_pos_dm.tif"))
dist_image = np.array(PIL.Image.open(folder_name + "dist_image.tif"))

x0 = 550
y0 = 484
radius = int(310)

zero_image_zeros = np.copy(image_ref_mirror)
image_control = image_ref_mirror

[ny,nx] = image_control.shape
x = np.linspace(1, nx, nx)
y = np.linspace(1, ny, ny)
xx, yy = np.meshgrid(x, y)
x_pcg, y_pcg = np.linspace(-1, 1, 2*radius), np.linspace(-1, 1, 2*radius)
xx_alg, yy_alg = np.meshgrid(x_pcg, y_pcg)
i, j = np.linspace(0, N-1, N), np.linspace(0, N-1, N)
ii, jj = np.meshgrid(i, j)

mask = [np.sqrt((xx_alg) ** 2 + (yy_alg) ** 2) >= 1]
orig = np.ma.array(inter_0[y0-radius:y0+radius, x0-radius:x0+radius], mask = mask)
orig /= np.max(orig)

j_range = np.arange(2, j_max+2)
power_mat = Zn.Zernike_power_mat(j_max+2)
Z_mat = Zn.Zernike_xy(xx_alg, yy_alg, power_mat, j_range)


x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image_zeros)
x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, image_control, xx, yy)
centre = Hm.centroid_centre(x_pos_flat, y_pos_flat)
x_pos_norm = ((x_pos_flat - centre[0]))/r_sh_px
y_pos_norm = ((y_pos_flat - centre[1]))/r_sh_px
inside = np.where(np.sqrt(x_pos_norm**2 + y_pos_norm**2) <= (1 + (box_len/r_sh_px))) #35 is the half the width of the pixel box aroudn a centroid and r_sh_px is the scaling factor
x_pos_zero_f, y_pos_zero_f, x_pos_flat_f, y_pos_flat_f, x_pos_norm_f, y_pos_norm_f = mc.filter_positions(inside, x_pos_zero, y_pos_zero, x_pos_flat, y_pos_flat, x_pos_norm, y_pos_norm)
integrate = False
inside_without_outsides = np.where(np.sqrt(x_pos_norm**2 + y_pos_norm**2) <= 1 - (box_len/r_sh_px))
x_pos_norm_f_wth, y_pos_norm_f_wth = mc.filter_positions(inside_without_outsides, x_pos_norm, y_pos_norm)

janss_args = (x_pos_zero_f, y_pos_zero_f, image_control, dist_image, px_size_sh, f_sh, j_max, wavelength, xx, yy, N, orig, mask, Z_mat, box_len, integrate)

x_norm_mat, y_norm_mat = np.meshgrid(x_pos_norm_f, y_pos_norm_f)

Z_mat_com_int = janssen.avg_complex_zernike(x_pos_norm_f, y_pos_norm_f, 200, 310)
Z_mat_com_non = Zn.complex_zernike(200, x_pos_norm_f, y_pos_norm_f)
Z_mat_int_wth, Z_mat_non_wth = janssen.avg_complex_zernike(x_pos_norm_f_wth, y_pos_norm_f_wth, 200, 310), Zn.complex_zernike(200, x_pos_norm_f_wth, y_pos_norm_f_wth)

rms_mat = np.sqrt((Z_mat_com_int - Z_mat_com_non)**2)/len(x_pos_norm_f)
rms = np.sum(rms_mat, axis = 0)
rms_mat_wth = np.sqrt((Z_mat_int_wth - Z_mat_non_wth)**2)/len(x_pos_norm_f_wth)
rms_wth = np.sum(rms_mat_wth, axis = 0)
#plot_z = Z_mat_com[:, 98].reshape(x_norm_mat)

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.scatter(x_pos_norm_f, y_pos_norm_f, rms_mat[:,189])
ax.set_zlim(bottom= 0)
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')
ax.set_zlabel(r'rms error')
ax.set_title(r'allowing outside spots')
ax3 = fig.add_subplot(122, projection = '3d')
ax3.scatter(x_pos_norm_f_wth, y_pos_norm_f_wth, rms_mat_wth[:,189])
ax3.set_zlim(bottom= 0)
ax3.set_xlabel(r'x')
ax3.set_ylabel(r'y')
ax3.set_zlabel(r'rms error')
ax3.set_title(r'not allowing outside spots')

fig.savefig(folder_name + "scatter_rms_j_189_with_outsides.png", bbox_inches = 'tight', dpi = dpi_num)
f2, ax2 = plt.subplots(1,2, sharey = True)
ax2[0].set_yscale('log')
ax2[0].plot(rms[1:])
ax2[0].set_xlabel(r'fringe number $j$')
ax2[0].set_ylabel(r'rms error')
ax2[0].set_title(r'allowing outside spots')
ax2[1].plot(rms_wth[1:])
ax2[1].set_xlabel(r'fringe number $j$')
ax2[1].set_ylabel(r'rms error')
ax2[1].set_title(r'not allowing outside spots')
f2.savefig(folder_name + "rms_per_j_with_outsides.png", bbox_inches = 'tight', dpi = dpi_num)
plt.show()
##print("checking 1")
##vars_janss, janss_rms = opt.fmin(rms_janss, [2.5, centre[0], centre[1], r_sh_px], args = janss_args, full_output = True, maxiter = 1000)[:2]
##a_janss = a_from_vars_int(vars_janss, x_pos_zero_f, y_pos_zero_f, image_control, dist_image, px_size_sh, f_sh, j_max, wavelength, xx, yy, N, orig, mask, Z_mat, box_len, integrate)
####
####print("checking 2")
####vars_janss_int, janss_int_rms = opt.fmin(rms_janss_int, [2.5, centre[0], centre[1], r_sh_px], args = janss_args, full_output = True, maxiter = 1000)[:2]
##integrate = True
##vars_janss_int = np.load(folder_name + "var_janss_int.npy")
##a_janss_int = a_from_vars_int(vars_janss_int, x_pos_zero_f, y_pos_zero_f, image_control, dist_image, px_size_sh, f_sh, j_max, wavelength, xx, yy, N, orig, mask, Z_mat, box_len, integrate)
##f, ax = plt.subplots(1,2)
##Zn.plot_interferogram(j_max, a_janss, f = f, ax = ax[0])
##Zn.plot_interferogram(j_max, a_janss_int, f = f, ax = ax[1])
##plt.show()
