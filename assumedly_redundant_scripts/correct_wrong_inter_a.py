import PIL.Image
import Hartmann as Hm
import displacement_matrix as Dm
import Zernike as Zn
import mirror_control as mc
import matplotlib.pyplot as plt
import phase_extraction as PE
import phase_unwrapping_test as pw
import scipy.optimize as opt
from matplotlib import rc
import numpy as np

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

# Define font for figures
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size' : 7})
rc('text', usetex=True)

## Given paramters for centroid gathering
px_size_sh = 5.2e-6     # width of pixels
px_size_int = 5.2e-6
f_sh = 17.6e-3            # focal length
r_int_px = 280
r_sh_px = 280
r_sh_m = r_sh_px * px_size_int
j_max= 30
j_range = np.arange(2, j_max +2)# maximum fringe order
wavelength = 632e-9 #[m]
box_len = 35.0 #half width of subaperture box in px

## plot making parameters
dpi_num = 600
int_im_size = (4.98, 3.07)
int_im_size_23 = (0.66 * 4.98, 3.07)
int_im_size_13 = (0.33 * 4.98, 3.07)

## centre and radius of interferogam. Done by eye, with the help of define_radius.py
x0 = 550
y0 = 484
radius = int(r_int_px)
N = 2 * radius

folder_name = "20170116_defocus/"
butter_unwr = np.load(folder_name + "filtered_phase.npy")

xi, yi = np.linspace(-1, 1, N), np.linspace(-1, 1, N)
xi, yi = np.meshgrid(xi, yi)

mask = [np.sqrt((xi) ** 2 + (xi) ** 2) >= 1]
mask_tile = np.tile(mask, (butter_unwr.shape[-1],1,1)).T
butter_mask = np.ma.array(butter_unwr, mask = mask_tile)
mean_butt = butter_mask.mean(axis=(0,1))
butter_unwr -= mean_butt

but_med = np.median(butter_unwr, axis = 2)
xy_inside = np.where(np.sqrt(xi**2 + yi**2) <= 1)
but_med_flat = but_med[xy_inside]
x_in, y_in = xi[xy_inside], yi[xy_inside]

j = np.arange(2, j_max) #start at 2, end at j_max-1
power_mat = Zn.Zernike_power_mat(j_max+2)
Zernike_3d = Zn.Zernike_xy(x_in, y_in, power_mat, j) ## create a 3d matrix with values of zernike polynomials at x_in and y_in 
Zernike_2d = np.zeros((len(x_in), j_max)) ## flatten to perform least squares fit
for i in range(len(j)):
    Zernike_2d[:, i] = Zernike_3d[...,i].flatten()

a_inter = np.linalg.lstsq(Zernike_2d, but_med_flat)[0]
Z_mat = Zn.Zernike_xy(xi, yi, power_mat, j_range)

inter_0 = np.array(PIL.Image.open(folder_name + "interferogram_0.tif"))
orig = np.ma.array(inter_0[y0-radius:y0+radius, x0-radius:x0+radius], mask = mask)
orig /= orig.max()

flipint = False
piston, inter_rms = opt.fmin(rms_piston, 0, args = (j_max, a_inter, N, Z_mat, orig, mask, flipint), full_output = True)[:2]

coeff_dict = np.load(folder_name+"coeff_dictionary.npy").item()
coeff_dict['coeff_inter'] = a_inter
np.save(folder_name + "coeff_dictionary.npy", coeff_dict)

vars_dict = np.load(folder_name+"vars_dictionary.npy").item()
vars_dict['pist_inter']= piston
np.save(folder_name + "vars_dictionary.npy", vars_dict)
