import numpy as np
import matplotlib.pyplot as plt
import Zernike as Zn
import PIL.Image
import scipy.optimize as opt
import detect_peaks as dp
from mpl_toolkits.mplot3d import Axes3D
import phase_unwrapping_test as pw


def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

folder_name = "20170126_single_actuators/8/"
### predetermined constants
x0 = 550
y0 = 484
radius = int(280)
N = 2 * radius
j_max = 30
j_range = np.arange(2, j_max +2)
dpi_num = 600
golden = (1 + 5**0.5)/2.0

xi, yi = np.linspace(-1, 1, N), np.linspace(-1, 1, N)
xi, yi = np.meshgrid(xi, yi)

## gather all variables
##variables_dict = np.load(folder_name + "vars_dictionary.npy").item()
##coeff_dict= np.load(folder_name + "coeff_dictionary.npy").item()

##a_janss =coeff_dict['coeff_janss']
##a_lsq = coeff_dict['coeff_lsq']
##a_inter = coeff_dict['coeff_inter']

##pist_janss = variables_dict['vars_janss'][0]
##pist_lsq = variables_dict['vars_lsq'][0]
### uncomment when available
##pist_inter = variables_dict['pist_inter']
mask = [np.sqrt((xi) ** 2 + (yi) ** 2) >= 1]

inter_0 = np.array(PIL.Image.open(folder_name + "interferogram_0.tif"))
orig = np.ma.array(inter_0[y0-radius:y0+radius, x0-radius:x0+radius], mask = mask)


## pad for filtering
[ny, nx] = orig.shape
res = [2**i for i in range(15)]
nx_pad = np.where( res > np.tile(nx, len(res)))
nx_pad = res[nx_pad[0][0]]
dif_x = (nx_pad - int(nx))/2
orig_pad = np.lib.pad(orig, dif_x, 'reflect')
orig_filt = pw.butter_filter_unwrapped(orig_pad, 2, 15)
orig_filt = orig_filt[dif_x:nx_pad - dif_x, dif_x:nx_pad - dif_x]
orig_filt = np.ma.array(orig_filt, mask = mask)
orig_filt /= orig_filt.max()
orig /= orig.max()


power_mat = Zn.Zernike_power_mat(j_max+2)
Z_mat = Zn.Zernike_xy(xi, yi, power_mat, j_range)
flipint = False

x_search = 150
y_search = 150
safety_fact = 1.25


##Intens_inter = (Zn.imshow_interferogram(j_max, a_inter, N, piston = pist_inter))
x_peaks = dp.detect_peaks(orig_filt[x_search, :], mpd = 100)
x_peak_i = orig_filt[x_search, x_peaks]
y_peaks = dp.detect_peaks(orig_filt[:, y_search], mpd = 50)
y_peak_i = orig_filt[y_peaks, y_search]

x_plot = np.linspace(0, N, N)
f, ax = plt.subplots(2,2)
ax[0,0].plot(orig_filt[x_search, :])
ax[0,0].plot(orig[x_search, :], 'r-')
ax[0,0].scatter(x_peaks, orig_filt[x_search, x_peaks])

ax[0,1].plot(orig_filt[:, y_search])
ax[0,1].plot(orig[:, y_search], 'r-')
ax[0,1].scatter(y_peaks, orig_filt[y_peaks, y_search])

n_x = len(x_peaks)
mean_x = np.sum(x_peaks*x_peak_i)/n_x                  #estimate mean
sigma_x = np.sum(x_peak_i*(x_peaks-mean_x)**2)/n_x**4        #note this correction
x_fit, x_cov = opt.curve_fit(gaus, x_peaks, x_peak_i, p0 = [1, mean_x, sigma_x])

n_y = len(y_peaks)
mean_y = np.sum(y_peaks*y_peak_i)/n_y                  #estimate mean
sigma_y = np.sum(y_peak_i*(y_peaks-mean_y)**2)/n_y**4       #note this correction
y_fit, y_cov = opt.curve_fit(gaus, y_peaks, y_peak_i, p0 = [1, mean_y, sigma_y])



x, y = np.linspace(0, N, N), np.linspace(0, N, N)
xx, yy = np.meshgrid(x, y)
gaus2d = safety_fact * gaus(xx, *x_fit) * gaus(yy, *y_fit)
orig_scale = orig / gaus2d
orig_scale[np.where(orig_scale > 1.01)] = 1


ax[0,0].plot(x_plot, gaus(x_plot, *x_fit), 'r--')
ax[0,1].plot(x_plot, gaus(x_plot, *y_fit), 'r--')
#ax[1,0].plot(Intens_inter[N/2, :])
#ax[1,1].plot(Intens_inter[:, N/2])
ax[1,0].plot(orig_scale[N/2, :])
ax[1,1].plot(orig_scale[:, N/2])

bin_hist = np.linspace(0, 1, 30)
width_hist = bin_hist[2]-bin_hist[1]
weight_mask = np.asarray(~mask[0], dtype = np.int32)
hist, bin_edges = np.histogram(orig_scale, bins = bin_hist, weights = weight_mask)

f = plt.figure()
ax = f.add_subplot(121)
ax2 = f.add_subplot(122)
ax.imshow(np.ma.array(orig_scale, mask = mask), vmin = 0, vmax = 1, cmap = 'gray')
ax2.bar(bin_edges[:-1], hist, width = width_hist)

plt.show()
PIL.Image.fromarray(orig_scale).save(folder_name + "orig_scale.tif")
