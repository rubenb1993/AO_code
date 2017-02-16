import numpy as np
import phase_unwrapping_test as pw
import matplotlib.pyplot as plt
import scipy.ndimage as img
from matplotlib import rc
import Zernike as Zn
import PIL.Image 
from mpl_toolkits.mplot3d import Axes3D



# Define font for figures
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size' : 7})
rc('text', usetex=True)

## plot making parameters
dpi_num = 600
int_im_size = (4.98, 3.07)
int_im_size_23 = (0.66 * 4.98, 3.07)
int_im_size_13 = (0.33 * 4.98, 3.07)
gold = (1 + np.sqrt(5))/2
folder = "AO_code/20140214_post_processing/"
#
#### Uncomment for wrapped phase plots
#nx = 480
#ny = 480
#x, y = np.linspace(-1, 1, nx), np.linspace(-1, 1, ny)
#xx, yy = np.meshgrid(x, y)
#i, j = np.arange(nx), np.arange(ny)
#ii, jj = np.meshgrid(i, j)
#N = nx
#phase = 15 * xx - 15 * yy
#phase_plt = np.copy(phase)
#phase += np.random.normal(0, 0.3, (nx, ny))
#wr_phase = pw.wrap_function(phase)
#
#wr_phase_mean = img.filters.uniform_filter(wr_phase, 3)
#wr_phase_spec = pw.filter_wrapped_phase(wr_phase, 3)
#
#unwr_phase_mean = pw.unwrap_phase_dct(wr_phase_mean, xx, yy, ii, jj, N, N)
#unwr_phase_spec = pw.unwrap_phase_dct(wr_phase_spec, xx, yy, ii, jj, N, N)
#
#f, ax = plt.subplots(1, 3, sharex = True, sharey = True, figsize =int_im_size)
#b_gold = np.pi/gold
#titles = [r'a)', r'b)', r'c)']
#for axes in ax:
#    axes.plot([-1,1], [np.pi, np.pi], 'k--', linewidth = 0.5)
#    axes.plot([-1, 1], [-np.pi, -np.pi], 'k--', linewidth = 0.5)
#    axes.set_yticks([-np.pi, np.pi])
#    axes.set_yticklabels([r'$-\pi$', r'$\pi$'])
#    axes.yaxis.labelpad = -5
#    axes.set_ylim([-np.pi-b_gold, np.pi+b_gold])
#    axes.set(adjustable = 'box-forced', aspect = 1./(np.pi+b_gold))
#    axes.tick_params(axis = 'both', which = 'major', labelsize = 7)
#
#for i in range(3):
#    ax[i].set_title(titles[i], fontsize = 8, loc = 'left')
#ax[0].set_xlabel(r'$x$')
#ax[0].set_ylabel(r'$\mathcal{W}(\varphi)$')
#ax[0].plot(x, wr_phase[:,N/2], 'o-', markersize= 2.5)
#ax[1].plot(x, wr_phase_mean[:,N/2], 'o-', markersize= 2.5)
#ax[2].plot(x, wr_phase_spec[:,N/2], 'o-', markersize= 2.5)
#f.savefig(folder + "wrapped_phase_filtering.png", dpi = dpi_num, bbox_inches = 'tight')
#f2, ax2 = plt.subplots(1,1, figsize = int_im_size_23)
#ax2.plot(x, phase_plt[:,N/2], label = 'orignal phase')
#ax2.plot(x, unwr_phase_mean[:,N/2], 'g-', label = 'uniform filter')
#ax2.plot(x, unwr_phase_spec[:,N/2], 'k', label = 'wrapped phase filter')
#ax2.set_xlabel(r'$x$')
#ax2.set_ylabel(r'$\varphi$')
#ax2.legend(fontsize = 6)
#ax2.set(adjustable = 'box-forced', aspect = 1070./(2620 * 15)) 
#f2.savefig(folder + "unwrapped_filt_phase.png", dpi = dpi_num, bbox_inches = 'tight')
#plt.show()

j_max = 8
a_def = np.zeros(j_max)
a_def[2] = -3
a_ast = np.zeros(j_max)
a_ast[4] = 2.5
a_com = np.zeros(j_max)
a_com[6] = 2.

a_stack = np.stack((a_def, a_ast, a_com)).T

fig = plt.figure()
for i in range(3):
    ax = fig.add_subplot('23' + str(i+4))
    Zn.imshow_interferogram(j_max, a_stack[...,i], N = 600, ax = ax)
    ax3d = fig.add_subplot('23' + str(i+1), projection = '3d')
    Zn.plot_zernike(j_max, a_stack[...,i], ax = ax3d)
plt.show()