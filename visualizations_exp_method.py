import numpy as np
import phase_unwrapping_test as pw
import matplotlib.pyplot as plt
import scipy.ndimage as img
from matplotlib import rc


# Define font for figures
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size' : 7})
rc('text', usetex=True)

## plot making parameters
dpi_num = 600
int_im_size = (4.98, 3.07)
int_im_size_23 = (0.66 * 4.98, 3.07)
int_im_size_13 = (0.33 * 4.98, 3.07)
gold = (1 + np.sqrt(5))/2
folder = "20140214_post_processing/"

###
nx = 480
ny = 480
x, y = np.linspace(-1, 1, nx), np.linspace(-1, 1, ny)
xx, yy = np.meshgrid(x, y)
i, j = np.arange(nx), np.arange(ny)
ii, jj = np.meshgrid(i, j)
N = nx
phase = 15 * xx - 15 * yy
phase_plt = np.copy(phase)
phase += np.random.normal(0, 0.3, (nx, ny))
wr_phase = pw.wrap_function(phase)

wr_phase_mean = img.filters.uniform_filter(wr_phase, 3)
wr_phase_spec = pw.filter_wrapped_phase(wr_phase, 3)

unwr_phase_mean = pw.unwrap_phase_dct(wr_phase_mean, xx, yy, ii, jj, N, N)
unwr_phase_spec = pw.unwrap_phase_dct(wr_phase_spec, xx, yy, ii, jj, N, N)

f, ax = plt.subplots(1, 3, sharex = True, sharey = True, figsize =int_im_size)
b_gold = np.pi/gold
titles = [r'a)', r'b)', r'c)']
for axes in ax:
    axes.plot([-1,1], [np.pi, np.pi], 'k--', linewidth = 0.5)
    axes.plot([-1, 1], [-np.pi, -np.pi], 'k--', linewidth = 0.5)
    axes.set_yticks([-np.pi, np.pi])
    axes.set_yticklabels([r'$-\pi$', r'$\pi$'])
    axes.yaxis.labelpad = -5
    axes.set_ylim([-np.pi-b_gold, np.pi+b_gold])
    axes.set(adjustable = 'box-forced', aspect = 1./(np.pi+b_gold))
    axes.tick_params(axis = 'both', which = 'major', labelsize = 7)

for i in range(3):
    ax[i].set_title(titles[i], fontsize = 8, loc = 'left')
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$\mathcal{W}(\varphi)$')
ax[0].plot(x, wr_phase[:,N/2], 'o-', markersize= 2.5)
ax[1].plot(x, wr_phase_mean[:,N/2], 'o-', markersize= 2.5)
ax[2].plot(x, wr_phase_spec[:,N/2], 'o-', markersize= 2.5)
f.savefig(folder + "wrapped_phase_filtering.png", dpi = dpi_num, bbox_inches = 'tight')
f2, ax2 = plt.subplots(1,1)
ax2.plot(x, phase_plt[:,N/2])
ax2.plot(x, unwr_phase_mean[:,N/2], 'go-', markersize = 3)
ax2.plot(x, unwr_phase_spec[:,N/2], 'k')
plt.show()
