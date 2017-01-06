import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import Zernike as Zn

fig = plt.figure(figsize = (8,8))

j_max = 50
a = np.zeros(50)
a_x = np.arange(1, 51)
a[3] = 1
piston = 1

gs1 = gridspec.GridSpec(3,3, width_ratios=[4,4,0.2], height_ratios = [3,3,2])
ax1 = fig.add_subplot(gs1[0,0])
ax2 = fig.add_subplot(gs1[0,1])
ax3 = fig.add_subplot(gs1[1,0])
ax4 = fig.add_subplot(gs1[1,1])

ax5 = fig.add_subplot(gs1[2,:])
ax6 = fig.add_subplot(gs1[:2,2])

Zn.plot_interferogram(j_max, a, piston = piston, ax = ax1, f = fig)
Zn.plot_interferogram(j_max, a, piston = piston, ax = ax2, f = fig)
Zn.plot_interferogram(j_max, a, piston = piston, ax = ax3, f = fig)
interferogram = Zn.plot_interferogram(j_max, a, piston = piston, ax = ax4, f = fig)

ax5.scatter(a_x, a, marker = 's', color = 'k')
plt.colorbar(interferogram, cax= ax6)


f, axes = plt.subplots(nrows = 1, ncols = 5, figsize = (4.98, 4.98/4.4), gridspec_kw={'width_ratios':[4, 4, 4, 4, 0.4]})#, 'height_ratios':[1,1,1,1,1]})
titles = [r'original', r'Interferogram', r'LSQ', r'Janssen']
for i in range(4):
    a = np.zeros(50)  
    a[i+40] += 1
    interf = Zn.plot_interferogram(j_max, a, piston = piston, ax = axes[i], f = f)
    axes[i].set(adjustable = 'box-forced', aspect = 'equal')
    axes[i].get_xaxis().set_ticks([])
    axes[i].get_yaxis().set_ticks([])
    axes[i].set_title(titles[i], fontsize = 9)
cbar = plt.colorbar(interf, cax = axes[-1], ticks = [0.0, 0.25, 0.5, 0.75, 1.0])
cbar.ax.tick_params(labelsize=7)
#bar.ax.set_yticklabels([0.0, 0.25, 0.5, 0.75, 1.0
f.savefig('test_gridspec.png', dpi = 600, bbox_inches = 'tight')

plt.show()
