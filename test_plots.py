import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import Zernike as Zn

fig = plt.figure(figsize = (8,8))

j_max = 10
a = np.zeros(10)
a_x = np.arange(1, 11)
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

plt.show()
