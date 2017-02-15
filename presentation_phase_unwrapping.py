import numpy as np
import phase_unwrapping_test as pu
import matplotlib.pyplot as plt

x, y = np.linspace(-1, 1, 500), np.linspace(-1, 1, 500)
xx, yy = np.meshgrid(x, y)
N = len(x)
i, j = np.linspace(0, N, N), np.linspace(0, N, N)
ii, jj = np.meshgrid(i, j)

phase = 12* xx - 11 * yy
wr_phase = pu.wrap_function(phase)

f, ax = plt.subplots(1,1, figsize = (4.98, 4.98/2.0))
ax.plot(x, phase[N/2, :], label = 'unwrapped phase')
ax.plot(x, wr_phase[N/2, :], label = 'wrapped phase')
ax.set_xlabel('x', fontsize = 5)
ax.set_ylabel('phase', fontsize = 5)
ax.legend(loc = 'upper center', ncol = 2, fontsize = 4.5)
ax.tick_params(axis = 'both', labelsize = 5)

f.savefig("presentation_unwrapping_example.png", dpi = 600, bbox_inches = 'tight')
plt.show()
