"""Quick script to quickly look at the nearest neighbour distance. Here not only
nearest neighbours are calculated, but all distances. By varying the range in the
histogram one can look at the lowest amount, those are the nearest neighbours.
"""
import numpy as np
import PIL.Image as pil
import Hartmann as Hm
import matplotlib.pyplot as plt

slm_sh = pil.open('SLM_codes_matlab/20170504_5_1/zero_pos_dm.tif')

x_pos_zero, y_pos_zero = Hm.zero_positions(np.copy(slm_sh))
triu_ind = np.triu_indices(len(x_pos_zero), 1)
box_lens = np.sqrt((x_pos_zero[triu_ind[0]] - x_pos_zero[triu_ind[1]])**2 + (y_pos_zero[triu_ind[0]] - y_pos_zero[triu_ind[1]])**2)
plt.hist(box_lens, bins = 100, range=(0, 120))
plt.show()
