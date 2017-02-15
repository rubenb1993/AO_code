import numpy as np

xx = np.diag([1.,2.,3.])
xx = np.tile(xx, (3, 1, 1)).T
xx = np.fliplr(xx)
print(xx)
