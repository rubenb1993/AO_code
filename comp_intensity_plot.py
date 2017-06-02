import numpy as np
import Zernike as Zn
import matplotlib.pyplot as plt
import json
from matplotlib.ticker import ScalarFormatter

x = np.linspace(0, 2000, 10)

j_lsq = 2 + x/3 + 2  * Zn.Zernike_j_2_nm(x, ordering = 'brug')[0]
j_janss = x/3

plt.plot(x, 6 * x * j_janss, 'go-', label = 'Janssen')
plt.plot(x, 2 * x * j_lsq, '2b-', label = 'LSQ')
plt.xlabel(r'$n_{spots}$')
plt.ylabel(r'Computational intensity [a.u.]')
plt.title(r'Order of computational intensity assuming $J = \frac{n_{spots}}{3}$')
plt.legend()
plt.show()
