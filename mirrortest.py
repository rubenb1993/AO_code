import numpy as np
import time

import edac40

mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here

voltages = 6.0 * np.ones(19)  # V = 0 to 12V
mirror.set(voltages)
time.sleep(5)

mirror.close()
