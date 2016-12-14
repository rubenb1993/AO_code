import numpy as np
import matplotlib.pyplot as plt

def phase_transf(Z):
    phase = np.mod(Z - np.pi, 2*np.pi) - np.pi
    return phase

def phase_old(Z):
    return np.abs(np.abs(Z) - np.floor(np.abs(Z))) * 2* np.pi

x = np.linspace(-3*np.pi, 3*np.pi, num = 1000)
y = phase_transf(x)
y2 = phase_old(x)

plt.plot(x, y, 'r-')
plt.plot(x, y2, 'c-')
plt.show()
