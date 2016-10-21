import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
import Zernike as Zn
import scipy.special as spec
import math
#import sympy.mpmath as mp

def cart2pol(x, y):
    "returns polar coordinates given x and y"
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def Zernike_nm(n, m, rho, theta):
    """Returns the values of the Zernike polynomial of radial and azimuthal order (n & m). 
    n and m should be integers,
    rho and theta np arrays with the same size
    out: array with values of at the points rho and theta"""
    if (n-np.abs(m)) %2 == 0 and (n >= 0) and (n >= np.abs(m)): #check if even
        s_max = int((n-abs(m))/2)
        radial = np.zeros(shape=rho.shape)
        for s in range(s_max + 1):
            prefactor = (-1)**s * math.factorial(n-s) / (math.factorial(s) * math.factorial((n+abs(m))/2 - s) * math.factorial((n-abs(m))/2 - s))
            radial += np.power(rho,(n - 2*s)) * prefactor
##        if m > 0:
##            angular = np.cos(m*theta)
##        elif m < 0:
##            angular = -1 * np.sin(m*theta)
##        else:
##            angular = 1
        angular = 1 #np.exp(1j * m * theta)
            
        return radial
        #return np.sqrt((2 - (m == 0)) * (n + 1)) * angular * radial
        #if m == 0:
        #    return np.sqrt(n+1) * angular * radial
        #else:
        #    return np.sqrt(2*n + 2) * angular * radial
    else:
        return np.zeros(shape=rho.shape)

def jacobi_pol(n, a, b, m, rho, rho2):
    xshape = list(rho.shape)
    xshape.append(len(n))
    xx = np.zeros(xshape)
    radial = np.zeros(xx.shape)
    for i in range(len(n)):
        nm = n[i]
        nm = int(nm)
        for j in range(nm+1):
            xx[:,i] += spec.comb(n[i]+a[i], j) * spec.comb(n[i]+b[i], n[i]-j) * np.power((rho2-1)/2.0, (n[i] - j)) * np.power((rho2+1)/2.0, j)
        radial[:,i] = np.power(rho, abs(m[i])) * xx[:,i]
        #radial[:,i] = np.power(-1, nm2[i]) * np.power(rho, m[i]) * xx[:,i]
    return radial


x, y = np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)
xx, yy = np.meshgrid(x, y)
j_max = 5
def complex_zernike(j_max, x, y):
    """Given the meshgrids for x and y, and given the maximum fringe order, complex zernike retursn
    an (len(x), len(y), j_max) sized matrix with values of the complex Zernike polynomial at the given points.
    THIS IS WITH THE NORMALISATION OF pi/(n+1)"""
    rho, theta = cart2pol(x, y)
    rho2 = 2 * rho**2 - 1
    j = np.arange(1, j_max+1)
    n, m = Zn.Zernike_j_2_nm(j)
    nm2 = (n - np.abs(m))/2
    xshape = list(rho.shape)
    xshape.append(j_max)
    Cnm = np.zeros(xshape, dtype = np.complex_)
    a = np.zeros(len(m))
    b = np.abs(m)
    for i in range(len(nm2)):
        nm = nm2[i]
        nm = int(nm)
        for jj in range(nm+1):
            print jj
            print nm2[i]
            Cnm[...,i] += spec.comb(nm+a[i], jj) * spec.comb(nm+b[i], nm-jj) * np.power((rho2-1)/2.0, (nm - jj)) * np.power((rho2+1)/2.0, jj)
        Cnm[...,i] = np.power(rho, abs(m[i])) * xx[:,i] * np.exp(1j * m[i] * theta)
    return Cnm


##def complex_zernike(j_max, x, y):
##    rho, theta = cart2pol(x, y)
##    rho2 = 2 * rho**2 - 1
##    j = np.arange(1, j_max+1)
##    n, m = Zn.Zernike_j_2_nm(j)
##    nm2 = (n - np.abs(m))/2
##    xshape = list(rho.shape)
##    xshape.append(j_max)
##    Cnm = np.zeros(xshape, dtype = np.complex_)
##    a = np.zeros(len(m))
##    b = np.abs(m)
##    for i in range(len(nm2)):
##        nm = nm2[i]
##        nm = int(nm)
##        for jj in range(nm+1):
##            print jj
##            print nm2[i]
##            Cnm[:,i] += spec.comb(nm2[i]+a[i], jj) * spec.comb(nm2[i]+b[i], nm2[i]-jj) * np.power((rho2-1)/2.0, (nm2[i] - jj)) * np.power((rho2+1)/2.0, jj)
##        Cnm[:,i] = np.power(rho, abs(m[i])) * xx[:,i] * np.exp(1j * m[i] * theta)
##    return Cnm

Cnm = complex_zernike(j_max, x, y)
print(Cnm.shape)

    
rho = np.linspace(-1, 1, 100)
rho2 = 2*rho**2 - 1
theta = np.zeros(rho.shape)
j = np.arange(2, 30)
n, m = Zn.Zernike_j_2_nm(j)
nm2 = (n - np.abs(m))/2
f, axarr = plt.subplots(14, 1, sharex = True, sharey = False)


R_janss = np.zeros((100, len(j)))
R_zern = np.zeros((100, len(j)))
list_of_poly = []
a = np.zeros(len(nm2))
R_janss = jacobi_pol(nm2, a, np.abs(m), m, rho, rho2)
for i in range(len(j)):
    #list_of_poly.append(spec.jacobi(nm2[i], 0, np.abs(m[i])))
    #poly_i = spec.jacobi(nm2[i], 0, np.abs(m[i]))
    #R_janss[:, i] = poly_i(rho2)
    if i%2 == 0:
        R_zern[:, i] = Zernike_nm(n[i], m[i], rho, theta)
        axarr[i/2].plot(rho, R_zern[:,i])
        axarr[i/2].plot(rho, R_janss[:,i])
        axarr[i/2].set_ylabel([str(n[i]), str(m[i])])

plt.show()
print(list_of_poly)
