### This file contains general functions to work with Zernike polynomials and their derivatives. 

import numpy as np
from scipy import special
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.special as spec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import sys

def cart2pol(x, y):
    "returns polar coordinates given x and y"
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def max_n_to_j(n_max, order = None):
    """ Converts a maximum index or array of maximum indices to a dictionary containing
    argument = str(n_max)
    value = numpy array containing all numbers between 2 and the maximum single index)
    ordeer must be 'fringe' or 'brug'
    """
    if isinstance(n_max, int) or isinstance(n_max, float):
        n_max = np.array([n_max])
        m = np.copy(n_max)
        j = {}
        max_j = np.max([Zernike_nm_2_j(n_max[0], m[0], ordering = order), Zernike_nm_2_j(n_max[0], -1*m[0], ordering = order)])
        j[str(n_max[0])] = np.arange(2, max_j+1)
        return j
    else:
        m = np.copy(n_max)
        j = {}
        for ijk in range(len(n_max)):
            max_j = np.max([Zernike_nm_2_j(n_max[ijk], m[ijk], ordering = order), Zernike_nm_2_j(n_max[ijk], -1*m[ijk], ordering = order)]) 
            j[str(n_max[ijk])] = np.arange(2, max_j+1)
        return j

def convert_fringe_2_brug(a):
    """ Take a fringe ordered zernike coefficient vector a
    and return a brug ordered zernike coefficient vector
    """
    j = np.arange(2, len(a)+2)
    n, m = Zernike_j_2_nm(j, ordering = 'fringe')
    j_brug = Zernike_nm_2_j(n, m, ordering = 'brug')
    index_brug = j_brug - 2
    index_brug = index_brug.astype(int)
    a_brug = np.zeros(len(max_n_to_j(np.max(n), order = 'brug')[str(np.max(n))]))
    a_brug[index_brug] = a[j - 2]
    return a_brug

def Zernike_j_2_nm(j, ordering = None):
    """Converts Zernike index j (FRINGE convention) to index n and m. 
    Based on code by Dr. Ir. S. van Haver (who based it on Herbert Gross, Handbook of Optical Systems, page 215).
    input j: either scalar or an array (lists not supported)
    output n, m: either a scalar or an array of corresponding n and m.
    Built in additional check if j is integer. If not, n = -1 will be generated s.t. any zernike function will be 0"""
    try:
        if ordering == 'brug' or ordering == 'Brug':
            n = np.ceil(-1.5 + 0.5 * np.sqrt(1+ j*8)).astype(int)
            m = 2 * j - (n*(n+2)) - 2
            return n, m
        elif ordering == 'fringe' or ordering == 'Fringe':
            q = np.floor(np.sqrt(j-1.0))
            p = np.floor((j - q**2 - 1.0)/2.0)
            r = j - q**2 - 2*p
            n = q+p
            m = np.power(-1, r+1) * (q-p)
            return n, m
        else:
            raise ValueError('ordering is not correct. Use Brug or Fringe')
    except ValueError as err:
        print(err.args)
        return sys.exit(1)

def Zernike_nm_2_j(n, m, ordering = None):
    """Converts Zernike index j (FRINGE convention) to index n and m. 
    Based on code by Dr. Ir. S. van Haver (who based it on Herbert Gross, Handbook of Optical Systems, page 215).
    input n, m: either scalar or an array (lists not supported)
    output j: either a scalar or an array of corresponding j"""
    try:
        if ordering == 'brug' or ordering == 'Brug':
            p = (n*(n+2) + m)/2 + 1
            return p
        elif ordering == 'fringe' or ordering == 'Fringe':        
            p = (n + np.abs(m))/2.0
            return p**2 + n - np.abs(m) + 1 + (m<0)
        else:
            raise ValueError('ordering is not correct. Use Brug or Fringe')
    except ValueError as err:
        print(err.args)
        return sys.exit(1)
    
def Zernike_power_mat(n_max, order):
    """Will create a matrix containing the poewrs for cartesian representation of zernike polynomials.
    taken from H.H. van Brug. Efficient cartesian representation of zernike polynomials in computer memory
    will create a matrix such that all zernikes of order n_max are represented. Fringe order will contain
    more zernieks as Z_4^0 comes before Z_3^-3.

    n_max = int
    order = 'brug' or 'fringe'
    """
    try:
        if order == 'brug' or order == 'Brug':
            j = max_n_to_j(n_max, order = order)
            j_max = np.max(j[str(n_max)]).astype(int)
        elif order == 'fringe' or order == 'Fringe':
            j_temp = max_n_to_j(n_max, order = order)
            print(j_temp)
            n_temp, m_temp = Zernike_j_2_nm(j_temp[str(n_max)], ordering = order)
            j_max = np.max(j_temp[str(n_max)]).astype(int)
            n_max = np.max(n_temp).astype(int)

            #n_max = np.max([Zernike_j_2_nm(j, ordering = order)[0] for j in range(1,j_max+1)]).astype(int) #find maximum n as n of j_max might not be the maximum n
        else:
            raise ValueError('ordering is not correct. Use Brug or Fringe')
    except ValueError as err:
        print(err.args)
        return sys.exit(1)

    fmat = np.zeros((n_max+1, n_max+1, j_max-1)) #allocate memory
    for jj in range(j_max-1):
        n, m_not_brug = Zernike_j_2_nm(jj+2, ordering = order) #brug uses different m, see later defined
        l = -m_not_brug
        if ( l > 0):
            p = 1
            q =  l/2.0 + 0.5 * (n%2.0) - 1.0
        else:
            p = 0
            q = -l/2.0 - 0.5*(n%2)
            
        #make integers for range    
        q = int(q)
        m = int((n-abs(l)) / 2)

        norm = np.sqrt( (2 - (m_not_brug == 0)) * (n+1))
        for i in range(q+1):
            for j in range(m+1):
                for k in range(m-j+1):
                    fact = np.power(-1, i+j)
                    fact *= special.binom( np.abs(l), 2*i + p)
                    fact *= special.binom( m-j, k)
                    fact *= math.factorial(n-j) / ( math.factorial(j) * math.factorial(m-j) * math.factorial(n-m-j))
                    ypow = int(2 * (i+k) + p)
                    xpow = int(n - 2 * (i+j+k) - p)
                    fmat[xpow, ypow, jj] += fact * norm 
    return fmat
        
def Zernike_xy(x, y, power_mat, j):
    """Computes the value of Zj at points x and y
    in: x, y: coordinate matrices
    power_mat: the power matrix as made by Zernike_power_mat
    j the fringe order of your polynomial (should be same length as power_mat.shape[-1]
    out: Z, a matrix containing the values of Zj at points x and y
    
    Normalized s.t. int(|Z_j|^2) = pi/(n+1)"""
    #x_list, y_list = np.nonzero(power_mat[...,j-1])
    j = np.array(j)
    dim1 = list(x.shape)
    dim1.append(len(j))
    Z = np.zeros(dim1)
    for i in range(len(j)):
        x_list, y_list = np.nonzero(power_mat[...,j[i]-2])
        for jj in range(len(x_list)):
            Z[...,i] += power_mat[x_list[jj], y_list[jj], j[i]-2] * np.power(x, x_list[jj]) * np.power(y, y_list[jj])
    return Z
    
def xder_brug(x, y, power_mat, j):
    """Computes the value of dZj/dx at points x and y
    in: x, y: coordinate matrices
    power_mat: the power matrix as made by Zernike_power_mat
    j the fringe order of your polynomial
    out: dZdx, a matrix containing the values of dZjdx at points x and y
    
    Normalized s.t. int(|Z_j|^2) = pi"""
    power_mat = power_mat[1:, :, :]
    dim1 = list(x.shape)
    dim1.append(len(j))
    dZdx = np.zeros(dim1)
    for jj in range(len(j)):
        x_list, y_list = np.nonzero(power_mat[...,j[jj]-2])
        for i in range(len(x_list)):
            dZdx[...,jj] += (x_list[i]+1)*power_mat[x_list[i], y_list[i], j[jj]-2] * np.power(x, x_list[i]) * np.power(y, y_list[i])
    return dZdx
    
def yder_brug(x, y, power_mat, j):
    """Computes the value of dZj/dy at points x and y
    in: x, y: coordinate matrices
    power_mat: the power matrix as made by Zernike_power_mat
    j the fringe order of your polynomial
    out: Z, a matrix containing the values of dZj/dy at points x and y
    
    Normalized s.t. int(|Z_j|^2) = pi/(n+1)"""
    power_mat = power_mat[:, 1:, :]
    dim1 = list(x.shape)
    dim1.append(len(j))
    dZdy = np.zeros(dim1)
    for jj in range(len(j)):
        x_list, y_list = np.nonzero(power_mat[...,j[jj]-2]) #-1 due to j = 1 in power_mat[...,0]
        for i in range(len(x_list)):
            dZdy[...,jj] += (y_list[i]+1)*power_mat[x_list[i], y_list[i], j[jj]-2] * np.power(x, x_list[i]) * np.power(y, y_list[i])
    return dZdy
 
def imshow_interferogram(j_max, a, N, ax = None, f = None, wantcbar = False, piston = 0, cmap = 'bone', wavelength = 632.8e-9, fliplr = False, Z_mat = None, power_mat = None):
    """ Algorithm to quickly show simulated interferogram,
    based on highest zernike polynomial (j_max), zernike coefficient vector a and show it using and NxN grid
    Z_mat and power_mat should be provided for quick plotting
    """
    if ax is None:
        ax = plt.gca()
    xi, yi = np.linspace(-1, 1, N), np.linspace(-1, 1, N)
    xi, yi = np.meshgrid(xi, yi)
    j_range = np.arange(2, j_max+2)
    if power_mat is None:
        power_mat = Zernike_power_mat(j_max+2)
    if Z_mat is None:
        Z_mat = Zernike_xy(xi, yi, power_mat, j_range)
    Z = np.sum(a * Z_mat, axis = 2)
    Z += piston
    phase = np.mod(Z - np.pi, 2*np.pi) - np.pi
    Intens = np.cos(phase/2.0)**2
    if fliplr:
        Intens = np.fliplr(Intens)
    interferogram = ax.imshow(np.ma.masked_where(xi**2 + yi**2 >= 1, Intens), vmin = 0, vmax = 1, cmap = cmap, origin = 'lower', interpolation = 'none')
    if wantcbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(interferogram, cax=cax)
    return interferogram
    

def int_for_comp(j_max, a, N, piston, Z_mat, wavelength = 632.8e-9, fliplr = False):
    """ Returns N x N grid of simulated intensity (normalized to 1) using Z_mat and a. I think j_max is not used"""
    Z = np.zeros(list(Z_mat.shape)[0:2])
    Z = np.sum(a * Z_mat, axis = 2)
    #Z /= wavelength
    Z += piston  
    phase = np.mod(Z - np.pi, 2*np.pi) - np.pi
    Intens = np.cos(phase/2.0)**2
    if fliplr:
        Intens = np.fliplr(Intens)
    return Intens    
#### Old functions          
def xderZ(j, x, y):
    """Calculate the x derivative of a Zernike polynomial of FRINGE ordering j according to [1].
    j: scalar value for the order of the polynomial
    x, y: 1d vectors containing x and y positions of where the derivative has to be evaluated
    out: 1d vector size of x with the values of the x derivative in those points
    
    [1] P.C.L. Stephenson, "Recurrence relations for the cartesian derivatives of the Zernike polynomials",
                J. Opt. Soc. Am. A, Vol. 31, No. 4, 708-714 (2014) """
    ## initialize values 
    n, m = Zernike_j_2_nm(j)
    rho, phi = cart2pol(x, y)
    
    ## Initial values
    if n == 0:
        return np.zeros(x.shape)
    if n == 1:
        if m == 1:
            return 2*np.ones(x.shape)
        else:
            return np.zeros(x.shape)
    
    #check if n and m are valid values for Zernike polynomials
    elif (n - np.abs(m)) %2 == 0 and n >= 0 and n >= np.abs(m): 
        am = np.sign(np.sign(m)+0.5)
        bfact1 = np.sqrt((2.0 - (m == 0))*(n+1.0)) / np.sqrt((2.0 - ((m-1.0) == 0))*n)
        bfact2 = np.sqrt((2.0 - (m == 0))*(n+1.0)) / np.sqrt((2.0 - ((m+1.0) == 0))*n)
        bfact3 = np.sqrt((2.0 - (m == 0))*(n+1.0)) / np.sqrt((2.0 - (m == 0))*(n-1.0))
        n_new = n - 2.0
        
        
        #check if the new n and m values are valid values (only add recursion if new n and m make sense)
        if (n_new - np.abs(m)) %2 == 0 and n_new >= 1 and n_new >= np.abs(m):
            j_new = Zernike_nm_2_j(n_new, m)        
            xder = n * (bfact1 * Zernike_nm(n-1, am*np.abs(m-1), rho, phi) + \
                    am * np.sign(m+1.0) * bfact2 * Zernike_nm(n-1.0, am*np.abs(m+1.0), rho, phi)) + \
                    bfact3 * xderZ(j_new, x, y)
            return xder
        else:
            xder = n * (bfact1 * Zernike_nm(n-1, am*np.abs(m-1), rho, phi) + \
                    am * np.sign(m+1.0) * bfact2 * Zernike_nm(n-1.0, am*np.abs(m+1.0), rho, phi))
            return xder
    else:
        return np.zeros(x.shape)
        
def yderZ(j, x, y):
    """Calculate the y derivative of a Zernike polynomial of FRINGE ordering j according to [1].
    j: scalar value for the order of the polynomial
    x, y: 1d vectors containing x and y positions of where the derivative has to be evaluated
    out: 1d vector size of x with the values of the x derivative in those points
    
    [1] P.C.L. Stephenson, "Recurrence relations for the cartesian derivatives of the Zernike polynomials",
                J. Opt. Soc. Am. A, Vol. 31, No. 4, 708-714 (2014) """
    ## initialize values 
    n, m = Zernike_j_2_nm(j)
    rho, phi = cart2pol(x, y)
    
    ## Initial values
    if n == 0:
        return np.zeros(x.shape)
    if n == 1:
        if m == -1:
            return 2.0*np.ones(x.shape)
        else:
            return np.zeros(x.shape)
    
    #check if n and m are valid values for Zernike polynomials
    elif (n - np.abs(m)) %2 == 0 and n >= 0 and n >= np.abs(m): 
        am = np.sign(np.sign(m)+0.5)
        bfact1 = np.sqrt((2.0 - (m == 0))*(n+1.0)) / np.sqrt((2.0 - ((m-1.0) == 0))*n)
        bfact2 = np.sqrt((2.0 - (m == 0))*(n+1.0)) / np.sqrt((2.0 - ((m+1.0) == 0))*n)
        bfact3 = np.sqrt((2.0 - (m == 0))*(n+1.0)) / np.sqrt((2.0 - (m == 0))*(n-1.0))
        n_new = n - 2.0
                
        #check if the new n and m values are valid values (only add recursion if new n and m make sense)
        if (n_new - np.abs(m)) %2 == 0 and n_new >= 1 and n_new >= np.abs(m): 
            j_new = Zernike_nm_2_j(n_new, m)
            yder = n*(-1*am*np.sign(m-1.0)*bfact1*Zernike_nm(n-1.0, -1*am*np.abs(m-1), rho, phi) + \
                        bfact2 * Zernike_nm(n-1.0, -1*am*np.abs(m+1.0), rho, phi)) + \
                        bfact3 * yderZ(j_new, x, y)
            return yder
        else:
            yder = n*(-1*am*np.sign(m-1.0)*bfact1*Zernike_nm(n-1.0, -1*am*np.abs(m-1), rho, phi) + \
                        bfact2 * Zernike_nm(n-1.0, -1*am*np.abs(m+1.0), rho, phi))
            return yder
    else:
        return np.zeros(x.shape)
        
def Zernike_nm(n, m, rho, theta):
    """Returns the values of the Zernike polynomial of radial and azimuthal order (n & m). 
    n and m should be integers,
    rho and theta np arrays with the same size
    out: array with values of at the points rho and theta"""
    if (n-np.abs(m)) %2 == 0 and (n >= 0) and (n >= np.abs(m)): #check if even
        s_max = int((n-abs(m))/2)
        radial = np.zeros(shape=rho.shape, dtype = complex_)
        for s in range(s_max + 1):
            prefactor = (-1)**s * math.factorial(n-s) / (math.factorial(s) * math.factorial((n+abs(m))/2 - s) * math.factorial((n-abs(m))/2 - s))
            radial += np.power(rho,(n - 2*s)) * prefactor
##        if m > 0:
##            angular = np.cos(m*theta)
##        elif m < 0:
##            angular = -1 * np.sin(m*theta)
##        else:
##            angular = 1
        angular = np.exp(1j * m * theta)
            
        return np.sqrt((2 - (m == 0)) * (n + 1)) * angular * radial
        #if m == 0:
        #    return np.sqrt(n+1) * angular * radial
        #else:
        #    return np.sqrt(2*n + 2) * angular * radial
    else:
        return np.zeros(shape=rho.shape)

def complex_zernike(n_max, x, y, order):
    """Given the meshgrids for x and y, and given the maximum fringe order, complex zernike retursn
    an (shape(x), j_max) sized matrix with values of the complex Zernike polynomial at the given points"""
    rho, theta = cart2pol(x, y)
    rho2 = 2 * rho**2 - 1
##    j = max_n_to_j(n_max, order = order)[str(n_max)]
    j = np.insert(max_n_to_j(n_max, order = order)[str(n_max)], 0, 1)
    n, m = Zernike_j_2_nm(j, ordering = order)
    nm2 = (n - np.abs(m))/2
    xshape = list(rho.shape)
    xshape.append(len(j))
    Cnm = np.zeros(xshape, dtype = np.complex_)
    a = np.zeros(len(m))
    b = np.abs(m)
    for i in range(len(nm2)):
        nm = nm2[i]
        nm = int(nm)
        for jj in range(nm+1):
            Cnm[...,i] += spec.comb(nm+a[i], jj) * spec.comb(nm+b[i], nm-jj) * np.power((rho2-1)/2.0, (nm - jj)) * np.power((rho2+1)/2.0, jj)
        Cnm[...,i] *= np.power(rho, abs(m[i])) * np.exp(1j * m[i] * theta)
    return Cnm

def complex_zernike_int(j_max, x, y):
    """Given the meshgrids for x and y, and given the maximum fringe order, complex zernike retursn
    an (shape(x), j_max) sized matrix with values of the complex Zernike polynomial at the given points"""
    rho, theta = cart2pol(x, y)
    rho2 = 2 * rho**2 - 1
    j = np.arange(1, j_max+2)
    n, m = Zernike_j_2_nm(j)
    nm2 = (n - np.abs(m))/2
    xshape = list(rho.shape)
    xshape.append(j_max+1)
    Cnm = np.zeros(xshape, dtype = np.complex_)
    a = np.zeros(len(m))
    b = np.abs(m)
    for i in range(len(nm2)):
        nm = nm2[i]
        nm = int(nm)
        for jj in range(nm+1):
            Cnm[...,i] += spec.comb(nm+a[i], jj) * spec.comb(nm+b[i], nm-jj) * np.power((rho2-1)/2.0, (nm - jj)) * np.power((rho2+1)/2.0, jj)
        Cnm[...,i] *= np.power(rho, abs(m[i])) * np.exp(1j * m[i] * theta)
    return Cnm

def plot_zernike(j_max, a, ax= None, wavelength = 632.8e-9, cmap = cm.jet, savefigure = False, title = 'zernike_plot', fliplr = False,**kwargs):
### plot zernikes according to coefficients
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    xi, yi = np.linspace(-1, 1, 600), np.linspace(-1, 1, 600)
    xi, yi = np.meshgrid(xi, yi)
    xn = np.ma.masked_where(xi**2 + yi**2 >= 1, xi)
    yn = np.ma.masked_where(xi**2 + yi**2 >= 1, yi)
    power_mat = Zernike_power_mat(j_max+2)
    j_range = np.arange(2, j_max+2)
    Z = np.zeros(xi.shape)
    Z_mat = Zernike_xy(xi, yi, power_mat, j_range)
    Z = np.sum(a * Z_mat, axis=2)
    #for jj in range(j_max):
    #    Z += a[jj] * Zernike_xy(xi, yi, power_mat, jj+2)
    #Z /= wavelength
    #Zn = np.ma.masked_where(xi**2 + yi**2 >=1, Z)
    outside = np.where(xi**2 + yi**2 >=1)
    inside= np.where(xi**2 + yi**2 < 1)
    Z[outside] = np.nan
    lev = np.linspace(Z[inside].min(), Z[inside].max())
    norml = colors.BoundaryNorm(lev, 256)
##    if 'v' in kwargs:
##        levels = kwargs['v']
##    else:
##        levels = np.linspace(np.min(Zn), np.max(Zn))
##    if fliplr:
##        Zn = np.fliplr(Zn)

    #fig = plt.figure(figsize = plt.figaspect(1.))
    plotje = ax.plot_surface(xi, yi, Z, cmap = 'jet', norm = norml, linewidth = 0.2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('phi')
    #ax.axis('off')
##    divider = make_axes_locatable(ax)
##    cax = divider.append_axes("right", size="5%", pad=0.1)
##    cbar = plt.colorbar(plotje, cax=cax)
    if savefigure:
        plt.savefig(folder_name + title + '.png', bbox_inches='tight', dpi = 600)
    return plotje

def plot_interferogram(j_max, a, piston = 0, ax = None, f = None, wantcbar = False, cmap = 'bone', wavelength = 632.8e-9, fliplr = False, savefigure = False, title = 'Interferogram according to a', **kwargs):
    if ax is None:
        ax = plt.gca()
    xi, yi = np.linspace(-1, 1, 300), np.linspace(-1, 1, 300)
    xi, yi = np.meshgrid(xi, yi)
    xn = np.ma.masked_where(xi**2 + yi**2 >= 1, xi)
    yn = np.ma.masked_where(xi**2 + yi**2 >= 1, yi)
    power_mat = Zernike_power_mat(j_max+2)
    Z = np.zeros(xi.shape)
    j_range = np.arange(2, j_max+2)
    Z_mat = Zernike_xy(xi, yi, power_mat, j_range)
    Z = np.sum(a * Z_mat, axis = 2)
    levels = np.linspace(0, 1, 15)
    
    #Z /= wavelength
    #Z += piston
    #fig = plt.figure(figsize = plt.figaspect(1.))
    Zn = np.ma.masked_where(xi**2 + yi**2 >=1, Z)
    Zn = Zn + piston
    if fliplr:
        Zn = np.fliplr(Zn)
    if 'want_phi_old' in kwargs:
        phase = np.abs(np.abs(Zn) - np.floor(np.abs(Zn))) * 2 * np.pi
    else:
        phase = np.mod(Zn - np.pi, 2*np.pi) - np.pi
    Intens = np.cos(phase/2.0)**2
    if 'v' in kwargs:
        levels = kwargs['v']
        interferogram = ax.contourf(xn, yn, Intens, levels= levels, rstride = 1, cstride = 1, cmap= cmap, linewidth=0)
    else:
        interferogram = ax.contourf(xn, yn, Intens,  rstride = 1, cstride = 1, cmap= cmap, linewidth=0, levels = levels)

        
    if wantcbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(interferogram, cax=cax)
    return interferogram


