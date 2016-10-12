### This file contains general functions to work with Zernike polynomials and their derivatives. 

import numpy as np
from scipy import special
import math

def cart2pol(x, y):
    "returns polar coordinates given x and y"
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def Zernike_j_2_nm(j):
    """Converts Zernike index j (FRINGE convention) to index n and m. 
    Based on code by Dr. Ir. S. van Haver (who based it on Herbert Gross, Handbook of Optical Systems, page 215).
    input j: either scalar or an array (lists not supported)
    output n, m: either a scalar or an array of corresponding n and m.
    Built in additional check if j is integer. If not, n = -1 will be generated s.t. any zernike function will be 0"""
    if (np.all(np.mod(j, 1) == 0)) and np.all(np.array(j) > 0): 
        q = np.floor(np.sqrt(j-1.0))
        p = np.floor((j - q**2 - 1.0)/2.0)
        r = j - q**2 - 2*p
        n = q+p
        m = np.power(-1, r+1) * (q-p)
        return n, m
    else:
        return -1, -1

def Zernike_nm_2_j(n, m):
    """Converts Zernike index j (FRINGE convention) to index n and m. 
    Based on code by Dr. Ir. S. van Haver (who based it on Herbert Gross, Handbook of Optical Systems, page 215).
    input n, m: either scalar or an array (lists not supported)
    output j: either a scalar or an array of corresponding j"""
    p = (n + np.abs(m))/2.0
    return p**2 + n - np.abs(m) + 1 + (m<0)
    
def Zernike_power_mat(j_max, Janssen = False):
    j_max = int(j_max)
    n_max = np.max([Zernike_j_2_nm(j)[0] for j in range(j_max+1)]) #find maximum n as n of j_max might not be the maximum n
    fmat = np.zeros((n_max+1, n_max+1, j_max)) #allocate memory
    for jj in range(j_max):
        n, m_not_brug = Zernike_j_2_nm(jj+1) #brug uses different m, see later defined
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
        
        if Janssen:
            norm = np.sqrt( (2 - (m_not_brug == 0))) #precompute normalization
        else:
            norm = np.sqrt( (2 - (m_not_brug == 0))*(n+1)) #precompute normalization
        
        
        for i in range(q+1):
            for j in range(m+1):
                for k in range(m-j+1):
                    fact = np.power(-1, i+j)
                    fact *= special.binom( np.abs(l), 2*i + p)
                    fact *= special.binom( m-j, k)
                    fact *= math.factorial(n-j) / ( math.factorial(j) * math.factorial(m-j) * math.factorial(n-m-j))
                    ypow = 2 * (i+k) + p
                    xpow = n - 2 * (i+j+k) - p
                    fmat[xpow, ypow, jj] += fact * norm     
    return fmat
        
def Zernike_xy(x, y, power_mat, j):
    """Computes the value of Zj at points x and y
    in: x, y: coordinate matrices
    power_mat: the power matrix as made by Zernike_power_mat
    j the fringe order of your polynomial
    out: Z, a matrix containing the values of Zj at points x and y
    
    Normalized s.t. int(|Z_j|^2) = pi"""
    x_list, y_list = np.nonzero(power_mat[...,j-1])
    Z = np.zeros(x.shape)
    for i in range(len(x_list)):
        Z += power_mat[x_list[i], y_list[i], j-1] * np.power(x, x_list[i]) * np.power(y, y_list[i])
    return Z
    
def xder_brug(x, y, power_mat, j):
    """Computes the value of dZj/dx at points x and y
    in: x, y: coordinate matrices
    power_mat: the power matrix as made by Zernike_power_mat
    j the fringe order of your polynomial
    out: dZdx, a matrix containing the values of dZjdx at points x and y
    
    Normalized s.t. int(|Z_j|^2) = pi"""
    power_mat = power_mat[1:, :, :]
    x_list, y_list = np.nonzero(power_mat[...,j-1])
    dZdx = np.zeros(x.shape)
    for i in range(len(x_list)):
        dZdx += (x_list[i]+1)*power_mat[x_list[i], y_list[i], j-1] * np.power(x, x_list[i]) * np.power(y, y_list[i])
    return dZdx
    
def yder_brug(x, y, power_mat, j):
    """Computes the value of dZj/dy at points x and y
    in: x, y: coordinate matrices
    power_mat: the power matrix as made by Zernike_power_mat
    j the fringe order of your polynomial
    out: Z, a matrix containing the values of dZj/dy at points x and y
    
    Normalized s.t. int(|Z_j|^2) = pi"""
    power_mat = power_mat[:, 1:, :]
    x_list, y_list = np.nonzero(power_mat[...,j-1]) #-1 due to j = 1 in power_mat[...,0]
    dZdy = np.zeros(x.shape)
    for i in range(len(x_list)):
        dZdy += (y_list[i]+1)*power_mat[x_list[i], y_list[i], j-1] * np.power(x, x_list[i]) * np.power(y, y_list[i])
    return dZdy
 
    
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

def plot_zernike(j_max, a, wavelength = 632.8e-9, savefigure = False, title = 'zernike_plot'):
    ### plot zernikes according to coefficients
    xi, yi = np.linspace(-1, 1, 300), np.linspace(-1, 1, 300)
    xi, yi = np.meshgrid(xi, yi)
    power_mat = Zn.Zernike_power_mat(j_max+1)
    Z = np.zeros(xi.shape)
    for jj in range(len(a)):
        Z += a[jj]*Zn.Zernike_xy(xi, yi, power_mat, jj+2)

    Z /= wavelength
    plt.contourf(xi, yi, Z, rstride=1, cstride=1, cmap=cm.YlGnBu_r, linewidth = 0)
    cbar = plt.colorbar()
    #plt.title("Defocus 10")
    cbar.ax.set_ylabel('lambda')
    if savefigure:
        plt.savefig(title + '.png', bbox_inches='tight')
    plt.show()
