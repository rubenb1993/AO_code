### This file contains general functions to work with Zernike polynomials and their derivatives. 

import numpy as np
from scipy import special
import math


def Zernike_j_2_nm(j):
    """Converts Zernike index j (FRINGE convention) to index n and m. 
    Based on code by Dr. Ir. S. van Haver (who based it on Herbert Gross, Handbook of Optical Systems, page 215).
    input j: either scalar or an array (lists not supported)
    output n, m: either a scalar or an array of corresponding n and m.
    Built in additional check if j is integer. If not, n = -1 will be generated s.t. any zernike function will be 0"""
    if (np.equal(np.mod(j, 1), 0)) and j > 0: 
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