# -*- coding: utf-8 -*-
## Find the real Zernike coefficients using the derivative of the wavefront

## Part of the Adaptive Optics code by Ruben Biesheuvel (ruben.biesheuvel@gmail.com)

## Code will go through:
#Gathering Centroids
#Centroids -> dWx and dWy
#Build "geometry matrix" (x and y derivative of Zernike pol. in all centroid positions)
#


## Note, this program normalizes accoring to ... $$$ fill in $$$

#import dmctr
import sys
if "C:\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Micro-Manager-1.4")
#import MMCorePy
import PIL.Image
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm


# Define font for figures
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

##### Make list of maxima given "flat" wavefront #####

def zero_positions(image, spotsize = 25):
    "From an image and ""spot box size"" in in pixels, make a list of x and y positions of the centroids"
    x_pos_flat = []
    y_pos_flat = []
    image[image<4] = 0
    while(np.amax(image) > 10):
        y_max, x_max = np.unravel_index(image.argmax(), image.shape)
        x_pos_flat.append(x_max)
        y_pos_flat.append(y_max)
        image[y_max - spotsize: y_max + spotsize, x_max - spotsize : x_max + spotsize] = 0
    return np.array(x_pos_flat), np.array(y_pos_flat)

##### Gather centroids ####

def centroid_positions(x_pos_flat, y_pos_flat, image, xx, yy, spot_size = 25):
    """Gather position (in px) of all maxima in image, given the position of maxima with a flat wavefront
    x_pos_flat & y_pos_flat: arrays with x and y coordinates of centroids
    image: tif snapshot of the SH
    xx & yy: meshgrids of pixels
    spot_size: approximate width of domain (in px) of one SH lenslet
    output: 2 arrays of x and y centroids in new figure"""
    centroids = np.zeros(shape = (len(x_pos_flat),2))
    image[image<4] = 0 #remove noisy pixels
    for i in range(len(x_pos_flat)):
        y_low = y_pos_flat[i] - spot_size
        y_high = y_pos_flat[i] + spot_size
        x_low = x_pos_flat[i] - spot_size
        x_high = x_pos_flat[i] + spot_size
        #Find centroids weighing them with intensity and position
        norm_photons = 1/np.sum(image[y_low: y_high, x_low: x_high])
        centroids[i,0] = norm_photons * np.sum(image[y_low: y_high, x_low: x_high] * xx[y_low: y_high, x_low: x_high])
        centroids[i,1] = norm_photons * np.sum(image[y_low: y_high, x_low: x_high] * yy[y_low: y_high, x_low: x_high])
    return centroids[:,0], centroids[:,1]
    
#### Centroid positions to slope on unit circle ####

def centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat, y_pos_flat, px_size, f, r_sh):
    "Given the positions of the disturbed wf and the flat wf, calculate the slope of the wf on unit disc"
    dx = (x_pos_dist - x_pos_flat) * px_size #displacement in px to mm
    dy = (y_pos_dist - y_pos_flat) * px_size
    slope_x = r_sh * (dx/f) #approximate wf slope as linearization and scale to unit disc by multiplying with r_sh
    slope_y = r_sh * (dy/f)
    return slope_x, slope_y
    
#### Functions necessary to create geometry matrix ####
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
        if m > 0:
            angular = np.cos(m*theta)
        elif m < 0:
            angular = -1 * np.sin(m*theta)
        else:
            angular = 1
            
        return np.sqrt((2 - (m == 0)) * (n + 1)) * angular * radial
        #if m == 0:
        #    return np.sqrt(n+1) * angular * radial
        #else:
        #    return np.sqrt(2*n + 2) * angular * radial
    else:
        return np.zeros(shape=rho.shape)
    
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
        
#### check the numerical derivatives with analytical ones from Stephenson [1] ####
def check_num_der(savefigure = False):
    """Function to check if the derivatives calculated by xderZ and yderZ comply with analytical values given by Stephenson [1]
    savefigure = True will save the figure generated."""
    rho = np.linspace(0, 1, 50)
    theta = 0.0 * np.pi/2 * np.ones(rho.shape)
    #theta = np.pi/2 * np.ones(rho.shape)
    x, y = pol2cart(rho, theta)
    n = np.array([4.0, 5.0, 5.0, 5.0, 5.0])
    m = np.array([0.0, -1.0, -3.0, 1.0, 3.0])
    j = Zernike_nm_2_j(n, m)
    
    #set 3d display parameters
    r_mat = np.linspace(0,1,20)
    theta_mat = np.linspace(0, 2*np.pi, 20)
    radius_matrix, theta_matrix = np.meshgrid(r_mat,theta_mat)
    X, Y = radius_matrix*np.cos(theta_matrix), radius_matrix*np.sin(theta_matrix)
    
    xderZ_com = np.zeros((len(x),len(j)))
    yderZ_com = np.zeros((len(x),len(j)))
    xderZ_ana = np.zeros((len(x),len(j)))
    yderZ_ana = np.zeros((len(x),len(j)))
    Z = np.zeros((len(r_mat), len(r_mat), len(j)))
    
    
    for i in range(len(j)):
        xderZ_com[:,i] = xderZ(j[i], x, y)
        yderZ_com[:,i] = yderZ(j[i], x, y)
        Z[:,:,i] = Zernike_nm(n[i], m[i], radius_matrix, theta_matrix)
        
    xderZ_ana[:,0] = 12 * np.sqrt(5) * x * (2 * x**2 + 2 * y**2 -1)
    xderZ_ana[:,1] = 16* np.sqrt(3) * x * y * (5 * x**2 + 5 * y**2 - 3)
    xderZ_ana[:,2] = 8 * np.sqrt(3) * x * y * (15 * x**2 + 5 * y**2 - 6)
    xderZ_ana[:,3] = 2 * np.sqrt(3) * (50 * x**4 + 12 * x**2 * (5 * y**2 - 3) + 10* y**4 - 12* y**2 + 3)
    xderZ_ana[:,4] = 2 * np.sqrt(3) * (25 * x**4 - 6 * x**2 * (y**2 + 2) - 3 * y**2 * (5 * y**2 -4))
    #2 * np.sqrt(3) * (50 * x**4 + 12 * x**2 * (5 * y**2 - 3) + 10* y**4 - 12* y**2 + 3)
    
    yderZ_ana[:,0] = 12 * np.sqrt(5) * y * (2 * x**2 + 2 * y**2 -1)
    yderZ_ana[:,1] = 2 * np.sqrt(3) * (50 * x**4 + 12 * x**2 * (5 * y**2 - 3) + 10* y**4 - 12* y**2 + 3)
    yderZ_ana[:,2] = 2 * np.sqrt(3) * ( 15 * x**4 - 6 * x**2 * (5* y**2 - 2) - y**2 * (25 * y**2 - 12))
    yderZ_ana[:,3] = 16* np.sqrt(3) * x * y * (5 * x**2 + 5 * y**2 - 3)
    yderZ_ana[:,4] = -8 * np.sqrt(3) * x * y * (5 * x**2 + 15 * y**2 - 6)
    #16* np.sqrt(3) * x * y * (5 * x**2 + 5 * y**2 - 3)
    
    #makefigure
    f, axarr = plt.subplots(5, 2, sharex='col', sharey='row')
    f2, axarr2 = plt.subplots(5, 1, sharex = True, sharey = True)
    for ii in range(len(j)):
        ana, = axarr[ii,0].plot(rho, xderZ_ana[:,ii], 'r-', label='Analytic')
        comp, = axarr[ii,0].plot(rho, xderZ_com[:,ii], 'bo', markersize = 2, label='computational')
        axarr[ii,1].plot(rho, yderZ_ana[:,ii], 'r-', rho, yderZ_com[:,ii], 'bo', markersize = 2)
            
    
        ZZ = axarr2[ii].contourf(X, Y, Z[:,:,ii], rstride=1, cstride=1, cmap=cm.YlGnBu_r)
            
        axarr2[ii].set_xlim([-1, 1])
        axarr2[ii].set_ylim([-1, 1]) 
    plt.colorbar(ZZ)      
    axarr[0,0].set_xlim([0, 1])

    axarr[4,0].set_xlabel(r'$ \rho $')
    axarr[0,0].set_ylabel(r'$Z_4^0$')
    axarr[1,0].set_ylabel(r'$Z_5^{-1}$')
    axarr[2,0].set_ylabel(r'$Z_5^{-3}$')
    axarr[3,0].set_ylabel(r'$Z_5^{1}$')
    axarr[4,0].set_ylabel(r'$Z_5^{3}$')
    axarr[0,0].set_title(r'$\partial / \partial x$')
    axarr[0,1].set_title(r'$\partial / \partial y$')
    
    plt.figlegend((ana, comp), ('Analytical', 'Computational') , 'lower right', ncol = 2, fontsize = 9 )
    if savefigure:
        f.savefig('AO_code/derivatives_comparison.pdf', bbox_inches='tight', pad_inches=0.1)
        plt.show()
    else:
        plt.show()
    return

##### Create the geometry matrix #### 
def geometry_matrix(x, y, j_max):
    """Creates the geometry matrix of the SH sensor according to the lecture slides of Imaging Physics (2015-2016)"""
    Bx = np.zeros((len(x), j_max))
    By = np.zeros((len(x), j_max))
    for jj in range(j_max):
        for ii in range(len(x)):
            Bx[ii, jj] = xderZ(jj+2, x[ii], y[ii])
            By[ii, jj] = yderZ(jj+2, x[ii], y[ii])
    B = np.vstack((Bx, By))
    return B

##### Solve dat system yo #####        
def solve_system(px_size, f, r_sh, j_max):
    #To do:
    #gather flat positions
    #take snapshot of image
    #make meshgrid of pixels
    #input: px_size, f, r_sh
    #determine amount of j you want
    x_pos_dist, y_pos_dist = centroid_positions(x_pos_flat, y_pos_flat, image, xx, yy, spot_size = 25)
    s = np.hstack(centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat, y_pos_flat, px_size, f, r_sh))
    Binv = np.linalg.pinv(geometry_matrix(x_pos_flat, y_pos_flat, j_max))
    a = np.dot(Binv, s)
    return a
    
    