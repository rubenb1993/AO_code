#import dmctr
import sys
if "C:\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Micro-Manager-1.4")
#import MMCorePy
import PIL.Image
import numpy as np
from math import factorial
import time
import matplotlib.pyplot as plt
from matplotlib import cm


#reference=np.asarray(PIL.Image.open("shref.tif")).astype(float)

def centroid_positions(x_max, y_max, image, spot_size = 25):    
    centroids = np.zeros(shape = (len(x_max),2))
    image[image<4]=0
    #spot_size = 25
    for i in range(len(x_max)):
        y_low = y_max[i] - spot_size
        y_high = y_max[i] + spot_size
        x_low = x_max[i] - spot_size
        x_high = x_max[i] + spot_size
        norm_photons = 1/np.sum(image[y_low: y_high, x_low: x_high])
        centroids[i,0] = norm_photons * np.sum(image[y_low: y_high, x_low: x_high] * xx[y_low: y_high, x_low: x_high])
        centroids[i,1] = norm_photons * np.sum(image[y_low: y_high, x_low: x_high] * yy[y_low: y_high, x_low: x_high])
    #print centroids[0,:]
    return np.reshape(centroids, -1)

def radial_zernike(n,m,rho):
    """Returns the radial part of the zernike polynomial. 
    n and m should be integers, while rho HAS to be a numpy array
    Returns an array the same size as rho with the radial coefficients."""
    if (n-m) & 0x1 == 0: #check if even
        s_max = int((n-abs(m))/2)
        radial = np.zeros(shape=rho.shape)
        for s in range(s_max + 1):
            prefactor = (-1)**s * factorial(n-s) / (factorial(s) * factorial((n+abs(m))/2 - s) * factorial((n-abs(m))/2 - s))
            radial += np.power(rho,(n - 2*s)) * prefactor
    else:
        radial = np.zeros(shape=rho.shape)
    return radial

def angular_zernike(m,theta):
    """Returns the angular part of the zernike polynomial. 
    m should be an integer, angle can be an array.
    returns an array the same size as theta (or a scalar if m = 0)
    """
    if m > 0:
        angular = np.cos(m*theta)
    elif m < 0:
        angular = np.sin(abs(m)*theta)
    else:
        angular = 1
    return angular
    
def zernike_m_n(n,m,rho,theta):
    "Returns the value of the zernike polynomial given n, m, r and theta"
    return radial_zernike(n,m,rho) * angular_zernike(m,theta)

def cart2pol(x, y):
    "returns polar coordinates given x and y"
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return[x, y]


def zernike_x_y(x, y, ai):
    """Returns the value of composed wavefield given the position x and y, 
    and the vector ai with the zernike polynomial weights
    """
    [r, theta] = cart2pol(x, y)
    Z = 0
    for i in range(len(ai)):
        Z += ai[i]*zernike_m_n(n[i],m[i],r,theta)
    return Z

n = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]
m = [0, 1, -1, 0, -2, 2, -1, 1, -3, 3, 0, 2, -2, 4, -4, 1, -1, 3, -3, 5, -5]
zj = np.arange(1,len(n)+1,1)


pi = np.pi
[ny,nx] = reference.shape
#img_pil = np.array(np.array(img_pil).reshape((ny,nx)))
xx, yy = np.meshgrid(np.linspace(1,nx,nx),np.linspace(1,ny,ny))
i,j = np.unravel_index(reference.argmax(), reference.shape)
img_pil_mask = reference > 3
img_pil_filtered = reference * img_pil_mask

list_of_maxima_x = []
list_of_maxima_y = []

while(np.amax(img_pil_filtered) > 10):
    y_max, x_max = np.unravel_index(img_pil_filtered.argmax(), img_pil_filtered.shape)
    list_of_maxima_x.append(x_max)
    list_of_maxima_y.append(y_max)
    img_pil_filtered[y_max - 40: y_max + 40, x_max - 40: x_max+40] = 0
    

x_max = np.array(list_of_maxima_x)
y_max = np.array(list_of_maxima_y)
reference_centroids = centroid_positions(x_max, y_max, reference)

aberration = np.asarray(PIL.Image.open("sh_abberation_disc.tif")).astype(float)
aberration_centroids = centroid_positions(x_max, y_max, aberration)

aberration_diff = aberration_centroids - reference_centroids


#Set lenslet array parameters
width_lenslet = 300e-6
width_lenslet_half = width_lenslet/2
f  = 18e-3

#Lenslet array
##r = [[0.75] for i in range(6)]
##theta = [[2*pi*i/6] for i in range(6)]
##lens_0 = np.array([[0, 0]])
##lens_array_pol = np.hstack((r,theta))
##lens_array_pol = np.concatenate((lens_array_pol,lens_0),axis=0)
##lens_array = np.array(pol2cart(lens_array_pol[:,0], lens_array_pol[:,1])).T
lens_array = np.vstack((list_of_maxima_x, list_of_maxima_y)).T




#set 3d display parameters
r = np.linspace(0,1,20)
theta = np.linspace(0, 2*pi, 20)
radius_matrix, theta_matrix = np.meshgrid(r,theta)
X, Y = radius_matrix*np.cos(theta_matrix), radius_matrix*np.sin(theta_matrix)

SH_pattern = np.zeros((len(x_max),2,len(zj)))

subplot_count = 1
for i in range(len(zj)):
    ai = np.zeros(len(zj))
    ai[i] = 1
    lens_x = lens_array[:,0]
    lens_y = lens_array[:,1]
    dx = np.array(zernike_x_y(lens_x + width_lenslet_half, lens_y, ai) - zernike_x_y(lens_x - width_lenslet_half, lens_y, ai))
    dy = np.array(zernike_x_y(lens_x, lens_y + width_lenslet_half, ai) - zernike_x_y(lens_x, lens_y- width_lenslet_half, ai))
    dr = np.vstack((dx,dy))
    dr_lens = dr*f / width_lenslet
    SH_pattern[...,i] = np.vstack((lens_x + dr_lens[0,:], lens_y + dr_lens[1,:])).T
    
    fig = plt.figure(figsize=plt.figaspect(5.))
    Z = zernike_m_n(n[i],m[i],radius_matrix, theta_matrix)
    ax = fig.add_subplot(len(zj), 2, subplot_count)
    subplot_count += 1
    # = fig.add_subplot(121, projection='3d')
    ax.contourf(X, Y, Z, rstride=1, cstride=1, cmap=cm.YlGnBu_r)
    #ax.set_zlim3d(-1, 1)
    
    #ax2 = fig.add_subplot(122)
    ax = fig.add_subplot(len(zj), 2, subplot_count)
    subplot_count += 1
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])    
    ax.scatter(lens_array[:,0], lens_array[:,1])
    ax.scatter(SH_pattern[:,0,i], SH_pattern[:,1,i], color = 'r')
    circle=plt.Circle((0,0),1,color='k',fill=False)
    ax2 = plt.gca()
    ax2.add_artist(circle)

plt.show()
