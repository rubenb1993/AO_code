### This file contains general functions to gather positions from Hartmanngramms and get the slope on the unit disc
import numpy as np

##### Make list of maxima given "flat" wavefront #####

def zero_positions(image, spotsize = 35):
    "From an image and ""spot box size"" in in pixels, make a list of x and y positions of the centroids"
    x_pos_flat = []
    y_pos_flat = []
    image[image<6] = 0
    while(np.amax(image) > 50):
        y_max, x_max = np.unravel_index(image.argmax(), image.shape)
        x_pos_flat.append(x_max)
        y_pos_flat.append(y_max)
        image[y_max - spotsize: y_max + spotsize, x_max - spotsize : x_max + spotsize] = 0
    return np.array(x_pos_flat), np.array(y_pos_flat)

##### Gather centroids ####

def centroid_positions(x_pos_flat, y_pos_flat, image, xx, yy, spot_size = 35):
    """Gather position (in px) of all maxima in image, given the position of maxima with a flat wavefront
    x_pos_flat & y_pos_flat: arrays with x and y coordinates of centroids
    image: tif snapshot of the SH
    xx & yy: meshgrids of pixels
    spot_size: approximate width of domain (in px) of one SH lenslet
    output: 2 arrays of x and y centroids in new figure"""
    centroids = np.zeros(shape = (len(x_pos_flat),2))
    image[image<6] = 0 #remove noisy pixels
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
    dx = (x_pos_dist - x_pos_flat) * px_size #displacement in px to m
    dy = (y_pos_dist - y_pos_flat) * px_size
    slope_x = r_sh * (dx/f) #approximate wf slope as linearization and scale to unit disc by multiplying with r_sh
    slope_y = r_sh * (dy/f)
    return slope_x, slope_y

#### Centroids centre and radius SH pattern
def centroid_centre(x_pos_flat, y_pos_flat, image, xx, yy, px_size):
    """Taken a SH pattern, find the centre of the SH circle and return the radius in meters.
    Requires flat centroid positions, the image itself, x and y meshgrids of 'pixel position' and the size of the pixels in meters
    """
    centre = np.zeros(2)
    image[image<4] = 0 #crudely remove noisy pixels
    norm_photons = 1.0/np.sum(image)
    centre[0] = norm_photons * np.sum(image * xx)
    centre[1] = norm_photons * np.sum(image * yy)
    r_sh = np.amax(np.sqrt((x_pos_flat - centre[0])**2 + (y_pos_flat - centre[1])**2))
    r_sh_real = r_sh * px_size
    return centre, r_sh, r_sh_real
    
