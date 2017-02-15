import edac40
import numpy as np
import Hartmann as Hm
import displacement_matrix as Dm
import Zernike as Zn
import sys
import os
import time
import matplotlib.pyplot as plt

if "C:\Program Files\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Program Files\Micro-Manager-1.4")
import MMCorePy
import PIL.Image

def filter_positions(inside, *args):
    """given a certain range of indices which are labeled inside, return an arbitrary number of vectors filtered with those indices"""
    new_positions = []
    for arg in args:
        new_positions.append(np.array(arg)[inside])
    return new_positions

def filter_nans(*args):
    """given vectors, filters out the NANs. Assumes that nans appear in the same place"""
    new_positions = []
    for arg in args:
        new_positions.append(arg[~np.isnan(arg)])
    return new_positions

def set_displacement(u_dm, mirror):
    """linearizes the deformable mirror control.
    u_dm is a vector in the range (-1, 1) with the size (actuators,)
    mirror is the deformable mirror that is controlled
    sets the voltages according to:
        sqrt((u_dm + 1.0) * 72) (such that 0 < V < 12) for nonlinear acts
        ((u_dm + 1.0) * 72)/12 (s.t. 0 < V < 12) for tip tilt
    also limits u_dm s.t. all values below -1 will be -1,
    and all values above 1 will be 1
    output: mirror is set to the voltages"""
    u_dm = u_dm * 72.0
    u_l = np.zeros(u_dm.shape)
    u_l = np.maximum(u_dm, -72.0 * np.ones(u_l.shape))
    u_l = np.minimum(u_l, 72.0 * np.ones(u_l.shape))
    actnum=np.arange(0,19,1)
    linacts=np.where(np.logical_or(actnum==4,actnum==7))
    others=np.where(np.logical_and(actnum!=4,actnum!=7))
    u_l += 72.0
    u_l[linacts]=(u_l[linacts])/12
    u_l[others]=np.sqrt(u_l[others])
    
    mirror.set(u_l)


def set_up_cameras():
    cam1=MMCorePy.CMMCore()
    cam1.loadDevice("cam","IDS_uEye","IDS uEye")
    cam1.initializeDevice("cam")
    cam1.setCameraDevice("cam")
    pixel_clock = cam1.getPropertyUpperLimit("cam", "Pixel Clock")
    if pixel_clock == 150.0:
        cam1.setProperty("cam","Pixel Clock",150)
        cam1.setProperty("cam", "PixelType", '8bit mono')
        cam1.setProperty("cam","Exposure",0.097)
        sh = cam1

        cam2=MMCorePy.CMMCore()
        cam2.loadDevice("cam","IDS_uEye","IDS uEye")
        cam2.initializeDevice("cam")
        cam2.setCameraDevice("cam")
        cam2.setProperty("cam","Pixel Clock", 43)
        cam2.setProperty("cam","Exposure", 13.226)
        int_cam = cam2
    else:
        cam1.setProperty("cam","Pixel Clock", 43)
        cam1.setProperty("cam","Exposure", 13.226)
        int_cam = cam1

        cam2=MMCorePy.CMMCore()
        cam2.loadDevice("cam","IDS_uEye","IDS uEye")
        cam2.initializeDevice("cam")
        cam2.setCameraDevice("cam")
        cam2.setProperty("cam","Pixel Clock", 150)
        cam2.setProperty("cam","PixelType", '8bit mono')
        cam2.setProperty("cam","Exposure", 0.097)
        sh = cam2
    return sh, int_cam

def flat_wavefront(u_dm, zero_image, image_control, r_sh_px, r_int_px, sh, mirror, scaling = 0.75, box_len = 35, show_accepted_spots = False, show_hist_voltage = False):
    """With an image of a flat wavefront, creates a flat wavefront with the mirror.
        In:
    zero_image: SH pattern of the flat wavefront (usually the reference mirror)
    image_control: SH pattern of the mirror at mid-stroke
    r_sh_px: radius of illuminated part shack hartmann sensor in px
    r_int_px: radius of illumated part of interferogram camera in px
    sh: MMcorepy camera of shack-hartmann sensor
    mirror: edac40 adress of mirror (???)
    scaling: scaling factor with which the new voltages are applied, for convergence purposes, standard at 0.75
    box_len: halfwidth of subaperture of SH sensor in px
    show_accepted_spots: plot a scatterplot of the accepted SH spots which lie within r_sh_px
    show_hist_voltage: show a histogram of the voltages on the mirror after it become flat
        Out:
    Bunch of information printed on screen and maybe some plots. A flat wavefront on the camera as well"""
    ### Create matrices needed for calculations
    [ny,nx] = zero_image.shape
    x = np.linspace(1, nx, nx)
    y = np.linspace(1, ny, ny)
    xx, yy = np.meshgrid(x, y)
    
    ### Gather centroids of current picture (u_dm 0) and voltage 2 distance (v2d) matrix
    x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image)
    x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, image_control, xx, yy)
    centre = Hm.centroid_centre(x_pos_flat, y_pos_flat)
    x_pos_norm = ((x_pos_flat - centre[0]))/r_sh_px
    y_pos_norm = ((y_pos_flat - centre[1]))/r_sh_px
    inside = np.where(np.sqrt(x_pos_norm**2 + y_pos_norm**2) <= (1 + (box_len/r_sh_px))) #35 is the half the width of the pixel box aroudn a centroid and r_sh_px is the scaling factor
    x_pos_zero_f, y_pos_zero_f, x_pos_flat_f, y_pos_flat_f, x_pos_norm_f, y_pos_norm_f = filter_positions(inside, x_pos_zero, y_pos_zero, x_pos_flat, y_pos_flat, x_pos_norm, y_pos_norm)

    if show_accepted_spots:
        f3 = plt.figure(figsize = plt.figaspect(1.))
        ax3 = f3.add_subplot(1,1,1)
        ax3.scatter(x_pos_norm, y_pos_norm, color='b')
        ax3.scatter(x_pos_norm_f, y_pos_norm_f, color ='r')
        ax3.set_xlim([-1, 1])
        ax3.set_ylim([-1, 1])
        plt.scatter(centre[0],centre[1], color = 'k')
        circle1 = plt.Circle([0,0] , 1, color = 'k', fill=False)
        ax3.add_artist(circle1)

    ### Make Voltage 2 displacement matrix
    actnum=np.arange(0,19,1)
    linacts=np.where(np.logical_or(actnum==4,actnum==7))
    others=np.where(np.logical_and(actnum!=4,actnum!=7))
    V2D = Dm.gather_displacement_matrix(mirror, sh, x_pos_zero_f, y_pos_zero_f)
    V2D_inv = np.linalg.pinv(V2D)

    #Calculate difference between flat and now, and adjust for it with scaling factor
    centroid_control = np.hstack((x_pos_flat_f, y_pos_flat_f))
    print("start making the wavefront flat")
    for i in range(30):
        sh.snapImage()
        zero_image = sh.getImage().astype(float)
        x_pos_flat_f, y_pos_flat_f = Hm.centroid_positions(x_pos_zero_f, y_pos_zero_f, zero_image, xx, yy)
        centroid_0 = np.hstack((x_pos_flat_f, y_pos_flat_f))
        d = centroid_control - centroid_0
        u_dm_diff = np.dot(V2D_inv, d)
        u_dm -= scaling * u_dm_diff
        set_displacement(u_dm, mirror)
        time.sleep(0.1)

    print("wavefront should be flat now")
    if show_hist_voltage:
        plt.hist(u_dm)

    if np.any(np.abs(u_dm) > 1.0):
        print("maximum deflection of mirror reached")
        print(u_dm)

    if (show_hist_voltage or show_accepted_spots):
        plt.show()

    return u_dm, x_pos_norm_f, y_pos_norm_f, x_pos_zero_f, y_pos_zero_f, V2D
