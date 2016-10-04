import sys
import os
import time
if "C:\Program Files\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Program Files\Micro-Manager-1.4")
import MMCorePy
import PIL.Image
import numpy as np
#from matplotlib import rc
import Hartmann as Hm
import matplotlib.pyplot as plt
from matplotlib import cm
import edac40

def set_displacement(u_dm, mirror):
    u_dm = u_dm * 72.0
    u_l = np.zeros(u_dm.shape)
    u_l = np.maximum(u_dm, -72.0 * np.ones(u_l.shape))
    u_l = np.minimum(u_dm, 72.0 * np.ones(u_l.shape))    
    actnum=np.arange(0,19,1)
    linacts=np.where(np.logical_or(actnum==4,actnum==7))
    others=np.where(np.logical_and(actnum!=4,actnum!=7))
    u_l[linacts]=(u_dm[linacts]+72.0)/12
    u_l[others]=np.sqrt(u_dm[others]+72.0)
    
    mirror.set(u_l)

def gather_displacement_matrix(mirror, sh, x_pos_zero, y_pos_zero):
    actuators = 19
    u_dm_0 = np.zeros(actuators)
    stroke = 0.5
    set_displacement(u_dm_0, mirror)
    time.sleep(0.2)
    sh.snapImage()
    zero_image = sh.getImage().astype(float) #re load image due to corruption

    ## Given paramters for centroid gathering
    [ny,nx] = zero_image.shape
    px_size = 5.2e-6     # width of pixels 
    f = 17.6e-3            # focal length
    x = np.linspace(1, nx, nx)
    y = np.linspace(1, ny, ny)
    xx, yy = np.meshgrid(x, y)
    j_max= 10           # maximum fringe order

    ### test linearity of non-linear actuators with new set_displacement function
    x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, zero_image, xx, yy)
    centroid_0 = np.hstack((x_pos_flat, y_pos_flat))

    V2D_plus = np.zeros(shape = (2 * len(x_pos_zero), actuators))
    V2D_min = np.zeros(shape = (2 * len(x_pos_zero), actuators))

    actnum=np.arange(0,19,1)
    linacts=np.where(np.logical_or(actnum==4,actnum==7))
    others=np.where(np.logical_and(actnum!=4,actnum!=7))

    for i in np.nditer(others):
        print(i)
        voltages = np.zeros(actuators)
        voltages[i] += stroke
        set_displacement(voltages, mirror)
        time.sleep(0.2)
        sh.snapImage()
        image = sh.getImage().astype(float)
        centroid_i = np.hstack(Hm.centroid_positions(x_pos_zero, y_pos_zero, image, xx, yy))
        V2D_plus[:, i] = (centroid_0 - centroid_i) / stroke

        voltages = np.zeros(actuators)
        voltages[i] -= stroke
        set_displacement(voltages, mirror)
        time.sleep(0.2)
        sh.snapImage()
        image = sh.getImage().astype(float)
        centroid_i = np.hstack(Hm.centroid_positions(x_pos_zero, y_pos_zero, image, xx, yy))
        V2D_min[:, i] = (centroid_0 - centroid_i) / stroke

    for i in np.nditer(linacts):
        print(i)
        voltages = np.zeros(actuators)
        voltages[i] += stroke
        set_displacement(voltages, mirror)
        time.sleep(0.2)
        sh.snapImage()
        image = sh.getImage().astype(float)
        centroid_i = np.hstack(Hm.centroid_positions(x_pos_zero, y_pos_zero, image, xx, yy))
        V2D_plus[:, i] = (centroid_0 - centroid_i) / stroke

        voltages = np.zeros(actuators)
        voltages[i] -= stroke
        set_displacement(voltages, mirror)
        time.sleep(0.2)
        sh.snapImage()
        image = sh.getImage().astype(float)
        centroid_i = np.hstack(Hm.centroid_positions(x_pos_zero, y_pos_zero, image, xx, yy))
        V2D_min[:, i] = (centroid_0 - centroid_i) / stroke

    V2D = (V2D_plus - V2D_min) / 2.0
    return V2D
