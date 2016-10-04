import edac40
import numpy as np

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
