import numpy as np
import edac40
import mirror_control as mc

global mirror
mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here

u_dm = np.zeros(19)
mc.set_displacement(u_dm, mirror)

i = 0
while True:
    i+=1
    defocus = np.sin(i/100.0)
    u_dm = np.ones(19) * defocus
    u_dm[4] = 0
    u_dm[7] = 0
    mc.set_displacement(u_dm, mirror)
