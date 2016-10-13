import numpy as np
import time
import mirror_control as mc

import edac40
import mirror_control as mc

mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here

def actuators_test():
    for i in range(19):
        print i
        raw_input("next?")
        voltages = np.ones(19)*6.0
        mirror.set(voltages)
        voltages[i]+=6.0
        time.sleep(1)
        mirror.set(voltages)

def defocus_cycle():
    i=1
    while i:
        i+=1
        voltage=np.sin(i%100*2*np.pi/100.0)*6.0
        voltages=np.ones(19)*voltage+6.0
        voltages[4]=6.0
        voltages[7]=6.0
        mirror.set(voltages)
##        print np.sin(i%100*2*np.pi/100.0)*6.0
        time.sleep(0.05)

def tt_test():
    i = 1
    while i:
        i+=1
        u_dm = np.zeros(19)

        if i%4 == 0:
            u_dm[4] = 0.5
            mc.set_displacement(u_dm, mirror)
            time.sleep(1)
        elif i%4 == 1:
            u_dm[4] = -0.5
            mc.set_displacement(u_dm, mirror)
            time.sleep(1)
        elif i%4 == 2:
            u_dm[7] = 0.5
            mc.set_displacement(u_dm, mirror)
            time.sleep(1)
        else:
            u_dm[7] = 0.5
            mc.set_displacement(u_dm, mirror)
            time.sleep(1)

mc.set_displacement(0.0 * np.ones(19), mirror)
##actuators_test()        
#defocus_cycle()   

#mirror.close()
