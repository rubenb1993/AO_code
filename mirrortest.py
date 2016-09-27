import numpy as np
import time

import edac40

mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here

voltages = 6.0 * np.ones(19)  # V = 0 to 12V
mirror.set(voltages)


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
##actuators_test()        
#defocus_cycle()   

#mirror.close()
