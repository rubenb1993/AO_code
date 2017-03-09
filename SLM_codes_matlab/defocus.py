import numpy as np
from PyQt4 import QtGui, QtCore
import time

class th(QtCore.QThread):
    def __init__(self,parent):        
        QtCore.QThread.__init__(self)
        self.parent=parent


    def add_defocus(self,a):
        i=0
        print(self.parent.movie_running)
        while self.parent.movie_running:
            a_def = a * np.sin(((i/20.0))*2*np.pi)
            print(a_def)
            phase=np.copy(self.parent.ramp)
            phase[self.parent.aperture]+=a_def * np.sqrt(2)*(2 * self.parent.r[self.parent.aperture]**2 - 1)

            self.parent.updateslm(phase)
            time.sleep(0.1)
            i += 1
        
