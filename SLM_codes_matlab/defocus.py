import numpy as np
from PyQt4 import QtGui, QtCore
import time
import PIL.Image as pil

class th(QtCore.QThread):
    def __init__(self,parent):        
        QtCore.QThread.__init__(self)
        self.parent=parent


    def add_defocus(self,a):
        ## 
        i=0
        print(self.parent.movie_running)
        while self.parent.movie_running:
            a_def = a * np.sin(((i/20.0))*2*np.pi)
            print(a_def)
            phase=np.copy(self.parent.ramp)
            phase[self.parent.aperture]+= a_def * np.sqrt(3)*(2 * self.parent.r[self.parent.aperture]**2 - 1)
            #phase[self.parent.nonaperture] = 0.0
            self.parent.updateslm(phase)
            time.sleep(0.1)
            i += 1

    def add_tip(self, a):
        ## 
        i=0
        print(self.parent.movie_running)
        while self.parent.movie_running:
            a_def = a * np.sin(((i/30.0))*2*np.pi)
            print(a_def)
            phase=np.copy(self.parent.ramp)
            phase[self.parent.aperture]+= a_def * 2 * (self.parent.x[self.parent.aperture]+960)/256.0
            #phase[self.parent.nonaperture] = 0.0
            self.parent.updateslm(phase)
            time.sleep(1)
            sh_spots = self.parent.img_thread.snapshot_sh()
            pil.fromarray(sh_spots.astype('uint8')).save('20170411_calibration/sh_spot_tip' + str(i) + '.tif')
            i += 1

    def add_tilt(self, a):
        ## 
        i=0
        print(self.parent.movie_running)
        while self.parent.movie_running:
            a_def = a * np.sin(((i/30.0))*2*np.pi)
            print(a_def)
            phase=np.copy(self.parent.ramp)
            phase[self.parent.aperture]+= a_def * 2 * (self.parent.y[self.parent.aperture]+(1080/2))/256.0
            #phase[self.parent.nonaperture] = 0.0
            self.parent.updateslm(phase)
            time.sleep(1)
            sh_spots = self.parent.img_thread.snapshot_sh()
            pil.fromarray(sh_spots.astype('uint8')).save('20170411_calibration/sh_spot_tilt' + str(i) + '.tif')
            i += 1

    def single_defocus(self, a):
        phase = np.copy(self.parent.ramp)
        phase[self.parent.aperture] += a * np.sqrt(3) * (2 * self.parent.r[self.parent.aperture]**2 - 1)
        self.parent.updateslm(phase)
