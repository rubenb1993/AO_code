import numpy as np
from PyQt4 import QtGui, QtCore
import time
import PIL.Image as pil

class th(QtCore.QThread):
    def __init__(self,parent):        
        QtCore.QThread.__init__(self)
        self.parent=parent

    def tilt_test(self, parent):
        """ Add a positive tip and tilt to the SLM in order to determine
        which x-and-y direction is positive with regards to the SH sensor
        """
        phase = np.copy(self.parent.ramp) #ramp is in positive x direction (pointing "left")
        self.parent.updateslm(phase)
        time.sleep(1)
        sh_flat = self.parent.img_thread.snapshot_sh(self.parent.sh_exp)
        yramp = np.zeros(phase.shape) #ramp in positive y direction (pointing "down")
        yramp[self.parent.aperture] = ((self.parent.y[self.parent.aperture]+(1080/2)).astype(float)%24)/24.0*2*np.pi
        xramp = np.zeros(phase.shape)
        xramp[self.parent.aperture] = ((self.parent.x[self.parent.aperture]+(1920/2)).astype(float)%24)/24.0 * 2 * np.pi
        phase += yramp
        phase += xramp #add extra ramp, should be going left
        self.parent.updateslm(phase)
        time.sleep(1)
        sh_tilt = self.parent.img_thread.snapshot_sh()
        pil.fromarray(sh_flat.astype('uint8')).save('snapshots/tilt_test_flat.tif')
        pil.fromarray(sh_tilt.astype('uint8')).save('snapshots/tilt_test_ramp.tif')

    def orientation_test(self, parent):
        """ Halves the spot, in order to simulate a knife-edge for aligning the setup
        """
        phase = np.copy(self.parent.ramp)
        L = np.zeros(phase.shape)
        [ny, nx] = phase.shape
        x = np.linspace(0, 2*np.pi, 20)
        y = np.ones(256*2)
        xx, yy = np.meshgrid(x, y)
        L[ny/2:, :] = 1
        phase *= L
        self.parent.updateslm(phase)
        time.sleep(1)
        int_quart = self.parent.img_thread.snapshot_int(800.0)
        sh_quart = self.parent.img_thread.snapshot_sh()
        pil.fromarray(int_quart.astype('uint8')).save('snapshots/quart_test_int.tif')
        pil.fromarray(sh_quart.astype('uint8')).save('snapshots/quart_test_sh.tif')

    def ramp_movie(self, parent):
        #adds a ramp between 0 and a half pi
        i = 0
        angle = np.linspace(0, 0.5*np.pi, 20)
        while self.parent.movie_running:
            phase = np.zeros(self.parent.x.shape)
            phase[self.parent.aperture] = angle[i%20] * (self.parent.x[self.parent.aperture]+960).astype(float)
            self.parent.updateslm(phase)
            time.sleep(0.1)
            i += 1

        
        
