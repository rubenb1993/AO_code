import numpy as np
import matlab.engine as mateng
from PyQt4 import QtGui, QtCore

class th(QtCore.QThread):
    def __init__(self, parent):
        QtCore.QThread.__init__(self)
        self.parent = parent
        eng = mateng.start_matlab()
        self.eng = eng
        
    def snapshot(self, exposure):
        ## makes a snapshot using matlab.
        ## Initiates the camera, sets the exposure time and closes the camera)
        img = self.eng.complete_image(exposure)
        img = np.asarray(img)
        return img
