import numpy as np
import matlab.engine as mateng
from PyQt4 import QtGui, QtCore
import sys
if "C:\Program Files\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Program Files\Micro-Manager-1.4")
import MMCorePy

class th(QtCore.QThread):
    def __init__(self, parent, sh_exp):
        QtCore.QThread.__init__(self)
        self.parent = parent
        eng = mateng.start_matlab()
        self.eng = eng

        sh = MMCorePy.CMMCore()
        sh.loadDevice("cam","ThorlabsUSBCamera","ThorCam")
        sh.initializeDevice("cam")
        sh.setCameraDevice("cam")
        #sh.setProperty("cam", "Pixel Clock", 30.)
        #sh.setProperty("cam", "PixelType", '8bit mono')
        sh.setProperty("cam", "Exposure", sh_exp)
        self.parent.sh = sh
        
    def snapshot_int(self, exposure):
        ## makes a snapshot using matlab.
        ## Initiates the camera, sets the exposure time and closes the camera)
        img = self.eng.complete_image(exposure)
        img = np.flipud(np.fliplr(np.asarray(img))) #flip up-down and lr to align SLM and camera
        return img

    def snapshot_sh(self):
        # use thorlabs software to make an image
        self.parent.sh.snapImage()
        img = self.parent.sh.getImage().astype(float)
        return img
