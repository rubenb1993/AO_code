import numpy as np
from PyQt4 import QtGui, QtCore
import time
import PIL.Image as pil

class th(QtCore.QThread):
    def __init__(self, parent):
        QtCore.QThread.__init__(self)
        self.parent = parent

    def acquire_images(self, parent, folder, shutter_time, a):
        ###---
        # Aproximate if phase is going to be too much
        ###---
        out = np.copy(self.parent.ramp)
        out[self.parent.aperture] += np.dot(self.parent.Z_SLM, a)
        doutdx = np.abs((out[:, 1:-1].astype('float') - out[:, :-2].astype('float')))
        doutdx = np.minimum(doutdx, 2*np.pi - doutdx)
        doutdy = np.abs(out[1:-1, :].astype('float') - out[:-2, :].astype('float'))
        doutdy = np.minimum(doutdy, 2*np.pi - doutdy)
        print("expected max x: " + str(np.max(doutdx[self.parent.inside_aperture])/np.pi) + "pi, max y: " + str(np.max(doutdy[self.parent.inside_aperture])/np.pi) + "pi.")

        ### make flat and save images
        phase, flat_wf, image_ref_mirror, zero_pos_dm = self.parent.temp_make_flat()
        pil.fromarray(flat_wf.astype('uint8')).save(folder + 'flat_wf.tif')
        pil.fromarray(image_ref_mirror.astype('uint8')).save(folder + 'image_ref_mirror.tif')
        pil.fromarray(zero_pos_dm.astype('uint8')).save(folder + "zero_pos_dm.tif")

        ### add phase and make photos
        phase_flat = np.copy(phase)
        phase[self.parent.aperture] += np.dot(self.parent.Z_SLM, a)
        self.parent.updateslm(phase)
        time.sleep(1)
         
        raw_input("block ref mirror")
        dist_image = self.parent.img_thread.snapshot_sh()
        pil.fromarray(dist_image.astype('uint8')).save(folder + 'dist_image.tif')

        raw_input("interferogram")
        image_i0 = self.parent.img_thread.snapshot_int(shutter_time)
        pil.fromarray(image_i0.astype('uint8')).save(folder + 'interferogram_0.tif')

        ### make tipped/tilted photos
        img_shape = list(flat_wf.shape)
        img_shape.append(4)
        imgs = np.zeros(img_shape)
        for i in range(4):
            raw_input("tip/tilt" + str(i+1))
            imgs[..., i] = self.parent.img_thread.snapshot_int(shutter_time)
            pil.fromarray(imgs[...,i].astype('uint8')).save(folder + 'interferogram_' + str(i+1) + '.tif')
        
