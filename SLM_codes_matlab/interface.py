import numpy as np
from PyQt4 import QtGui, QtCore
import PIL.Image as pil
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'H:/Desktop/AO_code')
import time, os
import defocus
import camera_control
import orientation_test
import Zernike as Zn
import Hartmann as Hm
import mirror_control as mc
import LSQ_method as LSQ
import json
import image_acquisition
import phase_unwrapping_test as pw

class MainWindow(QtGui.QMainWindow):
    def __init__(self, app):

        # Folder and the to be added aberrations
        #---
        self.folder = "20170515_leica_544_33/"
        self.shutter_time = 600.0 #interferogram camera [us]
        self.sh_exp = 0.125 #shack-hartmann sensor [us]
        self.j_max = 50

        self.n_max = 10
        self.j = Zn.max_n_to_j(self.n_max, order = 'brug')[str(self.n_max)]

        ### either load in or set the aberrations yourself. vector a will be fed to the image making algorithm
##        a = np.zeros(len(self.j))
##        #a[35] = 2.0
##        a[3] = 2.5
##        a[10] = 1.0
##        a[19] = 0.50
##        self.a = a
        a = np.load("objective_544_nm_3.3_pi_pv.npy")
        self.a = a
        np.save(self.folder + "reference_slm_vector.npy", self.a)

        ### start necessary sub-processes
        QtGui.QMainWindow.__init__(self)
        self.defocus_thread = defocus.th(self)
        self.img_thread = camera_control.th(self, self.sh_exp)
        self.test_thread = orientation_test.th(self)
        self.acquisition_th = image_acquisition.th(self)

        #---
        # Initialize ramp
        #---
        x=np.linspace(0,1919,1920)-960
        y=-1 * (np.linspace(0,1079,1080)-540)
        self.x,self.y=np.meshgrid(x,y)
        r2=self.x**2+self.y**2
        self.r=np.sqrt(r2)/256.0
        self.aperture=np.where(r2<(256**2))
        self.inside_aperture = np.where(r2<(253**2))
        self.nonaperture = np.where(r2>65536)
        self.ramp=np.zeros((1080,1920))
        self.ramp[self.aperture]=((self.x[self.aperture]+960).astype(float)%7)/7.0*2*np.pi


        #---
        # Initial constants
        #---
        self.Nint = 638
        self.NSLM = 2*256
        self.f_sh = 14.2e-3
        self.pitch_sh = 300.0e-6
        self.px_size_sh = 4.65e-6
        self.wavelength = 632.9e-9
        #self.r_sh_m = 2.048e-3 #initial guess based on SLM circle diameter
        self.r_sh_px = 439.32#
        self.r_sh_m = self.r_sh_px * self.px_size_sh #int(self.r_sh_m/self.px_size_sh)
        self.box_len = 32#34/2
        self.ny = 1024
        self.nx = 1280
        ## grid of x and y coordinates SH sensor
        i, j = np.arange(0, self.nx, 1), np.arange(self.ny-1, -1, -1)
        self.ii, self.jj = np.meshgrid(i, j)      

        #---
        # Initial matrices for computations on interferogram
        #---
        self.power_mat = Zn.Zernike_power_mat(self.n_max, order = 'brug')
        xi, yi = np.linspace(-1, 1, self.Nint), np.linspace(-1, 1, self.Nint)
        self.xi, self.yi = np.meshgrid(xi, yi)
        self.mask = [np.sqrt((self.xi)**2 + (self.yi)**2) >= 1]

        #---
        # Matrices for setting up SLM patterns
        #---
        self.x_slm, self.y_slm = -1 * self.x[self.aperture]/256.0, self.y[self.aperture]/256.0
        if os.path.isfile("Z_SLM_" + str(self.n_max) + ".npy"):
            print("found the SLM matrix!")
            self.Z_SLM = np.load("Z_SLM_" + str(self.n_max) + ".npy")
        else:
            self.Z_SLM = -1 * Zn.Zernike_xy(self.x_slm, self.y_slm, self.power_mat, self.j)
            np.save("Z_SLM_" + str(self.n_max) + ".npy", self.Z_SLM)
        assert(self.Z_SLM.shape[0] == self.ramp[self.aperture].shape[0])
        
        ##------------------------------------------------------------------------------------------------------------
        ## SLM output
        ##------------------------------------------------------------------------------------------------------------

        self.app=app
        self.desktop = QtGui.QDesktopWidget()
        screen = self.desktop.screenGeometry(1)

        self.hologramArea = QtGui.QScrollArea()
        self.hologramArea.setGeometry(QtCore.QRect(0, 0, 1920, 1080))
        self.hologramArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.hologramArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.hologramLabel = QtGui.QLabel()
        hologram = np.zeros((1920, 1080))
        imgHologram = QtGui.QImage(hologram.data, 1920, 1080, QtGui.QImage.Format_Indexed8)
        self.hologramLabel.setPixmap(QtGui.QPixmap.fromImage(imgHologram))
        self.hologramArea.setWidget(self.hologramLabel)
        self.hologramArea.move(screen.left(), screen.top())
        self.hologramArea.setStyleSheet("border: 0px")
        self.hologramArea.showFullScreen()

        #---
        # Make Layout & buttons
        #---
        Box = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()

        ##### -----
        ###   Test alignment with a defocus movie
        ##### -----
        self.defocus_button = QtGui.QPushButton('Defocus Movie')
        self.defocus_stop = QtGui.QPushButton('start/stop movie')
        self.defocus_mag = QtGui.QDoubleSpinBox()
        self.defocus_mag.setRange(0.0,80.0)
        self.defocus_mag.setValue(5.0)
        self.defocus_mag.setPrefix('a_2^0 = ')
        self.movie_running = True
        self.defocus_stop.clicked[bool].connect(self.stop_movie)
        self.defocus_button.clicked[bool].connect(self.defocus_movie)
        layout.addWidget(self.defocus_button)
        layout.addWidget(self.defocus_mag)
        layout.addWidget(self.defocus_stop)        

        #####  -----
        ###     Test if x-and-y are well defined by adding tip and tilt positive on the SLM
        #####  ----
        self.tilt_button = QtGui.QPushButton('tilt test')
        self.tilt_button.clicked[bool].connect(self.test_tilt)
        layout.addWidget(self.tilt_button)
        
        #### ---
        ###     Test alignment by creating a knife edge
        #### ---
        self.quart_button = QtGui.QPushButton('knife edge')
        self.quart_button.clicked[bool].connect(self.test_orientation)
        layout.addWidget(self.quart_button)

        #### ----
        ### Create interferograms / SH-o-grams using self.a
        #### ----
        self.get_imgs = QtGui.QPushButton('Get interferograms!')
        self.get_imgs.clicked[bool].connect(self.acq_images)
        layout.addWidget(self.get_imgs)

        #### ---
        ### Create interferogram / SH-o-grams by creating a random surface and smoothing that
        #### ---
        self.rand_button = QtGui.QPushButton('random surface')
        self.rand_button.clicked[bool].connect(self.random_surface)
        self.n_mag = QtGui.QSpinBox()
        self.f0_mag = QtGui.QDoubleSpinBox()
        self.n_mag.setRange(0,15)
        self.n_mag.setValue(5)
        self.n_mag.setPrefix('n = ')
        self.f0_mag.setRange(0.0, 20.0)
        self.f0_mag.setValue(3.0)
        self.f0_mag.setPrefix('f0 = ')
        layout.addWidget(self.n_mag)
        layout.addWidget(self.f0_mag)
        layout.addWidget(self.rand_button)

        #### ---
        ### Create a movie of a gradually increading ramp to check if you don't create more order
        #### ---
        self.ramp_button = QtGui.QPushButton('ramp test')
        self.ramp_button.clicked[bool].connect(self.ramp_movie)
        layout.addWidget(self.ramp_button)


        ##### ----
        ### Test shutter Time of SH sensor
        ##### ----
        self.sh_calib = QtGui.QPushButton('SH Calibration')
        self.shutter_time = QtGui.QDoubleSpinBox()
        self.shutter_time.setRange(0.0,10.0)
        self.shutter_time.setValue(0.250)
        self.shutter_time.setPrefix('Exposure time = ')
        self.sh_calib.clicked[bool].connect(self.test_shut_sh)
        layout.addWidget(self.shutter_time)
        layout.addWidget(self.sh_calib)

        
        Box.setLayout(layout)

        self.setCentralWidget(Box)
        self.updateslm(self.ramp)

    def updateslm(self, phase):
        #print "updating"
        out = np.zeros(phase.shape)
        out[self.aperture] = phase[self.aperture] - np.min(phase[self.aperture])
      
        out=(((out%(2*np.pi))/(2*np.pi))*255.0).astype('uint8')
        ###---
        # Check if phase difference does not exceed 0.5 pi
        doutdx = np.abs((out[:, 1:-1].astype('float') - out[:, :-2].astype('float')))*(2*np.pi/255)
        doutdx = np.minimum(doutdx, 2*np.pi - doutdx)
        doutdy = np.abs(out[1:-1, :].astype('float') - out[:-2, :].astype('float'))*(2*np.pi/255)
        doutdy = np.minimum(doutdy, 2*np.pi - doutdy)
##        f, ax = plt.subplots(1,3, sharex = True, sharey = True)
##        ax[0].imshow(out[:-2, :-2].astype('float'), origin = 'lower')
##        ax[1].imshow(doutdx[:-1, :], origin = 'lower', vmin = 0.0 * np.pi, vmax = 0.5 * np.pi)
##        ax[2].imshow(doutdy[:, :-1], origin = 'lower', vmin = 0.0 * np.pi, vmax = 0.5 * np.pi)
##        plt.show()
        print("max x stepsize: " + str(np.max(doutdx[self.inside_aperture]/np.pi)) + "pi")
        print("max y stepsize: " + str(np.max(doutdy[self.inside_aperture]/np.pi)) + "pi")
        #assert(np.all(np.less(doutdx[self.inside_aperture], 0.5*np.pi)))
        #assert(np.all(np.less(doutdy[self.inside_aperture], 0.5*np.pi)))
        ###---

        imgHologram = QtGui.QImage(out.data, 1920, 1080, QtGui.QImage.Format_Indexed8)
        self.hologramLabel.setPixmap(QtGui.QPixmap.fromImage(imgHologram))
        self.hologramArea.setWidget(self.hologramLabel)
        self.app.processEvents()

    def calc_out_slope(self, phase):
        out = np.zeros(phase.shape)
        out[self.aperture] = phase[self.aperture] - np.min(phase[self.aperture])
      
        out=(((out%(2*np.pi))/(2*np.pi))*255.0).astype('uint8')
        ###---
        # Check if phase difference does not exceed 0.5 pi
        doutdx = np.abs((out[:, 1:-1].astype('float') - out[:, :-2].astype('float')))*(2*np.pi/255)
        doutdx = np.minimum(doutdx, 2*np.pi - doutdx)
        doutdy = np.abs(out[1:-1, :].astype('float') - out[:-2, :].astype('float'))*(2*np.pi/255)
        doutdy = np.minimum(doutdy, 2*np.pi - doutdy)
        doutdx = np.max(doutdx[self.inside_aperture]/np.pi)
        doutdy = np.max(doutdy[self.inside_aperture]/np.pi)
        return doutdx, doutdy
        
    def random_surface(self):
        #temporary_vals
        n = self.n_mag.value()
        f0 = self.f0_mag.value()

        x_slm_check, y_slm_check = -1 * self.x/256.0, self.y/256.0
        mask = [np.sqrt((x_slm_check)**2 + (y_slm_check)**2) >= 1]
        if os.path.isfile("Z_mat_check_" + str(self.n_max) + ".npy"):
            print("found the matrix!")
            self.Z_mat_check = np.load("Z_mat_check_" + str(self.n_max) + ".npy")
        else:
            self.Z_mat_check = Zn.Zernike_xy(x_slm_check, y_slm_check, self.power_mat, self.j)
            np.save("Z_mat_check_" + str(self.n_max) + ".npy", self.Z_mat_check)
        
        doutdx = 100
        doutdy = 100
        while doutdx > (0.45 - 0.29) * np.pi and doutdy > (0.45) *np.pi:
            print("let's try again")
            random_surf = 2 * np.pi * np.random.randn(self.x.shape[0], self.x.shape[1])
            random_surf = np.flipud(np.fliplr(random_surf))
            butter_surf = pw.butter_filter_unwrapped(random_surf, n, f0, pad = True)
            butter_surf *= 3*np.pi/(np.max(butter_surf))
            a = -1 * np.linalg.lstsq(self.Z_SLM, butter_surf[self.aperture])[0]
            check_phase = np.dot(self.Z_mat_check, a)
            doutdx, doutdy = self.calc_out_slope(check_phase)
            print("max x stepsize: " + str(doutdx + 0.29) + "pi")
            print("max y stepsize: " + str(doutdy) + "pi")
        print(a)
        f, ax = plt.subplots(1,2)
        ax[0].set_title('original filtered surface')
        ax[0].imshow(np.ma.array(butter_surf, mask = mask), origin = 'lower')
        ax[1].set_title('Zernike fit up to ' + str(self.n_max) + ' order')
        ax[1].imshow(np.ma.array(np.dot(self.Z_mat_check, a), mask = mask), origin = 'lower', vmin = -3*np.pi, vmax = 3*np.pi)
        plt.show()
        self.a = a
        np.save(self.folder + "reference_slm_vector.npy", self.a)
        self.updateslm(self.ramp)
        self.acquisition_th.acquire_images(self, self.folder, self.shutter_time, self.a)


    def stop_movie(self):
        self.movie_running = False
        print(self.movie_running)
        return self.movie_running

    def closeEvent(self, event):
        self.hologramArea.hide()

    def defocus_movie(self):
        self.movie_running = True
        self.defocus_thread.add_defocus(self.defocus_mag.value())

    def test_tilt(self):
        self.test_thread.tilt_test(self)
##        wf = self.img_thread.snapshot_int(1000.0)
##        pil.fromarray(wf.astype('uint8')).save('snapshots/interferogram.tif')

    def acq_images(self):
        self.updateslm(self.ramp)
        self.acquisition_th.acquire_images(self, self.folder, self.shutter_time, self.a)
        
    def test_shut_sh(self):
        self.sh.setProperty("cam", "Exposure", self.shutter_time.value())
        sh_spots = self.img_thread.snapshot_sh()
        sh_spots = self.img_thread.snapshot_sh()
        pil.fromarray(sh_spots.astype('uint8')).save('snapshots/sh_calibration.tif')
        
    def test_orientation(self):
        self.test_thread.orientation_test(self)

    def ramp_movie(self):
        self.test_thread.ramp_movie(self)


    def temp_make_flat(self):
        raw_input("block SLM")
        ref_mir_sh = self.img_thread.snapshot_sh()
        ref_mir_sh = self.img_thread.snapshot_sh()
        ref_mir_int = self.img_thread.snapshot_int(self.shutter_time)
        
        raw_input("block referencemirror")
        initial_slm_sh = self.img_thread.snapshot_sh()

        self.x_pos_zero, self.y_pos_zero = Hm.zero_positions(np.copy(initial_slm_sh), spotsize = self.box_len)
        self.x_pos_flat, self.y_pos_flat = Hm.centroid_positions(self.x_pos_zero, self.y_pos_zero, np.copy(ref_mir_sh), self.ii, self.jj, spot_size = self.box_len)

        a_flat, G, inside = self.LSQ_calculation(initial_slm_sh, ref_mir_sh)
        x_pos_flat_f, y_pos_flat_f = mc.filter_positions(inside, self.x_pos_flat, self.y_pos_flat)
        print(a_flat)
        a_neutralize = -0.5 * a_flat
        
        phase = np.copy(self.ramp)
        phase[self.aperture] += np.dot(self.Z_SLM, a_neutralize)
        self.updateslm(phase)
        time.sleep(1)
        
        for i in range(3):
            slm_sh_iterate = self.img_thread.snapshot_sh()
            x_pos_dist, y_pos_dist = Hm.centroid_positions(self.x_pos_zero, self.y_pos_zero, slm_sh_iterate, self.ii, self.jj, spot_size = self.box_len)
            x_pos_dist_f, y_pos_dist_f = mc.filter_positions(inside, x_pos_dist, y_pos_dist)
            s = np.hstack(Hm.centroid2slope(x_pos_dist_f, y_pos_dist_f, x_pos_flat_f, y_pos_flat_f, self.px_size_sh, self.f_sh, self.r_sh_m, self.wavelength))
            a_iter = -0.5 * np.linalg.lstsq(G, s)[0]
            print(np.sum((-2. * a_iter)**2/30.))
            print(-2. * a_iter)
            phase[self.aperture] += np.dot(self.Z_SLM, a_iter)
            self.updateslm(phase)
            time.sleep(1)
            ### check the flat making ability of this algorithm, and test if it is better with more iterations
##            raw_input("interferogram")
##            interferogram = self.img_thread.snapshot_int(self.shutter_time)
##            pil.fromarray(interferogram.astype('float')).save(self.folder + "flat_int" + str(i) + ".tif")
##            raw_input("block ref")
        
        flat_slm_int = self.img_thread.snapshot_int(self.shutter_time)
        zero_pos_dm = self.img_thread.snapshot_sh()
        #pil.fromarray(flat_slm_int.astype('uint8')).save('snapshots/flat_slm_int.tif')
        
        raw_input("new interferogram")
        int_flat = self.img_thread.snapshot_int(self.shutter_time)
        #pil.fromarray(int_flat.astype('uint8')).save('snapshots/flat_wf_int.tif')
        return phase, int_flat, ref_mir_sh, zero_pos_dm

    def cutout_int(self, interferogram, x_0 = 212, y_0 = 282, N = 614):
        return interferogram[y_0:y_0 + N, x_0:x_0 + N]

    def LSQ_calculation(self, initial_slm_sh, ref_mir_sh, G = None):
        ### if spots in initial_slm_sh are more than self.box_len away from their corresponding centroid of ref_mir_sh, it can cause problems
     
        x_pos_dist, y_pos_dist = Hm.centroid_positions(self.x_pos_zero, self.y_pos_zero, np.copy(initial_slm_sh), self.ii, self.jj, spot_size = self.box_len)
        print(len(self.x_pos_flat))
        centre = Hm.centroid_centre(self.x_pos_flat, self.y_pos_flat)
        x_pos_norm, y_pos_norm = (self.x_pos_flat - centre[0])/self.r_sh_px, (self.y_pos_flat - centre[1])/self.r_sh_px
        inside = np.where(np.sqrt(x_pos_norm**2 + y_pos_norm**2)<= 1 + (self.box_len/self.r_sh_px))
        x_pos_zero_f, y_pos_zero_f, x_pos_flat_f, y_pos_flat_f, x_pos_norm_f, y_pos_norm_f, x_pos_dist_f, y_pos_dist_f = mc.filter_positions(inside, self.x_pos_zero, self.y_pos_zero, self.x_pos_flat, self.y_pos_flat, x_pos_norm, y_pos_norm, x_pos_dist, y_pos_dist)
        if type(G).__module__ != np.__name__:
            G = LSQ.matrix_avg_gradient(x_pos_norm_f, y_pos_norm_f, self.j, self.r_sh_px, self.power_mat, self.box_len)

        s = np.hstack(Hm.centroid2slope(x_pos_dist_f, y_pos_dist_f, x_pos_flat_f, y_pos_flat_f, self.px_size_sh, self.f_sh, self.r_sh_m, self.wavelength))
        
        a = np.linalg.lstsq(G,s)[0]
        return a, G, inside


app = QtGui.QApplication(sys.argv)
main = MainWindow(app)
main.show()

sys.exit(app.exec_())
