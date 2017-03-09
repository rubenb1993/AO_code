import numpy as np
from PyQt4 import QtGui, QtCore
import PIL.Image as pil
import sys
import time, os
import defocus


class MainWindow(QtGui.QMainWindow):
    def __init__(self, app):
        
        QtGui.QMainWindow.__init__(self)



        x=np.linspace(0,1919,1920)-960
        y=np.linspace(0,1079,1080)-540
        self.x,self.y=np.meshgrid(x,y)
        r2=self.x**2+self.y**2
        self.r=np.sqrt(r2)/256.0
        self.aperture=np.where(r2<65536)
        self.ramp=np.zeros((1080,1920))
        self.ramp[self.aperture]=((self.x[self.aperture]+960).astype(float)%9)/9.0*2*np.pi

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

        Box = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        
        self.defocus_thread = defocus.th(self)
        self.defocus_mag = QtGui.QDoubleSpinBox()
        self.defocus_mag.setMaximum(10.0)
        self.defocus_mag.setMinimum(0.0)
        self.defocus_button = QtGui.QPushButton('Defocus Movie')
        self.defocus_stop = QtGui.QPushButton('start/stop movie')
        self.movie_running = True
        self.defocus_stop.clicked[bool].connect(self.stop_movie)
        self.defocus_button.clicked[bool].connect(self.defocus_movie)
        layout.addWidget(self.defocus_button)
        layout.addWidget(self.defocus_mag)
        layout.addWidget(self.defocus_stop)


        
        Box.setLayout(layout)

        self.setCentralWidget(Box)
        self.updateslm(self.ramp)

    def updateslm(self, phase):
        #print "updating"
        out = phase - np.min(phase)
        out=(((out%(2*np.pi))/(2*np.pi))*255.0).astype('uint8')
        imgHologram = QtGui.QImage(out.data, 1920, 1080, QtGui.QImage.Format_Indexed8)
        self.hologramLabel.setPixmap(QtGui.QPixmap.fromImage(imgHologram))
        self.hologramArea.setWidget(self.hologramLabel)
        self.app.processEvents()

    def stop_movie(self):
        self.movie_running = not(self.movie_running)
        print(self.movie_running)
        return self.movie_running

    def closeEvent(self, event):
        self.hologramArea.hide()

    def defocus_movie(self):
        self.defocus_thread.add_defocus(self.defocus_mag.value())
    
app = QtGui.QApplication(sys.argv)
main = MainWindow(app)
main.show()

sys.exit(app.exec_())
