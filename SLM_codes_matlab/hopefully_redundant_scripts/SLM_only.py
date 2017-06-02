import numpy as np
from PyQt4 import QtGui, QtCore
import PIL.Image as pil
import sys
import time, os

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        
        QtGui.QMainWindow.__init__(self)


        ##------------------------------------------------------------------------------------------------------------
        ## SLM output
        ##------------------------------------------------------------------------------------------------------------

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

        ##------------------------------------------------------------------------------------------------------------
        ## Control Window
        ##------------------------------------------------------------------------------------------------------------

        self.resize(800, 600)
        self.setWindowTitle('SLM Controller')

        outerBox = QtGui.QWidget()
        bottomBox = QtGui.QWidget()

        layout = QtGui.QVBoxLayout()
        layoutBottom = QtGui.QHBoxLayout()

        # Display the Pattern
        self.display = QtGui.QScrollArea()
        self.displayLabel = QtGui.QLabel()
        self.displayLabel.setPixmap(QtGui.QPixmap.fromImage(imgHologram))
        self.display.setWidget(self.displayLabel)

        self.path = QtGui.QLineEdit(self)

        self.select = QtGui.QPushButton('Select')
        self.select.clicked[bool].connect(self.openDialog)

        self.load = QtGui.QPushButton('Load')
        self.load.clicked[bool].connect(self.loadPath)

        # Bottom Box Organisation
        layoutBottom.addWidget(self.path)
        layoutBottom.addWidget(self.select)
        layoutBottom.addWidget(self.load)
        bottomBox.setLayout(layoutBottom)

        # Overall Organisation
        layout.addWidget(self.display)
        layout.addWidget(bottomBox)
        outerBox.setLayout(layout)

        self.setCentralWidget(outerBox)

    # Shut down all windows when exiting
    def closeEvent(self, event):
        if self.desktop != None:
            self.hologramArea.hide()

    def openDialog(self):
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file',
                                            'c:\\', "Image files (*.jpg *.gif *.tif *.png)")
        hologram = self.convertToRGB(np.array(pil.open(str(fname))))
        imgHologram = QtGui.QImage(hologram.data, 1920, 1080, QtGui.QImage.Format_Indexed8)
        self.displayLabel.setPixmap(QtGui.QPixmap.fromImage(imgHologram))
        self.display.setWidget(self.displayLabel)
        app.processEvents()
        self.path.setText(fname)

    def loadPath(self):
        img = np.array(pil.open(str(self.path.text())))
        img = self.convertToRGB(img)
        self.putHologram(img)
        return

    # Update the Screen
    def putHologram(self, hologram):
        imgHologram = QtGui.QImage(hologram.data, 1920, 1080, QtGui.QImage.Format_Indexed8)
        self.hologramLabel.setPixmap(QtGui.QPixmap.fromImage(imgHologram))
        self.hologramArea.setWidget(self.hologramLabel)
        self.displayLabel.setPixmap(QtGui.QPixmap.fromImage(imgHologram))
        self.display.setWidget(self.displayLabel)
        app.processEvents()
        return

    # Convert to the correct bit encoding
    def convertToRGB(self, phase):
        # Convert the -pi to pi phase into bits
        out = phase - np.min(phase)
        out=(((out%(2*np.pi))/(2*np.pi))*255.0).astype('uint8')
          
        return out

app = QtGui.QApplication(sys.argv)
main = MainWindow()
main.show()

sys.exit(app.exec_())
