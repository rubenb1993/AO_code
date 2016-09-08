import clr

clr.AddReference("System.Runtime.InteropServices")

from System import Array,Double,Boolean
import System.Runtime.InteropServices
import Thorlabs.TLDFM_64.Interop
import Thorlabs.TLDFM_64.Interop.NativeMethods
import Ivi.Visa.Interop

import numpy
import time



class dm:
    
    def __init__(self):
        rm=Ivi.Visa.Interop.ResourceManager()

        resourcelist=rm.FindRsrc(Thorlabs.TLDFM_64.Interop.TLDFM.FindPattern)

        self.dm=Thorlabs.TLDFM_64.Interop.TLDFM(resourcelist[0],True,True)

        self.dm.reset()

        
        offsetact=Array[Double]([100.0]*40)

        offsettiptilt=Array[Double]([100.0]*3)

        self.dm.set_voltages(offsetact,offsettiptilt)


    def setvoltages(self,volt):

        voltact=volt[0:40]+100.0
        if voltact[voltact>200.0].size:
            print "out of bounds"
        voltact[voltact>200.0]=200.0
        if voltact[voltact<0.0].size:
            print "out of bounds"
        voltact[voltact<0.0]=0.0
        volttiptilt=volt[40:43]+100.0
        if volttiptilt[volttiptilt>200.0].size:
            print "out of bounds"
        volttiptilt[volttiptilt>200.0]=200.0
        if volttiptilt[volttiptilt<0.0].size:
            print "out of bounds"
        volttiptilt[volttiptilt<0.0]=0.0
        

        voltactarray=Array[Double](voltact)
        volttiptiltarray=Array[Double](volttiptilt)

        self.dm.set_voltages(voltactarray,volttiptiltarray)
        

    def relax(self):
        for i in range(1000):
            volttiptilt=numpy.ones(3)*((-1.0)**(i+1))*(100.0/(i+1))
            voltact=numpy.ones(40)*((-1.0)**(i+1))*(100.0/(i+1))

            self.setvoltages(numpy.concatenate((voltact,volttiptilt)))

            time.sleep(0.001)






    def getvoltages(self):
        voltact=Array[Double]([0.0]*40)

        volttiptilt=Array[Double]([0.0]*3)

        self.dm.get_voltages(voltact,volttiptilt)

        voltages=numpy.zeros(43)

        for i in range(40):
            voltages[i]=voltact[i]

        for i in range(3):
            voltages[40+i]=volttiptilt[i]

        return voltages

        







