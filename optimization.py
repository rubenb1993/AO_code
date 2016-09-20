import sys
if "C:\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Micro-Manager-1.4")
import MMCorePy
import PIL.Image
import numpy
import time

def cost_function(cam1):
    cam1.snapImage()
    image=cam1.getImage().astype(float)
    image_max=numpy.unravel_index(image.argmax(), image.shape)
    Airy_intensity = numpy.sum(image[image_max[0]-2:image_max[0]+2, image_max[1]-2:image_max[1]+2])
    total_intensity = numpy.sum(image[image_max[0]-20:image_max[0]+20, image_max[1]-20:image_max[1]+20])
    strehl = Airy_intensity/total_intensity

    return strehl,image[image_max[0]-20:image_max[0]+20, image_max[1]-20:image_max[1]+20]


### set paramters
half_voltage = 6
max_pert = half_voltage * 0.03
no_actuators = 19
voltages = half_voltage * numpy.ones(no_actuators)

mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here
mirror.set(voltages)

### Set up cameras 
cam1=MMCorePy.CMMCore()

cam1.loadDevice("cam","ThorlabsUSBCamera","ThorCam")
cam1.initializeDevice("cam")
cam1.setCameraDevice("cam")
cam1.setProperty("cam","PixelClockMHz",30)
cam1.setProperty("cam","Exposure",0.6)

cam2=MMCorePy.CMMCore()

cam2.loadDevice("cam","ThorlabsUSBCamera","ThorCam")
cam2.initializeDevice("cam")
cam2.setCameraDevice("cam")
cam2.setProperty("cam","PixelClockMHz",30)
cam2.setProperty("cam","Exposure",0.1)

cam1.snapImage()
cam2.snapImage()

cam1.snapImage()
cam2.snapImage()

PIL.Image.fromarray(cam1.getImage()).save("camera1.tif")
PIL.Image.fromarray(cam2.getImage()).save("camera2.tif")


## Optimization scheme
cost_old,image = cost_function(cam2)
cost_new = cost_old

for i in range(1000):
    perturbation=numpy.zeros(no_actuators)
    rand_int = numpy.random.randint(no_actuators)
    while rand_int == 4 or rand_int == 7:
        rand_int = numpy.random.randint(no_actuators)
    perturbation[rand_int]=1.0
    pert_size = numpy.random.uniform(-max_pert, max_pert)

    voltages_new = voltages + pert_size * perturbation
    mirror.set(voltages_new)
    time.sleep(0.005)
    cost_new,image = cost_function(cam2)

    if cost_new > cost_old:
        cost_old = cost_new
        voltages = voltages_new
    else:
        mirror.set(voltages)
        time.sleep(0.005)
        cost_old,image = cost_function(cam2)

    print cost_old, i

    if i%10==0:
        PIL.Image.fromarray(image).save("images\\camera"+str(i).zfill(4)+".tif")    

PIL.Image.fromarray(cam2.getImage()).save("shref.tif")
mirror.close()
