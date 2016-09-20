import dmctr
import sys
if "C:\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Micro-Manager-1.4")
import MMCorePy
import PIL.Image
import numpy as np
import time

def cost_function(cam1):
    cam1.snapImage()
    image=cam1.getImage().astype(float)
    image_max=np.unravel_index(image.argmax(), image.shape)
    Airy_intensity = np.sum(image[image_max[0]-2:image_max[0]+2, image_max[1]-2:image_max[1]+2])
    total_intensity = np.sum(image[image_max[0]-20:image_max[0]+20, image_max[1]-20:image_max[1]+20])
    strehl = Airy_intensity/total_intensity

    return strehl,image[image_max[0]-20:image_max[0]+20, image_max[1]-20:image_max[1]+20]

def centroid_positions(x_max, y_max, image, spot_size = 25):
    centroids = np.zeros(shape = (len(x_max),2))
    image[image<4]=0
    #spot_size = 25
    for i in range(len(x_max)):
        y_low = y_max[i] - spot_size
        y_high = y_max[i] + spot_size
        x_low = x_max[i] - spot_size
        x_high = x_max[i] + spot_size
        norm_photons = 1/np.sum(image[y_low: y_high, x_low: x_high])
        centroids[i,0] = norm_photons * np.sum(image[y_low: y_high, x_low: x_high] * xx[y_low: y_high, x_low: x_high])
        centroids[i,1] = norm_photons * np.sum(image[y_low: y_high, x_low: x_high] * yy[y_low: y_high, x_low: x_high])
    #print centroids[0,:]
    return np.reshape(centroids, -1)

def get_image(camera):
    camera.snapImage()
    image = camera.getImage().astype(float)
    return image


dm=dmctr.dm()

voltages=np.zeros(43)
##voltages=np.random.random(43)*50.0

dm.setvoltages(voltages)



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

reference=np.asarray(PIL.Image.open("shref.tif")).astype(float)

[ny,nx] = reference.shape
#img_pil = np.array(np.array(img_pil).reshape((ny,nx)))
xx, yy = np.meshgrid(np.linspace(1,nx,nx),np.linspace(1,ny,ny))
i,j = np.unravel_index(reference.argmax(), reference.shape)
img_pil_mask = reference > 3
img_pil_filtered = reference * img_pil_mask

list_of_maxima_x = []
list_of_maxima_y = []

while(np.amax(img_pil_filtered) > 10):
    y_max, x_max = np.unravel_index(img_pil_filtered.argmax(), img_pil_filtered.shape)
    list_of_maxima_x.append(x_max)
    list_of_maxima_y.append(y_max)
    img_pil_filtered[y_max - 40: y_max + 40, x_max - 40: x_max+40] = 0


x_max = np.array(list_of_maxima_x)
y_max = np.array(list_of_maxima_y)
reference_centroids = centroid_positions(x_max, y_max, reference)

voltage_standard = 50.0
voltages=np.zeros(43)
dm.setvoltages(voltages)
time.sleep(0.005)
cam1.snapImage()
image = cam1.getImage().astype(float)
centroid_0 = centroid_positions(x_max, y_max, image)

G = np.zeros(shape=(len(reference_centroids),len(voltages)))

for i in range(len(voltages)):
    voltages = np.zeros(43)
    voltages[i] = voltage_standard
    dm.setvoltages(voltages)
    time.sleep(0.005)
    cam1.snapImage()
    image = cam1.getImage().astype(float)
    centroid_i = centroid_positions(x_max, y_max, image)
    displacement = centroid_0 - centroid_i
    #print centroid_i
    G[:,i] = displacement
    #PIL.Image.fromarray(image).save("calibration_first_try\\actuator" + str(i) +".tif")

G_pinv = np.linalg.pinv(G)
raw_input('press button!')
corr = 0.0
y_abb = centroid_0 - reference_centroids
voltages_new = np.dot(G_pinv, y_abb) * voltage_standard * corr
voltages = voltages_new
dm.setvoltages(voltages)
time.sleep(0.005)
image = get_image(cam1)

times = np.zeros(100)
for i in range(100):
    times[i] = time.clock()
    centroid_i = centroid_positions(x_max, y_max, image)
    y_abb = centroid_i - reference_centroids
    voltage_update = np.dot(G_pinv, y_abb) * voltage_standard * corr
    voltages += voltage_update
    dm.setvoltages(voltages)
    time.sleep(0.005)
    image = get_image(cam1)
    image_saved = get_image(cam2)
    image_max=np.unravel_index(image_saved.argmax(), image_saved.shape)
    image_saved = image_saved[image_max[0]-30:image_max[0]+30, image_max[1]-30:image_max[1]+30]
    PIL.Image.fromarray(image_saved).save("faster_correction\\psf_moving_disc_bananas_no_gain" + str(i) +".tif")
    #PIL.Image.fromarray(image).save("calibration_iteration\\sh_pattern_no_gain" + str(i) +".tif")

print times
