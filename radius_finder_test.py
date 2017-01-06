import numpy as np
import os
import PIL.Image
import matplotlib.pyplot as plt
impath_int = os.path.abspath("dm_int.tif")
image_int = np.asarray(PIL.Image.open(impath_int)).astype(float)

[ny, nx] = image_int.shape
x = np.linspace(1, nx, nx)
y = np.linspace(1, ny, ny)
xx, yy = np.meshgrid(x, y)

int_num = 10
image_int[image_int<=int_num] = 0
image_int[image_int>int_num] = 1


centre = np.zeros(2)
centre[0] = np.sum(image_int * xx) / np.sum(image_int)
centre[1] = np.sum(image_int * yy) / np.sum(image_int)

radius = 300
mask = [np.sqrt((xx-centre[0])**2 + (yy-centre[1])**2) >= radius]
image_masked = image_int * np.squeeze(mask)

print(centre[0], centre[1])

f, ax = plt.subplots(1,1)
ax.imshow(image_masked)
ax.scatter(centre[0], centre[1])
plt.show()
