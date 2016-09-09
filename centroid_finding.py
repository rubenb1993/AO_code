
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

get_ipython().magic('matplotlib inline')


# In[3]:

img_pil = Image.open("zero_abb.tif")
[nx,ny] = img_pil.size
img_pil = np.array(np.array(img_pil).reshape((ny,nx)))
xx, yy = np.meshgrid(np.linspace(1,ny,ny),np.linspace(1,nx,nx))
i,j = np.unravel_index(img_pil.argmax(), img_pil.shape)
img_pil_mask = img_pil > 3
img_pil_filtered = img_pil * img_pil_mask

list_of_maxima_x = []
list_of_maxima_y = []

while(np.amax(img_pil_filtered) > 10):
    y_max, x_max = np.unravel_index(img_pil_filtered.argmax(), img_pil_filtered.shape)
    list_of_maxima_x.append(x_max)
    list_of_maxima_y.append(y_max)
    img_pil_filtered[y_max - 40: y_max + 40, x_max - 40: x_max+40] = 0
    

x_max = np.array(list_of_maxima_x)
y_max = np.array(list_of_maxima_y)
centroids = np.zeros(shape = (len(x_max),2))
spot_size = 25
for i in range(len(x_max)):
    y_low = y_max[i] - spot_size
    y_high = y_max[i] + spot_size
    x_low = x_max[i] - spot_size
    x_high = x_max[i] + spot_size
    norm_photons = 1/np.sum(img_pil[y_low: y_high, x_low: x_high])
    centroids[i,0] = norm_photons * np.sum(img_pil[y_low: y_high, x_low: x_high] * xx[y_low: y_high, x_low: x_high])
    centroids[i,1] = norm_photons * np.sum(img_pil[y_low: y_high, x_low: x_high] * yy[y_low: y_high, x_low: x_high])

plt.figure(figsize= (15,21))
plt.scatter(centroids[:,0], centroids[:,1], c = (0,0,0))
plt.imshow(img_pil)


# In[ ]:




# In[ ]:




# In[ ]:



