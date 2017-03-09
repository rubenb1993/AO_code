import numpy as np
import matlab.engine as mateng
import matplotlib.pyplot as plt

print("start your engines")
eng = mateng.start_matlab()
print("engines succesfully started")
img = eng.complete_image(300.0)
img = np.asarray(img)
print(img.shape)
plt.imshow(img)
plt.show()

