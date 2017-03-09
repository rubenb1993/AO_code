import numpy as np
import PIL.Image

x=np.linspace(0,1919,1920)
y=np.linspace(0,1079,1080)

x,y=np.meshgrid(x,y)


x=(x.astype(float)%9)/9.0*2*np.pi
print x

PIL.Image.fromarray(x).save("ramp.tif")
