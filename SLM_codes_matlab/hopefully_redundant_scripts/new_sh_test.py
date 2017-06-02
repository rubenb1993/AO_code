import numpy as np
import sys

if "C:\Program Files\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Program Files\Micro-Manager-1.4")
import MMCorePy
import PIL.Image as pil

sh = MMCorePy.CMMCore()
sh.loadDevice("cam", "IDS_uEye", "IDS uEye")
sh.initializeDevice("cam")
sh.setCameraDevice("cam")
sh.setProperty("cam", "Pixel Clock", 30.)
sh.setProperty("cam", "PixelType", '8bit mono')
sh.setProperty("cam", "Exposure", 0.190)
