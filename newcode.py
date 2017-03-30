import numpy as np
from skimage import data,img_as_float
from skimage import exposure
import picamera
from time import sleep
from skimage import io
camera=picamera.PiCamera()
camera.resolution = (1920,1080)
camera.capture("denemedamar.png")
img=io.imread("denemedamar.png",as_grey=True)
io.imsave("denemedamar1.png",img)
p2 = np.percentile(img, 2)
p98 = np.percentile(img, 98)
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
io.imsave("contraststretching.png" , img_rescale)
img_eq = exposure.equalize_hist(img)
io.imsave("histogrammed.png" , img_eq)
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
io.imsave("adaptivehist.png" , img_adapteq)
