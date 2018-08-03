import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from morphological import spur
from copy import deepcopy
from grouping import grouping
img = (cv2.imread('./1108.png',0)/255).astype(np.uint8)
img_spur = spur(img).astype(np.uint8)
plt.imshow(img_spur)
plt.show()