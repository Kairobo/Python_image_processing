import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from grouping import grouping

img = cv2.imread('./1108.png',0)
h,w = img.shape
#connect small part
kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
groups, ele_num, enumi = grouping(closing)
print(ele_num)
print(enumi)
for i in range(h):
    for j in range(w):
        if groups[i][j] == -1:
            groups[i][j] = 0
plt.imshow(groups)
plt.show()