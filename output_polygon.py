import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from copy import deepcopy
from grouping import grouping
from morphological import spur
img = cv2.imread('./1108.png',0)
h,w = img.shape
#connect small part
kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#split as groups
groups, ele_num, enumi = grouping(closing)
print(ele_num)
print(enumi)
for i in range(h):
    for j in range(w):
        if groups[i][j] == -1:
            groups[i][j] = 0
#show a single group
single_group = np.zeros(groups.shape)
select_i = 3
for i in range(h):
    for j in range(w):
        if groups[i][j] == select_i:
            single_group[i][j] = select_i
#remove spur
single_group_norm = (single_group/select_i).astype(np.uint8)
single_group_nospur = spur(single_group_norm).astype(np.uint8)
#thicken
kernel1 = np.ones((1,1),np.uint8)
dilation = cv2.dilate(single_group_nospur,kernel1,iterations = 1)
#print(sg_nospur_thick.shape)
plt.imshow(dilation)
plt.show()