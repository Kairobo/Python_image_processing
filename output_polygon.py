import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from copy import deepcopy
from grouping import grouping
from morphological import spur
from copy import deepcopy

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
#thicken by dilation
kernel1 = np.ones((1,1),np.uint8)
dilation = cv2.dilate(single_group_nospur,kernel1,iterations = 1)
#print(sg_nospur_thick.shape)

#leave just the boundary

I_edge = cv2.Laplacian(dilation, cv2.CV_8U)

#find the polygon
#go through the boundary and output the polygon
#logic find the closest no_mark point to go
I_mark = np.zeros([h,w])
is_go_through = 0
nzeros_indexes = np.nonzero(I_edge)
six = nzeros_indexes[0][0]
siy = nzeros_indexes[1][0]
dis_polypoint = 5
I_mark[six,siy] = 1
#point from distance 1 to sqrt(2)
c_x = [0,0, -1,1,-1,1,-1,1]
c_y = [-1,1, 0,0,-1,-1,1,1]
check_add_v = [0] * 8
cur_point_xy = [six,siy]
num_point_k = 1
onetime_bool = True
cur_point_xy_t = deepcopy(cur_point_xy)
polygon_index = [cur_point_xy_t]
counter_for_poly = 0
while is_go_through is not 1:
    #find the next point to go
    check_add_v = [0] * 8
    for i in range(8):
        curx = cur_point_xy[0] + c_x[i]
        cury = cur_point_xy[1] + c_y[i]
        if curx >= 0 and curx < h and cury >= 0 and cury < w:
            if I_edge[curx][cury] >= 1:
                check_add_v[i] = 1
                if I_mark[curx][cury] == 1:
                    check_add_v[i] = 0
    i_1 = int(np.nonzero(check_add_v)[0][0])

    cur_point_xy[0] = cur_point_xy[0] + c_x[i_1]
    cur_point_xy[1] = cur_point_xy[1] + c_y[i_1]
    I_mark[cur_point_xy[0]][cur_point_xy[1]] = 1
    num_point_k = num_point_k + 1
    counter_for_poly = counter_for_poly + 1
    if counter_for_poly == dis_polypoint:
        counter_for_poly = 0
        cur_point_xy_t = deepcopy(cur_point_xy)
        polygon_index.append(cur_point_xy_t)
    if num_point_k >= 4 and onetime_bool == True:
        onetime_bool = False
        I_mark[six][siy] = 0
    if cur_point_xy[0] == six and cur_point_xy[1] == siy:
        break
print("num_polygon_point",len(polygon_index))
I_out = np.zeros([h,w])
for i in range(len(polygon_index)):
    c_x = polygon_index[i][0]
    c_y = polygon_index[i][1]
    I_out[c_x][c_y] = 1


#draw the polygon

plt.imshow(I_out)
plt.show()