import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from copy import deepcopy
from grouping import grouping
from morphological import spur
from copy import deepcopy
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

def edge_crossing(img):
    e_img = deepcopy(img)
    h,w = img.shape
    c_x = [0, 0, -1, 1, -1, 1, -1, 1]
    c_y = [-1, 1, 0, 0, -1, -1, 1, 1]
    if_crossing = 0
    for i in range(h):
        for j in range(w):
            #check_if_crossing
            if_crossing = 0
            if img[i][j] == 1:
                for k in range(8):
                    ci = i + c_x[k]
                    cj = j + c_y[k]
                    if ci >= 0 and ci < h and cj >= 0 and cj < w:
                        if img[ci][cj] == 0:
                            if_crossing = 1
                            break
                if i == 0 or i == h - 1 or j == 0 or j == w - 1:
                    is_crossing = 1
                    break
                if if_crossing == 1:
                    e_img[i][j] = 1
                else:
                    e_img[i][j] = 0
            else:
                e_img[i][j] = 0
    return e_img



img = cv2.imread('./1108.png',0)
img = (img>200).astype(np.uint8) * 255
h,w = img.shape
#debug
plt.imshow(img)
plt.show()
#connect small part
kernel = np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#split as groups
groups, ele_num, enumi = grouping(closing)
print(ele_num)
print(enumi)
for i in range(h):
    for j in range(w):
        if groups[i][j] == -1:
            groups[i][j] = 0


print("num_groups",ele_num)
#show a single group
single_group = np.zeros(groups.shape)
select_i = 1
for i in range(h):
    for j in range(w):
        if groups[i][j] == select_i:
            single_group[i][j] = select_i

#remove spur
single_group_norm = (single_group/select_i).astype(np.uint8)
single_group_nospur = spur(single_group_norm).astype(np.uint8)
#debug
plt.imshow(single_group_norm)
plt.show()
#thicken by dilation
kernel1 = np.ones((2,2),np.uint8)
dilation = cv2.dilate(single_group_nospur,kernel1,iterations = 1)
#debug
plt.imshow(dilation)
plt.show()
#print(sg_nospur_thick.shape)
#save dilation to debug
cv2.imwrite('dilation.png',dilation)
#leave just the boundary
if False:
    sobelx = cv2.Sobel(dilation,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(dilation,cv2.CV_64F,0,1,ksize=3)
    I_edge_1 = sobelx + sobely
    #debug
    plt.imshow(I_edge_1)
    plt.show()
edge_detection_type = 'crossing'#'Canny'
if edge_detection_type is 'crossing':
    I_edge = edge_crossing(dilation);
else:
    I_edge = cv2.Laplacian(dilation, cv2.CV_8U)
#debug
plt.imshow(I_edge)
plt.show()
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
    try:
        print("current point",cur_point_xy)
        i_1 = int(np.nonzero(check_add_v)[0][0])
    except:
        print("break in",cur_point_xy)
        exit(0)

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

#real draw the polygon
image = Image.new("1", (w, h))
draw = ImageDraw.Draw(image)
polygon_tuple = []
for i in range(len(polygon_index)):
    polygon_tuple.append(tuple((polygon_index[i][1],polygon_index[i][0])))
print(tuple(polygon_tuple))

draw.polygon((tuple(polygon_tuple)), fill=200)
image.show()
I_mask = np.array(image).astype(np.uint8)
plt.imshow(I_mask)
plt.show()
#calculate the area using I_edge and
area_ratio_pre = np.sum(dilation)/np.sum(I_mask)
area_ratio = 1 if area_ratio_pre > 1 else area_ratio_pre
print('area_ratio',area_ratio)