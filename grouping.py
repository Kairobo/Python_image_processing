#kajia@umich.edu
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
def grouping(img):
    label = 1
    adjlabel = 0
    h,w = img.shape
    lims = (w * h) / 30000
    dx4 = [-1, 0, 1, 0]
    dy4 = [0, -1, 0, 1]
    new_clusters = (-1) * np.ones([h, w])
    counti = [];
    for i in range(h):
        for j in range(w):
            if new_clusters[i][j] == -1 and img[i][j] > 0:
                elements = []
                elements.append([i,j])
                #find an adjacent label, for possible use latter
                for k in range(4):
                    x = elements[0][0] + dx4[k]
                    y = elements[0][1] + dy4[k]
                    if x >= 0 and x < h and y >= 0 and y < w:
                        if new_clusters[x][y] >= 0:
                            adjlabel = new_clusters[x][y]
                count = 0
                c = 0
                while c <= count:
                    for k in range(4):
                        x = elements[c][0] + dx4[k]
                        y = elements[c][1] + dy4[k]
                        if x >= 0 and x < h and y >= 0 and y < w:
                            if new_clusters[x][y] == -1 and img[x][y] == img[i][j]:
                                elements.append([x,y])
                                new_clusters[x][y] = label
                                count = count + 1
                    c = c + 1
                #use the earlier found adjacent label if a segment size is smaller than a limit
                if count < lims/4 and adjlabel > 0:
                    for c in range(count+1):
                        new_clusters[elements[c][0]][elements[c][1]] = adjlabel;
                    label = label - 1
                    counti[adjlabel] = counti[adjlabel] + count + 1
                else:
                    counti.append(count+1)
                adjlabel = 0#can lose some pixels, background
                label = label + 1
    ele_num = label-1
    ele_num_pre = ele_num
    label = 1
    enumi = []
    x_i = 0
    y_i = 0
    #rearrange
    for i in range(ele_num_pre):
        if counti[i] < lims:
            for x_i in range(h):
                for y_i in range(w):
                    if new_clusters[x_i][y_i] == i+1:
                        new_clusters[x_i][y_i] = 0
            ele_num = ele_num - 1
        else:
            for x_i in range(h):
                for y_i in range(w):
                    if new_clusters[x_i][y_i] == i+1:
                        new_clusters[x_i][y_i] = label
            label = label + 1
            enumi.append(counti[i])

    return new_clusters, ele_num, enumi
if __name__=="__main__":
    img = cv2.imread('./1108.png', 0)
    h,w = img.shape
    # connect small part
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # grouping discriminate different
    groups, ele_num, enumi = grouping(closing)
    print(ele_num)
    print(enumi)
    for i in range(h):
        for j in range(w):
            if groups[i][j] == -1:
                groups[i][j] = 0
    plt.imshow(groups)
    plt.show()