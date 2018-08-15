from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img=mpimg.imread('./lawntest.jpg')
fig = plt.imshow(img)
x = plt.ginput(5)
print("clicked", x)
plt.show()
