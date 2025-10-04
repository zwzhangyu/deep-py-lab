import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 读取图像
img = cv.imread("images/avatar.jpg", cv.IMREAD_GRAYSCALE)

# 显示图像
# cv.imshow("img", img)
# cv.waitKey(0)

plt.imshow(img, cmap="gray")
plt.show()