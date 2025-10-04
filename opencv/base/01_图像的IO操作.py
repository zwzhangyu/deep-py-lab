import numpy as np
import cv2 as cv

# 读取图像
img = cv.imread("images/avatar.jpg", cv.IMREAD_GRAYSCALE)

# 显示图像
cv.imshow("img", img)
cv.waitKey(0)