# coding:utf8

import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
分水岭算法：

"""

# 读入图像
img = cv2.imread("../data/item2.jpg")
# 转成灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 黑白二值化
ret, thresh = cv2.threshold(
    gray,
    0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# 使用morphologyEx变换来去除噪声， 对图像先膨胀再腐蚀
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# morphologyEx变换后的图像进行膨胀，可以得到大部分都是背景的区域
sure_bg = cv2.dilate(opening, kernel, iterations=3)
# 通过distanceTransform来确定前景区域 结合阈值来决定哪些是前景
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

# 用sure_bg 和 sure_fg 相减确定前景和背景重合区域
sure_fg = np.uint8(sure_fg)
unkown = cv2.subtract(sure_bg, sure_fg)

# 定义栅栏 分割不同的山谷
ret, markers = cv2.connectedComponents(sure_fg)
# 背景区域加上1， unkown区域设为0
markers = markers + 1
markers[unkown == 255] = 0

# 打开栅栏 把栅栏设置成红色
markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]
plt.imshow(img)
plt.show()
