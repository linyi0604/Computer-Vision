# coding:utf8

import cv2
import numpy as np
from scipy import ndimage

# 3*3 的高通卷积核
kernel_3x3 = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

# 5*5 高通卷积核
kernel_5x5 = np.array([
    [-1, -1, -1, -1, -1],
    [-1, 1, 2, 1, -1],
    [-1, 3, 4, 2, -1],
    [-1, 1, 2, 1, -1],
    [-1, -1, -1, -1, -1]
])

# 按灰度值读入图像
img = cv2.imread("../data/mm1.jpg", 0)


# 进行卷积运算
k3 = ndimage.convolve(img, kernel_3x3)
k5 = ndimage.convolve(img, kernel_5x5)
"""
高通滤波器： 根据像素与临近像素的亮度差值来提升像素的亮度
"""


# 原图像运用高斯低通滤波器
blurred = cv2.GaussianBlur(img, (11, 11), 0)
"""
低通滤波器： 像素周围亮度小于一个特定值时候，平滑该像素的亮度，主要用于去噪和模糊化
    高斯滤波器是最常用的模糊滤波器之一，他是一个削弱强度的低通滤波器
"""
# 原图像减去低通
g_hpf = img - blurred

cv2.imshow("3x3", k3)
cv2.imshow("5x5", k5)
cv2.imshow("g_hpf", g_hpf)
cv2.imshow("origin", img)
cv2.waitKey()
cv2.destroyAllWindows()