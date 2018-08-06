# coding:utf-8

import cv2
import numpy as np

img = cv2.imread("../data/chess1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
# 角点检测
dst = cv2.cornerHarris(gray, 2, 23, 0.04)
"""
角点也是处在一个无论框框往哪边移动　框框内像素值都会变化很大的情况而定下来的点
cv2.cornerHarris() 
    img - 数据类型为 float32 的输入图像。
    blockSize - 角点检测中要考虑的领域大小。
    ksize - Sobel 求导中使用的窗口大小
    k - Harris 角点检测方程中的自由参数,取值参数为 [0,04,0.06].
"""

# 将检测到角点的位置标记为红色
img[dst > 0.01 * dst.max()] = (0, 0, 255)

cv2.imshow("corners", img)
cv2.waitKey()
cv2.destroyAllWindows()