# coding:utf-8

import cv2

img = cv2.imread("../data/mm1.jpg", cv2.IMREAD_GRAYSCALE)

img = cv2.putText(img, "hello world!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
"""
cv2.putText(图像, 文字, (x, y), 字体, 大小, (b, g, r), 宽度)
"""

cv2.imshow("", img)
cv2.waitKey()
cv2.destroyAllWindows()

