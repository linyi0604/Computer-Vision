# coding:utf8

import cv2
"""
canny边缘检测：
有五个步骤：
        1 高斯滤波器降噪
        2 计算梯度
        3 边缘上使用非最大抑制 nms
        4 边缘上使用双阈值去除假阳性
        5 分析所有边缘连接 消除不明显的边缘
"""

img = cv2.imread("../data/mm2.jpeg", 0)  # 按照灰度值读入
canny = cv2.Canny(img, 200, 300)
cv2.imshow("origin", img)
cv2.imshow("canny", canny)
cv2.waitKey()
cv2.destroyAllWindows()
