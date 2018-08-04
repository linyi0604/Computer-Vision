# coding:utf8

import cv2
import numpy as np


# 读入图像
img = cv2.imread("../data/line1.png")
# 转为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Canny边缘检测
edges = cv2.Canny(gray, 50, 100)
"""
canny边缘检测：
有五个步骤：
        1 高斯滤波器降噪
        2 计算梯度
        3 边缘上使用非最大抑制 nms
        4 边缘上使用双阈值去除假阳性
        5 分析所有边缘连接 消除不明显的边缘
"""

minLineLength = 20
maxLineGap = 5
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
"""
cv2.HoughLinesP
    作用：标准霍夫线变换， 找到图像中的所有直线
    参数：
        1 二值图
        2 半径精度
        3 角度精度
        4 最短检测长度
        5 允许的最大缺口
    返回：
        一个列表，每一项是一个四元组，分别是直线两个端点的坐标
"""
for line in lines:
    for x1, y1, x2, y2 in line:
        # 在图片上画直线
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("edges", edges)
cv2.imshow("lines", img)
cv2.waitKey()
cv2.destroyAllWindows()