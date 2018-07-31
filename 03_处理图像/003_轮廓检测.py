# coding:utf8

import cv2
import numpy as np

# 创建一个200*200 的黑色空白图像
img = np.zeros((200, 200), dtype=np.uint8)
# 在图像的中央位置 放置一个100*100的白色方块
img[50:150, 50: 150] = 255

cv2.imshow("image", img)
# 二值化操作
ret, thresh = cv2.threshold(img, 127, 255, 0)
"""
ret, dst = cv2.threshold(src, thresh, value, type)
参数:
    src: 原图像
    thresh: 阈值
    value: 新值 大于或小于阈值的值将赋新值
    type: 方法类型，有如下取值：
        cv2.THRESH_BINARY 黑白二值
        cv2.THRESH_BINARY_INV 黑白二值翻转
        cv2.THRESH_TRUNC 得到多像素值
        cv2.THRESH_TOZERO
        cv2.THRESH_TOZERO_INV
返回值：
    ret: 得到的阈值值
    dst: 阈值化后的图像
"""

# 得到 修改后的图像， 轮廓， 轮廓的层次
image, contours, hierarchy = cv2.findContours(
    thresh,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

"""
img, contours, hierarchy =  cv2.findContours(输入图像, 层次类型, 逼近方法)
参数：
    输入图像： 该方法会修改输入图像，建议传入输入图像的拷贝
    层次类型： 
        cv2.RETR_TREE 会得到图像中整体轮廓层次
        cv2.RETR_EXTERNAL 只得到最外面的轮廓
    逼近方法：

返回值：
    img: 修改后的图像
    contours: 图像的轮廓
    hierarchy: 图像和轮廓的层次
    
"""
# 原图像转换成bgr图像
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# 用绿色 在原图像上画出轮廓
img = cv2.drawContours(color, contours, -1, (0, 255, 255), 2)

cv2.imshow("contours", color)
cv2.waitKey()
cv2.destroyAllWindows()