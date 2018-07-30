# coding:utf8
import cv2
import numpy
import os


"""
随机字节的bytearray转为灰度图像和BGR图像
"""
randomByteArray = bytearray(os.urandom(120000))
flatNumpyArray = numpy.array(randomByteArray)
# 转换成400*300的灰度图像
grayImage = flatNumpyArray.reshape(400, 300)
cv2.imwrite("randomGray.png", grayImage)

"""
随机字节的bytearray转为400*100的彩色图像
"""
bgrImage = flatNumpyArray.reshape(400, 100, 3)
cv2.imwrite("randomColor.png", bgrImage)