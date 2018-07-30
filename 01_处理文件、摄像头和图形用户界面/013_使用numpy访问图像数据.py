# coding:utf8
import cv2

"""
将bgr在(0, 0)处改为白色像素

0号为green 1号为blue 2号为red
img的每一个位置存一个 3个长度的向量 分别表示gbr
"""
# img = cv2.imread("../data/mm2.jpeg")
# print(img[0, 0])    # [49 65 11]
# img[0, 0] = [255, 255, 255]
# cv2.imshow("", img)
# cv2.waitKey(0)


"""
将坐标(150, 120) 的蓝色值改为255

建议使用itemset函数完成， 这能避免访问原始索引
"""
# img = cv2.imread("../data/mm2.jpeg")
# print(img.item(150, 120, 0))    # 打印出这个坐标的blue值
# cv2.imshow("", img)
# img.itemset((150, 120, 0), 255) # 将这个位置的blue值设为255
# print(img.item(150, 120, 0))    # 打印这个坐标的blue值


"""
将图像所有的green值都设置为0

不推荐使用循环，，使用索引方式能提高程序实现的效率
"""
# img = cv2.imread("../data/mm1.jpg")
# img[:, :, 1] = 0
# cv2.imshow("", img)
# cv2.waitKey(0)


"""
将某个区域与变量绑定，将值分配给第二个区域
"""
# img = cv2.imread("../data/mm3.jpg")
# my_roi = img[0: 100, 0:100]     # 选定宽和高都是0到100的区域为感兴趣的区域
# img[300: 400, 300: 400] = my_roi    # 将my_roi区域的值赋给 宽和搞300到400像素的位置
# cv2.imshow("", img)
# cv2.waitKey()

"""
查看图像的一些属性
shape: 宽度 高度 和 通道数
size： 图像像素的大小
datatype： 图像的数据类型 一般为无符号整型
"""
img = cv2.imread("../data/mm3.jpg")
print(img.shape)    # (750, 1000, 3)
print(img.size)     # 2250000
print(img.dtype)    # uint8

