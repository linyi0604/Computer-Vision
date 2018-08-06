# coding:utf-8

import cv2

# 读取图片
img = cv2.imread("../data/walez1.jpg")
# 转为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 创建一个sift对象 并计算灰度图像
sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(gray, None)
"""
sift对象会使用DoG检测关键点，对关键点周围的区域计算向量特征，检测并计算
返回 关键点和描述符
关键点是点的列表
描述符是检测到的特征的局部区域图像列表

关键点的属性：
    pt: 点的x y坐标
    size： 表示特征的直径
    angle: 特征方向
    response: 关键点的强度
    octave: 特征所在金字塔层级
        算法进行迭代的时候， 作为参数的图像尺寸和相邻像素会发生变化
        octave属性表示检测到关键点所在的层级
    ID： 检测到关键点的ID

"""
# 在图像上绘制关键点
# DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS表示对每个关键点画出圆圈和方向
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))

cv2.imshow("sift_keypoints", img)
cv2.waitKey()
cv2.destroyAllWindows()