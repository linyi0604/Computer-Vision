# coding:utf-8

import cv2

img = cv2.imread("../data/walez1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create(8000)
"""
    创建surf对象，设置阈值，阈值越高检测到的特征就越少，
    通过调整阈值得到合适的关键点
"""
# 检测图像中的关键点和描述
keypoints, descriptor = surf.detectAndCompute(gray, None)
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
# 将关键点画在原图像上
cv2.drawKeypoints(image=img, outImage=img,
                  keypoints=keypoints,flags=4,
                  color=(51, 163, 236))

cv2.imshow("surf_detected", img)
cv2.waitKey()
cv2.destroyAllWindows()

