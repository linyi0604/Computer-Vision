# coding:utf-8

import cv2

"""
orb特征检测和匹配
    两幅图片分别是 乐队的logo 和包含该logo的专辑封面
    利用orb进行检测后进行匹配两幅图片中的logo
    
"""
# 按照灰度图像的方式读入两幅图片
img1 = cv2.imread("../data/logo1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("../data/album1.png", cv2.IMREAD_GRAYSCALE)

# 创建ORB特征检测器和描述符
orb = cv2.ORB_create()
# 对两幅图像检测特征和描述符
keypoint1, descriptor1 = orb.detectAndCompute(img1, None)
keypoint2, descriptor2 = orb.detectAndCompute(img2, None)
"""
keypoint 是一个包含若干点的列表
descriptor 对应每个点的描述符 是一个列表， 每一项都是检测到的特征的局部图像

检测的结果是关键点
计算的结果是描述符

可以根据监测点的描述符 来比较检测点的相似之处

"""
# 获得一个暴力匹配器的对象
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# 利用匹配器 匹配两个描述符的相近成都
maches = bf.match(descriptor1, descriptor2)
# 按照相近程度 进行排序
maches = sorted(maches, key=lambda x: x.distance)
# 画出匹配项
img3 = cv2.drawMatches(img1, keypoint1, img2, keypoint2, maches[: 30], img2, flags=2)

cv2.imshow("matches", img3)
cv2.waitKey()
cv2.destroyAllWindows()