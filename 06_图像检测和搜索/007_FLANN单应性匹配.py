# coding:utf-8

"""
单应性匹配：
    两幅图像中的一幅 出现投影畸变的时候，他们还能彼此匹配
"""

import cv2
import numpy as np
# 最小匹配数量设为10个， 大于这个数量从中筛选出10个最好的
MIN_MATCH_COUNT = 10

# 读入两幅图片 图片中有相同部分
img1 = cv2.imread("../data/logo1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("../data/album1.png", cv2.IMREAD_GRAYSCALE)

# 获取sift特征检测器
sift = cv2.xfeatures2d.SIFT_create()
# 检测关键点 计算描述符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# kdtree建立索引方式的常量参数
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50) # checks指定索引树要被遍历的次数
flann = cv2.FlannBasedMatcher(index_params, search_params)
# 进行匹配搜索
matches = flann.knnMatch(des1, des2, k=2)

# 寻找距离近的放入good列表
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

# 如果足够多  就筛选
if len(good) > MIN_MATCH_COUNT:
    # 通过距离近的描述符 找到两幅图片的关键点
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 单应性匹配图关键点匹配线。。不懂啥意思
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    """
    参数：
        src_pts: 第一张图片的检测特征点
        dst_pts: 第二张图片的检测特征点
        第三个参数：
            cv2.RANSAC RANSAC-based robust method
            cv2.LMEDS least-median robust method
        第四个参数： 1到10之间 阈值， 原图像点经过变换后超过误差就舍弃
    返回值:
        H: 变换矩阵
        mask: 掩膜
    """
    matchesMask = mask.ravel().tolist()

    h, w = img1.shape

    # 计算第二张图相对于第一张图的畸变  其实不太理解这是咋回事
    pts = np.float32([[0, 0], [0, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
else:
    matchesMask = None

draw_params = dict(
    matchColor=(0, 255, 0),
    singlePointColor=None,
    matchesMask=matchesMask,
    flags=2
)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
cv2.imshow("", img3)
cv2.waitKey()
