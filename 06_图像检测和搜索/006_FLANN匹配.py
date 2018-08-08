# coding:utf-8

import cv2


"""
FLANN是类似最近邻的快速匹配库
    它会根据数据本身选择最合适的算法来处理数据
    比其他搜索算法快10倍
"""
# 按照灰度图片读入
img1 = cv2.imread("../data/logo1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("../data/album1.png", cv2.IMREAD_GRAYSCALE)
# 创建sift检测器
sift = cv2.xfeatures2d.SIFT_create()
# 查找监测点和匹配符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
"""
keypoint是检测到的特征点的列表
descriptor是检测到特征的局部图像的列表
"""
# 获取flann匹配器
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)

"""
flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    传入两个参数以字典的形式
    FLANN内部会决定如何处理索引和搜索对象
    可以选择 LinearIndex KTreeIndex KMeansIndex CompositeIndex 和 AutotunelIndex
    KTreeIndex配置索引简单 只需要指定待处理核密度树的数量 最理想的数量是1到16 KTreeIndex非常灵活
    searchParams字段包含一个checks 用来指定索引树要被遍历的次数 次数越高花费时间越长 结果越准确
    
"""

# 进行匹配
matches = flann.knnMatch(des1, des2, k=2)
# 准备空的掩膜 画好的匹配项
matchesMask = [[0, 0] for i in range(len(matches))]

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i] = [1, 0]

drawPrams = dict(matchColor=(0, 255, 0),
                 singlePointColor=(255, 0, 0),
                 matchesMask=matchesMask,
                 flags=0)
# 匹配结果图片
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **drawPrams)


cv2.imshow("matches", img3)
cv2.waitKey()
cv2.destroyAllWindows()