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
cv.findContours()
    参数：
        1 要寻找轮廓的图像 只能传入二值图像，不是灰度图像
        2 轮廓的检索模式，有四种：
            cv2.RETR_EXTERNAL表示只检测外轮廓
            cv2.RETR_LIST检测的轮廓不建立等级关系
            cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，
                里面的一层为内孔的边界信息。
                如果内孔内还有一个连通物体，这个物体的边界也在顶层
            cv2.RETR_TREE建立一个等级树结构的轮廓
        3 轮廓的近似办法
            cv2.CHAIN_APPROX_NONE存储所有的轮廓点，
                相邻的两个点的像素位置差不超过1，
                即max（abs（x1-x2），abs（y2-y1））==1
            cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，
                只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    返回值:
        contours:一个列表，每一项都是一个轮廓， 不会存储轮廓所有的点，只存储能描述轮廓的点
        hierarchy:一个ndarray, 元素数量和轮廓数量一样， 
            每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，
            分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数
"""
# 原图像转换成bgr图像
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# 用绿色 在原图像上画出轮廓
img = cv2.drawContours(color, contours, -1, (0, 255, 255), 2)

cv2.imshow("contours", color)
cv2.waitKey()
cv2.destroyAllWindows()