# coding:utf8
import cv2


# 读取一张图片, 第二个参数可选
image = cv2.imread("../data/mm1.jpg")
# 设置窗口
cv2.namedWindow("show", cv2.WINDOW_AUTOSIZE)
# 图像窗口显示
cv2.imshow("show", image)
# 等待按键
cv2.waitKey(0)


'''
cv2.IMREAD_ANYCOLOR = 4
cv2.IMREAD_ANYDEPTH = 2
cv2.IMREAD_COLOT = 1
cv2.IMREAD_GRAYSCALE = 0
cv2.IMREAD_LOAD_GDAL = 8
cv2.IMREAD_UNCHANGED = -1
'''

# 加载一个图像为灰度图像（会丢失颜色） 再保存为png
image = cv2.imread("../data/mm1.jpg", cv2.IMREAD_GRAYSCALE)
# 保存图片 根据拓展名自动转换保存格式 保存一定是BGR图像
cv2.imwrite("mm1_gray.png", image)

