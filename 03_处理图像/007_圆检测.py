import cv2
import numpy as np

img_origin = cv2.imread("../data/circle.jpg")
img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
# 低同滤波进行平滑图像
img = cv2.medianBlur(img_gray, 5)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 120,
                           param1=100,param2=30,
                           minRadius=0, maxRadius=0)
"""
cv2.HoughCircles(image, method, dp, 
        minDist, circles, param1, param2, 
        minRadius, maxRadius)
参数：
    image:　输入图像　必须是灰度图像
    method:检测方法,常用CV_HOUGH_GRADIENT
    dp:检测内侧圆心的累加器图像的分辨率于输入图像之比的倒数，
        如dp=1，累加器和输入图像具有相同的分辨率，如果dp=2，
        累计器便有输入图像一半那么大的宽度和高度
    minDist: 两个圆心之间的最小距离
    param1: 默认100, 是method方法的参数
        在CV_HOUGH_GRADIENT表示传入canny边缘检测的阈值
    param2： 默认100,method的参数， 
        对当前唯一的方法霍夫梯度法cv2.HOUGH_GRADIENT，
        它表示在检测阶段圆心的累加器阈值，
        它越小，就越可以检测到更多根本不存在的圆，
        而它越大的话，能通过检测的圆就更加接近完美的圆形了
    minRadius:默认值0，圆半径的最小值
    maxRadius:默认值0，圆半径的最大值
返回值：
    
"""
# 整数化
circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    # 画出外边圆
    cv2.circle(img_origin, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # 画出圆心
    cv2.circle(img_origin, (i[0], i[1]), 2, (0, 0, 255), 3)


cv2.imshow("", img_origin)
cv2.waitKey()
cv2.destroyAllWindows()