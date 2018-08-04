# coding:utf-8

import cv2

filename = "../data/mm3.jpg"


def detect(filename):
    # 创建检测人脸的对象
    face_cascade = cv2.CascadeClassifier("../data/haarcascade_frontalface_default.xml")
    # 读取图像
    img = cv2.imread(filename)
    # 转为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 进行人脸检测
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    """
    faces = face_cascade.detectMultiScale(img, scaleFactor, minNeighbors)
    参数：
        img: 识别的原图
        scaleFactor： 迭代时图像的压缩率
        minNeighbors: 每个人脸矩形保留近邻数目的最小值
        
    返回值：
        一个列表，列表里边每一项是一个框起人脸的矩形(x, y, w, h)
        
    """
    print(faces)
    for (x, y, w, h) in faces:
        # 画出矩形框
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Vikings Detected", img)
    cv2.waitKey()


detect(filename)
