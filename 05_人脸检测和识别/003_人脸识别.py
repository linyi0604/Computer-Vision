# coding:utf-8

import cv2
import os
import numpy as np


# 1 生成人脸识别数据
#  图像是灰度格式，后缀名.pgm
#  图像是正方形 图像大小要一样 在这里使用200*200
def generate():
    face_cascade = cv2.CascadeClassifier("../data/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("../data/haarcascade_eye.xml")
    camera = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            f = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            cv2.imwrite("./data/%s.pgm" % str(count), f)
            count += 1

        cv2.imshow("camera", frame)
        if cv2.waitKey(50) & 0xff == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


def readImages():
    x, y = [], []
    path = "./data/faces/"
    image_file = os.listdir(path)
    image_files = [path + i for i in image_file]
    for file in image_files:
        images = os.listdir(file)
        label = file.split("/")[-1][1:]
        for i in images:
            img = cv2.imread(file + "/" + i, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (200, 200))
            x.append(np.asarray(img, dtype=np.uint8))
            y.append(int(label))

    y = np.asarray(y, dtype=np.int32)
    return x, y


# 检测人脸
def face_rec():
    # 获取数据
    x, y = readImages()
    # 训练模型
    model = cv2.face.EigenFaceRecognizer_create()
    model.train(np.asarray(x), np.asarray(y))
    # 开摄像头
    camera = cv2.VideoCapture(0)
    # 加载检测人脸对象
    face_cascade = cv2.CascadeClassifier("../data/haarcascade_frontalface_default.xml")
    while True:
        # 读取当前帧
        read, img = camera.read()
        # 当前帧下检测人脸z
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            # 画出人脸
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # 转成灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 拿出人脸部分
            roi = gray[x: x+w, y: y+h]
            try:
                # 更改大小
                roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                # 进行预测
                params = model.predict(roi)
                # 在图像上写预测结果
                # 1.2是字体大小 2是粗细
                img = cv2.putText(img, str(params[0]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                print(params)
            except Exception as e:
                print(e)
        cv2.imshow("detect face", img)
        if cv2.waitKey(5) & 0xff == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 调用摄像头 采集人脸照片数据
    # generate()

    # 检测人脸
    face_rec()
