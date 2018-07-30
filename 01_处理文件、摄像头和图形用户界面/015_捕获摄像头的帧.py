# coding:utf8

import cv2


"""
捕获摄像头10s的视频信息 写入一个avi文件
"""

cameraCapture = cv2.VideoCapture(0)     # 传入0代表0号摄像头
fps = 30
size = (
    int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)

videoWriter = cv2.VideoWriter(
    "outputVid.avi",
    cv2.VideoWriter_fourcc("I", "4", "2", "0"),
    fps,
    size
)

success, frame = cameraCapture.read()
numFramesRemaining = 10 * fps - 1
while success and numFramesRemaining:
    videoWriter.write(frame)
    success, frame = cameraCapture.read()
    numFramesRemaining -= 1

cameraCapture.release()


"""
如果使用一组摄像头或多个摄像头
用grab和retrieve方法代替

success0 = cameraCapture.grab()
success1 = cameraCapture.grab()
if success0 and success1:
    frame0 = cameraCapture0.retrieve()
    frame1 = cameraCapture1.retrieve()
    
    
"""
