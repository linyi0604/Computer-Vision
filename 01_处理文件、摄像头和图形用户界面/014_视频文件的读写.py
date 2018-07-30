# coding:utf8
import cv2

"""
读取视频文件的帧， 采用yuv颜色编码写入到另一个帧

VideoCapture和VideoWriter用于视频文件的读写
VideoCapture读的每一帧都是一个bgr格式的图像
"""
videoCapture = cv2.VideoCapture("../data/demo3.mp4")
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (
    int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)

videoWriter = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter_fourcc("I", "4", "2", "0"),  # 编码器
    fps,
    size
)
'''
编码器常用的几种：
cv2.VideoWriter_fourcc("I", "4", "2", "0") 
    压缩的yuv颜色编码器，4:2:0色彩度子采样 兼容性好，产生很大的视频 avi
cv2.VideoWriter_fourcc("P", I", "M", "1")
    采用mpeg-1编码，文件为avi
cv2.VideoWriter_fourcc("X", "V", "T", "D")
    采用mpeg-4编码，得到视频大小平均 拓展名avi
cv2.VideoWriter_fourcc("T", "H", "E", "O")
    Ogg Vorbis， 拓展名为ogv
cv2.VideoWriter_fourcc("F", "L", "V", "1")
    FLASH视频，拓展名为.flv
'''

success, frame = videoCapture.read()
while success:  # 循环直到没有帧了
    videoWriter.write(frame)
    success, frame = videoCapture.read()