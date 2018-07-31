# coding:utf8

import cv2
from managers import WindowManger, CaptureManager
import filters
"""
Cameo项目 面向对象方式 实现一个 人脸跟踪和图像处理
"""


class Cameo(object):
    def __init__(self):
        # 窗口管理器
        self._windowManager = WindowManger(
            "Cameo",
            self.onKeyPress
        )
        # 捕获管理器
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0),
            self._windowManager,
            True,
        )

        self._curveFilter = filters.BGRPortraCurveFilter()

    def run(self):
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            # 进行边缘检测
            filters.strokeEdge(frame, frame)
            self._curveFilter.apply(frame, frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeyPress(self, keycode):
        if keycode == 32:  # 空格 进行截屏
            self._captureManager.writeImage("screenshot.png")
        elif keycode == 9:  # tab 录像
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo("screencast.avi")
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27:  # esc键退出
            self._windowManager.destroyWindow()


if __name__ == '__main__':
    Cameo().run()
