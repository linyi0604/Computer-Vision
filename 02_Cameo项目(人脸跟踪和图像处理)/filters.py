# coding:utf8

import cv2
import numpy
import utils
import scipy
"""
边缘检测： 
    先平滑消除噪声
    再转为灰度图像
    再边缘检测
    
    1 模糊消除噪声
    2 转为灰度图像
    3 边缘检测 转为边缘亮色 非背景黑色
    4 将边缘检测结果转换为黑色边缘 被色背景的图像
    5 原图像边缘变黑
"""


def strokeEdge(src, dst, blurKsize=7, edgeKsize=5):
    if blurKsize >= 3:
        # 算数平均卷积消除噪声
        blurredSrc = cv2.medianBlur(src, blurKsize)
        # 转为灰度图像
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        # 如果卷积核太小，就直接转为灰度图像
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Laplacian边缘检测 会将非边缘变成黑色，边缘变成其他颜色
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)


# 一般的卷积滤波器
class VConvolutionFilter(object):
    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        # 做卷积运算， -1表示和输入图像的深度是一样的
        cv2.filter2D(src, -1, self._kernel, dst)


# 锐化滤波器
class SharpenFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ])
        VConvolutionFilter.__init__(self, kernel)


# 检测边缘滤波器 边缘为白色 非边缘为黑色
class FindEdgesFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ])
        VConvolutionFilter.__init__(self, kernel)


# 模糊滤波器
class BlurFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([
            [0.04, 0.04, 0.04, 0.04, 0.04],
            [0.04, 0.04, 0.04, 0.04, 0.04],
            [0.04, 0.04, 0.04, 0.04, 0.04],
            [0.04, 0.04, 0.04, 0.04, 0.04],
            [0.04, 0.04, 0.04, 0.04, 0.04],
        ])
        VConvolutionFilter.__init__(self, kernel)


# 浮雕效果滤波器
class EmbossFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ])
        VConvolutionFilter.__init__(self, kernel)


"""
原书第三版 缺少部分
"""


class BGRFuncFilter(object):

    def __init__(self, vFunc=None, bFunc=None, gFunc=None, rFunc=None, dtype=numpy.uint8):

        length = numpy.iinfo(dtype).max + 1
        self._bLookupArray = utils.createLookupArray(utils.createCompositeFunc(bFunc, vFunc), length)
        self._gLookupArray = utils.createLookupArray(utils.createCompositeFunc(gFunc, vFunc), length)
        self._rLookupArray = utils.createLookupArray(utils.createCompositeFunc(rFunc, vFunc), length)

    def apply(self, src, dst) :

        """Apply the filter with a BGR source/destination."""
        b, g, r = cv2.split(src)
        utils.applyLookupArray(self._bLookupArray, b, b)
        utils.applyLookupArray(self._gLookupArray, g, g)
        utils.applyLookupArray(self._rLookupArray, r, r)
        cv2.merge([ b, g, r ], dst)


class BGRCurveFilter(BGRFuncFilter):

    def __init__(self, vPoints = None, bPoints = None,gPoints = None, rPoints = None, dtype = numpy.uint8):
        BGRFuncFilter.__init__(self, utils.createCurveFunc(vPoints), utils.createCurveFunc(bPoints), utils.createCurveFunc(gPoints), utils.createCurveFunc(rPoints), dtype)


class BGRPortraCurveFilter(BGRCurveFilter):
    def __init__(self, dtype = numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            vPoints = [ (0, 0), (23, 20), (157, 173), (255, 255) ],
            bPoints = [ (0, 0), (41, 46), (231, 228), (255, 255) ],
            gPoints = [ (0, 0), (52, 47), (189, 196), (255, 255) ],
            rPoints = [ (0, 0), (69, 69), (213, 218), (255, 255) ],
            dtype = dtype)