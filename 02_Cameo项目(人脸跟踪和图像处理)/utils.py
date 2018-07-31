# coding:utf8

import cv2
import numpy
import scipy.interpolate


def createCurveFunc(points):
    """Return a function derived from control points."""
    if points is None:
        return None
    num_points = len(points)
    if num_points < 2:
        return None
    xs, ys = zip(*points)
    if num_points < 4:
        kind = 'linear'
        # 'quadratic' is not implemented.
    else:
        kind = 'cubic'
    return scipy.interpolate.interp1d(xs, ys, kind,bounds_error = False)


def createLookupArray(func, length = 256):
    """Return a lookup for whole-number inputs to a function. The lookup values are clamped to [0, length - 1]."""
    if func is None:
        return None
    lookup_array = numpy.empty(length)
    i = 0
    while i < length:
        func_i = func(i)
        lookup_array[i] = min(max(0, func_i), length - 1)
        i += 1
    return lookup_array

def applyLookupArray(lookup_array, src, dst):
    """Map a source to a destination using a lookup."""
    if lookup_array is None:
        return
    dst[:] = lookup_array[src]


def createCompositeFunc(func0, func1):
    """Return a composite of two functions."""
    if func0 is None:
        return func1
    if func1 is None:
        return func0
    return lambda x: func0(func1(x))


def createFlatView(array):
    """Return a 1D view of an array of any dimensionality."""
    flat_view = array.view()
    flat_view.shape = array.size
    return flat_view

