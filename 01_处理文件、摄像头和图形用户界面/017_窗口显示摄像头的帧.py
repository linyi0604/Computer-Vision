# coding:utf8

import cv2


"""
在窗口显示摄像头帧

    namedWindow() 指定窗口名
    imshow() 创建窗口
    DestroyWindow() 销毁所有窗口
    waitKey() 获取键盘输入
    setMouseCallback() 获取鼠标输入    

"""
"""
    opencv窗口只有调用waitKey()后才能实时更新
    waitKey() 只有窗口创建后才能捕获键盘
"""

clicked = False


def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True


cameraCapture = cv2.VideoCapture(0)     # 传入摄像头设备的索引
cv2.namedWindow("MyWindow")     # 设置窗口名称
cv2.setMouseCallback("MyWindow", onMouse)  # 传入窗口名称和响应捕获的函数
"""
setMouseCallback() 第二个参数接收一个回调函数
回调事件可以取值如下：
    cv2.EVENT_MOUSEMOVE 鼠标移动
    cv2.EVENT_LBUTTONDOWN 左按键按下
    cv2.EVENT_RBUTTONDOWN 右按键按下
    cv2.EVENT_MBUTTONDOWN 中间键按下
    cv2.EVENT_LBUTTONDBLCLK 双击左键
    cv2.EVENT_RBUTTONDBLCLK 双击右键
    cv2.EVENT_MBUTTONDBLCLK 双击中间
    
鼠标回调的标志参数可能是以下事件的桉位组合：
    cv2.EVENT_FLAG_LBUTTON 按下鼠标左键
    cv2.EVENT_FLAG_RBUTTON 按下鼠标右键
    cv2.EVENT_FLAG_MBUTTON 按下中间
    cv2.EVENT_FLAG_CTRLKEY 按下ctrl键
    cv2.EVENT_FLAG_SHITKEY 按下shift键
    cv2.EVENT_FLAG_ALTKEY 按下alt键
"""

print("点击窗口或者按键停止")

success, frame = cameraCapture.read()
while success and cv2.waitKey(1) == -1 and not clicked:
    cv2.imshow("MyWindow", frame)
    success, frame = cameraCapture.read()

cv2.destroyAllWindows()
cameraCapture.release()