# coding:utf-8

import cv2


# 检测i方框 包含o方框
def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and ox + ow < ix + iw and oy + oh < iy + ih


# 将人外面的方框画出来
def draw_person(image, person):
    x, y, w, h = person
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)


# 读入图片
img = cv2.imread("../data/people2.jpg")
# 获取hog检测器对象
hog = cv2.HOGDescriptor()
# 设置检测人的默认检测器
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# 在图片中检测人，
# 返回found列表 每个元素是一个(x, y, w, h)的矩形，w是每一个矩形的置信度
found, w = hog.detectMultiScale(img)
found_filtered = []
# 如果方框有包含，只留下内部的小方块
for ri, r in enumerate(found):
    for qi, q in enumerate(found):
        if ri != qi and is_inside(r, q):
            break
        else:
            found_filtered.append(r)

# 将每一个方块画出来
for person in found_filtered:
    draw_person(img, person)


cv2.imshow("person detection", img)
cv2.waitKey()
cv2.destroyAllWindows()