import numpy as np
import cv2
import matplotlib.pyplot as plt

# 读入图片
img = cv2.imread("../data/mm2.jpeg")
# 创建一个和加载图像一样形状的 填充为0的掩膜
mask = np.zeros(img.shape[:2], np.uint8)

# 创建以0填充的前景和背景模型
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 定义一个矩形
rect = (100, 50, 421, 378)

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

"""
cv2.grabCut()
参数：
    img: 输入图像
    mask: 蒙板图像，确定前景区域，背景区域，不确定区域，
        可以设置为cv2.GC_BGD,cv2.GC_FGD,cv2.GC_PR_BGD,cv2.GC_PR_FGD，
        也可以输入0,1,2,3
    rect: 包含前景的矩形 格式(x, y, w, h)
    bdgModel: 算法内部使用的数组. 你只需要创建两个大小为 (1,65)，数据类型为 np.float64 的数组
    fgdModel: 算法内部使用的数组. 你只需要创建两个大小为 (1,65)，数据类型为 np.float64 的数组
    iterCount: 算法的迭代次数 
    mode: 可以设置为 cv2.GC_INIT_WITH_RECT 或 cv2.GC_INIT_WITH_MASK，也可以联合使用。
        这是用来确定我们进行修改的方式，矩形模式或者掩模模式 

算法会修改掩模图像，在新的掩模图像中，
所有的像素被分为四类：背景，前景，可能是背景/前景使用 4 个不同的标签标记。
然后我们来修改掩模图像，
所有的 0 像素和 1 像素都被归为 0（例如背景），所有的 1 像素和 3 像素都被归为 1（前景）。
我们最终的掩模图像就这样准备好了。用它和输入图像相乘就得到了分割好的图像

原理：
    1 输入矩形框，矩形框外部区域都是背景。内部一定包含前景。
    2 电脑对输入图像进行初始化，标记前景和背景的像素。
    3 使用高斯混合模型（GMM）对前景和背景建模。
    4 根据输入，GMM会学习并创建新的像素分布。
        对未知的像素（前景或背景不确定），根据他们与已知的分类像素关系进行分类。（类似聚类操作）
    5 这样会根据像素的分布创建一幅图，图中节点是像素。
        除了像素点是节点以外，还有Source_node和Sink_node两个节点。
        所有的前景图像都与Source_node相连。背景与Sink_node相连。
    6 像素是否连接到Source_node/end_node依赖于权值，
        这个权值由像素属于同一类，也就是前景或者背景的概率来决定。
        如果像素的颜色有很大区别，那么他们之间的权重就很小。
    7 使用mincut算法对图像进行分割。
        它会根据最小代价方程对图像分成source_node和sink_node。
        代价方程是指裁剪所有边上权重的和。
        裁剪完成后，所有连接到source_node的判定为前景，sink_node上的为背景。
继续此过程，直到分类收敛。

    

"""
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
img = img*mask2[:, :, np.newaxis]

plt.subplot(121), plt.imshow(img)
plt.title("grabcut"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(cv2.imread("../data/mm2.jpeg"), cv2.COLOR_BGR2RGB))
plt.title("original"), plt.xticks([]), plt.yticks([])
plt.show()
