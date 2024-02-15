import cv2 # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# 读取图像
# img = cv2.imread('01.jpg')
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img = cv2.imread('01.jpg')
img = cv2.pyrDown(img)
cv_show('img', img)
# (854, 900)
print(img.shape)

img = cv2.imread('01.jpg', 0)
plt.hist(img.ravel(), 256)
plt.show()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
res_clahe = clahe.apply(img)
res = np.hstack((img, res_clahe))
cv_show('res', res)

# equ = cv2.equalizeHist(img)
# plt.hist(equ.ravel(), 256)
# plt.show()
#
# res = np.hstack((img, equ))
# cv_show('res', res)

# img = cv2.imread('01.jpg', 0)
# mask = np.zeros(img.shape[:2], np.uint8)
# mask[100:300, 100:400] = 255
# cv_show('mask', mask)
# img = cv2.imread('01.jpg', 0)
# cv_show('img', img)
# masked_img = cv2.bitwise_and(img, img, mask=mask) # 与操作
# cv_show('masked_img', masked_img)
# hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
# hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])
# plt.subplot(221), plt.imshow(img, 'gray')
# plt.subplot(222), plt.imshow(mask, 'gray')
# plt.subplot(223), plt.imshow(masked_img, 'gray')
# plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
# plt.xlim([0, 256])
# plt.show()


# # 0表示灰度图
# img = cv2.imread('01.jpg', 0)
# hist = cv2.calcHist([img], [0], None, [256], [0,256])
# # (256, 1)
# print(hist.shape)
# plt.hist(img.ravel(), 256)
# plt.show()
#
# img = cv2.imread('01.jpg')
# color = ('b', 'g', 'r')
# for i, col in enumerate(color):
#     histr = cv2.calcHist([img], [i], None, [256], [0, 256])
#     plt.plot(histr, color=col)
#     plt.xlim([0, 256])
# plt.show()

# img_rgb = cv2.imread('01.jpg')
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# template = cv2.imread('02.jpg', 0)
# h, w = template.shape[:2]
#
# res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
# threshold = 0.8
# # 取匹配程度大于80%的坐标
# loc = np.where(res >= threshold)
# for pt in zip(*loc[::-1]): # *号表示可选参数
#     bottom_right = (pt[0] + w, pt[1] + h)
#     cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)
# cv2.imshow('img_rgb', img_rgb)
# cv2.waitKey(0)


# img = cv2.imread('01.jpg', 0)
# template = cv2.imread('02.jpg', 0)
# h, w = template.shape[:2]
# # (854, 900)
# print(img.shape)
# # (1500, 1500)
# print(template.shape)
#
# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
# res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
# # (647, 601)
# print(res.shape)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# # 0.937662661075592
# print(min_val)
# # 0.9643488526344299
# print(max_val)
# # (389, 189)
# print(min_loc)
# # (0, 0)
# print(max_loc)
# for meth in methods:
#     img2 = img.copy()
#     # 匹配方法的真值
#     method = eval(meth)
#     print(method)
#     res = cv2.matchTemplate(img, template, method)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#     # 如果是平方差匹配TM_SQDIFF或归一化平方差匹配TM_SQDIFF_NORMED，取最小值
#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     # 画矩形
#     cv2.rectangle(img2, top_left, bottom_right, 255, 2)
#     plt.subplot(121), plt.imshow(res, cmap='gray')
#     plt.xticks([]), plt.yticks([]) # 隐藏坐标轴
#     plt.subplot(122), plt.imshow(img2, cmap='gray')
#     plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#     plt.show()

# 转换灰度图
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# cv_show('thresh', thresh)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cnt = contours[0]

# (x, y), radius = cv2.minEnclosingCircle(cnt)
# center = (int(x), int(y))
# radius = int(radius)
# img = cv2.circle(img, center, radius, (0, 255, 0), 2)
# cv_show('img', img)

# x, y, w, h = cv2.boundingRect(cnt)
# img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255.0), 2)
# cv_show('img', img)
# area = cv2.contourArea(cnt)
#
# x, y, w, h = cv2.boundingRect(cnt)
# rect_area = w * h
# extent = float(area) / rect_area
# # 54410616705699
# print('轮廓面积与边界矩形比', extent)


# draw_img = img.copy()
# res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
# cv_show('res', res)
#
# epsilon = 0.1 * cv2.arcLength(cnt, True)
# approx = cv2.approxPolyDP(cnt, epsilon, True)
# draw_img = img.copy()
# res = cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)
# cv_show('res', res)


# # 绘制轮廓，传入绘制图像，轮廓，轮廓索引，颜色模式，线条厚度
# # 注意需要copy，否则原图会变
# draw_img = img.copy()
# res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
# cv_show('res', res)
#
# draw_img = img.copy()
# res = cv2.drawContours(draw_img, contours, 0, (0, 0, 255), 2)
# cv_show('res', res)
#
# cnt = contours[0]
# # 面积 191274.0
# print(cv2.contourArea(cnt))
# # 周长，True表示闭合的 1750.0
# print(cv2.arcLength(cnt, True))

# up = cv2.pyrUp(img)
# # (1708, 1800)
# print(up.shape)
# cv_show('up', up)
#
# down = cv2.pyrDown(img)
# # (427, 450)
# print(down.shape)
# cv_show('down', down)

# v1 = cv2.Canny(img, 80, 150)
# v2 = cv2.Canny(img, 50, 100)
# res = np.hstack((v1, v2))
# cv_show('res', res)

# # 不同算子的差异
# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
# sobelx = cv2.convertScaleAbs(sobelx)
# sobely = cv2.convertScaleAbs(sobely)
# sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
#
# scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
# scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
# scharrx = cv2.convertScaleAbs(sobelx)
# scharry = cv2.convertScaleAbs(sobely)
# scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
#
# laplacian = cv2.Laplacian(img, cv2.CV_64F)
# laplacian = cv2.convertScaleAbs(laplacian)
#
# res = np.hstack((sobelxy, scharrxy, laplacian))
# cv_show('res', res)


#
# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# cv_show('sobelx', sobelx)
# # 白到黑是整数，黑到白就是负数了，所有的负数会被截断成0，所以要取绝对值
# sobelx = cv2.convertScaleAbs(sobelx)
# cv_show('sobelx', sobelx)
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
# sobely = cv2.convertScaleAbs(sobely)
# cv_show('sobely', sobely)
# # 分别计算x和y，再求和
# sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# cv_show('sobelxy', sobelxy)
#
# # 不建议直接计算
# sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
# sobelxy = cv2.convertScaleAbs(sobelxy)
# cv_show('sobelxy直接计算', sobelxy)
#
# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# sobelx = cv2.convertScaleAbs(sobelx)
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
# sobely = cv2.convertScaleAbs(sobely)
# sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# cv_show('sobelxy分别计算', sobelxy)


# # 礼帽=原始输入-开运算结果
# kernel = np.ones((7, 7), np.uint8)
# tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# cv2.imshow('tophat', tophat)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 黑帽=闭运算-原始输入
# blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
# cv2.imshow('blackhat', blackhat)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 梯度=膨胀-腐蚀
# kernel = np.ones((7, 7), np.uint8)
# dilate = cv2.dilate(img, kernel, iterations=5)
# erosion = cv2.erode(img, kernel, iterations=5)
# res = np.hstack((dilate, erosion))
# cv2.imshow('res', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# cv2.imshow('gradient', gradient)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 开：先腐蚀，再膨胀
# kernel = np.ones((5, 5), np.uint8)
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# cv2.imshow('opening', opening)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 闭：先膨胀，再腐蚀
# kernel =np.ones((5, 5), np.uint8)
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('closing', closing)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# kernel = np.ones((3, 3), np.uint8)
# dige_erosion = cv2.erode(img, kernel, iterations=1)
# cv2.imshow('erosion', dige_erosion)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# kernel = np.ones((3, 3), np.uint8)
# dige_dilate = cv2.dilate(dige_erosion, kernel, iterations=1)
# cv2.imshow('dilate', dige_dilate)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# kernel = np.ones((30, 30), np.uint8)
# dilate_1 = cv2.dilate(img, kernel, iterations=1)
# dilate_2 = cv2.dilate(img, kernel, iterations=2)
# dilate_3 = cv2.dilate(img, kernel, iterations=3)
# res = np.hstack((dilate_1, dilate_2, dilate_3))
# cv2.imshow('res', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# kernel = np.ones((5, 5), np.uint8)
# erosion = cv2.erode(img, kernel, iterations=1)
# cv2.imshow('erosion', erosion)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# erosion_1 = cv2.erode(img, kernel, iterations=1)
# erosion_2 = cv2.erode(img, kernel, iterations=2)
# erosion_3 = cv2.erode(img, kernel, iterations=3)
# res = np.hstack((erosion_1, erosion_2, erosion_3))
# cv2.imshow('res', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 均值滤波，简单的平均卷积操作
# blur = cv2.blur(img, (3, 3))
# cv2.imshow('blur', blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 方框滤波
# # 基本和均值一样，可以选择归一化
# box = cv2.boxFilter(img, -1, (3,3), normalize=True)
# cv2.imshow('box', box)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 方框滤波
# # 基本和均值一样，可以选择归一化，容易越界
# box = cv2.boxFilter(img, -1, (3,3), normalize=False)
# cv2.imshow('box', box)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 高斯滤波，高斯模糊的卷积核里的数值是满足高斯分布，相当于更重视中间的
# aussian = cv2.GaussianBlur(img, (5, 5), 1)
# cv2.imshow('aussian', aussian)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 中值滤波，相当于用中值代替
# median = cv2.medianBlur(img, 5)
# cv2.imshow('median', median)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 展示所有的
# res = np.hstack((blur, aussian, median))
# cv2.imshow('median vs average', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# (854, 900, 3)
# print(img.shape)
# print(img)
# # 图像的显示
# cv2.imshow('image', img)
# # 等待时间，毫秒级，0表示任意键终止
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# img = cv2.imread('01.jpg', cv2.IMREAD_GRAYSCALE)
# (854, 900)
# print(img.shape)
# print(img)
# # <class 'numpy.ndarray'>
# print(type(img))
# # 2305800
# print(img.size)
# # uint8
# print(img.dtype)

# cv2.imwrite('jienigui.jpg', img)

# img = cv2.imread('01.jpg')
# img01 = img[0:50, 0:200]
# cv_show('02.jpg', img01)

# b, g, r = cv2.split(img)
# print(b)
# img = cv2.merge((b, g, r))
# # (854, 900, 3)
# print(img.shape)


# ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
# ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
# ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
#
# titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
# for i in range(6):
#     plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

# # 只保留R
# cur_img = img.copy()
# cur_img[:, :, 0] = 0
# cur_img[:, :, 1] = 0
# cv_show('R', cur_img)
# # 只保留G
# cur_img = img.copy()
# cur_img[:, :, 0] = 0
# cur_img[:, :, 2] = 0
# cv_show('G', cur_img)
# # 只保留B
# cur_img = img.copy()
# cur_img[:, :, 1] = 0
# cur_img[:, :, 2] = 0
# cv_show('B', cur_img)

# top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
# replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
# reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT)
# reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT_101)
# wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_WRAP)
# constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT, value=0)
#
# plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
# plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
# plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
# plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
# plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
# plt.subplot(236), plt.imshow(img, 'gray'), plt.title('CONSTANT')
# plt.show()

# img01 = cv2.imread('01.jpg')
# img02 = cv2.imread('02.jpg')
# img03 = img01 - 10
# cv_show('result', img01)
# # img01 + img02 ValueError: operands could not be broadcast together with shapes (854,900,3) (1500,1500,3)
# # img01 + img02
# img01 = cv2.resize(img01, (1500, 1500))
# # (1500, 1500, 3)
# print(img01.shape)
# res = cv2.addWeighted(img01, 0.4, img02, 0.6, 0)
# cv_show('result', res)
# img01 = cv2.resize(img01, (0, 0), fx=4, fy=4)
# cv_show('result', img01)
# img01 = cv2.resize(img01, (0, 0), fx=1, fy=4)
# cv_show('result', img01)

# # [[255 255 255 ... 255 255 255]
# #  [255 255 255 ... 255 255 255]
# #  [255 255 255 ... 255 255 255]
# #  [255 255 255 ... 255 255 255]
# #  [255 255 255 ... 255 255 255]]
# print(img01[:5, :, 0])
# # [[245 245 245 ... 245 245 245]
# #  [245 245 245 ... 245 245 245]
# #  [245 245 245 ... 245 245 245]
# #  [245 245 245 ... 245 245 245]
# #  [245 245 245 ... 245 245 245]]
# print(img03[:5, :, 0])
# # [[244 244 244 ... 244 244 244]
# #  [244 244 244 ... 244 244 244]
# #  [244 244 244 ... 244 244 244]
# #  [244 244 244 ... 244 244 244]
# #  [244 244 244 ... 244 244 244]]
# # 相当于 % 256
# print((img01 + img03)[:5, :, 0])
# # [[255 255 255 ... 255 255 255]
# #  [255 255 255 ... 255 255 255]
# #  [255 255 255 ... 255 255 255]
# #  [255 255 255 ... 255 255 255]
# #  [255 255 255 ... 255 255 255]]
# print(cv2.add(img01, img03)[:5, :, 0])
