import cv2
import numpy as np

image = cv2.imread('01.jpg')
# (854, 900, 3)
print('image shape:', image.shape)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
# (854, 900)
print('dst.shape:', dst.shape)
image[dst > 0.01 * dst.max()] = [0, 0, 255]
cv2.imshow('dst', image)
cv2.waitKey(0)
cv2.destroyAllWindows()