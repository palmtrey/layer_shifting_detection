import numpy as np
import cv2
from matplotlib import pyplot as plt

cropping_standard = cv2.imread('standard_tight.jpg',0) # standard
target_image = cv2.imread('test.jpg',0) # image that will be cropped

sift = cv2.SIFT_create()

# find the keypoints and descriptors
standard_kp, standard_des = sift.detectAndCompute(cropping_standard,None)
target_kp, target_des = sift.detectAndCompute(target_image,None)


bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)

matches = bf.match(standard_des,target_des)

x = list()
y = list()
for m in matches:
  (x1, y1) = target_kp[m.trainIdx].pt
  x.append(x1)
  y.append(y1)
  
x.sort()
y.sort()

x_start = int(x[0])
x_end = int(x[-1])
y_start = int(y[0])
y_end = int(y[-1])

cropped = target_image[x_start:x_end, y_start:y_end]

plt.imshow(cropped, cmap='gray')
plt.savefig('cropped.jpg')