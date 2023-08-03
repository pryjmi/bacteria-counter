import cv2
import numpy as np
import matplotlib.pyplot as plt

# load the image
image = cv2.imread("src/img/bacteria_dark.jpg", 0)
rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# histogram equalization
image = cv2.equalizeHist(image)

# binary thresholding
ret, image = cv2.threshold(image, 253, 255, cv2.THRESH_BINARY)

# define erosion kernel
kernel = np.ones((3,3), np.uint8)

# apply erosion
thresh = cv2.erode(image, kernel, iterations=0)

# find the circles
circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=1.2, minDist=13, param1=50, param2=0.9, minRadius=1, maxRadius=5)

count = 0

if circles is not None:
  circles = np.round(circles[0, :]).astype("int")
  
  for (x,y,r) in circles:
    cv2.circle(rgb_image, (x,y), r, (0,255,0), 1)
    count += 1

print(count)

"""
cv2.imshow("image", rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(thresh, cmap="gray")
plt.title("Thresholded image")

plt.subplot(1,2,2)
plt.imshow(rgb_image)
plt.title("RGB image")

plt.show()