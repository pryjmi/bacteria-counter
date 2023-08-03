import cv2
import numpy as np

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

# find the contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# draw the contours
count = 0
for c in contours:
  if cv2.contourArea(c) < 0.05:
    cv2.drawContours(rgb_image, contours, -1, (0,255,0), 1)
    count += 1

print(count)

#cv2.imshow("image", thresh)
cv2.imshow("image", rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()