import numpy as np
import imutils
import cv2

# dict to count colonies
counter = {}

# load the image
image_orig = cv2.imread("src/img/bacteria_light.jpg")
height_orig, width_orig = image_orig.shape[:2]

# output image with contours
image_contours = image_orig.copy()

# detect white colonies
colors = ["white"]
for color in colors:
  # copy of original image
  image_to_process = image_orig.copy()
  counter[color] = 0
  if color == "white":
    image_to_process = (255-image_to_process)
    lower = np.array([50, 50, 40])
    upper = np.array([100, 120, 80])

# find the colors within the specified boundaries
image_mask = cv2.inRange(image_to_process, lower, upper)
# apply the mask
image_res = cv2.bitwise_and(image_to_process, image_to_process, mask=image_mask)

# load the image, convert to grayscale, and blur it slightly
image_gray = cv2.cvtColor(image_res, cv2.COLOR_BGR2GRAY)
image_gray = cv2.GaussianBlur(image_gray, (5,5), 0)

# edge detection, then perform a dilation + erosion to close gaps in between object edges
image_edged = cv2.Canny(image_gray, 50, 100)
image_edged = cv2.dilate(image_edged, None, iterations=1)
image_edged = cv2.erode(image_edged, None, iterations=1)

# find contours in the edge map
cnts = cv2.findContours(image_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# loop over the contours individually
for c in cnts:
  c = c.reshape(-1, 1, 2)
  if cv2.contourArea(c) < 5:
    continue

  hull = cv2.convexHull(c)
  if color == "white":
    cv2.drawContours(image_contours, [hull], 0, (0,255,0),1)

  counter[color] += 1

print(f"{color} colonies: {counter[color]}")
cv2.imshow("image", image_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()