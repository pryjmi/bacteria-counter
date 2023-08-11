import cv2
import numpy as np

# k-means
def kmeans_color_quantization(image, clusters=8, rounds=1):
  h, w = image.shape[:2]
  samples = np.zeros([h*w,3], dtype=np.float32)
  count = 0

  for x in range(h):
    for y in range(w):
      samples[count] = image[x][y]
      count += 1

  compactness, labels, centers = cv2.kmeans(samples, clusters, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), rounds, cv2.KMEANS_RANDOM_CENTERS)

  centers = np.uint8(centers)
  res = centers[labels.flatten()]
  return res.reshape((image.shape))

# load image
image = cv2.imread("img/bacteria_light_1.jpeg")
original = image.copy()

# perform k-means color segmentation, grayscale, Otsu's threshold
kmeans = kmeans_color_quantization(image, clusters=2)
gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# find contours, remove tiny specs using area filtering, gather points
points_list = []
size_list = []
cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
AREA_THRESHOLD = 2
for c in cnts:
  area = cv2.contourArea(c)
  if area < AREA_THRESHOLD:
    cv2.drawContours(thresh, [c], -1, 0, -1)
  else:
    (x,y), radius = cv2.minEnclosingCircle(c)
    points_list.append((int(x), int(y)))
    size_list.append(area)

# apply mask onto original image
result = cv2.bitwise_and(original, original, mask=thresh)
result[thresh==255] = (36, 255, 12)

# overlay on original
original[thresh==255] = (36, 255, 12)

print(f"Number of particles: {len(points_list)}")
print(f"Average size: {np.mean(size_list)}")

# display
cv2.imshow("kmeans", kmeans)
cv2.imshow("original", original)
cv2.imshow("thresh", thresh)
cv2.imshow("result", result)
cv2.waitKey(0)