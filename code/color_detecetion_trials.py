import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import io , measure
from scipy import ndimage as nd


img = io.imread("random.jpg")
plt.imshow(img)
plt.show()


hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
mask = cv2.inRange(hsv,(96, 100, 100), (125, 255, 255))
plt.imshow(mask)
plt.show()
# Apply morphological opening to remove small noise
opened_mask = nd.binary_opening(mask, np.ones((7,7)))
closed_mask= nd.binary_closing(opened_mask,np.ones((10,10)))

# Create an image that shows only the detected blue color and black background
color_only = np.zeros_like(img)  # Create a black image of the same size as the input
color_only[closed_mask] = img[closed_mask]  # Copy the color pixels to the black image

# Display the image showing only the detected color
plt.imshow(color_only)
plt.show()


label_image = measure.label(closed_mask)
plt.imshow(label_image)
plt.show()

# Find the center of mass (centroid) for each detected region
centroids = [prop.centroid for prop in measure.regionprops(label_image)]
print("Centroids (y, x) of detected regions:", centroids)

# Visualize the centroids on the color-only image
fig, ax = plt.subplots()
ax.imshow(color_only)

# Mark each centroid on the image
for centroid in centroids:
    ax.plot(centroid[1], centroid[0], 'bo')  # 'ro' marks the centroid with a red dot

plt.show()








