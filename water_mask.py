import cv2
import numpy as np

input_path = "sat_lidar_pansharpen.jpg"
output_path = "water_fixed.jpg"

img = cv2.imread(input_path, cv2.IMREAD_COLOR)


Convert to HSV for easier color thresholding
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#define a range of HSV values that (roughly) capture water.
lower_hsv = np.array([70,  0,   0])
upper_hsv = np.array([140, 255, 120])

# Create a mask for pixels in this HSV range
mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

water_color = (100, 50, 50)
result = img.copy()
result[mask == 255] = water_color

cv2.imwrite(output_path, result)
print("Saved water-masked image to", output_path)
