import cv2
import numpy as np

def convert_color_range(color_range, exclude_color=None):
    lower, upper = color_range
    if exclude_color is not None:
        lower = np.array(lower)
        upper = np.array(upper)
        lower_bound = lower - 10
        upper_bound = upper + 10
        exclude_mask = ((lower_bound <= exclude_color) & (exclude_color <= upper_bound)).all(axis=-1)
        lower[exclude_mask] = exclude_color
        upper[exclude_mask] = exclude_color
    return lower, upper

# Read the image
image = cv2.imread('images/1.png')

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define HSV ranges for green and brown colors
green_range = ([35, 50, 50], [85, 255, 255])
brown_range = ([10, 100, 100], [30, 255, 255])
red_range = ([0, 100, 100], [10, 255, 255])

# Define colors to exclude from green, brown, and red
exclude_brown_color = np.array([240, 100, 100])
exclude_red_color = np.array([0, 100, 100])

# Convert color ranges
lower_green, upper_green = convert_color_range(green_range, exclude_red_color)
lower_brown, upper_brown = convert_color_range(brown_range, exclude_brown_color)
lower_red, upper_red = convert_color_range(red_range)

# Create masks for green, brown, and red areas
green_mask = cv2.inRange(hsv_image, np.array(lower_green), np.array(upper_green))
brown_mask = cv2.inRange(hsv_image, np.array(lower_brown), np.array(upper_brown))
red_mask = cv2.inRange(hsv_image, np.array(lower_red), np.array(upper_red))

# Create a mask for blue regions
exclude_mask = green_mask | brown_mask | red_mask
blue_mask = np.logical_not(exclude_mask)

# Create blue and yellow channels based on the masks
blue_channel = np.zeros_like(hsv_image[:, :, 0])
blue_channel[blue_mask] = 120

yellow_channel = np.zeros_like(hsv_image[:, :, 0])
yellow_channel[brown_mask > 0] = 30

# Merge modified channels to create the final image
modified_hsv_image = hsv_image.copy()
modified_hsv_image[:, :, 0] = blue_channel + yellow_channel

# Convert the modified HSV image back to BGR color space
result_image = cv2.cvtColor(modified_hsv_image, cv2.COLOR_HSV2BGR)

# Display or save the result
cv2.imshow('Modified Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
