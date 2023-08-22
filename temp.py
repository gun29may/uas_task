import cv2
import numpy as np


# Function to update the border based on trackbar values
def update_border(x):
    global lower_hsv, upper_hsv

    # Update HSV thresholds based on trackbar values
    lower_hsv = np.array([cv2.getTrackbarPos('Hue', 'Adjust Border'),
                          cv2.getTrackbarPos('Saturation', 'Adjust Border'),
                          cv2.getTrackbarPos('Value', 'Adjust Border')])

    upper_hsv = np.array([lower_hsv[0] + cv2.getTrackbarPos('Hue Range', 'Adjust Border'),
                          lower_hsv[1] + cv2.getTrackbarPos('Saturation Range', 'Adjust Border'),
                          lower_hsv[2] + cv2.getTrackbarPos('Value Range', 'Adjust Border')])

    # Create mask for red and brown regions
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Apply Gaussian blur to the mask to reduce noise
    blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Perform edge detection on the blurred mask
    edges = cv2.Canny(blurred_mask, threshold1=30, threshold2=70)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Separated Regions', result)


# Load the image
image = cv2.imread('images/1.png')

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Initialize HSV thresholds
lower_hsv = np.array([0, 100, 100])
upper_hsv = np.array([10, 255, 255])

# Create a window for the trackbars
cv2.namedWindow('Adjust Border')

# Create trackbars for adjusting the border
cv2.createTrackbar('Hue', 'Adjust Border', lower_hsv[0], 180, update_border)
cv2.createTrackbar('Saturation', 'Adjust Border', lower_hsv[1], 255, update_border)
cv2.createTrackbar('Value', 'Adjust Border', lower_hsv[2], 255, update_border)
cv2.createTrackbar('Hue Range', 'Adjust Border', upper_hsv[0] - lower_hsv[0], 180, update_border)
cv2.createTrackbar('Saturation Range', 'Adjust Border', upper_hsv[1] - lower_hsv[1], 255, update_border)
cv2.createTrackbar('Value Range', 'Adjust Border', upper_hsv[2] - lower_hsv[2], 255, update_border)

# Call the update_border function initially to display the result
update_border(0)

# Start the loop to handle events
while True:
    key = cv2.waitKey(1)
    if key == 27:  # Press Esc to exit
        break

# Release resources
cv2.destroyAllWindows()

'''import cv2
import numpy as np


def on_slider_change(value):
    global lower_green, upper_green
    lower_green[1] = value
    mask_green = cv2.inRange(image, lower_green, upper_green)

    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_triangle_count = 0

    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 3:
            roi = image[approx[0][0][1]:approx[2][0][1], approx[0][0][0]:approx[1][0][0]]
            red_threshold = 100
            if np.mean(roi[:, :, 2]) > red_threshold and np.mean(roi[:, :, 1]) < red_threshold:
                red_triangle_count += 1

    print("Number of red triangles in the green region:", red_triangle_count)
    cv2.imshow('Image', image)


image = cv2.imread('images/1.png')
cv2.namedWindow('Image')

# Initialize green color threshold values
lower_green = np.array([0, 0, 0])
upper_green = np.array([100, 255, 100])

# Create a slider for adjusting the green color threshold
cv2.createTrackbar('Green Threshold', 'Image', lower_green[1], 255, on_slider_change)

# Display the initial image
cv2.imshow('Image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''


