'''import cv2 as cv
import numpy as np

img=cv.imread('images/1.png')
blank = np.zeros(img.shape, dtype='uint8')
blur=cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)
canny=cv.Canny(blur,100,10)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(img, 49, 150, cv.THRESH_BINARY)
#et, thresh = cv.threshold(img, 200, 150, cv.THRESH_BINARY)

cv.imshow('Thresh', thresh)
gray = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)

blur_1=cv.blur(thresh,(16,14),cv.BORDER_DEFAULT)
canny=cv.Canny(blur_1,100,100)

cv.imshow("edges ",canny)
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.imshow('Contours Drawn', blank)


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(src=gray, ksize=(1, 25), sigmaX=1.5)
canny = cv.Canny(blurred, 70, 135)
cv.imshow("edges",canny)
img_hsv=cv.cvtColor(img,cv.COLOR_BGR2LAB)

ret, thresh = cv.threshold(img, 200, 150, cv.THRESH_BINARY)
dest_and = cv.bitwise_and(img_hsv, thresh, mask = None)

cv.imshow("1",thresh)
cv.waitKey(0)'''
import cv2
import numpy as np


def find_blue_triangles(brown_threshold):
    image = cv2.imread("images/1.png")
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_brown = np.array([0, brown_threshold, 20])  # Adjust the second value for the slider
    upper_brown = np.array([30, 255, 255])

    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    brown_blue_mask = cv2.bitwise_and(brown_mask, blue_mask)

    contours, _ = cv2.findContours(brown_blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 3:
            cv2.drawContours(image, [approx], 0, (0, 255, 0), -1)

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def on_trackbar(val):
    brown_threshold = val
    find_blue_triangles(brown_threshold)


# Create a window with a slider
cv2.namedWindow("Brown Threshold")
cv2.createTrackbar("Threshold", "Brown Threshold", 0, 255, on_trackbar)

# Initialize with default threshold value
default_brown_threshold = 50  # You can set the initial value here
cv2.setTrackbarPos("Threshold", "Brown Threshold", default_brown_threshold)

# Call the function with the default threshold
find_blue_triangles(default_brown_threshold)

