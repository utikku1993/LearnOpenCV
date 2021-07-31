import numpy
import cv2

## OpenCV uses BGR format, not RGB

# Doesn't throw an error if the image is not read
img = cv2.imread('./data/1.jpg')

## ---------- Extract color channels ---------------------------------

blue = img[:,:,0]

green = img[:,:,1]

red = img[:,:,2]

## ----------------------------------------------------------------------

# Show images
cv2.imshow('Image', img)
cv2.imshow('Blue', blue)
cv2.imshow('Green', green)
cv2.imshow('Red', red)

# Wait until input is received, to prevent the image window from closing
cv2.waitKey(0)

# Now once the input is received, close the windows
cv2.destroyAllWindows()