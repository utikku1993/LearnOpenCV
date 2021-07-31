import numpy
import cv2

## OpenCV uses BGR format, not RGB

# Doesn't throw an error if the image is not read
img = cv2.imread('./data/1.jpg')

# Therefore check the size of image to ensure that it has actually been read
print(img.shape)

print(type(img))

# Show image
cv2.imshow('Image', img)

# Wait until input is received, to prevent the image window from closing
cv2.waitKey(0)

# Now once the input is received, close the windows
cv2.destroyAllWindows()