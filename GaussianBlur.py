import cv2
import numpy as np

#Reads the image
#Replace with desired input path
image = cv2.imread(r"inputpath.jpg", cv2.IMREAD_GRAYSCALE)

#Adds a gaussian blur to the image
#5x5 kernel size picked so image is not blurred too much
#sigmaX of 1 picked for moderate deviation of blur
gaussianBlurredImage = cv2.GaussianBlur(image, (5, 5), sigmaX=1)

#Saves gaussian blurred image to output path
#Replace with desired output path
cv2.imwrite(r"outputpath.jpg", gaussianBlurredImage)
