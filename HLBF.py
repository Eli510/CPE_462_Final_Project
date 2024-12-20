import cv2
import numpy as np
import random

def horizontalLinearBlur(input, output, kernelSize):
    """ Apply a random horizontal linear blur to an image."""

    #Reads image and states error if image path is wrong
    image = cv2.imread(input)
    if image is None:
        raise ValueError("The image path is incorrect.")

    #Kernel is created based on kernel size
    #Initially contains all zeroes
    kernel = np.zeros((kernelSize, kernelSize))
    #Middle row values all become 1/kernel_size (1/31 in this case) to create horizontal blur kernel 
    kernel[kernelSize // 2, :] = 1 / kernelSize 
    

    # 2D convolution is applied to utilize the horizontal blur kernel
    # Image is blurred horizontally
    blurredImage = cv2.filter2D(image, -1, kernel)

    #The resulting image is saved to where the output path indicates
    cv2.imwrite(output, blurredImage)


if __name__ == "__main__":
    """Main takes an input and output path (for input and output image)
       Horizontal Linear Blurring Function is applied to input image to produce output image"""
    
    #Input Image Path (replace with desired input path)
    inputImage = r"inputpath.jpg"
    #Output Image Path (replace with desired output path)
    outputImage = r"outputpath.jpg"
    
    # Kernel size for blurring is picked
    # Kernel size must be odd so that there is a center pixel
    # Odd height and width will allow this
    #For our example we picked 31x31 kernel (large kernel size)
    kernelSize = 31

    horizontalLinearBlur(inputImage, outputImage, kernelSize)
