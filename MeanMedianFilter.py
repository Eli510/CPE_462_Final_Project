import cv2
import numpy as np


balancePercentage = 0.2


def meanFilter(image, height, width):
    #Create the kernel
    kernel = np.ones((3, 3), np.float32) / 9

    #Goes through each pixel
    for row in range(1, height + 1):
        for column in range(1, width + 1):
            # Generates area to be filtered
            filter_area = image[row - 1:row + 2, column - 1:column + 2]
            #Adds the mean filter
            image[row][column] = np.sum(np.multiply(kernel, filter_area))

    return image


def medianFilter(image, height, width):
    #Goes through each pixel
    for row in range(1, height + 1):
        for column in range(1, width + 1):
            #Generates area to be filtered
            filter_area = image[row - 1:row + 2, column - 1:column + 2]
            #Adds the median filter
            image[row][column] = np.median(filter_area)

    return image


def balancedMeanMedianFilter(image, height, width):
    #Goes through each pixel
    for row in range(1, height + 1):
        for column in range(1, width + 1):
            #Creates kernel for mean filter
            kernel = np.ones((3, 3), np.float32) / 9
            #Generates area to be filtered
            filter_area = image[row - 1:row + 2, column - 1:column + 2]
            #Mean Filtering
            meanFilterVector = np.sum(np.multiply(kernel, filter_area))
            #Median Filtering
            medianFilterVector = np.median(filter_area)
            #Adds Mean and Median filter with a certain balance percentage considered
            image[row][column] = balancePercentage * meanFilterVector + (1 - balancePercentage) * medianFilterVector
    return image


def filter_image(image, filtering_function):
    # Get the image size for the kernel looping.
    height, width = image.shape[:2]

    # Add 1px reflected padding to allow kernels to work properly.
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)

    finalImage = filtering_function(image, height, width)

    return finalImage


if __name__ == "__main__":
    """Main Takes image with noise as input and applies mean, median, and a mean meadian balanced filter to image
        Saves final mean median balanced filter"""

    #Input Image path (replace with desired input image path)
    inputImage = r"inputpath.jpg"
    #Output Image path (replace with desired output image path)
    outputImage = r"outputpath.jpg"

    # Read and print the original image.
    image = cv2.imread(inputImage, 0)
    cv2.imshow('Original Image', image)

    # Calculate the mean filtered version and print the resulting image.
    meanFilterImage = filter_image(image, meanFilter)
    cv2.imshow('Mean filtered Image', meanFilterImage)

    # Calculate the median filtered version and print the resulting image.
    medianFiltermage = filter_image(image, medianFilter)
    cv2.imshow('Median filtered Image', medianFiltermage)

    # Calculate the balanced filtered version and print the resulting image.
    balancedMeanMedianFilterImage = filter_image(image, balancedMeanMedianFilter)
    cv2.imshow(f'Mean & Median with balance {balancePercentage} filtered Image', balancedMeanMedianFilterImage )

    #Save as jpg the balanced filter
    cv2.imwrite(outputImage, balancedMeanMedianFilterImage )

    # Destroy all the images on any key press.
    cv2.waitKey(0)
    cv2.destroyAllWindows()


