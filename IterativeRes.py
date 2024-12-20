import cv2
import numpy as np

def iterativeRestoration(image_path, kernel_size, iterations):
    """Takes a blurred image input and returns an iteratively restored image output
        Based on a known kernel size for the blur and a select number of iterations"""
    
    # Read the blurred image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("The image path is incorrect.")

    # Recreate what was the horizontal blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1 / kernel_size

    #Copy image and convert it to float32 to allow for higher precision
    # and prevent overflow and underflow
    tempImage = image.copy().astype(np.float32)

    #Run multiple iterations to reduce horizontal linear blurring
    for i in range(iterations):
        # Blur the current estimate using 2D convolution
        blur = cv2.filter2D(tempImage, -1, kernel)
        
        # Compute the error
        error = image - blur

        # Update the image using the error with 2D convolution
        tempImage += cv2.filter2D(error, -1, kernel)

    #Converts final image to uint8 so that image can be saved as jpg
    restoredImage = np.clip(tempImage, 0, 255).astype(np.uint8)

    return restoredImage


if __name__ == "__main__":
    """Main takes input image path, output image path, kernel size of previous blurring function, and number of iterations
        Iteratively restores blurred image back to original image"""

    #Input Image path (replace with desired input image path)
    inputImage = r"inputpath.jpg"
    #Output Image path (replace with desired output image path)
    outputImage = r"outputpath.jpg"

    #Size of blur kernel from previous horizontal linear blurring
    kernelSize = 31
    #Number of iterations to update image
    iterations = 100

    restoredImage = iterativeRestoration(inputImage, kernelSize, iterations)

    #Saves the restored image to output path
    cv2.imwrite(outputImage, restoredImage)

    cv2.imshow(f'Iteratively Restored Image with {iterations} iterations', restoredImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

