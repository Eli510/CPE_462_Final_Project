import cv2
import numpy as np

def whiteGaussianNoise(input, output, mean, stddev):
    """Takes an input image path, output image path, noise signal mean, noise signal stddev
        Returns an output image with a specific white gaussian noise added to it based on mean and stddev"""
    
    #Reads the image
    image = cv2.imread(input)
    if image is None:
        raise ValueError("The image path is incorrect.")

    #Creates the Gaussian Noise (converting to float32 for better precision)
    gaussianNoise = np.random.normal(mean, stddev, image.shape).astype(np.float32)

    # Adds the Gaussian Noise to the image (converting to float32 for better precision)
    noiseImage = cv2.add(image.astype(np.float32), gaussianNoise)

    # Clip the values to be in the valid range [0, 255] (convert to uint8 so to be saved as .jpg)
    noiseImage = np.clip(noiseImage, 0, 255).astype(np.uint8)

    # Image with noise is saved to output path
    cv2.imwrite(output, noiseImage)


if __name__ == "__main__":
    """Main takes an input and output path (for input and output image)
       White Gaussian Noise is applied to the image"""
    
    #Input Image Path (replace with desired input path)
    inputImage = r"inputpath.jpg"  
    #Output Image Path (replace with desired output path)
    outputImage = r"outputpath.jpg"  

    #Average Value of Noise Signal
    mean = 0
    #Standard Deviation of Noise Signal (increasing deviation causes increasing noise)
    stddev = 40

    whiteGaussianNoise(inputImage, outputImage, mean, stddev)