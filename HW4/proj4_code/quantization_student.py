from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import skimage.color as color
from PIL import Image
import numpy as np

def computeQuantizationError(orig_img, quantized_img):
    err = np.inf
    ######################################################################################
    ##                                                                                  ##
    ## TODO: We will be calculating the quantization error by finding the sum of        ##
    ## squared difference between the original and quantized images. Implement a        ##
    ## vectorized version of this error metric.                                         ##
    ##                                                                                  ##
    ######################################################################################                        

    raise NotImplementedError   #remove this line after you implement this function
    ######################################################################################
    return err



def quantizeRGB(origImage, k):
    random_state = 7
    
    ######################################################################################
    ##                                                                                  ##
    ## TODO: Quantize the RGB image along all 3 channels and assign the values of the   ## 
    ## nearest cluster center to each pixel. Return the quantized image and cluster     ##
    ## centers. Use the random_state variable to defined above. Otherwise your answers  ##
    ## may not match the expected output.                                               ##
    ##                                                                                  ##
    ######################################################################################

    raise NotImplementedError   #remove this line after you implement this function
    return None # modify this to return the required outputs.



def quantizeHSV(origImage, k):
    random_state = 7

    ######################################################################################
    ##                                                                                  ##
    ## TODO: Convert the image to HSV and quantize the Hue channel. assign the nearest  ## 
    ## cluster center to each pixel. Return the quantized image and cluster centers.    ##
    ## Use the random_state variable to defined above. Otherwise your answers may not   ##
    ## match the expected output. Remember to convert the HSV image back to RGB.        ##
    ##                                                                                  ##
    ######################################################################################
    
    raise NotImplementedError   #remove this line after you implement this function
    return None # modify this to return the required outputs.