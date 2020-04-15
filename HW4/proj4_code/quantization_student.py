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
    # err = np.sum(np.power((orig_img - quantized_img), 2 )) / (orig_img.shape[0] * orig_img[1] * orig_img[2])
    err = np.sum((orig_img[:,:,0:3] - quantized_img[:,:,0:3])**2)

    # raise NotImplementedError   #remove this line after you implement this function
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
    image = np.array(origImage, dtype=np.float64)
    w, h, d = image.shape
    image2 = np.reshape(image, (w*h, d))
    
    clus = KMeans(n_clusters = k)
    clus.fit(image2)

    meancolor = clus.cluster_centers_
    center = clus.predict(image2)

    imageclus = meancolor[center]
    imageout = np.reshape(imageclus, (w, h, 3))

    # raise NotImplementedError   #remove this line after you implement this function
    # return None # modify this to return the required outputs.
    return imageout, meancolor


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
    
    # raise NotImplementedError   #remove this line after you implement this function
    # return None # modify this to return the required outputs
    image = np.array(origImage, dtype=np.float64)
    w, h, d = image.shape

    hsv = color.rgb2hsv(origImage)
    hsv2 = np.copy(hsv)
    temp = np.reshape(hsv2[:,:,0],(-1,1))
    clus = KMeans(n_clusters=k)
    clus.fit(temp)
    label = clus.labels_
    meanH = clus.cluster_centers_
    for i in range(w * h):
        temp[i] = meanH[label[i]]
    hsv2[:,:,0] = np.reshape(temp, (w, h))
    imageout = color.hsv2rgb(hsv2)
    imageout = (255*imageout/np.amax(imageout)).astype(np.uint8)
    return imageout, meanH