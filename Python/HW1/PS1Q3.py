import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io

class Prob3():
    def __init__(self):
        """Load input color image inputPS1Q3.jpg here and assign it as a class variable"""
        ###### START CODE HERE ######
        self.img = plt.imread('inputPS1Q3.jpg')
        self.imgarray = np.asarray(self.img)

        ###### END CODE HERE ######
        pass
    
    def rgb2gray(self, rgb):
        """
        Do RGB to Gray image conversion here. Input is the RGB image and you must return the grayscale image as gray

        Returns:
            gray: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        gray = np.uint8(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]))

        ###### END CODE HERE ######
        pass
    
        return gray ######
        
    def prob_3_1(self):
        """
        Swap red and green color channels here, and return swapImg

        Returns:
            swapImg: RGB image with R and G channels swapped (3 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        img2 = self.img
        r = self.imgarray[:, :, 0].copy()
        g = self.imgarray[:, :, 1].copy()
        swapImg = self.imgarray.copy()
        swapImg[:, :, 0] = g
        swapImg[:, :, 1] = r

        fig = plt.figure(figsize = (10, 10))
        plt.subplot(2,3,1)
        plt.imshow(swapImg)
        plt.axis('off')
        plt.title('swapImg')
        # plt.show()

        ###### END CODE HERE ######
        pass
    
        return swapImg ######
    
    def prob_3_2(self):
        """
        This function would simply call your rgb2gray function and return the grayscale image.

        Returns:
            grayImg: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        grayImg = self.rgb2gray(self.img.copy())

        plt.subplot(2,3,2)
        plt.imshow(grayImg,cmap='gray')
        plt.axis('off')
        plt.title('grayImg')

        ###### END CODE HERE ######
        pass
    
        return grayImg ######
    
    def prob_3_3(self):
        """
        Convert grayscale image to its negative.

        Returns:
            negativeImg: negative image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        grayImg = self.prob_3_2()
        negativeImg = 255 - grayImg

        plt.subplot(2,3,3)
        plt.imshow(negativeImg,cmap='gray')
        plt.axis('off')
        plt.title('negativeImg')

        ###### END CODE HERE ######
        pass
    
        return negativeImg ######
    
    def prob_3_4(self):
        """
        Create mirror image of gray scale image here.
        
        Returns:
            mirrorImg: mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        grayImg = self.prob_3_2()
        mirrorImg = grayImg[:, ::-1]

        plt.subplot(2,3,4)
        plt.imshow(mirrorImg,cmap='gray')
        plt.axis('off')
        plt.title('mirrorImg')
        ###### END CODE HERE ######
        pass
    
        return mirrorImg ######
    
    def prob_3_5(self):
        """
        Average grayscale image with mirror image here.
        
        Returns:
            avgImg: average of grayscale and mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        mirrorImg = self.prob_3_4() / 255.0
        grayImg = self.prob_3_2() / 255.0
        avgImg = ((mirrorImg + grayImg) / 2)*255.0

        plt.subplot(2,3,5)
        plt.imshow(avgImg,cmap='gray')
        plt.axis('off')
        plt.title('avgImg')
        ###### END CODE HERE ######
        pass
    
        return avgImg ######
    
    def prob_3_6(self):
        """
        Create noise matrix N and save as noise.npy. Add N to grayscale image, clip to ensure that max value is 255.
        
        Returns:
            addNoiseImg: grayscale image after adding noise (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        ## build noise.npy
        # N = np.random.normal(0, 0.1**0.5, self.size)
        # N = np.clip(N, a_min = 0, a_max = 1) * 255
        # N  = N.astype('uint8')
        # np.save("noise.npy", N)
        ## implement with noise.npy
        grayImg = self.prob_3_2()/255.0
        N = np.load("noise.npy")/255.0
        addNoiseImg = np.uint32( (grayImg+ N) *255.0)
        addNoiseImg = np.clip(addNoiseImg, a_min = 0, a_max = 255)

        plt.subplot(2,3,6)
        plt.imshow(addNoiseImg,cmap='gray')
        plt.axis('off')
        plt.title('addNoiseImg')
        plt.show()
        
        ###### END CODE HERE ######
        pass
    
        return addNoiseImg ######
        
        
if __name__ == '__main__':
    
    p3 = Prob3()
    
    swapImg = p3.prob_3_1()
    grayImg = p3.prob_3_2()
    negativeImg = p3.prob_3_3()
    mirrorImg = p3.prob_3_4()
    avgImg = p3.prob_3_5()
    addNoiseImg = p3.prob_3_6()