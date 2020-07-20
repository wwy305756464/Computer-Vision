import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
import skimage

class Prob4():
    def __init__(self):
        """Load input color image indoor.png and outdoor.png here as class variables."""
        ###### START CODE HERE ######
        self.indoor = cv2.imread('indoor.png',1)
        self.indoorary = np.asarray(self.indoor)
        self.outdoor = cv2.imread('outdoor.png',1)
        self.outdoorary = np.asarray(self.outdoor)
        ###### END CODE HERE ######
        pass
    
    def prob_4_1(self):
        """Plot R,G,B channels separately and also their corresponding LAB space channels separately"""
        
        ###### START CODE HERE ######
        # change from BGR to RGB
        indoor = self.indoorary.copy()
        outdoor = self.outdoorary.copy()
        indoorred = indoor[:,:,2].copy()
        indoorblue = indoor[:,:,0].copy()
        indoor[:,:,2] = indoorblue
        indoor[:,:,0] = indoorred
        outdoorred = outdoor[:,:,2].copy()
        outdoorblue = outdoor[:,:,0].copy()
        outdoor[:,:,2] = outdoorblue
        outdoor[:,:,0] = outdoorred

        fig = plt.figure(figsize = (10, 10))
        plt.subplot(231)
        plt.imshow(indoor[:,:,0],cmap='gray')
        plt.axis('off')
        plt.title('indoor Red')
        plt.subplot(232)
        plt.imshow(indoor[:,:,1],cmap='gray')
        plt.axis('off')
        plt.title('indoor Green')
        plt.subplot(233)
        plt.imshow(indoor[:,:,2],cmap='gray')
        plt.axis('off')
        plt.title('indoor Blue')

        plt.subplot(234)
        plt.imshow(outdoor[:,:,0],cmap='gray')
        plt.axis('off')
        plt.title('outdoor Red')
        plt.subplot(235)
        plt.imshow(outdoor[:,:,1],cmap='gray')
        plt.axis('off')
        plt.title('outdoor Green')
        plt.subplot(236)
        plt.imshow(outdoor[:,:,2],cmap='gray')
        plt.axis('off')
        plt.title('outdoor Blue')

        plt.show()
        plt.close()

        indoorlab = indoor
        outdoorlab = outdoor
        indoorlab = cv2.cvtColor(indoorlab,cv2.COLOR_RGB2LAB)
        outdoorlab = cv2.cvtColor(outdoorlab,cv2.COLOR_RGB2LAB)
        fig = plt.figure(figsize = (10, 10))
        plt.subplot(231)
        plt.imshow(indoorlab[:,:,0],cmap='gray')
        plt.axis('off')
        plt.title('indoor Red - LAB')
        plt.subplot(232)
        plt.imshow(indoorlab[:,:,1],cmap='gray')
        plt.axis('off')
        plt.title('indoor Green - LAB')
        plt.subplot(233)
        plt.imshow(indoorlab[:,:,2],cmap='gray')
        plt.axis('off')
        plt.title('indoor Blue - LAB')

        plt.subplot(234)
        plt.imshow(outdoorlab[:,:,0],cmap='gray')
        plt.axis('off')
        plt.title('outdoor Red - LAB')
        plt.subplot(235)
        plt.imshow(outdoorlab[:,:,1],cmap='gray')
        plt.axis('off')
        plt.title('outdoor Green - LAB')
        plt.subplot(236)
        plt.imshow(outdoorlab[:,:,2],cmap='gray')
        plt.axis('off')
        plt.title('outdoor Blue - LAB')

        plt.show()
        plt.close()

        ###### END CODE HERE ######
        pass

    def prob_4_3(self):
        """
        Convert the loaded RGB image to HSV and return HSV matrix without using inbuilt functions. Return the HSV image as HSV. Make sure to use a 3 channeled RGB image with floating point values lying between 0 - 1 for the conversion to HSV.

        Returns:
            HSV image (3 channeled image with floating point values lying between 0 - 1 in each channel)
        """
    
        ###### START CODE HERE ######
        img = io.imread('inputPS1Q4.jpg') 
        imgarr = np.asarray(img, dtype='int64')
        size = imgarr.shape
        height = size[0]
        width = size[1]
        HSV = imgarr.copy() / 255.0
        cout = 0
        for x in range(height):
            for y in range(width):
                R = imgarr[x,y,0]
                G = imgarr[x,y,1] 
                B = imgarr[x,y,2]

                Rb = HSV[x,y,0]
                Gb = HSV[x,y,1] 
                Bb = HSV[x,y,2]

                ## calculate V
                V = max(R,G,B)
                Vb = max(Rb,Gb,Bb)

                ## calculate S
                if (R == 0 & G == 0 & B == 0):
                    S = 0
                else:
                    m = min(R,G,B)
                    C = V - m
                    S = C/V #* 255.0

                ## calculate H
                if C == 0:
                    H = 0
                else:
                    if V == R:
                        Hp = (G-B)/C
                    elif V == G:
                        Hp = (B-R)/C + 2
                    elif V == B:
                        Hp = (R-G)/C + 4

                    if Hp < 0:
                        H = (Hp/6 + 1) #* 255.0
                    else:
                        H = Hp/6 #* 255.0

                HSV[x,y,0] = H 
                HSV[x,y,1] = S 
                HSV[x,y,2] = Vb 
                cout = cout + 1
        HSV2 = HSV * 255
        cv2.imwrite('outputPS1Q4.jpg', HSV2)

        ###### END CODE HERE ######
        pass
    
        return HSV ######
        

        
if __name__ == '__main__':
    
    p4 = Prob4()
    
    p4.prob_4_1()

    HSV = p4.prob_4_3()





