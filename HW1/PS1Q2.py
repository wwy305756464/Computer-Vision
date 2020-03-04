import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mi

class Prob2():
    def __init__(self):
        """Load inputAPS1Q2.npy here as a class variable A."""
        ###### START CODE HERE ######
        A = np.load('inputAPS1Q2.npy')
        ###### END CODE HERE ######
        pass
    
    A = np.load('inputAPS1Q2.npy')

    def prob_2_1(self):
        """Do plotting of intensities of A in decreasing value."""
        ###### START CODE HERE ######
        size = self.A.shape
        self.A = self.A.reshape(1, size[0]*size[1])
        self.A.sort()
        B = np.flipud(self.A[0])
        self.A = [B]
        image = plt.imshow(self.A, aspect=800, cmap=plt.get_cmap('gray'))
        plt.yticks([])
        plt.xlim(0,10000)
        plt.show()
        plt.close()
        ###### END CODE HERE ######
        pass
    
    def prob_2_2(self):
        """Display histogram of A's intensities with 20 bins here."""
        ###### START CODE HERE ######
        B = [i for item in self.A for i in item]
        image2 = plt.hist(B, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
        plt.title("Histogram for given data with 20 bins")
        plt.show()
        plt.close()
        ###### END CODE HERE ######
        pass
    
    def prob_2_3(self):
        """
        Create a new matrix X that consists of the bottom left quadrant of A here.
        Returns:
            X: bottom left quadrant of A which is of size 50 x 50
        """
        ###### START CODE HERE ######
        A = np.load('inputAPS1Q2.npy')
        X = A[50:100, 0:50]
        ###### END CODE HERE ######
        pass 
    
        return X 
    
    def prob_2_4(self):
        """Create a new matrix Y, which is the same as A, but with A’s mean intensity value subtracted from each pixel.
        Returns:
            Y: A with A's mean intensity subtracted from each pixel. Output Y is of size 100 x 100.
        """
        ###### START CODE HERE ######
        A = np.load('inputAPS1Q2.npy')
        Y = A - np.mean(A)
        ###### END CODE HERE ######
        pass
    
        return Y 
    
    def prob_2_5(self):
        """
        Create your threshholded A i.e Z here.
        Returns:
            Z: A with only red pixels when the original value of the pixel is above the threshhold. Output Z is of size 100 x 100.
        """
        ###### START CODE HERE ######
        A = self.prob_2_4()
        Z = np.zeros((100, 100, 3), dtype = 'uint8')
        a,b = np.where(A > np.mean(A))
        Z [a,b, 0] = 1
        Zplot = Z * 255

        plt.imshow(Zplot)
        plt.savefig('outputZPS1Q2.png')
        plt.show()
        ###### END CODE HERE ######
        pass
    
        return Z 
        
        
        
if __name__ == '__main__':
    
    p2 = Prob2()
    
    p2.prob_2_1()
    p2.prob_2_2()
    
    X = p2.prob_2_3()
    Y = p2.prob_2_4()
    Z = p2.prob_2_5()