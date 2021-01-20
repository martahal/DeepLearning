import numpy as np
import matplotlib.pyplot as plt

class DataGeneration():
    def __init__(self, noiseParam=0, imgSize=10, setSize=100, flattening=False, trainValTest=(70,20,10)):
        self.noiseParam = noiseParam
        self.imgSize = imgSize
        self.setSize = setSize
        self.flattening = flattening
        self.trainValTest = trainValTest
        self.trainSet = {}
        self.valSet = {}
        self.testSet = {}

    def gen_array(self, n, shape):
        array = np.zeros((n,n))
        if shape == 'cross':
            #length of cross arms are random odd length
            l = (np.random.randint(2,np.floor(n/2))*2) - 1
            start_pos = [np.random.randint(0, n - l), np.random.randint(0, n - l)]
            array[start_pos[0] + int(np.floor(l/2))][start_pos[1] : start_pos[1] + l] = 1
            for i in range(start_pos[0], start_pos[0] + l):
                array[i][start_pos[1] + int(np.floor(l/2))] = 1

        if shape == 'rect':
            l = np.random.randint(2,n)
            w = np.random.randint(2,n)
            start_pos = [np.random.randint(0, n - l), np.random.randint(0, n - w)]
            array[start_pos[0]][start_pos[1]:start_pos[1] + w] = 1  # top side
            array[start_pos[0] + l][start_pos[1]:start_pos[1] + w] = 1  # bottom side
            for i in range(start_pos[0], start_pos[0] + l + 1):
                array[i][start_pos[1]]= 1  # left side
                array[i][start_pos[1] + w] = 1 #right side

        if shape == 'bars':
            start_col = np.random.randint(0, round(n/2))
            # Draw bars at random columns
            cols = np.zeros(n)
            cols[:np.random.randint(2, round(n/3))] = 1
            np.random.shuffle(cols)
            print(cols)
            for i in range(n):
                if cols[i] == 1:
                    array[i] = 1

        elif shape == 'blob':
            r = np.random.randint(3, n/2) # To avoid same shape as cross, min r = 2
            center = [np.random.randint(r, n - r), np.random.randint(r, n - r)]
            for i in range(n):
                for j in range(n):
                    if np.ceil(np.sqrt((center[0] - i)**2 + (center[1] - j)**2)) < r :
                        array[i][j] = 1
        return array

    def draw_image(self, array):
        plt.imshow(array, cmap='gray')
        plt.show()

if __name__ == '__main__':
    data1 = DataGeneration()
    #print(gen_array(10,'cross'))
    #draw_image(gen_array(10,'blob'))