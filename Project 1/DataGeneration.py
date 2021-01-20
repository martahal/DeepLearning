import numpy as np
import matplotlib.pyplot as plt


class DataGeneration:
    def __init__(self, noiseParam=0, imgSize=10, setSize=100, flatten=False, trainValTest=(0.7, 0.2, 0.1)):
        self.noiseParam = noiseParam
        self.imgSize = imgSize
        self.setSize = setSize
        self.flatten = flatten
        self.trainValTest = trainValTest
        self.trainSet = []
        self.valSet = []
        self.testSet = []

    def gen_dataset(self):
        """Generates a full dataset and splits it into training, validation and test set according to user specification
        the proportion of each class in each of the sets is random, but should be more or less equal over time"""
        full_set = []
        for i in range(self.setSize):
            # the full set is portioned with roughly 1/4 of each image category
            if i > self.setSize * 0.75:
                full_set.append(self.gen_image(self.imgSize, 'blob', self.flatten))
            elif i > self.setSize * 0.5:
                full_set.append(self.gen_image(self.imgSize, 'bars', self.flatten))
            elif i > self.setSize * 0.25:
                full_set.append(self.gen_image(self.imgSize, 'rect', self.flatten))
            else:
                full_set.append(self.gen_image(self.imgSize, 'cross', self.flatten))
        np.random.shuffle(full_set)

        if (sum(self.trainValTest)- 0.01)**2 < 1 or (sum(self.trainValTest)- 0.01)**2  == 1:
            # Dividing the shuffled full set into training set, validation set and test set
            train_proportion = int(round(self.trainValTest[0] * len(full_set)))
            val_proportion = int(round(self.trainValTest[1] * len(full_set)))
            test_proportion = int(round(self.trainValTest[2] * len(full_set)))
            self.trainSet = full_set[:train_proportion]
            self.valSet = full_set[:train_proportion + val_proportion]
            self.testSet = full_set[:train_proportion + val_proportion + test_proportion]
        else:
            print("trainValTest values must sum to exactly 1")

        draw_selection = self.testSet[:20]  # Drawing a selection from the test set
        #TODO: maybe just draw figures when we want it?
        for image in draw_selection:
            self.draw_image(image)

    @staticmethod
    def gen_image(self, n, shape, flatten):
        """Create one image with one of the four desired shapes.
        Positions and sizes of shapes are generated randomly for each image.
        Image is flattened if flattened = True
        Returns a tuple: (label, image)"""
        # TODO Implement noise
        image = np.zeros((n, n))
        if shape == 'cross':
            # length of cross arms are random odd length
            l = (np.random.randint(2,np.floor(n/2))*2) - 1
            start_pos = [np.random.randint(0, n - l), np.random.randint(0, n - l)]
            image[start_pos[0] + int(np.floor(l/2))][start_pos[1] : start_pos[1] + l] = 1
            for i in range(start_pos[0], start_pos[0] + l):
                image[i][start_pos[1] + int(np.floor(l/2))] = 1
            if flatten:
                flat_image = image.flatten()
                image = flat_image
        if shape == 'rect':
            l = np.random.randint(2,n)
            w = np.random.randint(2,n)
            start_pos = [np.random.randint(0, n - l), np.random.randint(0, n - w)]
            image[start_pos[0]][start_pos[1]:start_pos[1] + w] = 1  # top side
            image[start_pos[0] + l][start_pos[1]:start_pos[1] + w] = 1  # bottom side
            for i in range(start_pos[0], start_pos[0] + l + 1):
                image[i][start_pos[1]]= 1  # left side
                image[i][start_pos[1] + w] = 1 # right side
            if flatten:
                flat_image = image.flatten()
                image = flat_image
        if shape == 'bars':
            # Draw bars at random columns
            cols = np.zeros(n)
            cols[:np.random.randint(2, round(n/3))] = 1
            np.random.shuffle(cols)
            for i in range(n):
                if cols[i] == 1:
                    image[i] = 1
            if flatten:
                flat_image = image.flatten()
                image = flat_image
        elif shape == 'blob':
            r = np.random.randint(3, n/2)  # To avoid same shape as cross, min r = 2
            center = [np.random.randint(r, n - r), np.random.randint(r, n - r)]
            for i in range(n):
                for j in range(n):
                    if np.ceil(np.sqrt((center[0] - i)**2 + (center[1] - j)**2)) < r :
                        image[i][j] = 1
            if flatten:
                flat_image = image.flatten()
                image = flat_image
        return shape, image

    @staticmethod
    def draw_image(data):
        # TODO How to draw if array is flattened?
        plt.title(data[0])
        plt.imshow(data[1], cmap='gray')
        plt.show()


if __name__ == '__main__':
    data1 = DataGeneration(imgSize=10, setSize= 100)
    data1.gen_dataset()
    #print(gen_array(10,'cross'))
    #draw_image(gen_array(10,'blob'))