import numpy as np
import matplotlib.pyplot as plt


class DataGeneration:
    def __init__(self, noiseParam=0.0, imgSize=10, setSize=100, flatten=False, trainValTest=(0.7, 0.2, 0.1),
                 figCentered=False, draw=False):
        self.noiseParam = noiseParam
        self.imgSize = imgSize
        self.setSize = setSize
        self.flatten = flatten
        self.trainValTest = trainValTest
        self.figCentered = figCentered
        self.draw = draw
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
                full_set.append(self.gen_image(self.imgSize, 'blob', self.noiseParam, self.figCentered))
            elif i > self.setSize * 0.5:
                full_set.append(self.gen_image(self.imgSize, 'bars', self.noiseParam, self.figCentered))
            elif i > self.setSize * 0.25:
                full_set.append(self.gen_image(self.imgSize, 'rect', self.noiseParam, self.figCentered))
            else:
                full_set.append(self.gen_image(self.imgSize, 'cross', self.noiseParam, self.figCentered))
        np.random.shuffle(full_set)

        if (sum(self.trainValTest)- 0.01)**2 < 1 or (sum(self.trainValTest)- 0.01)**2  == 1:
            # Dividing the shuffled full set into training set, validation set and test set
            train_proportion = round(self.trainValTest[0] * len(full_set))
            val_proportion = round(self.trainValTest[1] * len(full_set))
            test_proportion = round(self.trainValTest[2] * len(full_set))
            self.trainSet = full_set[:train_proportion]
            self.valSet = full_set[:train_proportion + val_proportion]
            self.testSet = full_set[:train_proportion + val_proportion + test_proportion]
        else:
            print("trainValTest values must sum to exactly 1")

        draw_selection = self.testSet[:20]  # Drawing a selection from the test set
        #TODO: maybe just draw figures when we want it?
        if self.draw:
            for image in draw_selection:
                self.draw_image(image)

     #@staticmethod
    def gen_image(self, n, shape, noise, centered):
        """Create one image with one of the four desired shapes.
        Positions and sizes of shapes are generated randomly for each image.'
        Noise is implemented by setting a user-defined portion of array entries to the opposite value
        at the end of the image generation.
        Returns a tuple: (label, image, flattened image)"""
        image = np.zeros((n, n))

        if shape == 'cross':
            # length of cross arms are random odd length
            l = (np.random.randint(2,np.floor(n/2))*2) - 1
            if centered:
                start_pos = [int(np.floor(n / 2) - np.floor(l / 2)), int(np.floor(n / 2) - np.floor(l / 2))]
            else:
                start_pos = [np.random.randint(0, n - l), np.random.randint(0, n - l)]
            image[start_pos[0] + int(np.floor(l/2))][start_pos[1] : start_pos[1] + l] = 1
            for i in range(start_pos[0], start_pos[0] + l):
                image[i][start_pos[1] + int(np.floor(l/2))] = 1


        if shape == 'rect':
            l = np.random.randint(2, n)
            w = np.random.randint(2, n)
            if centered:
                start_pos = [int(np.floor(n/2) - np.floor(l/2) - 1), int(np.floor(n/2) - np.floor(w/2) - 1)]  # -1 to prevent index error
            else:
                start_pos = [np.random.randint(0, n - l), np.random.randint(0, n - w)]
            image[start_pos[0]][start_pos[1]:start_pos[1] + w] = 1  # top side
            image[start_pos[0] + l][start_pos[1]:start_pos[1] + w] = 1  # bottom side
            for i in range(start_pos[0], start_pos[0] + l + 1):
                image[i][start_pos[1]]= 1  # left side
                image[i][start_pos[1] + w] = 1 # right side
            flat_image = image.flatten()

        if shape == 'bars':
            # Draw bars at random columns
            #TODO:How to center bars?
            cols = np.zeros(n)
            cols[:np.random.randint(2, round(n/3))] = 1
            np.random.shuffle(cols)
            for i in range(n):
                if cols[i] == 1:
                    image[i] = 1

        elif shape == 'blob':
            r = np.random.randint(3, n/2)  # To avoid same shape as cross for small figures, min r = 3
            if centered:
                center = [np.floor(n/2), np.floor(n/2)]
            else:
                center = [np.random.randint(r, n - r), np.random.randint(r, n - r)]
            for i in range(n):
                for j in range(n):
                    if np.ceil(np.sqrt((center[0] - i)**2 + (center[1] - j)**2)) < r :
                        image[i][j] = 1


        if noise > 1 or noise < 0:
            print("Noise parameter must be between 0 and 1")
        else:
            #Select a portion of the array to flip
            flipped_pixels = round((len(image) * n) * noise)
            for i in range(flipped_pixels):
                rdm_pixel = (np.random.randint(0, len(image)), np.random.randint(0, n))
                if image[rdm_pixel[0], rdm_pixel[1]] == 0:
                    image[rdm_pixel[0], rdm_pixel[1]] = 1
                else:
                    image[rdm_pixel[0], rdm_pixel[1]] = 0

        flat_image = image.flatten()

        return shape, image, flat_image

    @staticmethod
    def draw_image(data):
        plt.title(data[0])
        plt.imshow(data[1], cmap='gray')
        plt.show()


if __name__ == '__main__':
    data1 = DataGeneration(noiseParam=0.0, imgSize=50, setSize= 100, figCentered=True)
    data1.gen_dataset()
    #print(gen_array(10,'cross'))
    #draw_image(gen_array(10,'blob'))