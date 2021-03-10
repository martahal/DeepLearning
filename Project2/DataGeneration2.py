import numpy as np
import matplotlib.pyplot as plt


class DataGeneration2:
    def __init__(self, noise=0.0, img_size=10, set_size=100, train_val_test=(0.7, 0.2, 0.1),
                 fig_centered=False, draw=False, flatten=False):
        self.noise = noise
        self.img_size = img_size
        self.set_size = set_size
        self.flatten = flatten
        self.train_val_test = train_val_test
        self.fig_centered = fig_centered
        self.draw = draw
        self.train_set = []
        self.val_set = []
        self.test_set = []

    def gen_dataset(self):
        """Generates a full dataset and splits it into training, validation and test set according to user specification
        the proportion of each class in each of the sets is random, but should be more or less equal over time
        Each entry in the data set is a dictionary with keys 'class', 'one_hot' 'image' and 'flat'image'"""
        full_set = []
        for i in range(self.set_size):
            # the full set is portioned with roughly 1/4 of each image category
            if i > self.set_size * 0.75:
                full_set.append(self._gen_image(self.img_size, 'blob', self.noise, self.fig_centered))
            elif i > self.set_size * 0.5:
                full_set.append(self._gen_image(self.img_size, 'bars', self.noise, self.fig_centered))
            elif i > self.set_size * 0.25:
                full_set.append(self._gen_image(self.img_size, 'rect', self.noise, self.fig_centered))
            else:
                full_set.append(self._gen_image(self.img_size, 'cross', self.noise, self.fig_centered))
        np.random.shuffle(full_set)

        if (sum(self.train_val_test) - 0.01)**2 < 1 or (sum(self.train_val_test) - 0.01)**2  == 1:
            # Dividing the shuffled full set into training set, validation set and test set
            train_proportion = round(self.train_val_test[0] * len(full_set))
            val_proportion = round(self.train_val_test[1] * len(full_set))
            test_proportion = round(self.train_val_test[2] * len(full_set))
            self.train_set = full_set[:train_proportion]
            self.val_set = full_set[train_proportion:train_proportion + val_proportion]
            self.test_set = full_set[train_proportion + val_proportion:train_proportion + val_proportion + test_proportion]
        else:
            print("trainValTest values must sum to exactly 1")

        draw_selection = self.test_set[:20]  # Drawing a selection from the test set
        if self.draw:
            for image in draw_selection:
                self.draw_image(image)

    def _gen_image(self, n, shape, noise, centered):
        """Create one image with one of the four desired shapes.
        Positions and sizes of shapes are generated randomly for each image.'
        Noise is implemented by setting a user-defined portion of array entries to the opposite value
        at the end of the image generation.
        Returns a dictionary: (label, one-hot-label, image, flattened image)"""
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
        one_hot_class = self._one_hot_encode(shape)
        multi_channel_image = np.array([image]) #making each image multi-channel

        return {'class': shape, 'one_hot': one_hot_class, 'image': multi_channel_image, 'flat_image': flat_image}

    def _one_hot_encode(self, label):
        """Converts class label to a one-hot-encoded vector\n
        One-hot-vector: ['cross', 'rect', 'bars', 'blob']"""
        if label == 'cross':
            return np.array([1, 0, 0, 0])
        elif label == 'rect':
            return np.array([0, 1, 0, 0])
        elif label == 'bars':
            return np.array([0, 0, 1, 0])
        elif label == 'blob':
            return np.array([0, 0, 0, 1])


    @staticmethod
    def draw_image(data):
        plt.title(data['class'])
        plt.imshow(data['image'], cmap='gray')
        plt.show()




if __name__ == '__main__':
    data1 = DataGeneration2(noise=0.0, img_size=10, set_size= 100, fig_centered=True, draw=True)
    data1.gen_dataset()
    #print(gen_array(10,'cross'))
    #draw_image(gen_array(10,'blob'))