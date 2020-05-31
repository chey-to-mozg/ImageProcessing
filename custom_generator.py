import numpy as np
import tensorflow.keras as keras
import rawpy
import cv2
import subprocess
from skimage.metrics import structural_similarity as ssim

path_to_dng = "Z:/Xlam/Dataset/fivek_dataset/dng/"
path_to_tif = "Z:/Xlam/Dataset/fivek_dataset/tif/"

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, n_channels=1, window_size=512,
                 n_classes=3, shuffle=True, debug=False):
        'Initialization'
        self.batch_size = batch_size
        self.window_size=window_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.debug = debug

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*int(self.batch_size*4):(index+1)*int(self.batch_size*4)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def process_dng(self, dng_name):
        metadata = subprocess.check_output(
            ["exiftool", dng_name, "-Orientation", "-BitsPerSample", "-DefaultCropOrigin",
             "-DefaultCropSize"])

        metadata = str(metadata).split('\\r\\n')
        metadata = metadata[0:(len(metadata) - 1)]
        data = []
        for row in metadata:
            data.append(row.split(': ')[1])
        Rotate = data[0]
        x_crop = int(data[2].split()[0].split('.')[0])
        y_crop = int(data[2].split()[1].split('.')[0])
        W_im = int(data[3].split()[0])
        H_im = int(data[3].split()[1])


        raw = rawpy.imread(dng_name)
        dng = raw.raw_image_visible


        #rotating

        if Rotate == "Rotate 90 CW":
            dng = np.rot90(dng, 3)
            W_im, H_im = H_im, W_im
        elif Rotate == "Rotate 270 CW":
            dng = np.rot90(dng)
            W_im, H_im = H_im, W_im
        elif Rotate == "Rotate 180":
            dng = np.rot90(dng, 2)

        # cropping

        image = dng[y_crop:y_crop + H_im, x_crop:x_crop + W_im]

        if image.shape[0] % 2 == 1:
            image = image[:image.shape[0] - 1, :]
        if image.shape[1] % 2 == 1:
            image = image[:, :image.shape[1] - 1]

        image = image / np.max(image)

        return image

    def compare(self, dng, tif):
        H = len(dng)
        W = len(dng[0])
        r = np.empty([H, W])
        g = np.empty([H, W])
        b = np.empty([H, W])

        r[ 0::2 , 0::2 ] = dng[ 0::2 , 0::2 ]
        r[0::2, 1::2] = dng[0::2, 0::2]
        r[1::2, 0::2] = dng[0::2, 0::2]
        r[1::2, 1::2] = dng[0::2, 0::2]
        g[ 0::2 , 0::2 ] = np.clip(dng[ 0::2 , 1::2 ] / 2 + dng[ 1::2 , 0::2 ] / 2 , 0 , 1)
        g[0::2, 1::2] = np.clip(dng[0::2, 1::2] / 2 + dng[1::2, 0::2] / 2, 0, 1)
        g[1::2, 0::2] = np.clip(dng[0::2, 1::2] / 2 + dng[1::2, 0::2] / 2, 0, 1)
        g[1::2, 1::2] = np.clip(dng[0::2, 1::2] / 2 + dng[1::2, 0::2] / 2, 0, 1)
        b[ 0::2 , 0::2 ] = dng[ 1::2 , 1::2 ]
        b[0::2, 1::2] = dng[1::2, 1::2]
        b[1::2, 0::2] = dng[1::2, 1::2]
        b[1::2, 1::2] = dng[1::2, 1::2]


        im1 = np.dstack([b, g, r])

        score = ssim(im1, tif, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
        print(score)

    def convert_image_to_array(self, ID):
        imagename = path_to_dng + ID + ".dng"

        dng = self.process_dng(imagename)

        imagename = path_to_tif + ID + ".tif"

        tif = cv2.imread(imagename) #tif = cv2.resize(cv2.imread(imagename), (dng.shape[1], dng.shape[0]))

        tif = tif[:dng.shape[0], :dng.shape[1], :] / 255

        #self.compare(dng, tif)

        return dng, tif

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.window_size, self.window_size), dtype=float)
        Y = np.empty((self.batch_size, self.window_size, self.window_size, self.n_classes), dtype=float)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            try:
                dng, tif = self.convert_image_to_array(ID)
                if self.debug:
                    print(ID)
                    print(dng.shape)
                    print(tif.shape)

                min_H = min(len(dng), len(tif))
                min_W = min(len(dng[0]), len(tif[0]))

                #1 batch = 1 crop
                rand_H = np.random.randint(len(dng) - self.window_size - 10)
                rand_W = np.random.randint(len(dng[0]) - self.window_size - 10)

                # Store sample
                #random crops from image
                X[i,] = (dng[(rand_H):(rand_H + self.window_size), (rand_W):(rand_W + self.window_size)])
                Y[i,] = (tif[(rand_H):(rand_H + self.window_size), (rand_W):(rand_W + self.window_size), :])

                #self.compare(X[i], Y[i])

                if i == (self.batch_size - 1):
                    break
            except Exception:
                i -= 1
                continue

        if self.debug:
            print(X.shape)
        X = np.expand_dims(X, -1)
        if self.debug:
            print(X.shape)
        #Y = np.asarray(Y)
        return X, Y

