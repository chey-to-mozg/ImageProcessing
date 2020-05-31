
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import math
import rawpy
import cv2
import os
from sklearn.model_selection import train_test_split
from custom_generator import DataGenerator


#%%

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

params = {'batch_size': 2,
          'n_classes': 3,
          'n_channels': 1,
          'window_size' : 256,
          'shuffle': True,
          'debug': False}

EPOCHS = 10 #->300

dng_dir = "Z:/Xlam/Dataset/fivek_dataset/dng"
tif_dir = "Z:/Xlam/Dataset/fivek_dataset/tif"

model_dir = "D:/Study/models/unet"
checkpoint_dir = "D:/Study/models/unet/checkpoint"

#%%

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303

#%%

def unet(pretrained_model = None, input_size = (256,256, 1)):

    if os.path.exists(pretrained_model):
        model = load_model(pretrained_model, custom_objects={'PSNR': PSNR})

    else:
        inputs = Input(input_size)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)

        # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        # conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        # drop4 = Dropout(0.5)(conv4)
        # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        #
        # conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        # conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        # drop5 = Dropout(0.5)(conv5)
        #
        # up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        # merge6 = concatenate([drop4,up6], axis = 3)
        # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        # up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        # merge7 = concatenate([conv3,up7], axis = 3)
        # conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        # conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv3))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

        conv10 = Conv2D(3, 1)(conv9)

        model = Model(inputs, conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_squared_error', metrics = [PSNR])

    model.summary()

    return model

#%%

def create_dataset_IDs(im_count=7000):
    if im_count > 5000:
        im_count = 5000
    imgs_from_dir = int(im_count / 7)

    print(f"reading {imgs_from_dir} images from each directory")

    dirs_dng = os.listdir(dng_dir)
    x_data = []

    for dir in dirs_dng:
        dir_path_dng = os.path.join(dng_dir, dir)
        files_dng = os.listdir(dir_path_dng)

        count = 0
        for image in files_dng[:imgs_from_dir]:
            imagename = os.path.splitext(image)[0]
            x_data.append(dir + "/" + imagename)
            print(f"{count * 100 / imgs_from_dir}%", end="\r")
            count += 1
        print(f"dng from {dir} finished")

    print("Dataset created")
    return x_data

#%%

print("reading started")
x_data = create_dataset_IDs(980)

#%%

net = unet(model_dir, input_size=(None, None, 1))

#%%

training_generator = DataGenerator(x_data, **params)

model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_dir, save_best_only=True)


#%%

history = net.fit(training_generator, epochs=EPOCHS, callbacks=[model_checkpoint_callback])

#%%

net.save(model_dir)

#%%
