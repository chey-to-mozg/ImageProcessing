
import rawpy
from tensorflow.keras.models import load_model
import numpy as np
import os
import subprocess
from tensorflow.keras import backend as K


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_dir = "D:/Study/models/unet"

#%%

def process_dng(dng_name):
    metadata = subprocess.check_output(
        ["exiftool", dng_name, "-Orientation", "-BitsPerSample", "-DefaultCropOrigin", "-DefaultCropSize"])

    metadata = str(metadata).split('\\r\\n')
    metadata = metadata[0:(len(metadata) - 1)]
    data = []
    for row in metadata:
        data.append(row.split(': ')[1])
    Rotate = data[0]
    bits = int(data[1])
    x_crop = int(data[2].split()[0].split('.')[0])
    y_crop = int(data[2].split()[1].split('.')[0])
    W_im = int(data[3].split()[0])
    H_im = int(data[3].split()[1])

    H_im = int(H_im / 8) * 8
    W_im = int(W_im / 8) * 8

    raw = rawpy.imread(dng_name)
    dng = raw.raw_image_visible


    #rotating

    if Rotate == "Rotate 90 CW":
        dng = np.rot90(dng, 3)
        x_crop, y_crop = y_crop, x_crop
        W_im, H_im = H_im, W_im
    elif Rotate == "Rotate 270 CW":
        dng = np.rot90(dng)
        W_im, H_im = H_im, W_im
        x_crop, y_crop = y_crop, x_crop
    elif Rotate == "Rotate 180":
        dng = np.rot90(dng, 2)

    # cropping

    image = dng[y_crop:y_crop + H_im, x_crop:x_crop + W_im]

    image = image / np.max(image)

    return image

#%%

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303

#%%

x = []
image = process_dng("Z:/Xlam/Dataset/fivek_dataset/dng/HQa4201to5000/a4999-DSC_0035.dng")
print(f'{image.shape[0]}x{image.shape[1]}')
x.append(image)
x = np.array(x)

#%%

model = load_model(model_dir, custom_objects={'PSNR': PSNR})

#%%

y = model.predict(x)

#%%

#%%

print(len(y[0]))
print(len(y[0,0]))
print(np.max(y))

#%%

y = np.clip(y * 256, 0, 255)


#%%

import cv2
image = y[0]

cv2.imwrite('D:/Study/dataset/tests/a4999-DSC_0035_rgb_20.jpg', image)


