import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import skimage.io
from skimage.filters import median
from PIL import Image
import torch
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim

images_collection = skimage.io.imread_collection(['/content/drive/MyDrive/Images/1min.jpg','/content/drive/MyDrive/Images/2min.jpg','/content/drive/MyDrive/Images/3min.jpg','/content/drive/MyDrive/Images/4min.jpg',
                                       '/content/drive/MyDrive/Images/5min.jpg','/content/drive/MyDrive/Images/6min.jpg','/content/drive/MyDrive/Images/11min.jpg'])

images_array_salt_pepper = np.array(images_collection)
original_images = np.array(images_collection)
print(original_images.shape)
np.max(original_images)

def add_noise(images_array_salt_pepper):
    row = 480
    col = 640

    # Randomly pick some pixels in the
    # image for coloring them white
    for i in range(7):
      for j in range(13000):

        # Pick a random y coordinate
        y_coord=np.random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord=np.random.randint(0, col - 1)

        # Color that pixel to white
        images_array_salt_pepper[i][y_coord][x_coord][0] = 255
        images_array_salt_pepper[i][y_coord][x_coord][1] = 255
        images_array_salt_pepper[i][y_coord][x_coord][2] = 255


    # Randomly pick some pixels in the
    # image for coloring them black
    for i in range(7):
      for j in range(13000):

        # Pick a random y coordinate
        y_coord=np.random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord=np.random.randint(0, col - 1)

        # Color that pixel to black
        images_array_salt_pepper[i][y_coord][x_coord][0] = 0
        images_array_salt_pepper[i][y_coord][x_coord][1] = 0
        images_array_salt_pepper[i][y_coord][x_coord][2] = 0

    return images_array_salt_pepper
noisy_images_salt_pepper = add_noise(images_array_salt_pepper)

def add_noise(noisy_images_salt_pepper):
      row = 480;
      col = 640;
      ch=3;
      mean = 5;
      var = 2000;
      sigma = var**0.5;
      # sigma = 20;
      noisy_img = np.zeros((7, 480, 640, 3));
      for i in range(7):
          gauss = np.random.normal(mean,sigma,(row,col,ch));
          gauss = gauss.reshape(row,col,ch);
          noisy = noisy_images_salt_pepper[i] + gauss;
          noisy_img[i] = noisy;
      return noisy_img;

noisy_images = add_noise(noisy_images_salt_pepper);

noisy_images = np.clip(noisy_images, 0, 255).astype(np.uint8)

plt.figure(figsize=(17,21))
x1 = plt.subplot(131)
x1.set_title("Original Image")
plt.imshow(original_images[0])
x2 = plt.subplot(132)
x2.set_title("Salt-pepper Noise")
plt.imshow(noisy_images_salt_pepper[0])
x3 = plt.subplot(133)
x3.set_title("Salt-pepper & Gaussian")
plt.imshow(noisy_images[0])
plt.show()

x_train,x_test,y_train,y_test = train_test_split(noisy_images,original_images,train_size = 0.88,random_state = 1)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = y_train.astype('float32') / 255
y_test = y_test.astype('float32') / 255

x_train = np.clip(x_train, 0., 1.)
x_test = np.clip(x_test, 0., 1.)
y_train = np.clip(y_train, 0., 1.)
y_test = np.clip(y_test, 0., 1.)


#Creating the model.
model = Sequential()
# Encoder
model.add(Conv2D(512, kernel_size=(3,3),activation='relu',input_shape=(480,640,3)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
# Decoder
model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))

# Adding output layer
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.summary()

# Training the model.
model.fit(x_train,
          y_train,
          epochs=100,
          batch_size=1,
          shuffle=True,
          validation_data=(x_test, y_test)
          )

# Prediction
no_noise_img = model.predict(noisy_images)

no_noise_imgs = np.clip(no_noise_img, 0, 255).astype(np.uint8)

plt.figure(figsize=(16,21))
x1 = plt.subplot(131)
x1.set_title("Original Image")
plt.imshow(original_images[5])
x2 = plt.subplot(132)
x2.set_title("Noisy Image")
plt.imshow(noisy_images[5])
x3 = plt.subplot(133)
x3.set_title("Denoised Image")
plt.imshow(no_noise_imgs[5])
plt.show()

img_true = original_images[5]
img_denoised = no_noise_imgs[5]

print(img_true.shape)
print(img_denoised.shape)

img_true = img_true.reshape(-1)
img_denoised = img_denoised.reshape(-1)

print(img_true.shape)
print(img_denoised.shape)

mse = mean_squared_error(img_true, img_denoised)
print(mse)

psnr = peak_signal_noise_ratio(img_true, img_denoised)
print(psnr)

sim = ssim(img_true, img_denoised)
print(sim)

print(f'MSE: {mse:.2f}')
print(f'PSNR: {psnr:.2f}')
print(f'SSIM: {sim:.2f}')
