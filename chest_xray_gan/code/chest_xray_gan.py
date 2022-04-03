'''
This python script performs the following:

    1. load and preprocess raw chest xray images into a numpy array
    2. create a deep convolution gan model
    3. train the model and save every 100th generated image
'''

import os
import glob
import re
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Dense, Flatten, Reshape, LeakyReLU
from tensorflow.keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# resizes images to desired dimensions
def resize(img, height=256, width=256):
    return cv2.resize(img, dsize=(height, width), interpolation=cv2.INTER_CUBIC)

# get names of original images
def get_image_names(jsrt_path='Downloads'):
    # path to raw images
    home = os.path.expanduser('~')
    image_dir = os.path.join(home, jsrt_path, 'jsrt/images/images')
    
    # get image names
    image_names =  os.listdir(image_dir)
    image_names.sort()
    
    return image_names

# loads and preprocesses images into numpy array
def get_images(jsrt_path='Downloads', height=256, width=256, original_size=False):
    if original_size:
        height = 1024
        width = 1024
    
    home = os.path.expanduser('~')
    image_dir = os.path.join(home, jsrt_path, 'jsrt/images/images')
    
    # load image names
    image_names = get_image_names()
    
    # create numpy array to hold image arrays
    images = np.zeros((len(image_names), height, width, 3))
    
    # load iamges
    for i in range(len(image_names)):
        img_path = os.path.join(image_dir, image_names[i])
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        # resize
        img = resize(img, height, width)
        #img = np.expand_dims(img, axis=2)
        
        # normalize
        img = img / 255
        
        images[i] = img
    
    return images.astype('float32')

# generator
def make_generator():
    latent_dim = 32
    channels = 3
    
    gen_input = Input(shape=(latent_dim))
    
    # transform input into 16x16x128 feature map
    x = Dense(16*16*128)(gen_input)
    x = LeakyReLU()(x)
    x = Reshape((16,16,128))(x)
    
    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(256, 4, strides=(2,2), padding='same')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(channels, 7, activation='tanh', padding='same')(x)
    generator = Model(gen_input, x)
    
    return generator

# discriminator
def make_discriminator():
    height = 32
    width = 32
    channels = 3
    
    dis_input = Input(shape=(height, width, channels))
    
    x = Conv2D(128, 3)(dis_input)
    x = LeakyReLU()(x)
    x = Conv2D(128, 4, strides=2)(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 4, strides=2)(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 4, strides=2)(x)
    x = LeakyReLU()(x)
    
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(1, activation='sigmoid')(x)
    
    discriminator = Model(dis_input, x)
    
    optimizer = RMSprop(learning_rate=0.0008,
                        clipvalue=1.0,
                        decay=1e-8)
    
    discriminator.compile(optimizer=optimizer, loss='binary_crossentropy')
    
    return discriminator

'''
Note: For labels, 0 is real, 1 is fake
'''
def train_gan():
    home = os.path.expanduser('~')
    # create directory to save generated images
    try:
        os.mkdir(os.path.join(home, 'Downloads/generated'))
    except:
        pass
    
    latent_dim = 32
    
    generator = make_generator()
    discriminator = make_discriminator()
    
    discriminator.trainable = False
    
    # create gan network
    gan_input = Input(shape=(latent_dim))
    gan_output = discriminator(generator(gan_input))
    
    gan = Model(gan_input, gan_output)
    
    # gan_optimizer = RMSprop(learning_rate=0.0004,
    #                     clipvalue=1.0,
    #                     decay=1e-8)
    
    gan_optimizer = RMSprop(learning_rate=0.0004,
                            clipvalue=1.0,
                            epsilon=1e-8)
    
    gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
    
    # start training gan
    iterations = 1200
    batch_size = 2
    save_dir = os.path.join(home, 'Downloads/generated')
    
    start = 0
    for step in range(iterations):
        # sample from latent space
        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
        
        # generate images
        generated_images = generator.predict(random_latent_vectors)
        
        stop = start + batch_size
        
        real_images = x_train[start:stop]
        combined_images = np.concatenate([generated_images, real_images])
        
        labels = np.concatenate([np.ones((batch_size, 1)),
                                 np.zeros((batch_size, 1))])
        labels += 0.05 * np.random.random(labels.shape)
        
        # train discriminator
        discriminator_loss = discriminator.train_on_batch(combined_images, labels)
        
        # sample from latent space
        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
        
        # create "all real" labels for 
        fake_labels = np.zeros((batch_size, 1))
        
        # train generator via gan model
        gan_loss = gan.train_on_batch(random_latent_vectors, fake_labels)
        
        start += batch_size
        
        if start > len(x_train) - batch_size:
            start = 0
        
        if step % 100 == 0:
            gan.save_weights('gan.h5')
            
            print('discriminator loss:', discriminator_loss)
            print('gan loss:', gan_loss)
            
            img = image.array_to_img(generated_images[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'real' + str(step) + '.png'))

# observe generated images
def see_generated_imgs(a=0, b=1200, c=100):
    '''Observe generated images.
    
    The generated images are saved to: /home/$USER/Downloads/generated
    For convenient viewing, either use your computer's image viewer or the jupyter notebook version of this script.
    '''
    home = os.path.expanduser('~')
    
    img_no = list(range(a,b,c))
    
    for i in img_no:
        img_path = os.path.join(home, 'Downloads/generated/real' + str(i) + '.png')
        img = plt.imread(img_path)
        plt.imshow(img)
        plt.show()

# main function
def main():
    # get images
    x_train = get_images(height=32, width=32)

    # train model
    train_gan()

if __name__ == "__main__":
    main()