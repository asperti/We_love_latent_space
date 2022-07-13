# Libraries
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

import lib.GAN as GAN
import lib.VAE as VAE

# VAE Loading
model_name = 'VAE64D1'
model_filepath = os.path.join(os.getcwd(), 'models', 'VAEs', model_name + '_model', model_name)

vae = VAE.load(model_filepath)

# VAE Generation Test
num_img = 7
random_latent_vectors = tf.random.normal(shape=(num_img, vae.latent_dim))
generated_images = vae.decoder(random_latent_vectors)
generated_images *= 255
generated_images.numpy()
plt.figure(figsize=(5*num_img, 5))
print("VAE Generation")
for i in range(num_img):
    img = keras.preprocessing.image.array_to_img(generated_images[i])
    plt.subplot(1, num_img, i+1)
    plt.xticks([])  
    plt.yticks([]) 
    plt.imshow(img)
plt.show()

# GAN Loading
model_name = 'GAN64D1'
model_filepath_fwd = os.path.join(os.getcwd(), 'models', 'GANs', model_name + '_model', 'forward', model_name)
model_filepath_bck = os.path.join(os.getcwd(), 'models', 'GANs', model_name + '_model', 'backward', model_name)

gan = GAN.load(model_filepath_fwd)
igan = GAN.load_inverse(model_filepath_bck, gan)

# GAN Generation Test
num_img = 7
random_latent_vectors = tf.random.normal(shape=(num_img, gan.latent_dim))
generated_images = gan.generator(random_latent_vectors)
generated_images *= 255
generated_images.numpy()
plt.figure(figsize=(5*num_img, 5))
print("GAN Generation")
for i in range(num_img):
    img = keras.preprocessing.image.array_to_img(generated_images[i])
    plt.subplot(1, num_img, i+1)
    plt.xticks([])  
    plt.yticks([]) 
    plt.imshow(img)
plt.show()

# VAE Training Example
model_name = 'VAE64D6'
input_shape = (64, 64, 3)
latent_dim = 64
parameters = {}
train_data = np.random.randint(255, size=(5, 64, 64, 3)).astype('float') #LOAD DATASET HERE (batches of images)
val_data = None # FID-based validation. Skipped if this is None

encoder, decoder = VAE.create(input_shape, latent_dim, **parameters)

info = {
    'dataset': "CelebA_align",
    'name': model_name
}
model_filepath = os.path.join(os.getcwd(), 'models', 'VAEs', model_name + '_model', model_name)
    
VAE.train(input_shape, latent_dim, train_data, model_filepath, encoder, decoder, info=info, parameters=parameters,
          val_data=val_data, fid_samples=1000, epochs=1, steps_per_epoch=1)

# GAN Training Example
model_name = 'GAN64D6'
input_shape = (64, 64, 3)
latent_dim = 64
parameters = {
    'recoder_args': {
        'base_filters_n': 128,
        'filters_multiplier': 2,
        'n_layers': 3,
        'stride': 2,
        'kernel_size': 4
    }
}
train_data = np.random.randint(255, size=(5, 64, 64, 3)).astype('float') # LOAD DATASET HERE (batches of images)
val_data = None # FID-based validation. Skipped if this is None

discriminator, generator, recoder = GAN.create(input_shape, latent_dim, **parameters)

info = {
    'dataset': "CelebA_align",
    'name': model_name
}
model_filepath_fwd = os.path.join(os.getcwd(), 'models', 'GANs', model_name + '_model', 'forward', model_name)
model_filepath_bck = os.path.join(os.getcwd(), 'models', 'GANs', model_name + '_model', 'backward', model_name)

GAN.train(input_shape, latent_dim, train_data, model_filepath_fwd, discriminator, generator, info=info, parameters=parameters,
          val_data=val_data, fid_samples=1000, epochs=1, steps_per_epoch=1)

gan = GAN.load(model_filepath_fwd)
GAN.train_inverse(input_shape, latent_dim, model_filepath_bck, recoder, gan.generator, info=info,
                  parameters=parameters, epochs=1, steps_per_epoch=1)
