import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input,Conv2D, Conv2DTranspose, Dense, Reshape, \
    BatchNormalization, GlobalAveragePooling2D, Flatten, Activation
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import wget
import os

#dowload pretrained weights
if not(os.path.exists("weights/celeba_split_one.hdf5")):
    import wget
    url = 'https://www.cs.unibo.it/~asperti/VAEcheckpoint/split_64_4SB_lat150_base128_512_k3.hdf5'
    filename = wget.download(url)
    os.rename(filename, "weights/celeba_split_one.hdf5")
    url = 'https://www.cs.unibo.it/~asperti/VAEcheckpoint/celeba_split_two.hdf5'
    filename = wget.download(url)
    os.rename(filename, "weights/celeba_split_two.hdf5")
    url = 'https://www.cs.unibo.it/~asperti/VAEcheckpoint/celeba_split_three.hdf5'
    filename = wget.download(url)
    os.rename(filename, "weights/celeba_split_three.hdf5")
       
def ResBlock(out_dim, depth=2, kernel_size=3, name='ResBlock'):
    def body(inputs, **kwargs):
      with K.name_scope(name):
        y = inputs
        for i in range(depth):
            y = BatchNormalization(momentum=.999,epsilon=1e-5)(y)
            y = Activation('swish')(y) #ReLU()(y)
            y = Conv2D(out_dim,kernel_size,padding='same')(y)
        s = Conv2D(out_dim, kernel_size,padding='same')(inputs)
      return y + s
    return(body)

def ResFcBlock(out_dim, depth=2, name='ResFcBlock'):
    def body(inputs, **kwargs):
      with K.name_scope(name):
        y = inputs
        for i in range(depth):
            y = BatchNormalization(momentum=.999,epsilon=1e-5)(y)
            y = Activation('swish')(y) #ReLU()(y)
            y = Dense(out_dim)(y)
        s = Dense(out_dim)(inputs)
      return y + s
    return(body)

def ScaleBlock(out_dim, block_per_scale=1, depth_per_block=2, kernel_size=3, name='ScaleBlock'):
    def body(inputs, **kwargs):
      with K.name_scope(name):
        y = inputs
        for i in range(block_per_scale):
            y = ResBlock(out_dim,depth=depth_per_block, kernel_size=kernel_size)(y)
      return y
    return (body)

def ScaleFcBlock(out_dim, block_per_scale=1, depth_per_block=2, name='ScaleFcBlock'):
    def body(inputs, **kwargs):
      with K.name_scope(name):
        y = inputs
        for i in range(block_per_scale):
            y = ResFcBlock(out_dim, depth=depth_per_block)(y)
      return y
    return(body)

# Model

def Encoder(input_shape, base_dim, kernel_size, num_scale, block_per_scale, depth_per_block,
             embedding_dim, name='Encoder'):
    with K.name_scope(name):
        dim = base_dim
        enc_input = Input(shape=input_shape)
        y = Conv2D(dim,kernel_size,padding='same',strides=2)(enc_input)
        for i in range(num_scale-1):
            y = ScaleBlock(dim, block_per_scale, depth_per_block, kernel_size)(y)
            if i != num_scale - 1:
                dim *= 2
                y = Conv2D(dim,kernel_size,strides=2,padding='same')(y)

        y = GlobalAveragePooling2D()(y)
        ySB = ScaleFcBlock(embedding_dim,1,depth_per_block)(y)
        
        encoder = Model(enc_input,ySB)
    return encoder

def Latent(embedding_dim,latent_dim, name="Latent"):
    with K.name_scope(name):
        emb = Input(shape=embedding_dim)
  
        mu_z = Dense(latent_dim)(emb)
        logvar_z = Dense(latent_dim)(emb)
        z = mu_z + K.random_normal(shape=K.shape(mu_z)) * K.exp(logvar_z*.5)

        back_to_emb = Dense(embedding_dim)
        emb_hat = back_to_emb(z)

        noise = Input(shape=latent_dim)
        emb_gen = back_to_emb(noise)
        
        through_latent = Model(emb,[emb_hat,z,mu_z,logvar_z])
        emb_generator = Model(noise,emb_gen)
        return through_latent,emb_generator

def Decoder(out_ch, embedding_dim, dims, scales, kernel_size, block_per_scale, depth_per_block, name='Decoder'):
  
    base_wh = 4
    data_depth = out_ch
    print("dims[0] is = ",dims[0])
    print("embedding_dim is ",embedding_dim)

    with K.name_scope(name):
        emb = Input(shape=(embedding_dim,))
        y = Dense(base_wh * base_wh * dims[0])(emb)
        y = Reshape((base_wh,base_wh,dims[0]))(y)

        for i in range(len(scales) - 1):
            y = Conv2DTranspose(dims[i+1], kernel_size, strides=2, padding='same')(y)
            y = ScaleBlock(dims[i+1],block_per_scale, depth_per_block, kernel_size)(y)

        x_hat = Conv2D(data_depth, kernel_size, 1, padding='same', activation='sigmoid')(y)
        decoder = Model(emb,x_hat)
    return(decoder)

def FullModel(input_shape,latent_dim,base_dim=32,emb_dim=512, kernel_size=3,num_scale=3,block_per_scale=1,depth_per_block=2):
    desired_scale = input_shape[1]
    scales, dims = [], []
    current_scale, current_dim = 4, base_dim
    while current_scale <= desired_scale:
        scales.append(current_scale)
        dims.append(current_dim)
        current_scale *= 2
        current_dim = min(current_dim * 2, 1024)
    assert (scales[-1] == desired_scale)
    dims = list(reversed(dims))
    print(dims,scales)

    encoder = Encoder(input_shape, base_dim, kernel_size, num_scale, block_per_scale, depth_per_block, emb_dim)
    through_latent,emb_generator = Latent(emb_dim,latent_dim)
    
    decoder = Decoder(input_shape[2]*2+1, emb_dim, dims, scales, kernel_size, block_per_scale, depth_per_block)
    

    x = Input(shape=input_shape)
    channels = input_shape[2]
    gamma = Input(shape=())
    emb = encoder(x)
    emb_hat, z, z_mean, z_log_var = through_latent(emb)
    dec = decoder(emb_hat)
    mask = dec[:,:,:,0:1]
    img1 = dec[:,:,:,1:1+channels]
    img2 = dec[:,:,:,1+channels:1+2*channels]
    x_hat = img1*mask + img2*(1-mask)
    #x_hat = dec
    vae = Model([x,gamma],x_hat)
    
    #loss
    beta = 3 
    L_rec =.5 * K.sum(K.square(x-x_hat), axis=[1,2,3]) / gamma 
    L_KL = .5 * K.sum(K.square(z_mean) + K.exp(z_log_var) - 1 - z_log_var, axis=-1)
    L_tot = K.mean(L_rec + beta * L_KL) 

    vae.add_loss(L_tot)

    return(vae,encoder,decoder,through_latent,emb_generator)


####################################################################
# create model
####################################################################


def get_model(dataset):
    if dataset == 'celeba':
        latent_dim = 150
        input_dim = (64,64,3)
    elif dataset == 'cifar10':
        latent_dim = 200
        input_dim = (32,32,3)
    elif dataset == 'mnist':
        latent_dim = 32
        input_dim = (32,32,1)
    return FullModel(input_dim,latent_dim,base_dim=64,num_scale=4,emb_dim=512)



    



