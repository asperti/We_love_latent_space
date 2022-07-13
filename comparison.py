#!/usr/bin/env python
# coding: utf-8

import lib.GAN as GAN
import lib.VAE as VAE
import lib.StyleGAN as StyleGAN
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import split

datadir = '../DL_models/data/img_align_celeba'
    
def show_images_3(imgs1,imgs2,imgs3):
    assert (imgs1.shape[0]==imgs2.shape[0]==imgs3.shape[0])
    n_imgs = imgs1.shape[0]

    plt.figure(figsize=(3*n_imgs, 3*3))
    for k in range(n_imgs):
        img = keras.preprocessing.image.array_to_img(imgs1[k])
        plt.subplot(3, n_imgs, k+1)
        plt.xticks([])  
        plt.yticks([])
        plt.imshow(img)
        img = keras.preprocessing.image.array_to_img(imgs2[k])
        plt.subplot(3, n_imgs, n_imgs+k+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)  
        img = keras.preprocessing.image.array_to_img(imgs3[k])
        plt.subplot(3, n_imgs, n_imgs*2+k+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)  
    plt.show()

#source and target generative models are specified by a class name [VAE,GAN,SVAE] and intance name

#choose model returns an encoder-decoder pair

def choose_model(classname,name):
    if classname=='VAE':
        latent_dim = 64
        VAEencoder, VAEdecoder = VAE.create(latent_dim=latent_dim)
        model_name = name #'VAE64D1' etc.
        vae_filepath = os.path.join('./', 'models', 'VAEs', model_name+'_model',model_name)
        vae = VAE.load(vae_filepath,latent_dim) #,VAEencoder,VAEdecoder)
        def vae_encoder(imgs):
          z_mean, _, _ = VAEencoder(imgs)
          return z_mean 
        return vae_encoder,VAEdecoder
    elif classname=='GAN':
        latent_dim = 64
        GANdiscriminator, GANgenerator, GANrecoder = GAN.create(latent_dim=latent_dim)
        model_name = name #'GAN64D1' etc.
        gan_filepath_fwd = os.path.join('./','models', 'GANs', model_name+'_model', 'forward', model_name)
        gan_filepath_bck = os.path.join('./', 'models', 'GANs', model_name+'_model', 'backward', model_name)

        gan = GAN.load(gan_filepath_fwd)
        igan = GAN.load_inverse(gan_filepath_bck)
        return igan.recoder,gan.generatorcd
    elif classname=='SVAE':
        svae,encoder,decoder,through_latent,emb_generator = split.get_model('celeba')
        model_name = name #'one', 'two' or 'three'
        svae.load_weights("weights/celeba_split_"+model_name+".hdf5")
        def svae_encoder(imgs):
           emb = encoder(imgs)
           _, _, z_mean, _ = through_latent(emb)
           return z_mean

        def svae_decoder(z):
           channels = 3
           emb = emb_generator(z)
           dec = decoder.predict(emb,batch_size=50)
           mask = dec[:,:,:,0:1]
           img1 = dec[:,:,:,1:1+channels]
           img2 = dec[:,:,:,1+channels:1+2*channels]
           generated = img1*mask + img2*(1-mask)
           return generated
        return svae_encoder,svae_decoder
    elif classname=='StyleGAN':
        latent_dim = stylegan.latent_dim
        model_filepath_fwd = os.path.join(os.getcwd(), 'models', 'Downloaded','StyleGAN_model', 'forward', 'Generator')
        model_filepath_bck = os.path.join(os.getcwd(), 'models', 'Downloaded', 'StyleGAN_model',
                                  'backward_' + ('from_w_' if from_w else '') + recoder_attempt_name, 'StyleGAN_CelebAHQ')
        stylegan = StyleGAN.load(model_filepath_fwd)
        igan = GAN.load_inverse(model_filepath_bck, stylegan)
            
        return igan.recoder,stylegan.generator

#-----------------------------------------------

def transform_with_support(sourceclass,sourcename, #e.g. 'SVAE','one'
                           targetclass,targetname, #e.g. 'VAE', 'VAE64D1',
                           support_set):
    #not supported for StyleGAN! use transform_from_W
    source = sourceclass+'_'+sourcename
    target = targetclass+'_'+targetname

    enc1,dec1 = choose_model(sourceclass,sourcename)
    enc2,dec2 = choose_model(targetclass,targetname)

    z_source = enc1(support_set)
    z_target = enc2(support_set)

    #mapping model

    input_dim = z_source.shape[1]
    target_dim = z_target.shape[1]

    weights_file_name = 'M_'+source+'_to_'+target+'.hdf5'
    adam = tf.keras.optimizers.Adam(learning_rate=0.0005)

    #REMARKS
    #1. 2000 iterations are usually enough. Tune it.
    #2. when one of the two models is a GAN augment regularization to 0.0015 

    tin = tf.keras.layers.Input(shape=(input_dim,))
    tout = tf.keras.layers.Dense(target_dim,use_bias=False,
             kernel_regularizer=tf.keras.regularizers.L2(0.0001),
             activity_regularizer=tf.keras.regularizers.L2(.0001))(tin)

    A = tf.keras.Model(tin,tout)
    A.compile(optimizer=adam,loss='mse')

    #A.load_weights('weights/Mappings/'+weights_file_name)

    train_steps = 2000

    for i in range(train_steps):
        loss = A.train_on_batch(z_source,z_target)
        print("loss = ",loss)

    A.save_weights('weights/Mappings/'+weights_file_name)
    return A,enc1,dec1,enc2,dec2

def transform_from_W(otherclass,othername,otherlatent_dim,direction):
    recoder,generator = choose_model('StyleGAN',None)
    otherenc,otherdec = choose_model(sourceclass,sourcename)
    
    direction = 'fromStyleGAn'  #'ToStyleGan
    if direction=='fromStyleGAn':
        source = 'StyleGAN'
        target = otherclass+'_'+othername
        input_dim,outout_dim = 512,otherlatent_dim
    else:
        source = otherclass+'_'+othername
        target = 'StyleGAN'
        input_dim,outout_dim = otherlatent_dim,512

    weights_file_name = 'M_'+source+'_to_'+target+'.hdf5'   
    adam = tf.keras.optimizers.Adam(learning_rate=0.0005)

    tin = tf.keras.layers.Input(shape=(input_dim,))
    tout = tf.keras.layers.Dense(out_dim,use_bias=False)(tin)
             #no need for regualrization
             #kernel_regularizer=regularizers.L2(.001),
             #activity_regularizer=regularizers.L2(.001))(tin)
    A = tf.keras.Model(tin,tout)
    A.compile(optimizer=adam,loss='mse')
    
    A.load_weights('weights/Mappings/'+weights_file_name+'.hdf5')
    training_stesp = 2000
    for i in range(200):
        random_latent_vectors = tf.random.normal((batch_size, latent_dim))
        W_sources = generator.w_embedder(random_latent_vectors)
        imgs = generator.outputs_to_images(generator.from_w(W_sources),out_size=None)
        start_images = generator.to_celeba(imgs,batch_size=n_imgs)*.5+.5
        z = otherenc(start_images)
        if direction == 'fromStyleGAn':
            loss[i] = A.train_on_batch(W_sources,z)
        else:
            #we simply invert source and target
            loss[i] = A.train_on_batch(z,W_sources)
        print("loss = ",loss[i])

    A.save_weights('weights/Mappings/'+weights_file_name)
    return A,recoder,generator,otherenc,otherdec
    

def compute_error_on_CelebA(A,enc1,dec1,enc2,dec2,iterno=100):
    print("computing error")
    data = tf.keras.utils.image_dataset_from_directory(datadir, labels=None,image_size=(64,64),batch_size=100)
    dataiter = iter(data)
    
    R_error = np.zeros(iterno)
    L_error = np.zeros(iterno)
    M_error = np.zeros(iterno)
    
    for i in range(iterno):
        print(i)
        imgs = next(dataiter)/255.
        z_source = enc1(imgs)
        z_target_exp = enc2(imgs)
        z_target_pred = A.predict(z_source)
        decoded1 = dec1(z_source)
        decoded2 = dec2(z_target_pred)
        R_error[i]=tf.reduce_mean(tf.square(imgs-decoded1),axis=[0,1,2,3])
        L_error[i]=tf.reduce_mean(tf.square(z_target_exp-z_target_pred),axis=[0,1])
        M_error[i]=tf.reduce_mean(tf.square(imgs-decoded2),axis=[0,1,2,3])
    print("R_mse = ",np.mean(R_error)," R_std = ",np.std(R_error))
    print("L_mse = ",np.mean(L_error)," L_std = ",np.std(L_error))
    print("M_mse = ",np.mean(M_error)," M_std = ",np.std(M_error))

def compute_error_on_W(A,recoder,generator,otherenc,otherdec,direction):
    print("computing error")
    error = np.zeros(dim)
    for i in range(dim):
        print(i)
        random_latent_vectors = tf.random.normal((batch_size, latent_dim))
        W_sources = generator.w_embedder(random_latent_vectors)
        imgs = generator.outputs_to_images(generator.from_w(W_sources),out_size=None)
        start_images = generator.to_celeba(imgs,batch_size=n_imgs)*.5+.5
        z = otherenc(start_images)
        if direction == 'fromStyleGAn':
            out_imgs = otherdec(z)
            z_hat = A.predict(W_sources)
            decoded2 = otherdec(z_hat)
            #other model reconstruction error
            R_error[i]=tf.reduce_mean(tf.square(start_imgs-out_imgs),axis=[0,1,2,3])
            L_error[i]=tf.reduce_mean(tf.square(z-z_hat),axis=[0,1])
            M_error[i]=tf.reduce_mean(tf.square(start_imgs-decoded2),axis=[0,1,2,3])
        else:
            out_imgs = otherdec(z)
            w_hat = A.predict(z)
            rec =  generator.outputs_to_images(generator.from_w(W_sources),out_size=None)
            decoded2 = generator.outputs_to_images(generator.from_w(W_hat),out_size=None)
            #other model reconstruction error
            R_error[i]=tf.reduce_mean(tf.square(imgs-rec),axis=[0,1,2,3])
            L_error[i]=tf.reduce_mean(tf.square(W_sources,w_hat),axis=[0,1])
            M_error[i]=tf.reduce_mean(tf.square(imgs-decoded2),axis=[0,1,2,3])
    print("mse = ",np.mean(error))
    print("std = ",np.std(error))


use_support_set = True
compute_error = False
visualize = True

if use_support_set:
    support_set = np.load('support_set/support_set_64.npy')
    #print(support_set.shape)
    #print(np.max(support_set, axis=(0,1,2,3)))
    #print(np.min(support_set, axis=(0,1,2,3)))
    #select source and target (not StyleGAN)
    sourceclass,sourcename = 'GAN','GAN4D1'
    #sourceclass,sourcename = 'SVAE','one'
    #sourceclass,sourcename = 'VAE', 'VAE64D1'
    targetclass,targetname = 'SVAE','two'
    
    A,enc1,dec1,enc2,dec2 = transform_with_support(sourceclass,sourcename,
                                                   targetclass,targetname,
                                                   support_set)
    if compute_errors:
        compute_error_on_CelebA(A,enc1,dec1,enc2,dec2)
    if visualize: #True to visualize
        data = tf.keras.utils.image_dataset_from_directory(datadir, labels=None,image_size=(64,64),batch_size=7)
        dataiter = iter(data)
        while True:
            imgs = next(dataiter)/255. 
            z_source = enc1(imgs)
            z_target = A.predict(z_source)
            #z_target = np.random.normal(size=(n_imgs,64))
            decoded1 = dec1(z_source)
            decoded2 = dec2(z_target)
            show_images_3(imgs,decoded1,decoded2)
else:
    # here one of the two models is supposed to be a pretrained StyleGAN
    otherclass,othername = 'SVAE','two'
    if otherclass == 'SVAE':
        otherlatent_dim = 150
    else:
        otherlatent_dim = 64
    direction = 'FromStyleGAN'
    A,recoder,generator,otherenc,otherdec = transform_from_W(otherclass,othername,otherlatent_dim,direction)
    compute_error_on_W(A,recoder,generator,otherenc,otherdec)
    if True: #True to visualize
        while True:
            n_imgs = 7
            random_latent_vectors = tf.random.normal((n_imgs, latent_dim))
            W_sources = generator.w_embedder(random_latent_vectors)
            imgs = generator.outputs_to_images(generator.from_w(W_sources),out_size=None)
            start_images = generator.to_celeba(imgs,batch_size=n_imgs)*.5+.5
            z = otherenc(start_images)
            decoded1 = otherdec(z)
            if direction == 'FromStyleGAN':
                z_hat = A.predict(W_sources)
                decoded2 = otherdec(z_hat)
                show_images_3(imgs,decode1,decode2)
            else:
                w_hat = A.predict(z)
                decoded2 = generator.outputs_to_images(generator.from_w(w_hat),out_size=None)
                show_images_3(imgs,decoded1,decoded2)



