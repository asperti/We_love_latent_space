import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import Sequence
from .keras_utils import describe_model, build_model
from . import DataEvaluator as DataEvaluator
from . import Metrics as Metrics
import math

class GAN(keras.Model):
    def __init__(self, discriminator, generator, img_shape, latent_dim, info, parameters, fid_samples=1000):
        super(GAN, self).__init__()
        
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.info = info # name, dataset_name
        self.fid_samples = fid_samples
        self.parameters = {}
        if 'discriminator_args' in parameters:
            self.parameters['discriminator_args'] = parameters['discriminator_args']
        else:
            self.parameters['discriminator_args'] = {}
        if 'generator_args' in parameters:
            self.parameters['generator_args'] = parameters['generator_args']
        else:
            self.parameters['generator_args'] = {}
        self.tf_version = tf.__version__
            
        self.fid_test_start = True

    def compile(self, d_optimizer, g_optimizer, loss_fn, **kwargs):
        super(GAN, self).compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        fid_model = Metrics.FrechetInceptionDistance.load_default_model()
        self.fid_tracker = Metrics.FrechetInceptionDistance(fid_model, 
                                                            Metrics.FrechetInceptionDistance.default_preprocess_fn,
                                                            shape=(299, 299, 3),
                                                            name="fid")

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker, self.fid_tracker]
    
    def test_step(self, real_images):
        if (self.fid_tracker.loaded_samples > 0 and self.fid_tracker.loaded_samples < self.fid_samples) or \
           (self.fid_tracker.loaded_samples == 0 and self.fid_test_start):
            self.fid_test_start = False
            batch_size = tf.shape(real_images)[0].numpy()
            pred_images = self.generator(tf.random.normal(shape=(batch_size, self.latent_dim))).numpy()
            pred_images = (pred_images*255).astype(int)
            pred_images = list(pred_images)
            true_images = (real_images*255).numpy().astype(int)
            true_images = list(true_images)
            self.fid_tracker.update_state(true_images, pred_images)
        else:
            self.fid_test_start = True
            return {'fid': self.fid_tracker.result()}
    
    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        
        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
        }

class GANInverse(keras.Model):
    def __init__(self, generator, recoder, img_shape, latent_dim, info, parameters):
        super(GANInverse, self).__init__()
        self.generator = generator
        self.recoder = recoder
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.info = info # name, dataset_name
        self.tf_version = tf.__version__
        self.parameters = {}
        if 'recoder_args' in parameters:
            self.parameters['recoder_args'] = parameters['recoder_args'] 
        else:
            self.parameters['recoder_args'] = {
                'n_layers': 3,
                'base_filters_n': 128,
                'filters_multiplier': 2,
                'stride': 2,
                'kernel_size': 4
            }
        self.loss_tracker = keras.metrics.Mean(
            name="loss"
        )

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, latent_vectors):
        # Get batch size
        batch_size = tf.shape(latent_vectors)[0]
        
        # Generate images from latent vectors
        generated_images = self.generator(latent_vectors)

        # Train the recoder (note that we should *not* update the weights
        # of the generator)!
        with tf.GradientTape() as tape:
            reconstruction = self.recoder(generated_images)
            loss = keras.losses.mean_squared_error(latent_vectors, reconstruction)
        grads = tape.gradient(loss, self.recoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.recoder.trainable_weights))

        # Update metrics
        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
        }

class GANLatentGenerator(Sequence):
    def __init__(self, batch_size=32, latent_dim=64, batches_per_epoch=30):
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.batches_per_epoch = batches_per_epoch

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, index):
        return tf.random.normal((self.batch_size, self.latent_dim))
    
class GenerateImagesCallback(keras.callbacks.Callback):
    def __init__(self, metadata_filepath, num_img=5, save_filepath=None):
        super(GenerateImagesCallback, self).__init__()
        self.num_img = num_img
        self.save_filepath = save_filepath
        self.metadata_filepath = metadata_filepath
        
    def on_epoch_end(self, epoch, logs=None):
        self.latent_dim = self.model.latent_dim
        
        data = DataEvaluator.JsonFile(self.metadata_filepath, "metadata")
        if data.exists_entry(['trained_epochs']):
            epoch = data.get_value(['trained_epochs'])
        else:
            epoch = 0
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        plt.figure(figsize=(5*self.num_img, 5))
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            plt.subplot(1, self.num_img, i+1)
            plt.xticks([])  
            plt.yticks([]) 
            plt.imshow(img)
        fig = plt.gcf()
        plt.show()
        if self.save_filepath:
            DataEvaluator.save_as_plot(self.save_filepath, 'samples_epoch_' + str(epoch), fig)

class WriteMetadataCallback(keras.callbacks.Callback):
    def __init__(self, filepath):
        self.filepath = filepath
        super(WriteMetadataCallback, self).__init__()
        
    def on_epoch_end(self, epoch, logs):
        self.latent_dim = self.model.latent_dim
        self.name = self.model.info['name']
        self.shape = self.model.img_shape
        self.inverse = True if isinstance(self.model, GANInverse) else False
        self.dataset_name = self.model.info['dataset']
        self.tf_version = self.model.tf_version
        self.parameters = self.model.parameters
        
        data = DataEvaluator.JsonFile(self.filepath, "metadata")
        data.set_value(['description'], "Training and Validation data of the model for each epoch.", metadata=True)
        if data.exists_entry(['trained_epochs']):
            data.set_value(['trained_epochs'], data.get_value(['trained_epochs'])+1)
        else:
            data.set_value(['trained_epochs'], 1)
            data.set_value(['latent_dim'], self.latent_dim)
            data.set_value(['name'], self.name)
            data.set_value(['dataset'], self.dataset_name)
            data.set_value(['library'], ['tensorflow', self.tf_version])
            data.set_value(['img_shape'], list(self.shape))
            data.set_value(['structure_parameters'], self.parameters)
            if not self.inverse:
                discriminator, generator, _ = create(self.shape, self.latent_dim, print_mode=True, **self.parameters)
                data.set_value(['structure', 'discriminator'], discriminator)
                data.set_value(['structure', 'generator'], generator)
            else:
                _, _, recoder = create(self.shape, self.latent_dim, print_mode=True, **self.parameters)
                data.set_value(['structure', 'recoder'], recoder)
        if not data.exists_entry(['per_epoch_data']):
            data.set_value(['per_epoch_data'], [logs])
        else:
            data.get_value(['per_epoch_data']).append(logs)
        data.save()

def create(input_shape=(64, 64, 3), latent_dim=64, verbose=True, print_mode=False, **kwargs):
    """Discriminator"""
    discriminator_list = [
        (keras.Input, {'shape':input_shape}, 'in')
    ]
    for i in range(3):
        discriminator_list.extend([
            (layers.Conv2D, {'filters':128*(2**i), 'kernel_size':4, 'activation':'relu',
                                      'strides':2, 'padding':'same'}, 'in' if i == 0 else 'x', 'x'),
            (layers.LeakyReLU, {'alpha':0.2}, 'x', 'x')
        ])
    discriminator_list.extend([
        (layers.Flatten, {}, 'x', 'x'),
        (layers.Dropout, {'rate':0.2}, 'x', 'x'),
        (layers.Dense, {'units':1, 'activation':'sigmoid'}, 'x', 'out')
    ])
    if not print_mode:
        discriminator = build_model(discriminator_list, ['in'], ['out'], 'discriminator')
        if verbose:
            discriminator.summary()
    else:
        discriminator = describe_model(discriminator_list, ['in'], ['out'], 'discriminator')
        
    """Generator"""
    generator_list = [
        (keras.Input, {'shape':latent_dim}, 'in'),
        (layers.Dense, {'units': 8*8*16, 'activation':'relu'}, 'in', 'x'),
        (layers.Reshape, {'target_shape':(8, 8, 16)}, 'x', 'x')
    ]
    for i in range(3):
        generator_list.extend([
            (layers.Conv2DTranspose, {'filters':128*(2**i), 'kernel_size':4, 'activation':'relu',
                                      'strides':2, 'padding':'same'}, 'x', 'x'),
            (layers.LeakyReLU, {'alpha':0.2}, 'x', 'x')
        ])
    generator_list.append((layers.Conv2D, {'filters':3, 'kernel_size':5, 'activation':'sigmoid', 'padding':'same'}, 'x', 'out'))
    if not print_mode:
        generator = build_model(generator_list, ['in'], ['out'], 'generator')
        if verbose:
            generator.summary()
    else:
        generator = describe_model(generator_list, ['in'], ['out'], 'generator')
                           
    """Recoder"""
    if 'recoder_args' in kwargs:
        n_layers = 3 if 'n_layers' not in kwargs['recoder_args'] else kwargs['recoder_args']['n_layers']
        base_filters_n = 128 if 'base_filters_n' not in kwargs['recoder_args'] else kwargs['recoder_args']['base_filters_n']
        filters_multiplier = 2 if 'filters_multiplier' not in kwargs['recoder_args'] else kwargs['recoder_args']['filters_multiplier']
        stride = 2 if 'stride' not in kwargs['recoder_args'] else kwargs['recoder_args']['stride']
        kernel_size = 4 if 'kernel_size' not in kwargs['recoder_args'] else kwargs['recoder_args']['kernel_size']
        extra_dense = False if 'extra_dense' not in kwargs['recoder_args'] else kwargs['recoder_args']['extra_dense']
    else:
        n_layers = 3
        base_filters_n = 128
        filters_multiplier = 2
        stride = 2
        kernel_size = 4
        extra_dense = False
        
    recoder_list = [
        (keras.Input, {'shape':input_shape}, 'in')
    ]
    for i in range(n_layers):
        n_filters = (base_filters_n*(filters_multiplier**i)) if filters_multiplier > 0 else (base_filters_n*(i+1))
        recoder_list.extend([
            (layers.Conv2D, {'filters':n_filters, 'kernel_size':kernel_size,
                             'activation':'relu', 'strides':stride, 'padding':'same'}, 'in' if i == 0 else 'x', 'x'),
            (layers.LeakyReLU, {'alpha':0.2}, 'x', 'x')
        ])
    recoder_list.extend([
        (layers.Flatten, {}, 'x', 'x'),
        (layers.Dropout, {'rate':0.2}, 'x', 'x')
    ])
    if extra_dense:
        recoder_list.extend([
            (layers.Dense, {'units': 4*latent_dim}, 'x', 'x'),
            (layers.LeakyReLU, {'alpha':0.2}, 'x', 'x'),
            (layers.Dense, {'units': latent_dim}, 'x', 'out')
        ])
    else:
        recoder_list.append(
            (layers.Dense, {'units': latent_dim}, 'x', 'out')
        )
    if not print_mode:
        recoder = build_model(recoder_list, ['in'], ['out'], 'recoder')
        if verbose:
            recoder.summary()
    else:
        recoder = describe_model(recoder_list, ['in'], ['out'], 'recoder')
    return discriminator, generator, recoder

def train(img_shape, latent_dim, dataset_train, model_filepath, discriminator, generator, info, parameters={},
          val_data=None, fid_samples=1000, epochs=1000, steps_per_epoch=None):
    if not os.path.exists(model_filepath.rsplit(os.path.sep, 1)[0]):
        os.makedirs(model_filepath.rsplit(os.path.sep, 1)[0], exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_filepath, verbose=1, monitor='val_fid',
                                                         save_weights_only=True, save_best_only=(val_data is not None))
    genimages_callback = GenerateImagesCallback(num_img=7, metadata_filepath=model_filepath.rsplit(os.path.sep, 1)[0],
                                                save_filepath=os.path.join(model_filepath.rsplit(os.path.sep, 1)[0], 'generated_examples'))
    metadata_callback = WriteMetadataCallback(filepath=model_filepath.rsplit(os.path.sep, 1)[0])
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_fid', mode='min', verbose=1, patience=3, min_delta=1)

    gan = GAN(discriminator=discriminator, generator=generator, img_shape=img_shape, latent_dim=latent_dim, info=info, parameters=parameters, fid_samples=fid_samples)
    if os.path.exists(model_filepath + ".index"):
        gan.load_weights(model_filepath) 
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss_fn=keras.losses.BinaryCrossentropy(),
        run_eagerly=True)
    gan.fit(dataset_train, epochs=epochs, callbacks=[genimages_callback, checkpoint_callback, metadata_callback, earlystop_callback], 
            validation_data=val_data, steps_per_epoch=steps_per_epoch)
    return gan

def train_inverse(img_shape, latent_dim, model_filepath, recoder, generator, info, parameters={}, sequence=None,
                  batch_size=32, steps_per_epoch=5382, epochs=1000):
    if not os.path.exists(model_filepath.rsplit(os.path.sep, 1)[0]):
        os.makedirs(model_filepath.rsplit(os.path.sep, 1)[0], exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_filepath, verbose=1, monitor='loss',
                                                         save_weights_only=True, save_best_only=True)
    metadata_callback = WriteMetadataCallback(filepath=model_filepath.rsplit(os.path.sep, 1)[0])
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=1, min_delta=2)
    igan = GANInverse(generator=generator, recoder=recoder, img_shape=img_shape, latent_dim=latent_dim, info=info, parameters=parameters)
    
    if os.path.exists(model_filepath + ".index"):
        igan.load_weights(model_filepath)
    igan.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01))
    if sequence is None:
        sequence = GANLatentGenerator(latent_dim=latent_dim, batch_size=batch_size, batches_per_epoch=steps_per_epoch)
    igan.fit(sequence, epochs=epochs, callbacks=[checkpoint_callback, metadata_callback], steps_per_epoch=steps_per_epoch)
    return igan

def load(model_filepath, verbose=True):
    data = DataEvaluator.JsonFile(model_filepath.rsplit(os.path.sep, 1)[0], "metadata")
    info = {
        'name': data.get_value(['name']),
        'dataset': data.get_value(['dataset'])
    }
    parameters = data.get_value(['structure_parameters'])
    img_shape = data.get_value(['img_shape'])
    latent_dim = data.get_value(['latent_dim'])
    discriminator, generator, _ = create(input_shape=img_shape,
                                         latent_dim=latent_dim,
                                         verbose=verbose, print_mode=False, **parameters)
    gan = GAN(discriminator=discriminator, generator=generator, img_shape=img_shape, latent_dim=latent_dim, info=info, parameters=parameters)
    gan.load_weights(model_filepath)
    return gan

def load_inverse(model_filepath, gan, verbose=True):
    data = DataEvaluator.JsonFile(model_filepath.rsplit(os.path.sep, 1)[0], "metadata")
    info = {
        'name': data.get_value(['name']),
        'dataset': data.get_value(['dataset'])
    }
    parameters = data.get_value(['structure_parameters'])
    img_shape = data.get_value(['img_shape'])
    latent_dim = data.get_value(['latent_dim'])
    _, _, recoder = create(input_shape=img_shape,
                           latent_dim=latent_dim,
                           verbose=verbose, print_mode=False, **parameters)
    igan = GANInverse(generator=gan.generator, recoder=recoder, img_shape=img_shape, latent_dim=latent_dim, info=info, parameters=parameters)
    igan.load_weights(model_filepath)
    return igan