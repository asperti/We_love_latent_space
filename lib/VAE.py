import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
import matplotlib.pyplot as plt
import os
from . import DataEvaluator as DataEvaluator
from . import Metrics as Metrics
from .keras_utils import build_model, describe_model

class GaussianSampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, img_shape, latent_dim, info, parameters, fid_samples=1000, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.info = info # name, dataset
        self.img_shape = img_shape
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.prev_gamma_squared = float('inf')
        self.fid_samples = fid_samples
        self.tf_version = tf.__version__
        self.parameters = {}
        if 'encoder_args' in parameters:
            self.parameters['encoder_args'] = parameters['encoder_args']
        else:
            self.parameters['encoder_args'] = {}
        if 'decoder_args' in parameters:
            self.parameters['decoder_args'] = parameters['decoder_args']
        else:
            self.parameters['decoder_args'] = {}
        
        self.fid_test_start = True
        fid_model = Metrics.FrechetInceptionDistance.load_default_model()
        self.fid_tracker = Metrics.FrechetInceptionDistance(fid_model, 
                                                            Metrics.FrechetInceptionDistance.default_preprocess_fn,
                                                            shape=(299, 299, 3),
                                                            name="fid")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.fid_tracker
        ]

    def test_step(self, real_images):
        if (self.fid_tracker.loaded_samples > 0 and self.fid_tracker.loaded_samples < self.fid_samples) or \
           (self.fid_tracker.loaded_samples == 0 and self.fid_test_start):
            self.fid_test_start = False
            batch_size = tf.shape(real_images)[0].numpy()
            pred_images = self.decoder(tf.random.normal(shape=(batch_size, self.latent_dim))).numpy()
            pred_images = (pred_images*255).astype(int)
            pred_images = list(pred_images)
            true_images = (real_images*255).numpy().astype(int)
            true_images = list(true_images)
            self.fid_tracker.update_state(true_images, pred_images)
        else:
            self.fid_test_start = True
            return {'fid': self.fid_tracker.result()}
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction),
                                                               axis=(1, 2)))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            gamma_squared = tf.math.minimum(backend.get_value(self.prev_gamma_squared), reconstruction_loss)
            total_loss = reconstruction_loss * self.img_shape[0] * self.img_shape[1] / (2 * gamma_squared) + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.prev_gamma_squared = gamma_squared
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "lambda": self.prev_gamma_squared
        }
    
class GenerateImagesCallback(keras.callbacks.Callback):
    def __init__(self, metadata_filepath, num_img=5, save_filepath=None):
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
        generated_images = self.model.decoder(random_latent_vectors)
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
        
    def on_epoch_end(self, epoch, logs):
        self.latent_dim = self.model.latent_dim
        self.name = self.model.info['name']
        self.shape = self.model.img_shape
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
            encoder, decoder = create(self.latent_dim, print_mode=True)
            data.set_value(['structure', 'encoder'], encoder)
            data.set_value(['structure', 'decoder'], decoder)
        if not data.exists_entry(['per_epoch_data']):
            data.set_value(['per_epoch_data'], [logs])
        else:
            data.get_value(['per_epoch_data']).append(logs)
        data.save()

def create(img_shape=(64, 64, 3), latent_dim=64, verbose=True, print_mode=False, **kwargs):
    """Encoder"""
    encoder_list = [
        (keras.Input, {'shape':img_shape}, 'in')
    ]
    for i in range(3):
        encoder_list.extend([
            (layers.Conv2D, {'filters':128*(2**i), 'kernel_size':4, 'activation':'relu',
                             'strides':2, 'padding':'same'}, 'in' if i == 0 else 'x', 'x'),
            (layers.LeakyReLU, {'alpha':0.2}, 'x', 'x')
        ])
    encoder_list.extend([
        (layers.Flatten, {}, 'x', 'x'),
        (layers.Dropout, {'rate':0.2}, 'x', 'x'),
        (layers.Dense, {'units':latent_dim}, 'x', 'z_mean'),
        (layers.Dense, {'units':latent_dim}, 'x', 'z_log_var'),
        (GaussianSampling, {}, ['z_mean', 'z_log_var'], 'out'),
    ])
    if not print_mode:
        encoder = build_model(encoder_list, ['in'], ['z_mean', 'z_log_var', 'out'], 'encoder')
        if verbose:
            encoder.summary()
    else:
        encoder = describe_model(encoder_list, ['in'], ['z_mean', 'z_log_var', 'out'], 'encoder')
        
    """Decoder"""
    decoder_list = [
        (keras.Input, {'shape':latent_dim}, 'in'),
        (layers.Dense, {'units': 8*8*512, 'activation':'relu'}, 'in', 'x'),
        (layers.Reshape, {'target_shape':(8, 8, 512)}, 'x', 'x')
    ]
    for i in range(3):
        decoder_list.extend([
            (layers.Conv2DTranspose, {'filters':256//(2**i), 'kernel_size':4, 'activation':'relu',
                                      'strides':2, 'padding':'same'}, 'x', 'x'),
            (layers.LeakyReLU, {'alpha':0.2}, 'x', 'x')
        ])
    decoder_list.append((layers.Conv2D, {'filters':3, 'kernel_size':5, 'activation':'sigmoid', 'padding':'same'}, 'x', 'out'))
    if not print_mode:
        decoder = build_model(decoder_list, ['in'], ['out'], 'decoder')
        if verbose:
            decoder.summary()
    else:
        decoder = describe_model(decoder_list, ['in'], ['out'], 'decoder')
        
    return encoder, decoder
    
def train(img_shape, latent_dim, dataset_train, model_filepath, encoder, decoder, info, parameters={},
          val_data=None, fid_samples=1000, epochs=1000, steps_per_epoch=None):
    if not os.path.exists(model_filepath.rsplit(os.path.sep, 1)[0]):
        os.makedirs(model_filepath.rsplit(os.path.sep, 1)[0], exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_filepath, verbose=1, monitor='val_fid',
                                                         save_weights_only=True, save_best_only=(val_data is not None))
    genimages_callback = GenerateImagesCallback(num_img=7,
                                                metadata_filepath=model_filepath.rsplit(os.path.sep, 1)[0],
                                                save_filepath=os.path.join(model_filepath.rsplit(os.path.sep, 1)[0], 'generated_examples'))
    metadata_callback = WriteMetadataCallback(filepath=model_filepath.rsplit(os.path.sep, 1)[0])
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_fid', mode='min', verbose=1, patience=3, min_delta=1)
    
    vae = VAE(encoder, decoder, img_shape, latent_dim, info=info, parameters=parameters, fid_samples=fid_samples)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), run_eagerly=True)
    if os.path.exists(model_filepath + ".index"):
        vae.load_weights(model_filepath)
    vae.fit(dataset_train, epochs=epochs, callbacks=[genimages_callback, checkpoint_callback, metadata_callback, earlystop_callback], 
            validation_data=val_data, steps_per_epoch=steps_per_epoch)
    return vae

def load(model_filepath, verbose=True):
    data = DataEvaluator.JsonFile(model_filepath.rsplit(os.path.sep, 1)[0], "metadata")
    info = {
        'name': data.get_value(['name']),
        'dataset': data.get_value(['dataset'])
    }
    parameters = data.get_value(['structure_parameters'])
    img_shape = data.get_value(['img_shape'])
    latent_dim = data.get_value(['latent_dim'])
    encoder, decoder = create(input_shape=img_shape,
                              latent_dim=latent_dim,
                              verbose=verbose, print_mode=False, **parameters)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim, img_shape=img_shape, info=info, parameters=parameters)
    vae.load_weights(model_filepath)
    return vae   