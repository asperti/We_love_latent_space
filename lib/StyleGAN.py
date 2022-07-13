from lib.external.tf2_StyleGAN.stylegan import StyleGAN_G
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence

class StyleGANWLatentGenerator(Sequence):
    def __init__(self, w_embedder, batch_size=32, latent_dim=64, batches_per_epoch=30):
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.batches_per_epoch = batches_per_epoch
        self.w_embedder = w_embedder

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, index):
        return self.w_embedder(tf.random.normal((self.batch_size, self.latent_dim)))

class StyleGAN:
    class Discriminator:
        def __init__(self, discriminator, latent_dim, img_shape):
            self.packed_model = discriminator
            self.latent_dim = latent_dim
            self.img_shape = img_shape
        
        def __call__(self, images_tf):
            raise NotImplementedError()
        
        def predict(self, images_np):
            raise NotImplementedError()
    
    class Generator:
        def outputs_to_images(self, outputs, out_size, to_zero_one=False):
            images = tf.transpose(outputs, [0, 2, 3, 1])
            if to_zero_one:
                images = (images + 1) / 2
            if out_size != None:
                images = tf.image.resize(images, size=out_size[:2], method='nearest')
            return images

        def __init__(self, pickle_generator, latent_dim, img_shape):
            self._packed_model = pickle_generator
            self.img_shape = img_shape
            self.latent_dim = latent_dim
            # Define sub-model going from z to w
            self.w_embedder = keras.Model(self._packed_model.layers[0].input, self._packed_model.layers[0].layers[-4].output)
            # Define sub-model going from w to images
            w_submodel0 = keras.Model(self._packed_model.layers[0].layers[-4].input,
                                      self._packed_model.layers[0].layers[-1].output)
            w0 = keras.Input((self.latent_dim))
            w1 = w_submodel0(w0)
            w2 = self._packed_model.layers[1](w1)
            self.from_w = keras.Model(w0, w2)
            
        def __call__(self, latent_vectors_tf, to_images=False, to_zero_one=False, out_size=None):
            images = self._packed_model(latent_vectors_tf)
            if to_images:
                images = self.outputs_to_images(images, to_zero_one=to_zero_one, out_size=out_size)
            return images
            
        def predict(self, latent_vectors_np, to_images=False, out_size=None):
            images = self.__call__(latent_vectors_np, to_images=to_images, out_size=out_size)
            return images.numpy()
                
    def __init__(self, generator, discriminator):
        self.latent_dim = 512
        self.img_shape = (1024, 1024, 3)
        self.generator = self.Generator(generator, self.latent_dim, self.img_shape)
        self.discriminator = self.Discriminator(discriminator, self.latent_dim, self.img_shape)

def load(model_filepath):
    generator = StyleGAN_G()
    generator.build(input_shape=(None, 512))
    generator.load_weights(model_filepath + '.h5')
    stylegan = StyleGAN(generator, None)
    return stylegan