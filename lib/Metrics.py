import numpy as np
import scipy as sp
from tensorflow import keras

# Keras Metric class computing the FID. Requires Tensorflow Eager Execution!
class FrechetInceptionDistance(keras.metrics.Metric):
    default_preprocess_fn = keras.applications.inception_v3.preprocess_input
    
    def load_default_model(shape=(299, 299, 3)):
        return keras.applications.InceptionV3(include_top=False, 
                                              weights="imagenet", 
                                              pooling='avg',
                                              input_shape=shape)
    
    def __init__(self, model, preprocess_fn, shape, name='frechet_inception_distance', **kwargs):
        super(FrechetInceptionDistance, self).__init__(name=name, **kwargs)
        self.true_embeddings = None
        self.pred_embeddings = None
        self.loaded_samples = 0
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.shape = shape
        
        self._cached_result = None
        self._cached_loaded_samples = 0

    def _scale_images(images, new_shape):
        resized_images = []
        for image in images:
            # resize with nearest neighbor interpolation
            new_image = skimage.transform.resize(image, new_shape, 0)
            resized_images.append(new_image)
        return resized_images
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            raise Exception("Weights should not be provided for this metric!")
        
        assert(len(y_true) == len(y_pred))
        
        true_imgs = FrechetInceptionDistance._scale_images(y_true, self.shape)
        gen_imgs = FrechetInceptionDistance._scale_images(y_pred, self.shape)
        preprocessed_true_imgs = self.preprocess_fn(
            np.array(true_imgs), data_format=None
            )
        preprocessed_gen_imgs = self.preprocess_fn(
            np.array(gen_imgs), data_format=None
            )
        new_true_embeddings = self.model(preprocessed_true_imgs).numpy()
        new_image_embeddings = self.model(preprocessed_gen_imgs).numpy()
        if self.true_embeddings is None:
            self.true_embeddings = new_true_embeddings
        else:
            self.true_embeddings = np.concatenate([self.true_embeddings, new_true_embeddings],
                                                  axis=0)
        if self.pred_embeddings is None:
            self.pred_embeddings = new_image_embeddings
        else:
            self.pred_embeddings = np.concatenate([self.pred_embeddings, new_image_embeddings],
                                                  axis=0)
        self.loaded_samples += len(y_true)

    def result(self):
        if not self._cached_result or self.loaded_samples != self._cached_loaded_samples:
            # calculate mean and covariance statistics
            mu1, sigma1 = self.true_embeddings.mean(axis=0), np.cov(self.true_embeddings, rowvar=False)
            mu2, sigma2 = self.pred_embeddings.mean(axis=0), np.cov(self.pred_embeddings, rowvar=False)
            # calculate sum squared difference between means
            ssdiff = np.sum((mu1 - mu2)**2.0)
            # calculate sqrt of product between cov
            covmean = sp.linalg.sqrtm(sigma1.dot(sigma2))
            # check and correct imaginary numbers from sqrt
            if np.iscomplexobj(covmean):
                covmean = covmean.real
             # calculate score
            fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
            self._cached_result = fid
            self._cached_loaded_samples = self.loaded_samples
        return self._cached_result

    def reset_state(self):
        self.true_embeddings = None
        self.pred_embeddings = None
        self.loaded_samples = 0
        self._cached_result = None
        self._cached_loaded_samples = 0