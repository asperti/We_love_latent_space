# We love latent space
Repository relative to the article "Comparing the latent space of generative models", joint work with Valerio Tonelli

In this work we
address the more general problem of comparing the latent spaces of different models,
looking for transformations between them. 

<p align="center"><img src="mapping.png" width="500"/><p>

Given a generative model, it is usually possible to have an encoder-decoder pair mapping
the visible space to the latent one (even GANs can be inverted, see Section 2.2.1). From this assumption, it is always possible to map an internal
representation in a space Z1 to the corresponding internal representation in a different space Z2 by
passing through the visible domain. This provides a supervised set of input/output pairs: we can try
to learn a direct map, as simple as possible. 

The astonishing fact is that a simple linear map gives
excellent results, in many situations. This is quite surprising, given that both encoder and decoder
functions are modeled by deep, non-linear transformations.

We tested mapping between latent spaces of:
- different training of a same model (Type 1)
- different generative models in a same class, e.g different VAEs (Type2)
- generative models with different learning obkectives, e.g. GAN vs. VAE (Type3)

In all cases, a linear map is enough to pass from a space to another preserving most of the information.

Some examples are provided below:

<p align="center"><img src="space1.png" width="500"/><p>
<p align="center"><img src="space_type2.png" width="500"/><p>
<p align="center"><img src="space_type3.png" width="500"/><p>

