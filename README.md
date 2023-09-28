This repository aims to provide a code capable of creating a generative model of galaxies, making use of Variational AutoEncoders and Normalizing Flows. The model is inspired by the work carried out by [Lanusse, et al., 2021](https://arxiv.org/abs/2008.03833). The framework used for the models is [JAX](https://jax.readthedocs.io/en/latest/). 

## Installing JAX (GPU version)

First of all, we have to upgrade pip:
```
pip install --upgrade pip
```

Then we install JAX based on the CUDA version, in this case because it's CUDA 12.0 we use the following command: 
```
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Also we need to install FLAX in order to create NN with JAX:
```
pip install --quiet flax
```

Tensorflow_datasets is necessary to load the data:
```
pip install tensorflow-datasets
```

## Variational AutoEncoders (VAEs)

In order to learn an approximate probability distribution to characterize the morphology of galaxies, this deep generative model is used. The idea is to maximize the Evidence Lower Bound (ELBO), in order to maximize the marginal log-likelihood of galaxies. The details of this type of architecture are described in more detail by [Kingma and Welling, 2013](https://arxiv.org/abs/1312.6114)

Two approaches were performed: the first one based on the VAE proposed by [Lanusse, et al., 2021](https://arxiv.org/abs/2008.03833) and the second one based on the VAE used by the StableDiffusion model, proposed by [Rombach, et al., 2022](https://arxiv.org/abs/2112.10752). The codes to train each VAE are found in [VAE_Resnet_train_C.py](VAE_Resnet_train_C.py) and [VAE_SD_C.py](VAE_SD_C.py) respectively. 

## Normalizing Flows (NFs)

Once the VAE (both Encoder and Decoder) has been trained, the correct prior distribution is learned in order to create a purely generative model. Once the correct prior distribution is learned, the Decoder is used to automatically generate galaxy images. Notebook [NF_Galsim](notebooks/NF_Galsim.ipynb) presents the model used using NFs for this purpose. 

Likewise, the [NF_Sampling](NF_Sampling.py) code shows examples of samples obtained from the distribution learned using NF.

Finally, notebook [NF_Galsim_moments](notebooks/NF_Galsim_moments.ipynb) shows how to calculate some physical parameters of the galaxies obtained, such as ellipticity, to compare with the results of the COSMOS dataset and thus evaluate the performance of the galaxies created.  