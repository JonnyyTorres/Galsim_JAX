# Installing JAX (GPU version)

## First of all, we have to upgrade pip
pip install --upgrade pip

# Then we install JAX based on the CUDA version, in this case because it's CUDA 12.0 we use the following command: 
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Also we need to install FLAX in order to create NN with JAX
pip install --quiet flax

# Tensorflow.datasets is necessary to load the data
pip install tensorflow-datasets