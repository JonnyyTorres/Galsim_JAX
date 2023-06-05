import jax
import tensorflow_datasets as tfds
import hsc_photoz
import tensorflow as tf
import numpy as np
import jax.numpy as jnp
import optax
import wandb
import subprocess
import os
import msgpack

from jax.lib import xla_bridge
from astropy.stats import mad_std
from tensorflow_probability.substrates import jax as tfp
from flax import linen as nn  # Linen API
from jax import random
from typing import Optional
from flax.serialization import (
    to_state_dict, msgpack_serialize, from_bytes
)
from tqdm.auto import tqdm

#Checking for GPU access
print('Device: {}'.format(xla_bridge.get_backend().platform))

# Checking the GPU available
gpus = jax.devices('gpu')
print('Number of avaliable devices : {}'.format(len(gpus)))

# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU')

# Loading the dataset and transforming it to NumPy Arrays
train_dset, info = tfds.load(name="hsc_photoz", with_info=True, split='train')

# What's in our dataset:
# info

# Let's collect a few examples to check their distributions
cutouts=[]
specz = []
for (batch, entry) in enumerate(train_dset.take(1000)):
    specz.append(entry['attrs']['specz_redshift'])
    cutouts.append(entry['image'])

cutouts = np.stack(cutouts)
specz = np.stack(specz)

scaling = []

for i,b in enumerate(['g', 'r', 'i', 'z', 'y']):
    sigma = mad_std(cutouts[...,i].flatten()) # Capturing the std devation of each band
    scaling.append(sigma)

# Using a mapping function to apply preprocessing to our data
def preprocessing(example):
    img = tf.math.asinh(example['image'] / tf.constant(scaling) / 3. )
    # We return the image as our input and output for a generative model
    return img

def input_fn(mode='train', batch_size=64):
    """
    mode: 'train' or 'test'
    """
    if mode == 'train':
        dataset = tfds.load('hsc_photoz', split='train[:80%]')
        dataset = dataset.repeat()
        dataset = dataset.shuffle(10000)
    else:
        dataset = tfds.load('hsc_photoz', split='train[80%:]')
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(preprocessing) # Apply data preprocessing
    dataset = dataset.prefetch(-1) # fetch next batches while training current one (-1 for autotune)
    return dataset

# Dataset as a numpy iterator
dset = input_fn().as_numpy_iterator()

# Loading distributions and bijectors from TensorFlow Probability (JAX version)
tfd = tfp.distributions
tfb = tfp.bijectors

class ResNetBlock(nn.Module):
    """Creates a block of a CNN with ResNet architecture to encode images."""
    act_fn : callable  # Activation function
    c_out : int   # Output feature size
    subsample : bool = False  # If True, we apply a stride inside F

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        print("Input X Resnet", x.shape)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    padding='SAME',
                    strides=(2, 2))(x)
        print("Z Resnet", z.shape)
        z = self.act_fn(z)
        x = nn.Conv(self.c_out, kernel_size=(3, 3),
                    padding='SAME',
                    strides=(2, 2))(x)
        print("Sum X Resnet", x.shape)
        x_out = self.act_fn(z + x)
        return x_out

class ResNetEnc(nn.Module):
    """"Creates a small convolutional encoder using ResNet blocks as intermediate layers"""
    act_fn : callable
    block_class : nn.Module
    num_blocks : tuple = (1, 1, 1)
    c_hidden : tuple = (64, 128, 256)
    latent_dim : int = 64
    prob_output : bool = True # If True, the output will be an image instead of a 
                               # probability distribution

    @nn.compact
    def __call__(self, x, train=True):
        # A first convolution on the original image to scale up the channel size
        print(x.shape)
        x = nn.Conv(self.latent_dim, kernel_size=(3, 3), padding='SAME', strides=(2, 2))(x)
        x = self.act_fn(x)
        print(x.shape)
        
        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = (bc == 0 and block_idx > 0)
                # ResNet block
                x = self.block_class(c_out=self.c_hidden[block_idx],
                                     act_fn=self.act_fn,
                                     subsample=subsample)(x, train=train)
       
        if self.prob_output:
            net = nn.Dense(features=self.latent_dim*2)(x)
            # Image is now 4x4x128
            print("Dense shape", net.shape, "\n")
            
            q = tfd.MultivariateNormalDiag(loc=net[..., :self.latent_dim], 
                               scale_diag=net[..., self.latent_dim:])
        else:
            net = nn.Dense(features=self.latent_dim)(x)
            print("Dense shape", net.shape, "\n")
            # Image is now 4x4x64
            q = net        
        
        return q
    
prob_output = False

Encoder = ResNetEnc(act_fn=nn.leaky_relu, block_class=ResNetBlock, prob_output=prob_output)

# Generating a random key for JAX
rng = random.PRNGKey(0)
# Size of the input to initialize the parameters
batch_enc = jnp.ones((1, 64, 64, 5))
# Initializing the VAE
params = Encoder.init(rng, batch_enc)

# Taking 64 images of the dataset
batch_im = next(dset)
# Generating new keys to use them for inference
rng_1, rng_2 = random.split(rng)

z = Encoder.apply(params, batch_im)

# print(z)

class ResNetBlockD(nn.Module):
    """Creates a block of a CNN with ResNet architecture to decode images."""
    act_fn : callable  # Activation function
    c_out : int   # Output feature size
    subsample : bool = False  # If True, we apply a stride inside F

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        print("Input X Resnet", x.shape)
        z = nn.ConvTranspose(self.c_out, kernel_size=(3, 3),
                    padding='SAME',
                    strides=(2, 2))(x)
        print("Z Resnet", z.shape)
        z = self.act_fn(z)
        x = nn.ConvTranspose(self.c_out, kernel_size=(3, 3),
                    padding='SAME',
                    strides=(2, 2))(x)
        print("Sum X Resnet", x.shape)
        x_out = self.act_fn(z + x)
        return x_out

class ResNetDec(nn.Module):
    """"Creates a small convolutional decoder using ResNet blocks as intermediate layers"""
    act_fn : callable
    block_class : nn.Module
    num_blocks : tuple = (1, 1, 1, 1)
    c_hidden : tuple = (128, 64, 32, 5)
    # num_channels : int = 5

    @nn.compact
    def __call__(self, x, train=True):
        
        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = (bc == 0 and block_idx > 0)
                # ResNet block
                x = self.block_class(c_out=self.c_hidden[block_idx],
                                     act_fn=self.act_fn,
                                     subsample=subsample)(x, train=train)

        x = nn.activation.softplus(x)
        # Image is now 64x64x5
        r =tfd.MultivariateNormalDiag(loc=x,
                                      scale_diag=[0.01, 0.01, 0.01, 0.01, 0.01])

        return r

Decoder = ResNetDec(act_fn=nn.leaky_relu, block_class=ResNetBlockD)

# print(z)

# Generating a random key for JAX
rng = random.PRNGKey(0)

# Generating new keys to use them for inference
rng_1, rng_2 = random.split(rng)

if prob_output:
    code_sample = z.sample(seed=rng_2) 

else:
    code_sample = z
    
# Size of the input to initialize the parameters
batch_dec = jnp.ones((1, 4, 4, 64))

# Initializing the resnet_dec
params = Decoder.init(rng_2, batch_dec)

# Decoding the image
decoded_img = Decoder.apply(params, code_sample)

# print(decoded_img)


# Generating a random key for JAX
rng = random.PRNGKey(0)
# Size of the input to initialize the parameters
batch_enc = jnp.ones((1, 64, 64, 5))
# Initializing the VAE
Encoder = ResNetEnc(act_fn=nn.leaky_relu, block_class=ResNetBlock, prob_output=prob_output)
params_enc = Encoder.init(rng, batch_enc)

# Taking 64 images of the dataset
batch_im = next(dset)
# Generating new keys to use them for inference
rng, rng_1 = random.split(rng)

# Size of the input to initialize the parameters
batch_dec = jnp.ones((1, 4, 4, 64))

# Initializing the resnet_dec
Decoder = ResNetDec(act_fn=nn.leaky_relu, block_class=ResNetBlockD)
params_dec = Decoder.init(rng, batch_dec)

@jax.jit
def loss_fn(params, rng_key, batch, kl_reg_w): #state, rng_key, batch):
    
    params_enc, params_dec = params
    
    x = batch

    # Autoencode an example
    q = Encoder.apply(params_enc, x)
    
    # # Sample from the posterior
    # z = q.sample(seed=rng_key)
    
#     # Apply or not KL divergence
#     p = lax.cond(kl_reg_w > 0, prob_branch, pure_branch, (rng_key, q, params_dec), (q, params_dec))
    
    # Decode the sample
    p = Decoder.apply(params_dec, q)

    # KL divergence between the prior distribution and p
    kl = tfd.kl_divergence(p, tfd.MultivariateNormalDiag(jnp.zeros((1, 64, 64, 5))))
    
    # Compute log-likelihood
    log_likelihood = p.log_prob(x)
    
    # Calculating the ELBO value
    elbo = log_likelihood - kl_reg_w*0.0001*kl # Here we apply a regularization factor on the KL term
    
    loss = -elbo.mean()
    return loss

# Apply Regularization
if prob_output:
    kl_reg_w = 1
else:
    kl_reg_w = 0

# Defining a general list of the parameters
params = [params_enc, params_dec]
# Veryfing that the 'value_and_grad' works fine
loss, grads = jax.value_and_grad(loss_fn)(params, rng, batch_im, kl_reg_w)

def lr_schedule(step):
    """Linear scaling rule optimized for 90 epochs."""
    steps_per_epoch = 40000 // 64 

    current_epoch = step / steps_per_epoch  # type: float
    boundaries = jnp.array((40, 80, 120)) * steps_per_epoch
    values = jnp.array([1., 0.1, 0.01, 0.001])

    index = jnp.sum(boundaries < step)
    return jnp.take(values, index)


optimizer = optax.chain(
      optax.adam(1e-3),
      optax.scale_by_schedule(lr_schedule))

opt_state = optimizer.init(params)

@jax.jit
def update(params, rng_key, opt_state, batch):
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(loss_fn)(params, rng_key, batch, kl_reg_w)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state 

loss, params, opt_state = update(params, rng, opt_state, batch_im)

# Login to wandb
wandb.login()

# Initializing a Weights & Biases Run
wandb.init(
    project="galsim-jax-resnet",
    name="first-model",
)

# Setting the configs of our experiment using `wandb.config`.
# This way, Weights & Biases automatcally syncs the configs of 
# our experiment which could be used to reproduce the results of an experiment.
config = wandb.config
config.seed = 42
config.batch_size = 64
# config.validation_split = 0.2
# config.pooling = "avg"
config.learning_rate = 1e-3
config.epochs = 25000

def get_git_commit_version():
    """"Allows to get the Git commit version to tag each experiment"""
    try:
        commit_version = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        return commit_version
    except subprocess.CalledProcessError:
        return None

# # Call the function to get the Git commit version
# commit_version = get_git_commit_version()

# if commit_version:
#     print("Git commit version:", commit_version)
# else:
#     print("Unable to get Git commit version.")


def save_checkpoint(ckpt_path, state, epoch):
    """Saves a Wandb checkpoint."""
    with open(ckpt_path, "wb") as outfile:
        outfile.write(msgpack_serialize(to_state_dict(state)))
    artifact = wandb.Artifact(
        f'{wandb.run.name}-checkpoint', type='model'
    )
    artifact.add_file(ckpt_path)
    wandb.log_artifact(artifact, aliases=["best", f"epoch_{epoch}", f"commit_{get_git_commit_version()}"])


def load_checkpoint(ckpt_file, state):
    """Loads the best Wandb checkpoint."""
    artifact = wandb.use_artifact(
        f'{wandb.run.name}-checkpoint:best'
    )
    artifact_dir = artifact.download()
    ckpt_path = os.path.join(artifact_dir, ckpt_file)
    with open(ckpt_path, "rb") as data_file:
        byte_data = data_file.read()
    return from_bytes(state, byte_data)


losses = []
losses_test = []
losses_test_epoch = []
best_eval_loss = 1e6

kl_reg = False

# Train the model as many epochs as indicated initially
for epoch in tqdm(range(1, config.epochs + 1)):
    rng, rng_1 = random.split(rng)
    batch_im = next(dset)
    loss, params, opt_state = update(params, rng_1, opt_state, batch_im)
    losses.append(loss)
     
    # Log metrics inside your training loop to visualize model performance
    wandb.log({ 
        "loss": loss,
        }, step=epoch)
    
    # Saving checkpoint
    if loss < best_eval_loss:
        best_eval_loss = loss
        save_checkpoint("checkpoint.msgpack", params, epoch)
    
    # Calculating the loss for all the test images 
    if epoch % 2500 == 0 :

        dataset_eval = input_fn('test')
        test_iterator = dataset_eval.as_numpy_iterator()

        for_list_mean = []

        for img in test_iterator:
            rng, rng_1 = random.split(rng)
            loss_test = loss_fn(params, rng_1, img, kl_reg_w)
            for_list_mean.append(loss_test)

        losses_test.append(np.mean(for_list_mean))
        losses_test_epoch.append(epoch)
        
        wandb.log({ 
        "test_loss": losses_test[-1],
        }, step=epoch)
            
        print("Epoch: {}, loss: {:.2f}, loss test: {:.2f}".format(epoch, loss, losses_test[-1]))
        
params = load_checkpoint("checkpoint.msgpack", params)
            
loss_min = min(losses)
best_epoch = losses.index(loss_min) + 1

print("\nBest Epoch: {}, loss: {:.2f}".format(best_epoch, loss_min))

wandb.finish()