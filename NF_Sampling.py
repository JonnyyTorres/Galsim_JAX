import tensorflow as tf
import os
import jax
import numpy as np
import sys
import wandb
import jax.numpy as jnp
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from jax.lib import xla_bridge
from galsim_jax.datasets import cosmos
from flax import linen as nn  # Linen API

from galsim_jax.dif_models import AutoencoderKLModule
from galsim_jax.utils import (
    load_checkpoint_wandb,
    plot_examples,
    get_git_commit_version,
)
from galsim_jax.convolution import convolve_kpsf
from galsim_jax.nf_models import (
    encode,
    decode,
    NeuralSplineFlowLogProb,
    NeuralSplineFlowSampler,
)

from absl import app
from absl import flags

flags.DEFINE_integer("gpu", 0, "Index of the GPU to use, e.g.: 0, 1, 2, etc.")
flags.DEFINE_string("project", "NF-results", "Name of the project, e.g.: 'NF-results'")
flags.DEFINE_string("name", "exp-1", "Name for the experiment, e.g.: 'exp-1'")

FLAGS = flags.FLAGS


def main(_):
    # Checking for GPU access
    print("Device: {}".format(xla_bridge.get_backend().platform))

    # Checking the GPU available
    gpus = jax.devices("gpu")
    print("Number of avaliable devices : {}".format(len(gpus)))

    # Ensure TF does not see GPU and grab all GPU memory.
    tf.config.set_visible_devices([], device_type="GPU")

    # Loading the dataset and transforming it to NumPy Arrays
    train_dset, info = tfds.load(name="Cosmos/25.2", with_info=True, split="train")

    # Input function to preprocess the data
    def input_fn(mode="train", batch_size=32):
        """
        mode: 'train' or 'test'
        """

        def preprocess_image(data):
            # Reshape 'psf' and 'image' to (128, 128, 1)
            data["kpsf_real"] = tf.expand_dims(data["kpsf_real"], axis=-1)
            data["kpsf_imag"] = tf.expand_dims(data["kpsf_imag"], axis=-1)
            data["image"] = tf.expand_dims(data["image"], axis=-1)
            return data

        if mode == "train":
            dataset = tfds.load("Cosmos/25.2", split="train")
            dataset = dataset.repeat()
            dataset = dataset.shuffle(10000)
        else:
            dataset = tfds.load("Cosmos/25.2", split="test")

        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.map(preprocess_image)  # Apply data preprocessing
        dataset = dataset.prefetch(
            -1
        )  # fetch next batches while training current one (-1 for autotune)
        return dataset

    # Dataset as a numpy iterator
    dset = input_fn().as_numpy_iterator()

    # Generating a random key for JAX
    rng, rng_2 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
    # Size of the input to initialize the encoder parameters
    batch_autoenc = jnp.ones((1, 128, 128, 1))

    latent_dim = 128
    act_fn = nn.gelu

    # Initializing the AutoEncoder
    Autoencoder = AutoencoderKLModule(
        ch_mult=(1, 2, 4, 8, 16),
        num_res_blocks=3,
        double_z=True,
        z_channels=1,
        resolution=latent_dim,
        in_channels=1,
        out_ch=1,
        ch=1,
        embed_dim=1,
        act_fn=act_fn,
    )

    params_auto = Autoencoder.init(rng, x=batch_autoenc, seed=rng_2)

    # Login to wandb
    wandb.login()

    # Initializing a Weights & Biases Run
    wandb.init(
        project=FLAGS.project,
        name=FLAGS.name,
    )

    # Setting the configs of our experiment using `wandb.config`.
    # This way, Weights & Biases automatically syncs the configs of
    # our experiment which could be used to reproduce the results of an experiment.
    config = wandb.config
    config.commit_version = get_git_commit_version()

    # Calling the Wandb API
    api = wandb.Api()
    # Calling Best VAE model with bottleneck size=64
    run = api.run("jonnyytorres/VAE-SD-NewELBO/ypziyp89")
    # Downloading artifacts from Wandb
    artifact = api.artifact(
        "jonnyytorres/VAE-SD-NewELBO/ypziyp89-checkpoint:best", type="model"
    )
    artifact_dir = artifact.download()

    # Loading checkpoint for the best step
    params_auto = load_checkpoint_wandb(run.id, "checkpoint.msgpack", params_auto)

    # Calling the encoder
    enc = encode(
        ch_mult=(1, 2, 4, 8, 16),
        num_res_blocks=3,
        double_z=True,
        z_channels=1,
        resolution=latent_dim,
        in_channels=1,
        out_ch=1,
        ch=1,
        embed_dim=1,
        act_fn=act_fn,
    )

    # Iterating over the TFDS dataset
    dataset_eval = input_fn("train")
    test_iterator = dataset_eval.as_numpy_iterator()
    batch = next(test_iterator)

    # Taking a batch of images
    x = batch["image"]
    # Applying the encoder
    posterior = enc.apply(params_auto, x)
    # Generating a random key for JAX
    rng, rng_2 = jax.random.split(rng_2)
    # Sampling over the posterior
    z = posterior.sample(seed=rng)

    # Plotting examples of the samplings
    plot_examples(z, "Samplings over Latent space", "Pixel value", "z_samples")

    # Plotting examples of original galaxies
    plot_examples(x, "Galaxies examples", "Pixel value", "original_galaxies")

    # Calling the decoder
    dec = decode(
        ch_mult=(1, 2, 4, 8, 16),
        num_res_blocks=3,
        double_z=True,
        z_channels=1,
        resolution=latent_dim,
        in_channels=1,
        out_ch=1,
        ch=1,
        embed_dim=1,
        act_fn=act_fn,
    )
    # Applying the decoder
    x_galaxie = dec.apply(params_auto, z)

    kpsf_real = batch["kpsf_real"]
    kpsf_imag = batch["kpsf_imag"]
    kpsf = kpsf_real + 1j * kpsf_imag
    std = batch["noise_std"].reshape((-1, 1, 1, 1))

    # Converting array into float32
    std = np.float32(std)

    # Generating a random key for JAX
    rng, rng_2 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
    rng, rng_1 = jax.random.split(rng)

    # Convolving the decoded image with the PSF
    x_convolve_ = jax.vmap(convolve_kpsf)(x_galaxie[..., 0], kpsf[..., 0])
    x_convolve_ = tf.expand_dims(x_convolve_, axis=-1)

    # Plotting examples of decoded galaxies
    plot_examples(x_convolve_, "Decoded Galaxies", "Pixel value", "decoded_galaxies")

    # Calling the NF model
    model = NeuralSplineFlowLogProb()

    # Random seed for initializing network and sampling training data
    seed = jax.random.PRNGKey(42)

    # Initializes the weights of the model
    params_nf = model.init(seed, jnp.zeros((1, 64)))

    # Calling the NF Sampler
    sampler = NeuralSplineFlowSampler()

    # Obtaining the id for the best NF experiment
    id_run = "53oggvli"

    # Downloading best checkpoint params
    run = api.run("jonnyytorres/NF_experiments/{}".format(id_run))
    artifact = api.artifact(
        "jonnyytorres/NF_experiments/{}-checkpoint:best".format(id_run), type="model"
    )
    artifact_dir = artifact.download()

    # Loading checkpoint for the best step
    params_nf = load_checkpoint_wandb(run.id, "checkpoint.msgpack", params_nf)

    # Applying the Sampler for 32 examples
    samps = sampler.apply(params_nf, jax.random.PRNGKey(1), 32)

    # Resahping to original bottleneck size
    samps_r = samps.reshape(-1, 8, 8, 1)

    # Applying the decoder to the samples of the learnt posterior
    q_nf = dec.apply(params_auto, samps_r)

    # Generating a random key for JAX
    rng, rng_2 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
    rng, rng_1 = jax.random.split(rng)

    # X estimated distribution
    x_convolve = jax.vmap(convolve_kpsf)(q_nf[..., 0], kpsf[..., 0])
    x_convolve = tf.expand_dims(x_convolve, axis=-1)

    # z3 = x_convolve

    # p3 = tfd.MultivariateNormalDiag(loc=p3, scale_diag=[0.01])

    # Plotting the decoded examples from the NF
    plot_examples(
        x_convolve, "Galaxies from Z_NF", "Pixel Value", "reconstructed_galaxies"
    )

    wandb.finish()


if __name__ == "__main__":
    # Parse the command-line flags
    app.FLAGS(sys.argv)

    # Set the CUDA_VISIBLE_DEVICES environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda-12.1"
    os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
    app.run(main)
