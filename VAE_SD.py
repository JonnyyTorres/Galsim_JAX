import jax
import tensorflow_datasets as tfds

import tensorflow as tf
import numpy as np
import jax.numpy as jnp
import optax
import wandb
import os

import logging
from galsim_jax.dif_models import AutoencoderKLModule
from galsim_jax.utils import (
    save_checkpoint,
    load_checkpoint,
    get_wandb_local_dir,
    create_folder,
    save_plot_as_image,
    save_samples,
    get_git_commit_version,
    get_activation_fn,
    get_optimizer,
    new_optimizer,
)

from jax.lib import xla_bridge
from astropy.stats import mad_std
from tensorflow_probability.substrates import jax as tfp
from flax import linen as nn  # Linen API
from jax import random

from tqdm.auto import tqdm
from absl import app
from absl import flags

# logging.getLogger("tfds").setLevel(logging.ERROR)

# flags.DEFINE_string("input_folder", "/data/tensorflow_datasets/", "Location of the input images")
flags.DEFINE_string("dataset", "hsc_photoz", "Suite of simulations to learn from")
# flags.DEFINE_string("output_dir", "./weights/gp-sn1v5", "Folder where to store model.")
flags.DEFINE_integer("batch_size", 64, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 50000, "Number of training steps to run.")
# flags.DEFINE_string("train_split", "90%", "How much of the training set to use.")
# flags.DEFINE_boolean('prob_output', True, 'The encoder has or not a probabilistic output')
flags.DEFINE_float("reg_value", 1e-6, "Regularization value of the KL Divergence.")
flags.DEFINE_integer("gpu", 0, "Index of the GPU to use, e.g.: 0, 1, 2, etc.")
flags.DEFINE_string(
    "experiment", "model_1", "Type of experiment, e.g. 'model_1', 'model_2', etc."
)
flags.DEFINE_string("project", "VAE-SD", "Name of the project, e.g.: 'VAE-SD'")
flags.DEFINE_string(
    "name", "first-model", "Name for the experiment, e.g.: 'dim_64_kl_0.01'"
)
flags.DEFINE_string(
    "act_fn", "gelu", "Activation function, e.g.: 'gelu', 'leaky_relu', etc."
)
flags.DEFINE_string("opt", "adam", "Optimizer, e.g.: 'adam', 'adamw'")
flags.DEFINE_integer("resblocks", 2, "Number of resnet blocks.: 1, 2.")
flags.DEFINE_integer("step_sch", 40000, "Steps for the lr_schedule")

FLAGS = flags.FLAGS

# Loading distributions and bijectors from TensorFlow Probability (JAX version)
tfd = tfp.distributions
tfb = tfp.bijectors


def main(_):
    # Set the CUDA_VISIBLE_DEVICES environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"

    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    # config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # os.chdir(FLAGS.input_folder)

    # Checking for GPU access
    print("Device: {}".format(xla_bridge.get_backend().platform))

    # Checking the GPU available
    gpus = jax.devices("gpu")
    print("Number of avaliable devices : {}".format(len(gpus)))

    # Ensure TF does not see GPU and grab all GPU memory.
    tf.config.set_visible_devices([], device_type="GPU")

    device = gpus[FLAGS.gpu]  # Select the GPU device of interest
    print("Device used: {}".format(device))

    # Loading the dataset and transforming it to NumPy Arrays
    train_dset, info = tfds.load(name=FLAGS.dataset, with_info=True, split="train")

    # What's in our dataset:
    # info

    # Let's collect a few examples to check their distributions
    cutouts = []
    specz = []
    for entry in train_dset.take(1000):
        specz.append(entry["attrs"]["specz_redshift"])
        cutouts.append(entry["image"])

    cutouts = np.stack(cutouts)
    specz = np.stack(specz)

    scaling = []

    for i, _ in enumerate(["g", "r", "i", "z", "y"]):
        sigma = mad_std(
            cutouts[..., i].flatten()
        )  # Capturing the std devation of each band
        scaling.append(sigma)

    # Using a mapping function to apply preprocessing to our data
    def preprocessing(example):
        img = tf.math.asinh(example["image"] / tf.constant(scaling) / 3.0)
        # We return the image as our input and output for a generative model
        return img

    def input_fn(mode="train", batch_size=FLAGS.batch_size):
        """
        mode: 'train' or 'test'
        """
        if mode == "train":
            dataset = tfds.load(FLAGS.dataset, split="train[:80%]")
            dataset = dataset.repeat()
            dataset = dataset.shuffle(10000)
        else:
            dataset = tfds.load(FLAGS.dataset, split="train[80%:]")

        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.map(preprocessing)  # Apply data preprocessing
        dataset = dataset.prefetch(
            -1
        )  # fetch next batches while training current one (-1 for autotune)
        return dataset

    # Dataset as a numpy iterator
    dset = input_fn().as_numpy_iterator()

    # Generating a random key for JAX
    rng, rng_2 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
    # Size of the input to initialize the encoder parameters
    batch_autoenc = jnp.ones((1, 64, 64, 5))

    """if FLAGS.experiment == "model_1":
        latent_dim = 64
        c_hidden_enc = (64, 128, 256)
        num_blocks_enc = (1, 1, 1)
        c_hidden_dec = (128, 64, 32, 5)
        num_blocks_dec = (1, 1, 1, 1)

        # Size of the input to initialize the decoder parameters
        batch_dec = jnp.ones((1, 4, 4, 64))

    if FLAGS.experiment == "model_2":
        latent_dim = 32
        c_hidden_enc = (32, 64, 128, 256)
        num_blocks_enc = (1, 1, 1, 1)
        c_hidden_dec = (64, 32, 16, 8, 5)
        num_blocks_dec = (1, 1, 1, 1, 1)

        # Size of the input to initialize the decoder parameters
        batch_dec = jnp.ones((1, 2, 2, 32))

    if FLAGS.experiment == "model_3":
        latent_dim = 16
        c_hidden_enc = (16, 32, 64, 128, 256)
        num_blocks_enc = (1, 1, 1, 1, 1)
        c_hidden_dec = (32, 16, 8, 4, 2, 5)
        num_blocks_dec = (1, 1, 1, 1, 1, 1)

        # Size of the input to initialize the decoder parameters
        batch_dec = jnp.ones((1, 1, 1, 16))"""

    """if FLAGS.experiment == "model_1":
        latent_dim = 64
        # Size of the input to initialize the decoder parameters
        batch_dec = jnp.ones((1, 4, 4, 64))

    if FLAGS.experiment == "model_2":
        latent_dim = 32
        # Size of the input to initialize the decoder parameters
        batch_dec = jnp.ones((1, 4, 4, 32))

    if FLAGS.experiment == "model_3":
        latent_dim = 16
        # Size of the input to initialize the decoder parameters
        batch_dec = jnp.ones((1, 4, 4, 16))

    if FLAGS.experiment == "model_4":
        latent_dim = 64
        c_hidden_enc = (32, 64, 128, 256)
        num_blocks_enc = (1, 1, 1, 1)
        c_hidden_dec = (128, 64, 32, 16, 5)
        num_blocks_dec = (1, 1, 1, 1, 1)
        # Size of the input to initialize the decoder parameters
        batch_dec = jnp.ones((1, 2, 2, 64))"""

    latent_dim = 64
    act_fn = get_activation_fn(FLAGS.act_fn)

    # Initializing the AutoEncoder
    Autoencoder = AutoencoderKLModule(
        ch_mult=(1, 2, 4),
        num_res_blocks=FLAGS.resblocks,
        double_z=True,
        z_channels=5,
        resolution=latent_dim,
        in_channels=5,
        out_ch=5,
        ch=5,
        embed_dim=5,
        act_fn=act_fn,
    )

    params = Autoencoder.init(rng, x=batch_autoenc, seed=rng_2)

    # Taking 64 images of the dataset
    batch_im = next(dset)
    # Generating new keys to use them for inference
    rng_1, rng_2 = jax.random.split(rng_2)

    # # Initializing the Decoder
    # Decoder = ResNetDec(
    #     act_fn=nn.leaky_relu,
    #     block_class=ResNetBlock,
    #     c_hidden=c_hidden_dec,
    #     num_blocks=num_blocks_dec,
    # )
    # params_dec = Decoder.init(rng_1, batch_dec)

    # # Defining a general list of the parameters
    # params = [params_enc, params_dec]

    # Initialisation
    optimizer = get_optimizer(FLAGS.opt, FLAGS.learning_rate, FLAGS.step_sch)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, rng_key, batch, reg_term):  # state, rng_key, batch):
        """Function to define the loss function"""

        x = batch

        # Autoencode an example
        q = Autoencoder.apply(params, x=x, seed=rng_key)

        # # Sample from the posterior
        # z = q.sample(seed=rng_key)

        # # Decode the sample
        # p = Decoder.apply(params_dec, z)

        # KL divergence between the prior distribution and p
        kl = tfd.kl_divergence(q, tfd.MultivariateNormalDiag(jnp.zeros((1, 64, 64, 5))))

        # Compute log-likelihood
        log_likelihood = q.log_prob(x)

        # Calculating the ELBO value
        elbo = (
            log_likelihood - reg_term * kl
        )  # Here we apply a regularization factor on the KL term

        loss = -elbo.mean()
        return loss, -log_likelihood.mean()

    """    # Veryfing that the 'value_and_grad' works fine
    loss, grads = jax.value_and_grad(loss_fn)(params, rng, batch_im, kl_reg_w)
    """

    @jax.jit
    def update(params, rng_key, opt_state, batch):
        """Single SGD update step."""
        (loss, log_likelihood), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, rng_key, batch, FLAGS.reg_value
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, log_likelihood, new_params, new_opt_state

    """loss, log_likelihood, params, opt_state = update(params, rng_1, opt_state, batch_im)"""

    # Login to wandb
    wandb.login()

    # Initializing a Weights & Biases Run
    wandb.init(
        project=FLAGS.project,
        name=FLAGS.name,
        # tags="kl_reg={:.4f}".format(reg),
    )

    # Setting the configs of our experiment using `wandb.config`.
    # This way, Weights & Biases automatcally syncs the configs of
    # our experiment which could be used to reproduce the results of an experiment.
    config = wandb.config
    config.seed = 42
    config.batch_size = FLAGS.batch_size
    # config.validation_split = 0.2
    # config.pooling = "avg"
    config.learning_rate = FLAGS.learning_rate
    config.steps = FLAGS.training_steps
    config.kl_reg = FLAGS.reg_value
    config.using_kl = False if FLAGS.reg_value == 0 else True
    config.latent_dim = latent_dim
    config.type_model = FLAGS.experiment
    config.commit_version = get_git_commit_version()
    config.act_fn = FLAGS.act_fn
    config.opt = FLAGS.opt
    config.resnet_blocks = FLAGS.resblocks
    config.steps_schedule = FLAGS.step_sch
    # config.lr_method = "Warmup Cosine"
    config.interpolation = "Bicubic"

    # Define the metrics we are interested in the minimum of
    wandb.define_metric("loss", summary="min")
    wandb.define_metric("log_likelihood", summary="min")
    wandb.define_metric("test_loss", summary="min")
    wandb.define_metric("test_log_likelihood", summary="min")

    losses = []
    losses_test = []
    losses_test_step = []

    log_liks = []
    log_liks_test = []

    best_eval_loss = 1e6

    # Train the model as many steps as indicated initially
    for step in tqdm(range(1, config.steps + 1)):
        rng, rng_1 = random.split(rng)
        # Iterating over the dataset
        batch_im = next(dset)
        loss, log_likelihood, params, opt_state = update(
            params, rng_1, opt_state, batch_im
        )
        losses.append(loss)
        log_liks.append(log_likelihood)

        # Log metrics inside your training loop to visualize model performance
        wandb.log(
            {
                "loss": loss,
                "log_likelihood": log_likelihood,
            },
            step=step,
        )

        # Saving best checkpoint
        if loss < best_eval_loss:
            best_eval_loss = loss
            save_checkpoint("checkpoint.msgpack", params, step)

        # Calculating the loss for all the test images
        if step % (config.steps // 10) == 0:
            dataset_eval = input_fn("test")
            test_iterator = dataset_eval.as_numpy_iterator()

            for_list_mean = []

            for img in test_iterator:
                rng, rng_1 = random.split(rng)
                loss_test, log_likelihood_test = loss_fn(
                    params, rng_1, img, FLAGS.reg_value
                )
                for_list_mean.append(loss_test)

            losses_test.append(np.mean(for_list_mean))
            losses_test_step.append(step)
            log_liks_test.append(log_likelihood_test)

            wandb.log(
                {
                    "test_loss": losses_test[-1],
                    "test_log_likelihood": log_liks_test[-1],
                },
                step=step,
            )

            print(
                "Step: {}, loss: {:.2f}, loss test: {:.2f}".format(
                    step, loss, losses_test[-1]
                )
            )

    # Loading checkpoint for the best step
    params = load_checkpoint("checkpoint.msgpack", params)

    # Obtaining the step with the lowest loss value
    loss_min = min(losses)
    best_step = losses.index(loss_min) + 1
    print("\nBest Step: {}, loss: {:.2f}".format(best_step, loss_min))

    # Obtaining the step with the lowest log-likelihood value
    log_lik_min = min(log_liks)
    best_step_log = log_liks.index(log_lik_min) + 1
    print("\nBest Step: {}, log-likelihood: {:.2f}".format(best_step_log, log_lik_min))

    best_steps = {
        "best_step_loss": best_step,
        "best_step_log_lik": best_step_log,
    }

    wandb.log(best_steps)

    total_steps = np.arange(1, config.steps + 1)

    # Creating the 'results' folder to save all the plots as images (or validating that the folder already exists)
    results_folder = "results/{}".format(get_wandb_local_dir(wandb.run.dir))
    create_folder(results_folder)

    # Saving the loss plots
    save_plot_as_image(
        folder_path=results_folder,
        plot_title="Loglog of the Loss function - Train (KL reg value = {})".format(
            FLAGS.reg_value
        ),
        x_data=total_steps,
        y_data=losses,
        plot_type="loglog",
        file_name="loglog_loss.png",
    )
    save_plot_as_image(
        folder_path=results_folder,
        plot_title="Loglog of the Loss function - Test (KL reg value = {})".format(
            FLAGS.reg_value
        ),
        x_data=losses_test_step,
        y_data=losses_test,
        plot_type="loglog",
        file_name="loglog_loss_test.png",
    )

    # Saving the log-likelihood plots
    save_plot_as_image(
        folder_path=results_folder,
        plot_title="Loglog of the Log-likelihood - Train (KL reg value = {})".format(
            FLAGS.reg_value
        ),
        x_data=total_steps,
        y_data=log_liks,
        plot_type="loglog",
        file_name="loglog_log_likelihood.png",
    )
    save_plot_as_image(
        folder_path=results_folder,
        plot_title="Loglog of the Log-likelihood - Test (KL reg value = {})".format(
            FLAGS.reg_value
        ),
        x_data=losses_test_step,
        y_data=log_liks_test,
        plot_type="loglog",
        file_name="loglog_log_likelihood_test.png",
    )

    # Predicting over an example of data
    dataset_eval = input_fn("test")
    test_iterator = dataset_eval.as_numpy_iterator()
    batch = next(test_iterator)
    # Taking 16 images as example
    batch = batch[:16, ...]

    """# Dividing the list of parameters obtained before
    params_enc, params_dec = params
    # Distribution of latent space calculated using the batch of data
    q = ResNetEnc(
        act_fn=nn.leaky_relu,
        block_class=ResNetBlock,
        latent_dim=latent_dim,
        c_hidden=c_hidden_enc,
        num_blocks=num_blocks_enc,
    ).apply(params_enc, batch)
    # Sampling from the distribution
    z = q.sample(seed=rng_1)"""

    rng, rng_1 = random.split(rng)
    # X estimated distribution
    q = Autoencoder.apply(params, x=batch, seed=rng_1)
    # Sample some variables from the posterior distribution
    rng, rng_1 = random.split(rng)
    z = q.sample(seed=rng_1)

    # Saving the samples of the predicted images and their difference from the original images
    save_samples(folder_path=results_folder, z=z, batch=batch)

    wandb.finish()


if __name__ == "__main__":
    app.run(main)
