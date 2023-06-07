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
import logging
import matplotlib.pyplot as plt
import datetime

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
from absl import app
from absl import flags
# from jax import lax

logging.getLogger('tfds').setLevel(logging.ERROR)

# flags.DEFINE_string("input_folder", "/data/tensorflow_datasets/", "Location of the input images")
flags.DEFINE_string("dataset", "hsc_photoz", "Suite of simulations to learn from")
# flags.DEFINE_string("output_dir", "./weights/gp-sn1v5", "Folder where to store model.")
flags.DEFINE_integer("batch_size", 64, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 25000, "Number of training steps to run.")
# flags.DEFINE_string("train_split", "90%", "How much of the training set to use.")
flags.DEFINE_boolean('prob_output', True, 'The encoder has or not a probabilistic output')


FLAGS = flags.FLAGS

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
    # prob_output : bool = True # If True, the output will be an image instead of a 
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
       

        net = nn.Dense(features=self.latent_dim*2)(x)
        # Image is now 4x4x128
        print("Dense shape", net.shape, "\n")
        
        q = tfd.MultivariateNormalDiag(loc=net[..., :self.latent_dim], 
                            scale_diag=net[..., self.latent_dim:])
        
        return q
    

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
    
def create_folder(folder_path='results'):
    try:
        # Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created: {folder_path}")
        else:
            # Folder already exists
            print("Folder already exists!")
    except Exception as e:
        print(f"Error creating folder: {str(e)}")

def lr_schedule(step):
    """Linear scaling rule optimized for 90 epochs."""
    steps_per_epoch = 40000 // 64 

    current_epoch = step / steps_per_epoch  # type: float
    boundaries = jnp.array((40, 80, 120)) * steps_per_epoch
    values = jnp.array([1., 0.1, 0.01, 0.001])

    index = jnp.sum(boundaries < step)
    return jnp.take(values, index)


def get_git_commit_version():
    """"Allows to get the Git commit version to tag each experiment"""
    try:
        commit_version = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        return commit_version
    except subprocess.CalledProcessError:
        return None

def save_checkpoint(ckpt_path, state, step):
    """Saves a Wandb checkpoint."""
    with open(ckpt_path, "wb") as outfile:
        outfile.write(msgpack_serialize(to_state_dict(state)))
    artifact = wandb.Artifact(
        f'{wandb.run.id}-checkpoint', type='model'
    )
    artifact.add_file(ckpt_path)
    wandb.log_artifact(artifact, aliases=["best", f"step_{step}", f"commit_{get_git_commit_version()}"])


def load_checkpoint(ckpt_file, state):
    """Loads the best Wandb checkpoint."""
    artifact = wandb.use_artifact(
        f'{wandb.run.id}-checkpoint:best'
    )
    artifact_dir = artifact.download()
    ckpt_path = os.path.join(artifact_dir, ckpt_file)
    with open(ckpt_path, "rb") as data_file:
        byte_data = data_file.read()
    return from_bytes(state, byte_data)

def save_plot_as_image(folder_path, plot_title, x_data, y_data, plot_type='loglog', file_name='plot.png', **kwargs):
    # Generate plot based on plot_type
    if plot_type == 'line':
        plt.plot(x_data, y_data, **kwargs)
    elif plot_type == 'loglog':
        plt.loglog(x_data, y_data, **kwargs)
    elif plot_type == 'semilogy':
        plt.semilogy(x_data, y_data, **kwargs)
    elif plot_type == 'semilogx':
        plt.semilogx(x_data, y_data, **kwargs)
    elif plot_type == 'scatter':
        plt.scatter(x_data, y_data, **kwargs)
    else:
        print("Invalid plot type!")
        return

    plt.title(plot_title)
    plt.xlabel('Step')
    plt.ylabel('Value')

    # Save plot as image within the folder
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path)
    wandb.log({"{}".format(file_name.split('.')[0]): wandb.Image(plt)})
    plt.close()

    print(f"Plot saved as {file_path}")

def save_samples(folder_path, z, batch):
    # Plotting the original and estimated image for 15 examples
    plt.figure(figsize=(6,2))
    for i in range(15):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9,3))
        ax1.imshow(batch[i,...].mean(axis=-1))
        ax1.axis('off')
        ax2.imshow(z[i,...].mean(axis=-1))
        ax2.axis('off')
        ax3.imshow(z[i,...].mean(axis=-1) - batch[i,...].mean(axis=-1))
        ax3.axis('off')
        fig.suptitle('Comparison between original and predicted images', fontsize=12)
    # Save plot as image within the folder
    file_path = os.path.join(folder_path, "difference_pred.png")
    plt.savefig(file_path)
    wandb.log({"difference_pred": wandb.Image(plt)})
    plt.close()

    print(f"Plot saved as {file_path}")

    # 16 images of the estimated shape of galaxies
    plt.figure(figsize=(10,10))
    for i in range(4):
        for j in range(4):
            plt.subplot(4,4,i+4*j+1)
            plt.imshow(z[i+4*j,...].mean(axis=-1))
            plt.axis('off')
    plt.title('Samples of predicted galaxies', fontsize=12)

    # Save plot as image within the folder
    file_path = os.path.join(folder_path, "samples_pred.png")
    plt.savefig(file_path)
    wandb.log({"samples_pred": wandb.Image(plt)})
    plt.close()

    print(f"Plot saved as {file_path}")

def get_date_time():
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    return formatted_datetime


def main(_):

    # os.chdir(FLAGS.input_folder)
    
    # Checking for GPU access
    print('Device: {}'.format(xla_bridge.get_backend().platform))

    # Checking the GPU available
    gpus = jax.devices('gpu')
    print('Number of avaliable devices : {}'.format(len(gpus)))

    # Ensure TF does not see GPU and grab all GPU memory.
    tf.config.set_visible_devices([], device_type='GPU')

    # Loading the dataset and transforming it to NumPy Arrays
    train_dset, info = tfds.load(name=FLAGS.dataset, with_info=True, split='train')

    # What's in our dataset:
    # info

    # Let's collect a few examples to check their distributions
    cutouts=[]
    specz = []
    for entry in train_dset.take(1000):
        specz.append(entry['attrs']['specz_redshift'])
        cutouts.append(entry['image'])

    cutouts = np.stack(cutouts)
    specz = np.stack(specz)

    scaling = []

    for i,_ in enumerate(['g', 'r', 'i', 'z', 'y']):
        sigma = mad_std(cutouts[...,i].flatten()) # Capturing the std devation of each band
        scaling.append(sigma)

    # Using a mapping function to apply preprocessing to our data
    def preprocessing(example):
        img = tf.math.asinh(example['image'] / tf.constant(scaling) / 3. )
        # We return the image as our input and output for a generative model
        return img

    def input_fn(mode='train', batch_size=FLAGS.batch_size):
        """
        mode: 'train' or 'test'
        """
        if mode == 'train':
            dataset = tfds.load(FLAGS.dataset, split='train[:80%]')
            dataset = dataset.repeat()
            dataset = dataset.shuffle(10000)
        else:
            dataset = tfds.load(FLAGS.dataset, split='train[80%:]')
        
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.map(preprocessing) # Apply data preprocessing
        dataset = dataset.prefetch(-1) # fetch next batches while training current one (-1 for autotune)
        return dataset
    
    # Dataset as a numpy iterator
    dset = input_fn().as_numpy_iterator()

    # Defining if we want a probabilistic output or not
    prob_output = FLAGS.prob_output

    # Defining KL regularization values based in prob_output
    kl_reg = [0.1**(x+1) for x in range(4)] if prob_output else [0]

    # Running the model as much times as KL regularization values
    for reg in kl_reg:

        # Generating a random key for JAX
        rng = random.PRNGKey(0)
        # Size of the input to initialize the parameters
        batch_enc = jnp.ones((1, 64, 64, 5))

        # Initializing the Encoder
        Encoder = ResNetEnc(act_fn=nn.leaky_relu, block_class=ResNetBlock)
        params_enc = Encoder.init(rng, batch_enc)

        # Taking 64 images of the dataset
        batch_im = next(dset)
        # Generating new keys to use them for inference
        rng, rng_1 = random.split(rng)

        # Size of the input to initialize the parameters
        batch_dec = jnp.ones((1, 4, 4, 64))

        # Initializing the Decoder
        Decoder = ResNetDec(act_fn=nn.leaky_relu, block_class=ResNetBlockD)
        params_dec = Decoder.init(rng_1, batch_dec)

        # Defining a general list of the parameters
        params = [params_enc, params_dec]

        # Initialisation
        optimizer = optax.chain(
            optax.adam(FLAGS.learning_rate),
            optax.scale_by_schedule(lr_schedule))

        opt_state = optimizer.init(params)

        @jax.jit
        def loss_fn(params, rng_key, batch, reg_term): #state, rng_key, batch):
            """Function to define the loss function"""
            
            params_enc, params_dec = params
            
            x = batch

            # Autoencode an example
            q = Encoder.apply(params_enc, x)

            # Sample from the posterior
            z = q.sample(seed=rng_key)
            
            # Decode the sample
            p = Decoder.apply(params_dec, z)

            # KL divergence between the prior distribution and p
            kl = tfd.kl_divergence(p, tfd.MultivariateNormalDiag(jnp.zeros((1, 64, 64, 5))))
            
            # Compute log-likelihood
            log_likelihood = p.log_prob(x)
            
            # Calculating the ELBO value
            elbo = log_likelihood - reg_term*kl # Here we apply a regularization factor on the KL term
            
            loss = -elbo.mean()
            return loss, -log_likelihood.mean()

        # # Apply KL Regularization
        # kl_reg_w = 1 if prob_output else 0

        '''    # Veryfing that the 'value_and_grad' works fine
        loss, grads = jax.value_and_grad(loss_fn)(params, rng, batch_im, kl_reg_w)
        '''

        @jax.jit
        def update(params, rng_key, opt_state, batch):
            """Single SGD update step."""
            (loss, log_likelihood), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, rng_key, batch, reg)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return loss, log_likelihood, new_params, new_opt_state 

        '''loss, log_likelihood, params, opt_state = update(params, rng_1, opt_state, batch_im)'''

        # Login to wandb
        wandb.login()

        # Initializing a Weights & Biases Run
        wandb.init(
            project="galsim-jax-resnet",
            name="first-model",
            # tags="kl_reg={:.4f}".format(reg),
        )

        # run.tags.append("kl_reg={:.4f}".format(reg))
        # run.update()

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
        config.using_kl = True if prob_output else False
        config.kl_reg = reg if prob_output else None
        
        losses = []
        losses_test = []
        losses_test_step = []

        log_liks = []
        log_liks_test = []

        best_eval_loss = 1e6

        # Train the model as many steps as indicated initially
        for step in tqdm(range(1, config.steps + 1)):
            rng, rng_1 = random.split(rng)
            batch_im = next(dset)
            loss, log_likelihood, params, opt_state = update(params, rng_1, opt_state, batch_im)
            losses.append(loss)
            log_liks.append(log_likelihood)
            
            # Log metrics inside your training loop to visualize model performance
            wandb.log({ 
                "loss": loss,
                "log_likelihood": log_likelihood,
                }, step=step)
            
            # Saving best checkpoint
            if loss < best_eval_loss:
                best_eval_loss = loss
                save_checkpoint("checkpoint.msgpack", params, step)
            
            # Calculating the loss for all the test images 
            if step % 2500 == 0 :

                dataset_eval = input_fn('test')
                test_iterator = dataset_eval.as_numpy_iterator()

                for_list_mean = []

                for img in test_iterator:
                    rng, rng_1 = random.split(rng)
                    loss_test, log_likelihood_test = loss_fn(params, rng_1, img, reg)
                    for_list_mean.append(loss_test)

                losses_test.append(np.mean(for_list_mean))
                losses_test_step.append(step)
                log_liks_test.append(log_likelihood_test)
                
                wandb.log({ 
                "test_loss": losses_test[-1],
                "test_log_likelihood": log_liks_test[-1],
                }, step=step)
                    
                print("Step: {}, loss: {:.2f}, loss test: {:.2f}".format(step, loss, losses_test[-1]))
                
        params = load_checkpoint("checkpoint.msgpack", params)
                    
        loss_min = min(losses)
        best_step = losses.index(loss_min) + 1

        total_steps = np.arange(1, config.steps+1)

        print("\nBest Step: {}, loss: {:.2f}".format(best_step, loss_min))

        # Getting the date and time of the experiment
        datetime = get_date_time()
        # Creating the 'results' folder to save all the plots as images (or validating that the folder already exists)

        results_folder = 'results/run-{}-{}'.format(datetime, wandb.run.id)
        create_folder(results_folder)

        # Saving the loss plots
        save_plot_as_image(folder_path=results_folder, 
                           plot_title='Loglog of the Loss function - Train',
                           x_data=total_steps,  
                           y_data=losses, 
                           plot_type='loglog', 
                           file_name='loglog_loss.png')
        save_plot_as_image(folder_path=results_folder, 
                           plot_title='Loglog of the Loss function - Test', 
                           x_data=losses_test_step, 
                           y_data=losses_test, 
                           plot_type='loglog', 
                           file_name='loglog_loss_test.png')
        
        # Saving the log-likelihood plots
        save_plot_as_image(folder_path=results_folder, 
                           plot_title='Loglog of the Log-likelihood - Train',
                           x_data=total_steps, 
                           y_data=log_liks, 
                           plot_type='loglog', 
                           file_name='loglog_log_likelihood.png')
        save_plot_as_image(folder_path=results_folder, 
                           plot_title='Loglog of the Log-likelihood - Test', 
                           x_data=losses_test_step,
                           y_data=log_liks_test, 
                           plot_type='loglog', 
                           file_name='loglog_log_likelihood_test.png')
        
        # Predicting over an example of data
        dataset_eval = input_fn('test')
        test_iterator = dataset_eval.as_numpy_iterator()
        batch = next(test_iterator)
        # Taking 16 images as example
        batch = batch[:16,...]

        # Dividing the list of parameters obtained before
        params_enc, params_dec = params
        # Distribution of latent space calculated using the batch of data
        q = ResNetEnc(act_fn=nn.leaky_relu, block_class=ResNetBlock).apply(params_enc, batch)
        # Sampling from the distribution
        z = q.sample(seed=rng_1)

        # Posterior distribution
        p = ResNetDec(act_fn=nn.leaky_relu, block_class=ResNetBlockD).apply(params_dec, z)
        # Sample some variables from the posterior distribution
        z = p.sample(seed=rng_1)         
        
        # Saving the samples of the predicted images and their difference from the original images
        save_samples(folder_path=results_folder, z=z, batch=batch)

        wandb.finish()

    # print(log_liks)

if __name__ == "__main__":
  app.run(main)