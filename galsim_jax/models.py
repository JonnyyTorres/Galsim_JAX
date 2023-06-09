from flax import linen as nn  # Linen API
from tensorflow_probability.substrates import jax as tfp

# Loading distributions and bijectors from TensorFlow Probability (JAX version)
tfd = tfp.distributions
tfb = tfp.bijectors


class ResNetBlock(nn.Module):
    """Creates a block of a CNN with ResNet architecture to encode images."""

    act_fn: callable  # Activation function
    c_out: int  # Output feature size
    subsample: bool = False  # If True, we apply a stride inside F

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        print("Input X Resnet", x.shape)
        z = nn.Conv(self.c_out, kernel_size=(3, 3), padding="SAME", strides=(2, 2))(x)
        print("Z Resnet", z.shape)
        z = self.act_fn(z)
        x = nn.Conv(self.c_out, kernel_size=(3, 3), padding="SAME", strides=(2, 2))(x)
        print("Sum X Resnet", x.shape)
        x_out = self.act_fn(z + x)
        return x_out


class ResNetEnc(nn.Module):
    """ "Creates a small convolutional encoder using ResNet blocks as intermediate layers"""

    act_fn: callable
    block_class: nn.Module
    num_blocks: tuple = (1, 1, 1)
    c_hidden: tuple = (64, 128, 256)
    latent_dim: int = 64
    # prob_output : bool = True # If True, the output will be an image instead of a
    # probability distribution

    @nn.compact
    def __call__(self, x, train=True):
        # A first convolution on the original image to scale up the channel size
        print(x.shape)
        x = nn.Conv(
            self.latent_dim, kernel_size=(3, 3), padding="SAME", strides=(2, 2)
        )(x)
        x = self.act_fn(x)
        print(x.shape)

        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = bc == 0 and block_idx > 0
                # ResNet block
                x = self.block_class(
                    c_out=self.c_hidden[block_idx],
                    act_fn=self.act_fn,
                    subsample=subsample,
                )(x, train=train)

        net = nn.Dense(features=self.latent_dim * 2)(x)
        # Image is now 4x4x128
        print("Dense shape", net.shape, "\n")

        q = tfd.MultivariateNormalDiag(
            loc=net[..., : self.latent_dim], scale_diag=net[..., self.latent_dim :]
        )

        return q


class ResNetBlockD(nn.Module):
    """Creates a block of a CNN with ResNet architecture to decode images."""

    act_fn: callable  # Activation function
    c_out: int  # Output feature size
    subsample: bool = False  # If True, we apply a stride inside F

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        print("Input X Resnet", x.shape)
        z = nn.ConvTranspose(
            self.c_out, kernel_size=(3, 3), padding="SAME", strides=(2, 2)
        )(x)
        print("Z Resnet", z.shape)
        z = self.act_fn(z)
        x = nn.ConvTranspose(
            self.c_out, kernel_size=(3, 3), padding="SAME", strides=(2, 2)
        )(x)
        print("Sum X Resnet", x.shape)
        x_out = self.act_fn(z + x)
        return x_out


class ResNetDec(nn.Module):
    """ "Creates a small convolutional decoder using ResNet blocks as intermediate layers"""

    act_fn: callable
    block_class: nn.Module
    num_blocks: tuple = (1, 1, 1, 1)
    c_hidden: tuple = (128, 64, 32, 5)
    # num_channels : int = 5

    @nn.compact
    def __call__(self, x, train=True):
        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = bc == 0 and block_idx > 0
                # ResNet block
                x = self.block_class(
                    c_out=self.c_hidden[block_idx],
                    act_fn=self.act_fn,
                    subsample=subsample,
                )(x, train=train)

        x = nn.activation.softplus(x)
        # Image is now 64x64x5
        r = tfd.MultivariateNormalDiag(loc=x, scale_diag=[0.01, 0.01, 0.01, 0.01, 0.01])

        return r
