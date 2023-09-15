import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp

# Loading distributions from TensorFlow Probability (JAX version)
tfd = tfp.distributions


def Normalize(num_groups=10):
    return nn.GroupNorm(num_groups=num_groups, epsilon=1e-6, use_scale=True)


class Downsample(nn.Module):
    in_channels: int

    def setup(self):
        self.conv = nn.Conv(
            self.in_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((0, 1), (0, 1)),
        )

    def __call__(self, x):
        pad = ((0, 0), (0, 1), (0, 0), (0, 1))
        x = jnp.pad(x, pad, mode="constant", constant_values=0)
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    in_channels: int
    out_channels: int
    act_fn: callable = nn.gelu  # Activation function

    def setup(self):
        self.norm1 = Normalize(num_groups=1)
        self.conv1 = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
        )
        self.norm2 = Normalize(num_groups=1)
        self.conv2 = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
        )

        self.nin_shortcut = nn.Conv(
            self.out_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=((0, 0), (0, 0)),
        )

    def __call__(self, x):
        h = x
        h = self.norm1(h)
        h = self.act_fn(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act_fn(h)
        h = self.conv2(h)
        x = self.nin_shortcut(x)
        x_ = x + h

        return x + h


class DownsamplingBlock(nn.Module):
    ch: int
    ch_mult: tuple
    num_res_blocks: int
    resolution: int
    block_idx: int
    act_fn: callable = nn.gelu  # Activation function

    def setup(self):
        self.ch_mult_ = self.ch_mult
        self.num_resolutions = len(self.ch_mult_)
        in_ch_mult = (1,) + tuple(self.ch_mult_)
        block_in = self.ch * in_ch_mult[self.block_idx]
        block_out = self.ch * self.ch_mult_[self.block_idx]

        res_blocks = []
        for _ in range(self.num_res_blocks):
            res_blocks.append(ResnetBlock(block_in, block_out, self.act_fn))
        block_in = block_out
        self.block = res_blocks

        self.downsample = None
        if self.block_idx != self.num_resolutions - 1:
            self.downsample = Downsample(block_in)

    def __call__(self, h):
        for i, res_block in enumerate(self.block):
            h = res_block(h)

        if self.downsample is not None:
            h = self.downsample(h)

        return h


class MidBlock(nn.Module):
    in_channels: int

    def setup(self):
        self.block_1 = ResnetBlock(
            self.in_channels,
            self.in_channels,
        )
        self.block_2 = ResnetBlock(
            self.in_channels,
            self.in_channels,
        )

    def __call__(self, h):
        h = self.block_1(h)
        h = self.block_2(h)

        return h


class Encoder(nn.Module):
    ch: int
    out_ch: int
    ch_mult: tuple
    num_res_blocks: int
    in_channels: int
    resolution: int
    z_channels: int
    double_z: bool
    act_fn: callable = nn.gelu  # Activation function

    def setup(self):
        self.num_resolutions = len(self.ch_mult)

        # downsampling
        self.conv_in = nn.Conv(
            self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
        )

        curr_res = self.resolution
        downsample_blocks = []

        for i_level in range(self.num_resolutions):
            downsample_blocks.append(
                DownsamplingBlock(
                    ch=self.ch,
                    ch_mult=self.ch_mult,
                    num_res_blocks=self.num_res_blocks,
                    resolution=self.resolution,
                    block_idx=i_level,
                    act_fn=self.act_fn,
                )
            )
            if i_level != self.num_resolutions - 1:
                curr_res = curr_res // 2

        self.down = downsample_blocks

        # middle
        mid_channels = self.ch * self.ch_mult[-1]
        self.mid = MidBlock(mid_channels)
        # end
        self.norm_out = Normalize(num_groups=1)
        self.conv_out = nn.Conv(
            self.z_channels * 2 if self.double_z else self.z_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
        )

    def __call__(self, x):
        # downsampling
        print("x :", x.shape)
        hs = self.conv_in(x)
        print("Conv_in :", hs.shape)
        for block in self.down:
            hs = block(hs)
        print("Down :", hs.shape)

        # middle
        hs = self.mid(hs)
        print("Mid :", hs.shape)

        # end
        hs = self.norm_out(hs)
        hs = self.act_fn(hs)
        hs = self.conv_out(hs)
        print("Conv_out :", hs.shape)

        return hs


class Upsample(nn.Module):
    in_channels: int

    def setup(self):
        self.conv = nn.Conv(
            self.in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
        )

    def __call__(self, hs):
        batch, height, width, channels = hs.shape
        hs = jax.image.resize(
            hs,
            shape=(batch, height * 2, width * 2, channels),
            method="bicubic",
        )
        hs = self.conv(hs)
        return hs


class UpsamplingBlock(nn.Module):
    ch: int
    ch_mult: tuple
    num_res_blocks: int
    resolution: int
    block_idx: int
    act_fn: callable = nn.gelu  # Activation function

    def setup(self):
        self.ch_mult_ = self.ch_mult
        self.num_resolutions = len(self.ch_mult_)

        if self.block_idx == self.num_resolutions - 1:
            block_in = self.ch * self.ch_mult_[-1]
        else:
            block_in = self.ch * self.ch_mult_[self.block_idx + 1]

        block_out = self.ch * self.ch_mult_[self.block_idx]

        res_blocks = []
        for _ in range(self.num_res_blocks + 1):
            res_blocks.append(ResnetBlock(block_in, block_out, self.act_fn))

        block_in = block_out

        self.block = res_blocks

        self.upsample = None
        if self.block_idx != 0:
            self.upsample = Upsample(block_in)

    def __call__(self, h):
        for i, res_block in enumerate(self.block):
            h = res_block(h)

        if self.upsample is not None:
            h = self.upsample(h)

        return h


class Decoder(nn.Module):
    ch: int
    out_ch: int
    ch_mult: tuple
    num_res_blocks: int
    in_channels: int
    resolution: int
    z_channels: int
    double_z: bool
    act_fn: callable = nn.gelu  # Activation function

    def setup(self):
        self.num_resolutions = len(self.ch_mult)

        block_in = self.ch * self.ch_mult[self.num_resolutions - 1]
        curr_res = self.resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, self.z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv(
            self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
        )

        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # middle
        self.mid = MidBlock(block_in)

        # upsampling
        upsample_blocks = []

        for i_level in reversed(range(self.num_resolutions)):
            upsample_blocks.append(
                UpsamplingBlock(
                    ch=self.ch,
                    ch_mult=self.ch_mult,
                    num_res_blocks=self.num_res_blocks,
                    resolution=self.resolution,
                    block_idx=i_level,
                    act_fn=self.act_fn,
                )
            )
            if i_level != 0:
                curr_res = curr_res * 2
        self.up = list(reversed(upsample_blocks))  # reverse to get consistent order

        # end
        self.norm_out = Normalize(num_groups=1)
        self.conv_out = nn.Conv(
            self.out_ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
        )

    def __call__(self, z):
        # z to block_in
        hs = self.conv_in(z)

        # middle
        hs = self.mid(hs)

        # upsampling
        for block in reversed(self.up):
            hs = block(hs)

        # end
        hs = self.norm_out(hs)
        hs = self.act_fn(hs)
        hs = self.conv_out(hs)

        return hs


class AutoencoderKLModule(nn.Module):
    ch: int
    out_ch: int
    ch_mult: tuple
    num_res_blocks: int
    in_channels: int
    resolution: int
    z_channels: int
    double_z: bool
    embed_dim: int
    act_fn: callable = nn.gelu  # Activation function

    def setup(self):
        self.encoder = Encoder(
            self.ch,
            self.out_ch,
            self.ch_mult,
            self.num_res_blocks,
            self.in_channels,
            self.resolution,
            self.z_channels,
            self.double_z,
            self.act_fn,
        )
        self.decoder = Decoder(
            self.ch,
            self.out_ch,
            self.ch_mult,
            self.num_res_blocks,
            self.in_channels,
            self.resolution,
            self.z_channels,
            self.double_z,
            self.act_fn,
        )
        self.quant_conv = nn.Conv(
            2 * self.embed_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
        )
        self.post_quant_conv = nn.Conv(
            self.z_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
        )

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        print("Moments shape :", moments.shape)
        posterior = tfd.MultivariateNormalDiag(
            loc=moments[..., : self.z_channels],
            scale_diag=moments[..., self.z_channels :],
        )
        print("Posterior :", posterior)

        return posterior

    def decode(self, h):
        h = self.post_quant_conv(h)
        h = self.decoder(h)
        return h

    def __call__(self, x, seed):
        posterior = self.encode(x)

        h = posterior.sample(seed=seed)
        print('h', h.shape)
        log_prob = posterior.log_prob(h)
        
        q = self.decode(h)
        
        return q, log_prob, h
