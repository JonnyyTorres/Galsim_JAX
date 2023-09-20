from galsim_jax.dif_models import AutoencoderKLModule
from flax import linen as nn

import numpy as np
import jax.numpy as jnp
import tensorflow_probability as tfp

tfp = tfp.substrates.jax

tfd = tfp.distributions
tfb = tfp.bijectors


class encode(AutoencoderKLModule):
    @nn.compact
    def __call__(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        print("Moments shape :", moments.shape)
        posterior = tfd.MultivariateNormalDiag(
            loc=moments[..., : self.z_channels],
            scale_diag=moments[..., self.z_channels :],
        )
        print("Posterior :", posterior)

        return posterior


class decode(AutoencoderKLModule):
    @nn.compact
    def __call__(self, h):
        h = self.post_quant_conv(h)
        h = self.decoder(h)
        h = nn.softplus(h)

        return h


class AffineCoupling(nn.Module):
    @nn.compact
    def __call__(self, x, nunits):
        net = nn.gelu(nn.Dense(256)(x))
        net = nn.gelu(nn.Dense(256)(net))

        # Shift and scale parameters
        shift = nn.Dense(nunits)(net)
        scale = nn.softplus(nn.Dense(nunits)(net)) + 1e-3  # For numerical stability

        return tfb.Chain([tfb.Shift(shift), tfb.Scale(scale)])


def make_nvp_fn(n_layers=4, d=64):
    # We alternate between permutations and flow layers
    layers = [
        tfb.Permute(np.flip(np.arange(d)))(
            tfb.RealNVP(d // 2, bijector_fn=AffineCoupling(name="affine%d" % i))
        )
        for i in range(n_layers)
    ]

    # We build the actual nvp from these bijectors and a standard Gaussian distribution
    nvp = tfd.TransformedDistribution(
        tfd.MultivariateNormalDiag(loc=jnp.zeros(d), scale_diag=jnp.ones(d)),
        bijector=tfb.Chain(layers),
    )
    return nvp


class NeuralSplineFlowLogProb(nn.Module):
    @nn.compact
    def __call__(self, x):
        nvp = make_nvp_fn()
        return nvp.log_prob(x)


class NeuralSplineFlowSampler(nn.Module):
    @nn.compact
    def __call__(self, key, n_samples):
        nvp = make_nvp_fn()
        return nvp.sample(n_samples, seed=key)
