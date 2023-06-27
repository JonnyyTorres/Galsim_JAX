# This module is taken from
# https://github.com/EiffL/Quarks2CosmosDataChallenge/blob/main/quarks2cosmos/galjax.py
# author: EiffL

import jax.numpy as jnp
import jax

def convolve(image, psf, return_Fourier=False):
  """ Convolves given image by psf.
  Args: 
    image: a JAX array of size [nx, ny], either in real or Fourier space.
    psf: a JAX array, must have same shape as image.
    return_Fourier: whether to return the real or Fourier image.
  Returns:
    The resampled kimage.

  Note: This assumes both image and psf are sampled with same pixel scale!
  """

  if image.dtype in ['complex64', 'complex128']:
    kimage = image
  else:
    kimage = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.fftshift(image)))  
  imkpsf = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.fftshift(psf)))

  im_conv = kimage * imkpsf

  if return_Fourier:
    return im_conv
  else:
    return jnp.fft.ifftshift(jnp.fft.ifft2(jnp.fft.ifftshift(im_conv)).real)