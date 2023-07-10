# This module is taken from
# https://github.com/EiffL/Quarks2CosmosDataChallenge/blob/main/quarks2cosmos/galjax.py
# author: EiffL

import jax.numpy as jnp
import jax
import numpy as np

def convolve(image, psf, return_Fourier=False):
    """Convolves given image by psf.
    Args:
      image: a JAX array of size [nx, ny], either in real or Fourier space.
      psf: a JAX array, must have same shape as image.
      return_Fourier: whether to return the real or Fourier image.
    Returns:
      The resampled kimage.

    Note: This assumes both image and psf are sampled with same pixel scale!
    """

    if image.dtype in ["complex64", "complex128"]:
        kimage = image
    else:
        kimage = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.fftshift(image)))
    imkpsf = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.fftshift(psf)))

    im_conv = kimage * imkpsf

    if return_Fourier:
        return im_conv
    else:
        return jnp.fft.ifftshift(jnp.fft.ifft2(jnp.fft.ifftshift(im_conv)).real)

def k_wrapping(kimage, wrap_factor=1):
    """Wraps kspace image of a real image to decrease its resolution by specified
    factor
    
    Args:
    kimage: `Tensor`, image in Fourier space of shape [batch_size, nkx, nky]
    wrap_factor: `float`, wrap factor
    
    Returns:
    kimage: `Tensor`, kspace image with descreased resolution by wrap factor
    """
    
    Nkx, Nky = kimage.shape

    # First wrap around the non hermitian dimension
    rkimage = kimage + jnp.roll(kimage, shift=Nkx // wrap_factor, axis=0)

    # Now take care of the hermitian part
    revrkimage = jnp.flip(jnp.conjugate((jnp.flip(rkimage, axis=1))), 0)

    # These masks take care of the special case of the 0th frequency
    mask = np.ones([Nkx, Nky])
    mask[0, :] = 0
    mask[Nkx//wrap_factor-1, :] = 0
    rkimage2 = rkimage + revrkimage*mask
    mask = np.zeros([Nkx, Nky])
    mask[Nkx//wrap_factor-1,:] = 1
    rkimage2 = rkimage2 + jnp.roll(revrkimage, shift=-1, axis=0) * mask

    kimage = rkimage2[:Nkx//wrap_factor, :(Nky-1)//wrap_factor+1]

    return kimage

def kconvolve(kimage, kpsf,
             zero_padding_factor=1,
             interp_factor=1):
    """
    Convolution of provided k-space images and psf tensor.
    
    Careful! This function doesn't remove zero padding.
    Careful! When using a kimage and kpsf from GalSim,
           one needs to apply an fftshift at the output.
    
    This function assumes that the k-space tensors are already prodided with the
    stepk and maxk corresponding to the specified interpolation and zero padding
    factors.
    
    Args:
        kimages: `Tensor`, image in Fourier space of shape [batch_size, nkx, nky]
        kpsf: `Tensor`, PSF image in Fourier space
        zero_padding_factor: `int`, amount of zero padding
        interp_factor: `int`, interpolation factor
    
    Returns:
        `Tensor`, real image after convolution
    """

    Nkx, Nky = kimage.shape
    Nx = Nkx // zero_padding_factor // interp_factor

    # Perform k-space convolution
    imk = kimage * kpsf

    # Apply frequency wrapping to reach target image resolution
    if interp_factor > 1:
        imk = k_wrapping(imk, interp_factor)

    # Perform inverse Fourier Transform
    conv_images = jnp.fft.irfft2(imk)

    return conv_images

def convolve_kpsf(image, kpsf,
             x_interpolant=jax.image.ResizeMethod.LANCZOS5,
             zero_padding_factor=1,
             interp_factor=1,
             ):
    """
    Convolution of input images with provided k-space psf tensor.
    
    This function assumes that the k-space PSF is already prodided with the
    stepk and maxk corresponding to the specified interpolation and zero padding
    factors.
    
    Args:
        images: `Tensor`, input image in real space of shape [nkx, nky]
        x_interpolant: `string`, argument returned by `tf.image.ResizeMethod.BICUBIC` 
          or `tf.image.ResizeMethod.BILINEAR` for instance
        zero_padding_factor: `int`, amount of zero padding
        interp_factor: `int`, interpolation factor
    
    Returns:
        `Tensor`, real image after convolution
    """
    Nx, Ny = image.shape

    assert Nx == Ny
    assert Nx % 2 == 0

    # First, we interpolate the image on a finer grid
    if interp_factor > 1:
        im = jax.image.resize(image,
                              [Nx*interp_factor,
                              Ny*interp_factor],
                              method = x_interpolant)
        # since we lower the resolution of the image, we also scale the flux
        # accordingly
        image = im / interp_factor**2

    # Second, we pad as necessary
    #im = image
    pad_size = Nx * interp_factor * (zero_padding_factor - 1) // 2 
    im = jnp.pad(image, pad_size)

    # Compute DFT
    imk = jnp.fft.rfft2(im)
                 
    # Performing k space convolution
    imconv = kconvolve(imk, kpsf,
                       zero_padding_factor=zero_padding_factor,
                       interp_factor=interp_factor)

    a, b = imconv.shape
    return imconv[a//2-Nx//2:a//2+Nx//2, b//2-Nx//2:b//2+Nx//2]