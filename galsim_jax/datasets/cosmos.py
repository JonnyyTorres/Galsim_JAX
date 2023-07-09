""" TensorFlow Dataset of COSMOS images. """
import tensorflow_datasets as tfds
import numpy as np
import galsim as gs
from galsim.bounds import _BoundsI

import tensorflow as tf

from tensorflow_datasets.core.utils import gcs_utils

# disable internet connection
gcs_utils.gcs_dataset_info_files = lambda *args, **kwargs: None
gcs_utils.is_dataset_on_gcs = lambda *args, **kwargs: False

_CITATION = """
"""

_DESCRIPTION = """
"""


class CosmosConfig(tfds.core.BuilderConfig):
    """BuilderConfig for Cosmos."""

    def __init__(self, *, sample="25.2", stamp_size=128, pixel_scale=0.03, **kwargs):
        """BuilderConfig for Cosmos.
        Args:
        sample: which Cosmos sample to use, "25.2".
        stamp_size: image stamp size in pixels.
        pixel_scale: pixel scale of stamps in arcsec.
        **kwargs: keyword arguments forwarded to super.
        """
        v1 = tfds.core.Version("0.0.1")
        super(CosmosConfig, self).__init__(
            description=(
                "Cosmos stamps from %s sample in %d x %d resolution, %.2f arcsec/pixel."
                % (sample, stamp_size, stamp_size, pixel_scale)
            ),
            version=v1,
            **kwargs
        )
        self.stamp_size = stamp_size
        self.pixel_scale = pixel_scale
        self.sample = sample


class Cosmos(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Cosmos dataset."""

    VERSION = tfds.core.Version("0.0.1")
    RELEASE_NOTES = {
        "0.0.1": "Initial release.",
    }

    BUILDER_CONFIGS = [CosmosConfig(name="25.2", sample="25.2")]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(kappatng): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "image": tfds.features.Tensor(
                        shape=[
                            self.builder_config.stamp_size,
                            self.builder_config.stamp_size,
                        ],
                        dtype=np.float32,
                    ),
                    "kpsf_real": tfds.features.Tensor(
                        shape=[
                            self.builder_config.stamp_size,
                            self.builder_config.stamp_size//2+1,
                        ],
                        dtype=np.float32,
                    ),
                    "kpsf_imag": tfds.features.Tensor(
                        shape=[
                            self.builder_config.stamp_size,
                            self.builder_config.stamp_size//2+1,
                        ],
                        dtype=np.float32,
                    ),
                    # "noise_std": tfds.features.Scalar(dtype=tf.float32),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("image", "image"),
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "offset": 0,
                    "size": 4000,
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "offset": 4000,
                    "size": 1000,
                },
            ),
        ]

    def _generate_examples(self, offset, size):
        """Yields examples."""
        # Loads the galsim COSMOS catalog
        cat = gs.COSMOSCatalog(sample=self.builder_config.sample)
        ngal = size

        for i in range(ngal):
            gal = cat.makeGalaxy(i + offset)
            cosmos_gal = gs.Convolve(gal, gal.original_psf)

            cosmos_stamp = cosmos_gal.drawImage(
                nx=self.builder_config.stamp_size,
                ny=self.builder_config.stamp_size,
                scale=self.builder_config.pixel_scale,
                method="no_pixel",
            ).array.astype("float32")

            interp_factor = 1
            padding_factor = 1
            Nk = self.builder_config.stamp_size * interp_factor * padding_factor
            bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)
            imkpsf = gal.original_psf.drawKImage(bounds=bounds,
                            scale=2.*np.pi/(self.builder_config.stamp_size*padding_factor*self.builder_config.pixel_scale),
                            recenter=False)

            kpsf = np.fft.fftshift(imkpsf.array, 0).astype('complex64')
            kpsf_real = kpsf.real
            kpsf_imag = kpsf.imag


            # noise_std = np.sqrt(cosmos_gal.noise.getVariance())

            yield "%d" % i, {
                "image": cosmos_stamp,
                "kpsf_real": kpsf_real,
                "kpsf_imag": kpsf_imag,
                # "noise_std": noise_std,
            }
