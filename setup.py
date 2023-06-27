import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="galsim-jax",
    version="0.0.1",
    author="Jonnyy Torres",
    author_email="wjatr777@gmail.com",
    description="Generative Models for Galaxy Image Simulations in JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JonnyyTorres/Galsim_JAX/galsim-jax",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        'flax',
        'optax',
        'tensorflow_probability',
        'tensorflow-datasets',
        'wandb',
        'tqdm',
        'astropy',
        'clu',
        ]
)