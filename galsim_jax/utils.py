import os
import matplotlib.pyplot as plt
import jax.numpy as jnp
import subprocess
import wandb
import optax

from flax.serialization import to_state_dict, msgpack_serialize, from_bytes
from flax import linen as nn  # Linen API


def create_folder(folder_path="results"):
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
    steps_per_epoch = 50000 // 64

    current_epoch = step / steps_per_epoch  # type: float
    boundaries = jnp.array((40, 80, 120)) * steps_per_epoch
    values = jnp.array([1.0, 0.1, 0.01, 0.001])

    index = jnp.sum(boundaries < step)
    return jnp.take(values, index)


def get_git_commit_version():
    """ "Allows to get the Git commit version to tag each experiment"""
    try:
        commit_version = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        return commit_version
    except subprocess.CalledProcessError:
        return None


def save_checkpoint(ckpt_path, state, step):
    """Saves a Wandb checkpoint."""
    with open(ckpt_path, "wb") as outfile:
        outfile.write(msgpack_serialize(to_state_dict(state)))
    artifact = wandb.Artifact(f"{wandb.run.id}-checkpoint", type="model")
    artifact.add_file(ckpt_path)
    wandb.log_artifact(
        artifact, aliases=["best", f"step_{step}", f"commit_{get_git_commit_version()}"]
    )


def load_checkpoint(ckpt_file, state):
    """Loads the best Wandb checkpoint."""
    artifact = wandb.use_artifact(f"{wandb.run.id}-checkpoint:best")
    artifact_dir = artifact.download()
    ckpt_path = os.path.join(artifact_dir, ckpt_file)
    with open(ckpt_path, "rb") as data_file:
        byte_data = data_file.read()
    return from_bytes(state, byte_data)


def save_plot_as_image(
    folder_path,
    plot_title,
    x_data,
    y_data,
    plot_type="loglog",
    file_name="plot.png",
    **kwargs,
):
    # Generate plot based on plot_type
    if plot_type == "line":
        plt.plot(x_data, y_data, **kwargs)
    elif plot_type == "loglog":
        plt.loglog(x_data, y_data, **kwargs)
    elif plot_type == "semilogy":
        plt.semilogy(x_data, y_data, **kwargs)
    elif plot_type == "semilogx":
        plt.semilogx(x_data, y_data, **kwargs)
    elif plot_type == "scatter":
        plt.scatter(x_data, y_data, **kwargs)
    else:
        print("Invalid plot type!")
        return

    plt.title(plot_title)
    plt.xlabel("Step")
    plt.ylabel("Value")

    # Save plot as image within the folder
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path)
    wandb.log({"{}".format(file_name.split(".")[0]): wandb.Image(plt)})
    plt.close()

    print(f"Plot saved as {file_path}")


def save_samples(folder_path, z, batch):
    # Plotting the original, predicted and their differences for 8 examples
    num_rows, num_cols = 8, 3

    plt.figure(figsize=(9, 28))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(9, 24))

    for i, (ax1, ax2, ax3) in enumerate(zip(axes[:, 0], axes[:, 1], axes[:, 2])):
        batch_img = batch[i, ...]
        z_img = z[i, ...]

        # Plotting original image
        ax1.imshow(batch_img.mean(axis=-1))
        ax1.axis("off")
        # Plotting predicted image
        ax2.imshow(z_img.mean(axis=-1))
        ax2.axis("off")
        # Plotting difference between original and predicted image
        ax3.imshow(z_img.mean(axis=-1) - batch_img.mean(axis=-1))
        ax3.axis("off")

    # Add a title to the figure
    fig.suptitle(
        "Comparison between original and predicted images", fontsize=12, y=0.99
    )

    # Adjust the layout of the subplots
    fig.tight_layout()

    # Save plot as image within the folder
    file_path = os.path.join(folder_path, "difference_pred.png")
    plt.savefig(file_path)
    wandb.log({"difference_pred": wandb.Image(plt)})
    plt.close(fig)

    print(f"Plot saved as {file_path}")

    # Plotting 16 images of the estimated shape of galaxies
    num_rows, num_cols = 4, 4

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    for ax, z_img in zip(axes.flatten(), z):
        ax.imshow(z_img.mean(axis=-1))
        ax.axis("off")

    # Add a title to the figure
    fig.suptitle("Samples of predicted galaxies", fontsize=16)

    # Adjust the layout of the subplots
    fig.tight_layout()
    # Save plot as image within the folder
    file_path = os.path.join(folder_path, "samples_pred.png")
    plt.savefig(file_path)
    wandb.log({"samples_pred": wandb.Image(plt)})
    plt.close(fig)

    print(f"Plot saved as {file_path}")


def get_wandb_local_dir(wandb_local_dir):
    # Extract the substring between 'run-' and '/files'
    start_index = wandb_local_dir.find("wandb/") + len("wandb/")
    end_index = wandb_local_dir.find("/files")
    run_string = wandb_local_dir[start_index:end_index]

    return run_string


def get_activation_fn(name):
    """JAX built-in activation functions"""

    activation_functions = {
        "linear": lambda: lambda x: x,
        "relu": nn.relu,
        "relu6": nn.relu6,
        "elu": nn.elu,
        "gelu": nn.gelu,
        "prelu": nn.PReLU,
        "leaky_relu": nn.leaky_relu,
        "hardtanh": nn.hard_tanh,
        "sigmoid": nn.sigmoid,
        "tanh": nn.tanh,
        "log_sigmoid": nn.log_sigmoid,
        "softplus": nn.softplus,
        "softsign": nn.soft_sign,
        "swish": nn.swish,
    }

    if name not in activation_functions:
        raise ValueError(
            f"'{name}' is not included in activation_functions. use below one. \n {activation_functions.keys()}"
        )

    return activation_functions[name]


def get_optimizer(name, lr, num_steps):
    """JAX built-in activation functions"""

    optimizer = {
        "adam": optax.chain(optax.adam(lr), optax.scale_by_schedule(lr_schedule)),
        "adamw": optax.chain(optax.adamw(lr), optax.scale_by_schedule(lr_schedule)),
    }

    if name not in optimizer:
        raise ValueError(
            f"'{name}' is not included in optimizer names. use below one. \n {optimizer.keys()}"
        )

    return optimizer[name]


def new_optimizer(name):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1.0,
        warmup_steps=5_000,
        decay_steps=95_000,
        end_value=0.0,
    )

    optimizer = {
        "adam": optax.chain(optax.clip(1.0), optax.adam(learning_rate=schedule)),
        "adamw": optax.chain(optax.clip(1.0), optax.adamw(learning_rate=schedule)),
    }

    if name not in optimizer:
        raise ValueError(
            f"'{name}' is not included in optimizer names. use below one. \n {optimizer.keys()}"
        )

    return optimizer[name]
