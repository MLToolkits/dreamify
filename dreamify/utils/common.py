from pathlib import Path

import numpy as np
import PIL.Image
import tensorflow as tf
from IPython import display

from dreamify.lib.models.base.constants import PREPROCESSORS, ModelType


def show(img):
    """Display an image."""
    img = np.array(img)
    img = np.squeeze(img)
    display.display(PIL.Image.fromarray(img))


def preprocess_image(img, model_name: "inception_v3"):
    """Dynamically applies the correct preprocessing function based on the model."""
    img = tf.keras.utils.img_to_array(img)

    if img.ndim == 3 and img.shape[-1] == 1:  # Grayscale images
        img = tf.repeat(img, 3, axis=-1)

    img = tf.expand_dims(img, axis=0)

    try:
        model_enum = ModelType[model_name.upper()]
    except KeyError:
        raise ValueError(f"Model '{model_name}' not found in preprocessing map.")

    preprocess_fn = PREPROCESSORS().get(model_enum, None)

    if preprocess_fn is None:
        raise ValueError(f"Preprocessing function for '{model_name}' not found.")

    return preprocess_fn(img)


def deprocess_image(img):
    """Normalize image for display."""
    img = np.squeeze(img)
    img = 255 * (img + 1.0) / 2.0
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def get_image(source, max_dim=None):
    """Retrieve an image from a URL or a local path and load it as a NumPy array."""

    if source.startswith("http"):  # A URL to some image
        name = source.split("/")[-1]
        image_path = tf.keras.utils.get_file(name, origin=source)
        img = PIL.Image.open(image_path)
    else:  # A directory path to some image
        image_path = Path(source)
        img = tf.keras.utils.load_img(image_path)

    if max_dim:
        img.thumbnail((max_dim, max_dim))

    img = np.array(img)
    return img


def save_output(img, output_path, config):
    tf.keras.utils.save_img(output_path, img)
    print(f"Dream image saved to {output_path}")

    framer = config.framer

    if config.save_video:
        video_path = output_path.parent / (output_path.stem + ".mp4")
        framer.to_video(video_path, config.vid_duration, config.mirror_video)

    if config.save_gif:
        gif_path = output_path.parent / (output_path.stem + ".gif")
        framer.to_gif(gif_path, config.gif_duration, config.mirror_video)
