import numpy as np
import PIL.Image
import tensorflow as tf
from IPython import display


def show(img):
    """Display an image."""
    img = np.array(img)
    img = np.squeeze(img)
    display.display(PIL.Image.fromarray(img))


def deprocess(img):
    """Normalize image for display."""
    img = tf.squeeze(img)
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)


def get_image(source, max_dim=None):
    """Retrieve an image from a URL or a local path and load it as a NumPy array."""

    source = str(source)

    if source.startswith("http"):  # A URL to some image
        name = source.split("/")[-1]
        image_path = tf.keras.utils.get_file(name, origin=source)
        img = PIL.Image.open(image_path)
    else:  # A directory path to some image
        img = tf.keras.utils.load_img(source)

    if max_dim:
        img.thumbnail((max_dim, max_dim))

    img = np.array(img)
    return img


def preprocess_image(image_path):
    img = tf.keras.utils.load_img(image_path)
    img = tf.keras.utils.img_to_array(img)
    img = tf.np.expand_dims(img, axis=0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img
