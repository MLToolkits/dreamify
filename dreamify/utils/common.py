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


def download(url, max_dim=None):
    """Download an image and load it as a NumPy array."""
    name = url.split("/")[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)


def preprocess_image(image_path):
    img = keras.utils.load_img(image_path)
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.inception_v3.preprocess_input(img)
    return img
