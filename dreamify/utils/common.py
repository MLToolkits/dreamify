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
