import IPython.display as display
import PIL.Image
import tensorflow as tf


def download(url, max_dim=None):
    """Download an image and load it as a NumPy array."""
    name = url.split("/")[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)


def deprocess(img):
    """Normalize image for display."""
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)


def show(img):
    """Display an image."""
    display.display(PIL.Image.fromarray(np.array(img)))
