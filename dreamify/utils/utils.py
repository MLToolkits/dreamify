import numpy as np
import tensorflow as tf
from moviepy.video.VideoClip import DataVideoClip
from tensorflow import keras
from tqdm import trange

is_configured = False
feature_extractor = None
layer_settings = None


def configure(extractor, settings):
    global is_configured, feature_extractor, layer_settings
    if not is_configured:
        feature_extractor = extractor
        layer_settings = settings
        is_configured = True


def preprocess_image(image_path):
    img = keras.utils.load_img(image_path)
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.inception_v3.preprocess_input(img)
    return img


def deprocess_image(img):
    img = img.reshape((img.shape[1], img.shape[2], 3))
    img /= 2.0
    img += 0.5
    img *= 255.0
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def compute_loss(input_image):
    features = feature_extractor(input_image)
    loss = tf.zeros(shape=())
    for name in features.keys():
        coeff = layer_settings[name]
        activation = features[name]
        loss += coeff * tf.reduce_mean(tf.square(activation[:, 2:-2, 2:-2, :]))
    return loss


@tf.function
def _gradient_ascent_step(image, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = compute_loss(image)
    grads = tape.gradient(loss, image)
    grads = tf.math.l2_normalize(grads)
    image += learning_rate * grads
    return loss, image


def gradient_ascent_loop(image, iterations, learning_rate, max_loss=None):
    for i in trange(
        iterations, desc="Gradient Ascent", unit="step", ncols=75, mininterval=0.5
    ):
        loss, image = _gradient_ascent_step(image, learning_rate)
        if max_loss is not None and loss > max_loss:
            print(f"Terminating early: Loss exceeded max_loss ({max_loss:.2f}).")
            break
    return image


def to_video(imgs, output_name, fps=30):
    vid = DataVideoClip(imgs, fps=fps)
    vid.write_videofile(output_name)
