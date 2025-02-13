import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import trange


def preprocess_image(image_path):
    img = keras.utils.load_img(image_path)
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.inception_v3.preprocess_input(img)
    return img


def compute_loss(input_image, config):
    features = config.feature_extractor(input_image)
    loss = tf.zeros(shape=())
    for name in features.keys():
        coeff = config.layer_settings[name]
        activation = features[name]
        loss += coeff * tf.reduce_mean(tf.square(activation[:, 2:-2, 2:-2, :]))
    return loss


@tf.function
def gradient_ascent_step(image, learning_rate, config):
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = compute_loss(image, config)
    grads = tape.gradient(loss, image)
    grads = tf.math.l2_normalize(grads)
    image += learning_rate * grads
    image = tf.clip_by_value(image, -1, 1)
    return loss, image


def gradient_ascent_loop(image, iterations, learning_rate, max_loss=None, config=None):
    for i in trange(
        iterations, desc="Gradient Ascent", unit="step", ncols=75, mininterval=0.1
    ):
        loss, image = gradient_ascent_step(image, learning_rate, config)

        if max_loss is not None and loss > max_loss:
            print(
                f"\nTerminating early: Loss ({loss:.2f}) exceeded max_loss ({max_loss:.2f}).\n"
            )
            break

        framer = config.framer

        if config.enable_framing and framer.continue_framing():
            framer.add_to_frames(image)

    return image


__all__ = [
    preprocess_image,
    gradient_ascent_loop,
]
