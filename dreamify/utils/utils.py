import numpy as np
import tensorflow as tf
from moviepy.video.VideoClip import DataVideoClip
from tensorflow import keras
from tqdm import trange

from dreamify.utils.configure import Config

config: Config = None


def configure(
    feature_extractor, layer_settings, original_shape, frames_for_vid, iterations
):
    global config
    config = Config(
        feature_extractor=feature_extractor,
        layer_settings=layer_settings,
        original_shape=original_shape,
        frames_for_vid=frames_for_vid,
        max_frames_to_sample=iterations,
    )


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
    features = config.feature_extractor(input_image)
    loss = tf.zeros(shape=())
    for name in features.keys():
        coeff = config.layer_settings[name]
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
    global config

    for i in trange(
        iterations, desc="Gradient Ascent", unit="step", ncols=75, mininterval=0.1
    ):
        loss, image = _gradient_ascent_step(image, learning_rate)

        if max_loss is not None and loss > max_loss:
            print(
                f"\nTerminating early: Loss ({loss:.2f}) exceeded max_loss ({max_loss:.2f}).\n"
            )
            break

        if config.curr_frame_idx < config.max_frames_to_sample - 1:
            frame = tf.image.resize(image, config.original_shape)
            frame = deprocess_image(image.numpy())
            config.frames_for_vid.append(frame)
            config.curr_frame_idx += 1

    return image


def to_video(output_path, fps=1):
    global config

    print(f"Number of images to frame: {len(config.frames_for_vid)}")

    def identity(x):
        return x

    vid = DataVideoClip(config.frames_for_vid, identity, fps=fps)
    vid.write_videofile(output_path)

    config = Config()  # Reset the configuration
