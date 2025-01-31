import numpy as np
import tensorflow as tf
from moviepy.video.fx import AccelDecel
from moviepy.video.VideoClip import DataVideoClip
from tensorflow import keras
from tqdm import trange

from dreamify.utils.configure import Config

config: Config = None


def configure_settings(
    feature_extractor,
    layer_settings,
    original_shape,
    enable_framing,
    frames_for_vid,
    iterations,
):
    global config
    config = Config(
        feature_extractor=feature_extractor,
        layer_settings=layer_settings,
        original_shape=original_shape,
        enable_framing=enable_framing,
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
def gradient_ascent_step(image, learning_rate):
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
        loss, image = gradient_ascent_step(image, learning_rate)

        if max_loss is not None and loss > max_loss:
            print(
                f"\nTerminating early: Loss ({loss:.2f}) exceeded max_loss ({max_loss:.2f}).\n"
            )
            break

        if (
            config.enable_framing
            and config.curr_frame_idx < config.max_frames_to_sample - 1
        ):
            frame = tf.image.resize(image, config.original_shape)
            frame = deprocess_image(image.numpy())
            config.frames_for_vid.append(frame)
            config.curr_frame_idx += 1

    return image


def to_video(output_path, duration, fps=60):
    global config

    upsample()

    print(f"Number of images to frame: {len(config.frames_for_vid)}")

    vid = DataVideoClip(config.frames_for_vid, lambda x: x, fps=fps)
    vid = AccelDecel(new_duration=duration).apply(vid)
    vid.write_videofile(output_path)

    config = Config()  # Reset the configuration


@tf.function
def interpolate_frames(frame1, frame2, num_frames):
    alphas = tf.linspace(0.0, 1.0, num_frames + 2)[1:-1]  # Avoid 0 and 1
    return tf.cast(
        (1 - alphas[:, None, None, None]) * frame1
        + alphas[:, None, None, None] * frame2,
        tf.uint8,
    )


def upsample():
    global config
    NUM_FRAMES_TO_INSERT = 60

    new_frames = []

    # Upsample via frame-frame interpoliation
    for i in range(len(config.frames_for_vid) - 1):
        frame1 = tf.convert_to_tensor(config.frames_for_vid[i], dtype=tf.float32)
        frame2 = tf.convert_to_tensor(config.frames_for_vid[i + 1], dtype=tf.float32)

        new_frames.append(config.frames_for_vid[i])
        new_frames.extend(
            interpolate_frames(frame1, frame2, NUM_FRAMES_TO_INSERT).numpy()
        )

    new_frames.extend(
        [config.frames_for_vid[-1]] * 60 * 3
    )  # Lengthen end frame by 3 secs
    config.frames_for_vid = new_frames


__all__ = [
    configure_settings,
    preprocess_image,
    deprocess_image,
    gradient_ascent_loop,
    to_video,
]
