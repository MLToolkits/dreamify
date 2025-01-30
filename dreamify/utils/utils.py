import numpy as np
import tensorflow as tf
from moviepy.video.VideoClip import DataVideoClip
from tensorflow import keras
from tqdm import trange

is_configured = False
feature_extractor = None
layer_settings = None
original_shape = None


def configure(extractor, settings, original_shape_of_image):
    global is_configured, feature_extractor, layer_settings, original_shape
    if not is_configured:
        feature_extractor = extractor
        layer_settings = settings
        original_shape = original_shape_of_image
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


def gradient_ascent_loop(image, iterations, learning_rate, max_loss=None, images_for_vid=None):
    prev_frame = None
    enable_framing = True
    video_diff_threshold = tf.size(image) * 1.5
    print(video_diff_threshold)


    for i in trange(
        iterations, desc="Gradient Ascent", unit="step", ncols=75, mininterval=0.1
    ):
        loss, image = _gradient_ascent_step(image, learning_rate)

        if max_loss is not None and loss > max_loss:
            print(f"\nTerminating early: Loss ({loss:.2f}) exceeded max_loss ({max_loss:.2f}).\n")
            break
        

        curr_frame = image.numpy()

        if enable_framing or prev_frame is not None:
            frame_diff = calculate_frame_difference(curr_frame, prev_frame)

            if frame_diff > video_diff_threshold:
                disable_framing = False

            prev_frame = curr_frame

        if enable_framing:
            frame_for_vid = tf.image.resize(image, original_shape)
            frame_for_vid = deprocess_image(image.numpy())
            images_for_vid.append(frame_for_vid)

    return image


def to_video(images_for_vid, output_path, fps=2):
    vid = DataVideoClip(images_for_vid, lambda x: x, fps=fps)
    vid.write_videofile(output_path)


def calculate_frame_difference(curr_frame, prev_frame):
    """Calculate the absolute difference between two frames."""
    return tf.reduce_sum(tf.abs(curr_frame - prev_frame))