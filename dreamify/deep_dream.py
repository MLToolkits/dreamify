import IPython.display as display
import numpy as np
import tensorflow as tf

from dreamify.lib import DeepDream, TiledGradients, validate_dream
from dreamify.utils.common import deprocess, show
from dreamify.utils.configure import Config
from dreamify.utils.deep_dream_utils import download

config: Config = None


def configure_settings(**kwargs):
    global config
    config = Config(**kwargs)
    return config


@validate_dream
def deep_dream_simple(
    img,
    dream_model,
    output_path="dream.png",
    iterations=100,
    learning_rate=0.01,
    save_video=False,
    duration=3,
    mirror_video=False,
):
    global config

    if config is None:
        config = configure_settings(
            feature_extractor=dream_model,
            layer_settings=dream_model.model.layers,
            original_shape=img.shape[:-1],
            enable_framing=save_video,
            max_frames_to_sample=iterations,
        )

    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)

    learning_rate = tf.convert_to_tensor(learning_rate)
    iterations_remaining = iterations
    iteration = 0
    while iterations_remaining:
        run_iterations = tf.constant(min(100, iterations_remaining))
        iterations_remaining -= run_iterations
        iteration += run_iterations

        loss, img = dream_model.gradient_ascent_loop(
            img, run_iterations, tf.constant(learning_rate), config
        )

        display.clear_output(wait=True)
        show(deprocess(img))
        print("Iteration {}, loss {}".format(iteration, loss))

    print(len(config.framer.frames_for_vid))

    return deprocess(img)


@validate_dream
def deep_dream_octaved(
    img,
    dream_model,
    output_path="dream.png",
    iterations=100,
    learning_rate=0.01,
    save_video=False,
    duration=3,
    mirror_video=False,
):
    global config

    config = configure_settings(
        feature_extractor=dream_model,
        layer_settings=dream_model.model.layers,
        original_shape=img.shape[:-1],
        enable_framing=save_video,
        max_frames_to_sample=iterations,
    )

    OCTAVE_SCALE = 1.30
    img = tf.constant(np.array(img))
    float_base_shape = tf.cast(tf.shape(img)[:-1], tf.float32)

    for n in range(-2, 3):
        new_shape = tf.cast(float_base_shape * (OCTAVE_SCALE**n), tf.int32)
        img = tf.image.resize(img, new_shape).numpy()
        img = deep_dream_simple(
            img=img,
            dream_model=dream_model,
            iterations=iterations,
            learning_rate=learning_rate,
            save_video=save_video,
            duration=duration,
            mirror_video=mirror_video,
        )

    return img


@validate_dream
def deep_dream_rolled(
    img,
    get_tiled_gradients,
    output_path="dream.png",
    iterations=100,
    learning_rate=0.01,
    octaves=range(-2, 3),
    octave_scale=1.3,
    save_video=False,
    duration=3,
    mirror_video=False,
    config=None,
):

    config = configure_settings(
        feature_extractor=dream_model,
        layer_settings=dream_model.model.layers,
        original_shape=img.shape[:-1],
        enable_framing=save_video,
        max_frames_to_sample=iterations,
    )

    base_shape = tf.shape(img)
    img = tf.keras.utils.img_to_array(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    initial_shape = img.shape[:-1]
    img = tf.image.resize(img, initial_shape)

    for octave in octaves:
        new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32) * (
            octave_scale**octave
        )
        img = tf.image.resize(img, tf.cast(new_size, tf.int32))

        for iteration in range(iterations):
            gradients = get_tiled_gradients(img, new_size)
            img = img + gradients * learning_rate
            img = tf.clip_by_value(img, -1, 1)

            if iteration % 10 == 0:
                display.clear_output(wait=True)
                show(deprocess(img))
                print("Octave {}, Iteration {}".format(octave, iteration))

            if config.enable_framing and config.framer.continue_framing():
                config.framer.add_to_frames(img)

    return deprocess(img)


def main(save_video=False, duration=3, mirror_video=False):
    url = (
        "https://storage.googleapis.com/download.tensorflow.org/"
        "example_images/YellowLabradorLooking_new.jpg"
    )

    original_img = download(url, max_dim=500)
    original_shape = original_img.shape[1:3]
    show(original_img)

    base_model = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet"
    )

    names = ["mixed3", "mixed5"]
    layers = [base_model.get_layer(name).output for name in names]

    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    global config
    config = configure_settings(
        feature_extractor=dream_model,
        layer_settings=layers,
        original_shape=original_shape,
        enable_framing=True,
        max_frames_to_sample=100,
    )

    deepdream = DeepDream(dream_model, config)

    # Single Octave
    img = deep_dream_simple(
        img=original_img,
        dream_model=deepdream,
        iterations=100,
        learning_rate=0.01,
        save_video=True,
    )

    img = tf.image.resize(img, original_img.shape[:-1])
    img = tf.image.convert_image_dtype(img / 255.0, dtype=tf.uint8)
    show(img)

    if save_video:
        config.framer.to_video("dream.mp4", duration, mirror_video)


def main2(save_video=False, duration=3, mirror_video=False):
    url = (
        "https://storage.googleapis.com/download.tensorflow.org/"
        "example_images/YellowLabradorLooking_new.jpg"
    )

    original_img = download(url, max_dim=500)
    original_shape = original_img.shape[1:3]
    show(original_img)

    base_model = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet"
    )

    names = ["mixed3", "mixed5"]
    layers = [base_model.get_layer(name).output for name in names]

    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    global config
    config = configure_settings(
        feature_extractor=dream_model,
        layer_settings=layers,
        original_shape=original_shape,
        enable_framing=True,
        max_frames_to_sample=100,
    )

    deepdream = DeepDream(dream_model, config)

    # Multi-Octave
    img = deep_dream_octaved(
        img=original_img,
        dream_model=deepdream,
        iterations=50,
        learning_rate=0.01,
        save_video=True,
    )
    img = tf.image.resize(img, original_img.shape[:-1])
    img = tf.image.convert_image_dtype(img / 255.0, dtype=tf.uint8)
    display.clear_output(wait=True)
    show(img)

    if save_video:
        config.framer.to_video("dream.mp4", duration, mirror_video)


def main3(save_video=False, duration=3, mirror_video=False):
    url = (
        "https://storage.googleapis.com/download.tensorflow.org/"
        "example_images/YellowLabradorLooking_new.jpg"
    )

    original_img = download(url, max_dim=500)
    original_shape = original_img.shape[1:3]
    show(original_img)

    base_model = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet"
    )

    names = ["mixed3", "mixed5"]
    layers = [base_model.get_layer(name).output for name in names]

    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    global config
    config = configure_settings(
        feature_extractor=dream_model,
        layer_settings=layers,
        original_shape=original_shape,
        enable_framing=True,
        max_frames_to_sample=100,
    )

    # Rolling/Multi-Octave with Tiling
    get_tiled_gradients = TiledGradients(dream_model)
    img = deep_dream_rolled(
        img=original_img,
        get_tiled_gradients=get_tiled_gradients,
        learning_rate=0.01,
        save_video=True,
        config=config,
    )
    img = tf.image.resize(img, original_img.shape[:-1])
    img = tf.image.convert_image_dtype(img / 255.0, dtype=tf.uint8)
    display.clear_output(wait=True)
    show(img)

    if save_video:
        config.framer.to_video("dream.mp4", duration, mirror_video)


if __name__ == "__main__":
    main()
