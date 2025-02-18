# import IPython.display as display
from pathlib import Path

import numpy as np
import tensorflow as tf

from dreamify.lib import DeepDream, TiledGradients, validate_dream
from dreamify.utils.common import deprocess, get_image, show
from dreamify.utils.configure import Config


# @validate_dream
# def deep_dream_simple(
#     image_path=None,
#     img=None,
#     output_path="dream.png",
#     model_name="inception_v3",
#     learning_rate=0.01,
#     iterations=100,
#     save_video=False,
#     duration=3,
#     mirror_video=False,
#     config=None,
# ):
#     if image_path is None and img is None:
#         raise TypeError("Missing image_path or img argument")

#     base_image_path = Path(image_path)
#     output_path = Path(output_path)

#     base_model = tf.keras.applications.InceptionV3(
#         include_top=False, weights="imagenet"
#     )

#     names = ["mixed3", "mixed5"]
#     layers = [base_model.get_layer(name).output for name in names]

#     ft_ext = tf.keras.Model(inputs=base_model.input, outputs=layers)
#     dream_model = DeepDream(ft_ext)

#     img = get_image(base_image_path)
#     img = tf.keras.applications.inception_v3.preprocess_input(img)
#     img_shape = img.shape[1:3]

#     if config is None:
#         config = Config(
#             feature_extractor=dream_model,
#             layer_settings=dream_model.model.layers,
#             original_shape=img_shape,
#             save_video=save_video,
#             enable_framing=save_video,
#             max_frames_to_sample=iterations,
#         )

#     img = tf.keras.applications.inception_v3.preprocess_input(img)
#     img = tf.convert_to_tensor(img)

#     learning_rate = tf.convert_to_tensor(learning_rate)
#     iterations_remaining = iterations
#     iteration = 0
#     while iterations_remaining:
#         run_iterations = tf.constant(min(100, iterations_remaining))
#         iterations_remaining -= run_iterations
#         iteration += run_iterations

#         loss, img = dream_model.gradient_ascent_loop(
#             img, run_iterations, tf.constant(learning_rate), config
#         )

#         # display.clear_output(wait=True)
#         show(deprocess(img))
#         print("Iteration {}, loss {}".format(iteration, loss))

#     tf.keras.utils.save_img(output_path, img)
#     print(f"Dream image saved to {output_path}")

#     if save_video:
#         config.framer.to_video(output_path.stem + ".mp4", duration, mirror_video)

#     return img


# @validate_dream
# def deep_dream_octaved(
#     image_path,
#     output_path="dream.png",
#     model_name="inception_v3",
#     learning_rate=0.01,
#     iterations=100,
#     save_video=False,
#     duration=3,
#     mirror_video=False,
# ):
#     base_image_path = Path(image_path)
#     output_path = Path(output_path)

#     base_model = tf.keras.applications.InceptionV3(
#         include_top=False, weights="imagenet"
#     )

#     names = ["mixed3", "mixed5"]
#     layers = [base_model.get_layer(name).output for name in names]

#     ft_ext = tf.keras.Model(inputs=base_model.input, outputs=layers)
#     dream_model = DeepDream(ft_ext)

#     img = get_image(base_image_path)
#     img = tf.keras.applications.inception_v3.preprocess_input(img)
#     img_shape = img.shape[1:3]

#     config = Config(
#         feature_extractor=dream_model,
#         layer_settings=dream_model.model.layers,
#         original_shape=img_shape,
#         save_video=False,
#         enable_framing=save_video,
#         max_frames_to_sample=iterations * 5,  # 5 octaves
#     )

#     OCTAVE_SCALE = 1.30
#     img = tf.constant(np.array(img))
#     float_base_shape = tf.cast(tf.shape(img)[:-1], tf.float32)

#     for n in range(-2, 3):
#         if n == 2:
#             config.save_video = save_video

#         new_shape = tf.cast(float_base_shape * (OCTAVE_SCALE**n), tf.int32)
#         img = tf.image.resize(img, new_shape).numpy()
#         img = deep_dream_simple(
#             img=img,
#             model_name=model_name,
#             iterations=iterations,
#             learning_rate=learning_rate,
#             save_video=False,
#             duration=duration,
#             mirror_video=mirror_video,
#             config=config,
#         )

#     tf.keras.utils.save_img(output_path, img)
#     print(f"Dream image saved to {output_path}")

#     if save_video:
#         config.framer.to_video(output_path.stem + ".mp4", duration, mirror_video)

#     return img


@validate_dream
def deep_dream(
    image_path,
    output_path="dream.png",
    iterations=100,
    learning_rate=0.01,
    octaves=range(-2, 3),
    octave_scale=1.3,
    save_video=False,
    duration=3,
    mirror_video=False,
):
    base_image_path = Path(image_path)
    output_path = Path(output_path)

    base_model = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet"
    )

    names = ["mixed3", "mixed5"]
    layers = [base_model.get_layer(name).output for name in names]

    ft_ext = tf.keras.Model(inputs=base_model.input, outputs=layers)
    get_tiled_gradients = TiledGradients(ft_ext)

    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    img = get_image(base_image_path)
    base_shape = tf.shape(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img_shape = img.shape[1:3]

    config = Config(
        feature_extractor=dream_model,
        layer_settings=layers,
        original_shape=img_shape,
        save_video=save_video,
        enable_framing=True,
        max_frames_to_sample=100,
    )

    for octave in octaves:
        new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32) * (
            octave_scale**octave
        )
        new_size = tf.cast(new_size, tf.int32)
        img = tf.image.resize(img, tf.cast(new_size, tf.int32))

        for iteration in range(iterations):
            gradients = get_tiled_gradients(img, new_size)
            img = img + gradients * learning_rate
            img = tf.clip_by_value(img, -1, 1)

            if iteration % 10 == 0:
                # display.clear_output(wait=True)
                show(deprocess(img))
                print("Octave {}, Iteration {}".format(octave, iteration))

            if config.enable_framing and config.framer.continue_framing():
                config.framer.add_to_frames(img)

    tf.keras.utils.save_img(output_path, img)
    print(f"Dream image saved to {output_path}")
    
    if save_video:
        config.framer.to_video(output_path.stem + ".mp4", duration, mirror_video)

    return img


# def main(save_video=False, duration=3, mirror_video=False):
#     url = (
#         "https://storage.googleapis.com/download.tensorflow.org/"
#         "example_images/YellowLabradorLooking_new.jpg"
#     )

#     # Single Octave
#     deep_dream_simple(
#         image_path=url,
#         iterations=100,
#         save_video=save_video,
#     )


# def main2(save_video=False, duration=3, mirror_video=False):
#     url = (
#         "https://storage.googleapis.com/download.tensorflow.org/"
#         "example_images/YellowLabradorLooking_new.jpg"
#     )

#     # Multi-Octave
#     deep_dream_octaved(
#         image_path=url,
#         iterations=50,
#         save_video=save_video,
#     )


def main(save_video=False, duration=3, mirror_video=False):
    url = (
        "https://storage.googleapis.com/download.tensorflow.org/"
        "example_images/YellowLabradorLooking_new.jpg"
    )

    deep_dream(
        image_path=url,
        save_video=save_video,
    )


if __name__ == "__main__":
    main()
