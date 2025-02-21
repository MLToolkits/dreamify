from pathlib import Path

import IPython.display as display
import tensorflow as tf

from dreamify.decorators import optional_global_determinism, validate_dream_params
from dreamify.lib import Config, FeatureExtractor
from dreamify.lib.models import TiledGradients
from dreamify.utils import (
    deprocess_image,
    get_image,
    preprocess_image,
    save_output,
    show,
)


@optional_global_determinism
@validate_dream_params
def deepdream(
    image_path,
    output_path="deepdream.png",
    model_name="inception_v3",
    iterations=100,
    learning_rate=0.01,
    octaves=range(-2, 3),
    octave_scale=1.3,
    save_video=False,
    save_gif=False,
    duration=3,
    vid_duration=3,
    gif_duration=3,
    mirror_video=False,
    seed=None,
):
    output_path = Path(output_path)

    ft_ext = FeatureExtractor(model_name, dream_style="deep", layer_settings=None)
    get_tiled_gradients = TiledGradients(ft_ext.feature_extractor)

    img = get_image(image_path)
    img = preprocess_image(img, model_name)

    original_shape = img.shape[1:-1]

    config = Config(
        feature_extractor=ft_ext,
        layer_settings=ft_ext.layer_settings,
        original_shape=original_shape,
        save_video=save_video,
        save_gif=save_gif,
        enable_framing=True,
        duration=duration,
        vid_duration=vid_duration,
        gif_duration=gif_duration,
        mirror_video=mirror_video,
        max_frames_to_sample=iterations * len(octaves),
    )

    for octave in octaves:
        new_size = tf.cast(tf.convert_to_tensor(original_shape), tf.float32) * (
            octave_scale**octave
        )
        new_size = tf.cast(new_size, tf.int32)
        img = tf.image.resize(img, tf.cast(new_size, tf.int32))

        for iteration in range(iterations):
            gradients = get_tiled_gradients(tf.squeeze(img), new_size)
            img = img + gradients * learning_rate
            img = tf.clip_by_value(img, -1, 1)

            if iteration % 10 == 0:
                display.clear_output(wait=True)
                show(deprocess_image(img))
                print("Octave {}, Iteration {}".format(octave, iteration))

            if config.enable_framing and config.framer.continue_framing():
                config.framer.add_to_frames(img)

    img = deprocess_image(img)
    show(img)

    save_output(img, output_path, config)

    return img


def main(img_path, save_video=False, save_gif=False, duration=3, mirror_video=False):
    if img_path is None:
        img_path = (
            "https://storage.googleapis.com/download.tensorflow.org/"
            "example_images/YellowLabradorLooking_new.jpg"
        )

    deepdream(
        image_path=img_path,
        save_video=save_video,
        save_gif=save_gif,
        duration=duration,
        mirror_video=mirror_video,
    )


if __name__ == "__main__":
    main()
