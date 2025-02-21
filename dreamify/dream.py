import warnings
from pathlib import Path

import tensorflow as tf

from dreamify.lib import Config, FeatureExtractor
from dreamify.lib.misc import validate_dream_params

# from dreamify.lib.dream_model import DreamModel
from dreamify.utils import (
    deprocess_image,
    get_image,
    preprocess_image,
    save_output,
    show,
)
from dreamify.utils.dream_utils import gradient_ascent_loop

warnings.filterwarnings(
    "ignore", category=UserWarning, module="keras.src.models.functional"
)


@validate_dream_params
def dream(
    image_path,
    output_path="dream.png",
    model_name="inception_v3",
    learning_rate=20.0,
    iterations=30,
    octaves=3,
    octave_scale=1.4,
    max_loss=15.0,
    save_video=False,
    save_gif=False,
    duration=3,
    mirror_video=False,
):
    output_path = Path(output_path)

    ft_ext = FeatureExtractor(
        model_name=model_name, dream_style="shallow", layer_settings=None
    )

    original_img = get_image(image_path)
    original_img = preprocess_image(original_img, model_name)

    original_shape = original_img.shape[1:-1]

    config = Config(
        feature_extractor=ft_ext,
        layer_settings=ft_ext.layer_settings,
        original_shape=original_shape,
        save_video=save_video,
        save_gif=save_gif,
        enable_framing=save_video,
        duration=duration,
        mirror_video=mirror_video,
        max_frames_to_sample=iterations,
    )

    successive_shapes = [original_shape]
    for i in range(1, octaves):
        shape = tuple([int(dim / (octave_scale**i)) for dim in original_shape])
        successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]

    shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])

    img = tf.identity(original_img)
    for i, shape in enumerate(successive_shapes):
        print(
            f"\n\n{'_'*20} Processing octave {i + 1} with shape {successive_shapes[i]} {'_'*20}\n\n"
        )
        img = tf.image.resize(img, successive_shapes[i])
        img = gradient_ascent_loop(
            img,
            iterations=iterations,
            learning_rate=learning_rate,
            max_loss=max_loss,
            config=config,
        )
        upscaled_shrunk_original_img = tf.image.resize(
            shrunk_original_img, successive_shapes[i]
        )
        same_size_original = tf.image.resize(original_img, successive_shapes[i])
        lost_detail = same_size_original - upscaled_shrunk_original_img
        img += lost_detail
        shrunk_original_img = tf.image.resize(original_img, successive_shapes[i])

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

    dream(
        image_path=img_path,
        save_video=save_video,
        save_gif=save_gif,
        duration=duration,
        mirror_video=mirror_video,
    )


if __name__ == "__main__":
    main()
