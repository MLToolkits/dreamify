import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from dreamify.lib import FeatureExtractor, validate_dream
from dreamify.utils.common import deprocess, get_image, show
from dreamify.utils.configure import Config
from dreamify.utils.dream_utils import gradient_ascent_loop

# from dreamify.utils.compare import main

warnings.filterwarnings(
    "ignore", category=UserWarning, module="keras.src.models.functional"
)


@validate_dream
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
    duration=3,
    mirror_video=False,
):
    base_image_path = Path(image_path)
    output_path = Path(output_path)

    ft_ext = FeatureExtractor(model_name)

    original_img = get_image(base_image_path)
    original_img = np.expand_dims(original_img, axis=0)
    original_img = keras.applications.inception_v3.preprocess_input(original_img)
    original_shape = original_img.shape[1:3]

    config = Config(
        feature_extractor=ft_ext,
        layer_settings=ft_ext.layer_settings,
        original_shape=original_shape,
        save_video=save_video,
        enable_framing=save_video,
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

    img = deprocess(img)
    keras.utils.save_img(output_path, img)
    print(f"Dream image saved to {output_path}")

    show(img)

    if save_video:
        config.framer.to_video(output_path.stem + ".mp4", duration, mirror_video)


def main():
    dream("examples/example0.jpg", octaves=2, iterations=10)


# Compares all models and layer settings on an image
if __name__ == "__main__":
    # main()  # current implementation of comparison,py has circular import
    pass
