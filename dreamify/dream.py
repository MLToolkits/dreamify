import warnings
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from dreamify.lib.feature_extractor import FeatureExtractor
from dreamify.utils.dream_utils import (
    configure_settings,
    deprocess_image,
    gradient_ascent_loop,
    preprocess_image,
    show,
    to_video,
)

# from dreamify.utils.compare import main

warnings.filterwarnings(
    "ignore", category=UserWarning, module="keras.src.models.functional"
)


def generate_dream_image(
    image_path,
    output_path="dream.png",
    model_name="inception_v3",
    learning_rate=5.0,
    num_octave=5,
    octave_scale=1.3,
    iterations=100,
    max_loss=15.0,
    save_video=False,
    duration=3,
):
    base_image_path = Path(image_path)
    output_path = Path(output_path)

    feature_extractor = FeatureExtractor(model_name)

    original_img = preprocess_image(base_image_path)
    original_shape = original_img.shape[1:3]

    configure_settings(
        feature_extractor, layer_settings, original_shape, save_video, [], iterations
    )

    successive_shapes = [original_shape]
    for i in range(-2, num_octave - 3):
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
        )
        upscaled_shrunk_original_img = tf.image.resize(
            shrunk_original_img, successive_shapes[i]
        )
        same_size_original = tf.image.resize(original_img, successive_shapes[i])
        lost_detail = same_size_original - upscaled_shrunk_original_img
        img += lost_detail
        shrunk_original_img = tf.image.resize(original_img, successive_shapes[i])

    image = img.numpy()
    keras.utils.save_img(output_path, deprocess_image(image))
    show(image)

    print(f"Dream image saved to {output_path}")

    if save_video:
        to_video(output_path.stem + ".mp4", duration)


def main():
    generate_dream_image("examples/example0.jpg", num_octave=2, iterations=10)


# Compares all models and layer settings on an image
if __name__ == "__main__":
    # main()  # current implementation of comparison has circular import
    pass
