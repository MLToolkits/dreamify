import warnings
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from dreamify.utils.models import choose_model
from dreamify.utils.utils import (
    configure_settings,
    deprocess_image,
    gradient_ascent_loop,
    preprocess_image,
    to_video,
)
# from dreamify.utils.compare import main

warnings.filterwarnings(
    "ignore", category=UserWarning, module="keras.src.models.functional"
)


def generate_dream_image(
    image_path,
    output_path="dream.png",
    model_name="xception",
    step=20.0,
    num_octave=3,
    octave_scale=1.4,
    iterations=30,
    max_loss=15.0,
    save_video=False,
    duration=10,
):
    base_image_path = Path(image_path)
    output_path = Path(output_path)

    model, layer_settings = choose_model(model_name)

    outputs_dict = {
        layer.name: layer.output
        for layer in [model.get_layer(name) for name in layer_settings.keys()]
    }
    feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

    original_img = preprocess_image(base_image_path)
    original_shape = original_img.shape[1:3]

    configure_settings(
        feature_extractor, layer_settings, original_shape, save_video, [], iterations
    )

    successive_shapes = [original_shape]
    for i in range(1, num_octave):
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
            learning_rate=step,
            max_loss=max_loss,
        )
        upscaled_shrunk_original_img = tf.image.resize(
            shrunk_original_img, successive_shapes[i]
        )
        same_size_original = tf.image.resize(
            original_img, successive_shapes[i])
        lost_detail = same_size_original - upscaled_shrunk_original_img
        img += lost_detail
        shrunk_original_img = tf.image.resize(
            original_img, successive_shapes[i])

    keras.utils.save_img(output_path, deprocess_image(img.numpy()))
    print(f"Dream image saved to {output_path}")

    if save_video:
        to_video(output_path.stem + ".mp4", duration)


# Compares all models and layer settings on an image
if __name__ == "__main__":
    # main()  # current implementation of comparison has circular import
    pass
