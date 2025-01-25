import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import inception_v3
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.models.functional")


from dreamify.utils.utils import preprocess_image, deprocess_image, compute_loss, gradient_ascent_loop, configure

def generate_dream_image(
    image_path,
    output_path="dream.png",
    layer_settings=None,
    step=20.0,
    num_octave=3,
    octave_scale=1.4,
    iterations=30,
    max_loss=15.0,
):
    if layer_settings is None:
        layer_settings = {
            "mixed4": 1.0,
            "mixed5": 1.5,
            "mixed6": 2.0,
            "mixed7": 2.5,
        }

    base_image_path = Path(image_path)

    model = inception_v3.InceptionV3(weights="imagenet", include_top=False)

    outputs_dict = {
        layer.name: layer.output
        for layer in [model.get_layer(name) for name in layer_settings.keys()]
    }
    feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

    configure(feature_extractor, layer_settings)


    original_img = preprocess_image(base_image_path)
    original_shape = original_img.shape[1:3]

    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale**i)) for dim in original_shape])
        successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]

    shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])

    img = tf.identity(original_img)
    for i, shape in enumerate(successive_shapes):
        print(f"\n\n{'_'*30} Processing octave {i} with shape {shape} {'_'*30}\n\n")
        img = tf.image.resize(img, shape)
        img = gradient_ascent_loop(
            img, iterations=iterations, learning_rate=step, max_loss=max_loss
        )
        upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img, shape)
        same_size_original = tf.image.resize(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img
        img += lost_detail
        shrunk_original_img = tf.image.resize(original_img, shape)

    keras.utils.save_img(output_path, deprocess_image(img.numpy()))
    print(f"Dream image saved to {output_path}")

def main():
    generate_dream_image("examples/example0.jpg", output_path="examples/dream0.png")
    generate_dream_image("examples/example1.jpg", output_path="examples/dream1.png")
    generate_dream_image("examples/example2.jpg", output_path="examples/dream2.png")
    generate_dream_image("examples/example3.jpg", output_path="examples/dream3.png")
    generate_dream_image("examples/example4.jpg", output_path="examples/dream4.png")

if __name__ == "__main__":
    main()
