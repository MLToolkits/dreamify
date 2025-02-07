import IPython.display as display
import numpy as np
import tensorflow as tf

from dreamify.utils.common import deprocess, show
from dreamify.utils.deep_dream_utils import DeepDream, TiledGradients, download


def deep_dream_simple(img, dream_model, steps=100, step_size=0.01):
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)

    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining > 100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps

        loss, img = dream_model(img, run_steps, tf.constant(step_size))

        display.clear_output(wait=True)
        show(deprocess(img))
        print("Step {}, loss {}".format(step, loss))

    result = deprocess(img)

    return result


def deep_dream_octaved(img, dream_model, steps_per_octave=100, step_size=0.01):
    OCTAVE_SCALE = 1.30

    img = tf.constant(np.array(img))
    base_shape = tf.shape(img)[:-1]
    float_base_shape = tf.cast(base_shape, tf.float32)

    for n in range(-2, 3):
        new_shape = tf.cast(float_base_shape * (OCTAVE_SCALE**n), tf.int32)

        img = tf.image.resize(img, new_shape).numpy()
        img = deep_dream_simple(
            img=img,
            dream_model=dream_model,
            steps=steps_per_octave,
            step_size=step_size,
        )

    return img


def deep_dream_rolled(
    img,
    get_tiled_gradients,
    steps_per_octave=100,
    step_size=0.01,
    octaves=range(-2, 3),
    octave_scale=1.3,
):
    base_shape = tf.shape(img)
    img = tf.keras.utils.img_to_array(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    initial_shape = img.shape[:-1]
    img = tf.image.resize(img, initial_shape)
    for octave in octaves:
        # Scale the image based on the octave
        new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32) * (
            octave_scale**octave
        )
        new_size = tf.cast(new_size, tf.int32)
        img = tf.image.resize(img, new_size)

        for step in range(steps_per_octave):
            gradients = get_tiled_gradients(img, new_size)
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

            if step % 10 == 0:
                display.clear_output(wait=True)
                show(deprocess(img))
                print("Octave {}, Step {}".format(octave, step))

    result = deprocess(img)
    return result


def main():
    url = (
        "https://storage.googleapis.com/download.tensorflow.org/"
        "example_images/YellowLabradorLooking_new.jpg"
    )

    original_img = download(url, max_dim=500)
    show(original_img)

    base_model = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet"
    )

    names = ["mixed3", "mixed5"]
    layers = [base_model.get_layer(name).output for name in names]

    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    deepdream = DeepDream(dream_model)

    # Single Octave
    img = deep_dream_simple(
        img=original_img, dream_model=deepdream, steps=100, step_size=0.01
    )

    img = tf.image.resize(img, original_img.shape[:-1])
    img = tf.image.convert_image_dtype(img / 255.0, dtype=tf.uint8)
    show(img)


def main2():
    url = (
        "https://storage.googleapis.com/download.tensorflow.org/"
        "example_images/YellowLabradorLooking_new.jpg"
    )

    original_img = download(url, max_dim=500)
    show(original_img)

    base_model = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet"
    )

    names = ["mixed3", "mixed5"]
    layers = [base_model.get_layer(name).output for name in names]

    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    deepdream = DeepDream(dream_model)

    # Multi-Octave
    img = deep_dream_octaved(
        img=original_img, dream_model=deepdream, steps_per_octave=50, step_size=0.01
    )
    img = tf.image.resize(img, original_img.shape[:-1])
    img = tf.image.convert_image_dtype(img / 255.0, dtype=tf.uint8)
    display.clear_output(wait=True)
    show(img)


def main3():
    url = (
        "https://storage.googleapis.com/download.tensorflow.org/"
        "example_images/YellowLabradorLooking_new.jpg"
    )

    original_img = download(url, max_dim=500)
    show(original_img)

    base_model = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet"
    )

    names = ["mixed3", "mixed5"]
    layers = [base_model.get_layer(name).output for name in names]

    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    # Rolling/Multi-Octave with Tiling
    get_tiled_gradients = TiledGradients(dream_model)
    img = deep_dream_rolled(
        img=original_img,
        get_tiled_gradients=get_tiled_gradients,
        step_size=0.01,
    )
    img = tf.image.resize(img, original_img.shape[:-1])
    img = tf.image.convert_image_dtype(img / 255.0, dtype=tf.uint8)
    display.clear_output(wait=True)
    show(img)


if __name__ == "__main__":
    main()
