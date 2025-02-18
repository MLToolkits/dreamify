# import IPython.display as display
from pathlib import Path

import tensorflow as tf

from dreamify.lib import TiledGradients, validate_dream
from dreamify.utils.common import deprocess, get_image, show
from dreamify.utils.configure import Config


@validate_dream
def deepdream(
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
    img_shape = img.shape[:-1]

    config = Config(
        feature_extractor=dream_model,
        layer_settings=layers,
        original_shape=img_shape,
        save_video=save_video,
        enable_framing=True,
        max_frames_to_sample=iterations * len(octaves),
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


def main(save_video=False, duration=3, mirror_video=False):
    url = (
        "https://storage.googleapis.com/download.tensorflow.org/"
        "example_images/YellowLabradorLooking_new.jpg"
    )

    deepdream(
        image_path=url,
        save_video=save_video,
    )


if __name__ == "__main__":
    main()
