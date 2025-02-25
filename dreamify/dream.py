import warnings
from pathlib import Path

import tensorflow as tf

from dreamify.decorators import optional_global_determinism, validate_dream_params
from dreamify.lib import Config, FeatureExtractor

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


@optional_global_determinism
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
    vid_duration=3,
    gif_duration=3,
    mirror_video=False,
    seed=None,
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
        vid_duration=vid_duration,
        gif_duration=gif_duration,
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


class Dream:
    def __init__(
        self,
        model_name="inception_v3",
        learning_rate=20.0,
        iterations=30,
        octaves=3,
        octave_scale=1.4,
        max_loss=15.0,
        seed=None,
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.octaves = octaves
        self.octave_scale = octave_scale
        self.max_loss = max_loss
        self.seed = seed

    def __call__(
        self,
        image_path,
        output_path="dream.png",
        save_video=False,
        save_gif=False,
        duration=None,
        vid_duration=None,
        gif_duration=None,
        mirror_video=False,
    ):
        params = vars(self).copy()
        params.update(
            image_path=image_path,
            output_path=output_path,
            save_video=save_video,
            save_gif=save_gif,
            duration=duration,
            vid_duration=vid_duration,
            gif_duration=gif_duration,
            mirror_video=mirror_video,
        )
        return dream(**params)

    def save_video(self, output_path, duration, vid_duration, mirror_video):
        return self(
            output_path=output_path,
            save_video=True,
            duration=duration,
            vid_duration=vid_duration,
            mirror_video=mirror_video,
        )

    def save_gif(self, output_path, duration, gif_duration, mirror_video):
        return self(
            output_path=output_path,
            save_gif=True,
            duration=duration,
            gif_duration=gif_duration,
            mirror_video=mirror_video,
        )


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
