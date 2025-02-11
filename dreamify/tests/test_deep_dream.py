import tensorflow as tf

from dreamify.deep_dream import deep_dream_simple
from dreamify.lib import DeepDream
from dreamify.utils.common import show
from dreamify.utils.configure import Config
from dreamify.utils.deep_dream_utils import download


def configure_settings(**kwargs):
    global config
    config = Config(**kwargs)
    return config


def test_mock_deepdream():
    url = (
        "https://storage.googleapis.com/download.tensorflow.org/"
        "example_images/YellowLabradorLooking_new.jpg"
    )

    original_img = download(url, max_dim=500)
    original_shape = original_img.shape[1:3]
    show(original_img)

    base_model = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet"
    )

    names = ["mixed3", "mixed5"]
    layers = [base_model.get_layer(name).output for name in names]

    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    global config
    config = configure_settings(
        feature_extractor=dream_model,
        layer_settings=layers,
        original_shape=original_shape,
        enable_framing=True,
        max_frames_to_sample=100,
    )

    deepdream = DeepDream(dream_model, config)

    # Single Octave
    deep_dream_simple(
        img=original_img,
        dream_model=deepdream,
        iterations=5,
        learning_rate=0.01,
    )
