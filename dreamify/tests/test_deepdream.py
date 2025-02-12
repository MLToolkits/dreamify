import pytest
import tensorflow as tf

from dreamify.deep_dream import deep_dream_octaved, deep_dream_simple
from dreamify.lib import DeepDream
from dreamify.utils.common import show
from dreamify.utils.configure import Config
from dreamify.utils.deep_dream_utils import download

config = None


def configure_settings(**kwargs):
    global config
    config = Config(**kwargs)
    return config


@pytest.fixture
def deepdream_inputs(request):
    iterations = getattr(request, "param", 100)

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

    config = configure_settings(
        feature_extractor=dream_model,
        layer_settings=layers,
        original_shape=original_shape,
        enable_framing=True,
        max_frames_to_sample=iterations,
    )

    deepdream = DeepDream(dream_model, config)

    return deepdream, original_img, iterations


@pytest.mark.parametrize("deepdream_inputs", [10], indirect=True)
def test_mock_deepdream(deepdream_inputs):
    deepdream, original_img, iterations = deepdream_inputs

    # Single Octave
    deep_dream_simple(
        img=original_img,
        dream_model=deepdream,
        iterations=iterations,
        learning_rate=0.1,
        save_video=True,
        output_path="deepdream_simple.mp4",
    )


@pytest.mark.parametrize("deepdream_inputs", [5], indirect=True)
def test_mock_deepdream_octaved(deepdream_inputs):
    deepdream, original_img, iterations = deepdream_inputs

    # Multi-Octave
    deep_dream_octaved(
        img=original_img,
        dream_model=deepdream,
        iterations=iterations,
        learning_rate=0.1,
        save_video=True,
        output_path="deepdream_octaved.mp4",
    )
