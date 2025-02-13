import pytest
import tensorflow as tf

from dreamify.deep_dream import deep_dream_octaved, deep_dream_rolled, deep_dream_simple
from dreamify.lib import DeepDream, TiledGradients
from dreamify.utils.common import show
from dreamify.utils.configure import Config
from dreamify.utils.deep_dream_utils import download


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

    config = Config(
        feature_extractor=dream_model,
        layer_settings=layers,
        original_shape=original_shape,
        save_video=True,
        enable_framing=True,
        max_frames_to_sample=iterations,
    )

    deepdream = DeepDream(dream_model)

    return deepdream, original_img, iterations, config


@pytest.mark.parametrize("deepdream_inputs", [2], indirect=True)
def test_mock_deepdream(deepdream_inputs):
    deepdream, original_img, iterations = deepdream_inputs

    # Single Octave
    deep_dream_simple(
        img=original_img,
        dream_model=deepdream,
        iterations=iterations,
        learning_rate=0.1,
        save_video=True,
        output_path="/mock/deepdream_simple.mp4",
    )


@pytest.mark.parametrize("deepdream_inputs", [1], indirect=True)
def test_mock_deepdream_octaved(deepdream_inputs):
    deepdream, original_img, iterations, _ = deepdream_inputs

    # Multi-Octave
    deep_dream_octaved(
        img=original_img,
        dream_model=deepdream,
        iterations=iterations,
        learning_rate=0.1,
        save_video=True,
        output_path="/mock/deepdream_octaved.mp4",
    )


@pytest.mark.parametrize("deepdream_inputs", [1], indirect=True)
def test_mock_deepdream_rolled(deepdream_inputs):
    deepdream, original_img, iterations, config = deepdream_inputs

    get_tiled_gradients = TiledGradients(deepdream.model)

    # Multi-Octave
    deep_dream_rolled(
        img=original_img,
        get_tiled_gradients=get_tiled_gradients,
        iterations=iterations,
        learning_rate=0.1,
        save_video=True,
        output_path="/mock/deepdream_rolled.mp4",
        config=config,
    )
