from pathlib import Path

import pytest

from dreamify.deepdream import deepdream


@pytest.fixture
def deepdream_fixture(request):
    iterations = getattr(request, "param", 100)

    url = (
        "https://storage.googleapis.com/download.tensorflow.org/"
        "example_images/YellowLabradorLooking_new.jpg"
    )

    return url, iterations


@pytest.mark.parametrize("deepdream_fixture", [1], indirect=True)
def test_mock_deepdream(deepdream_fixture):
    img_src, iterations = deepdream_fixture

    # Rolled
    deepdream(
        image_path=img_src,
        iterations=iterations,
        learning_rate=0.1,
        save_video=True,
        output_path="mock.png",
    )
    Path("mock.png").unlink(missing_ok=True)
    Path("mock.mp4").unlink(missing_ok=True)
