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
        output_path="mock_deepdream.png",
        iterations=iterations,
        learning_rate=0.1,
        save_video=True,
        save_gif=True,
        mirror_video=True,
    )
    Path("mock_deepdream.png").unlink(missing_ok=True)
    Path("mock_deepdream.mp4").unlink(missing_ok=True)
    Path("mock_deepdream.gif").unlink(missing_ok=True)
