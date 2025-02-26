from pathlib import Path

import pytest

from dreamify.deepdream import DeepDream, deepdream


@pytest.fixture
def deepdream_fixture(request):
    iterations = getattr(request, "param", 100)

    url = "examples/mock_rgb.jpg"

    return url, iterations


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("deepdream_fixture", [1], indirect=True)
def test_mock_deepdream_rgb(deepdream_fixture):
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


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("deepdream_fixture", [1], indirect=True)
def test_mock_deepdream_grayscale(deepdream_fixture):
    _, iterations = deepdream_fixture

    img_src = "examples/mock_grayscale.jpg"

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


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("deepdream_fixture", [1], indirect=True)
def test_classed_deepdream(deepdream_fixture):
    img_src, iterations = deepdream_fixture

    deepdream = DeepDream(iterations=iterations)

    deepdream(img_src)
    Path("deepdream.png").unlink(missing_ok=True)
