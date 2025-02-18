import pytest

from dreamify.deep_dream import deep_dream
# deep_dream_octaved, 
# , deep_dream_simple


@pytest.fixture
def deepdream_fixture(request):
    iterations = getattr(request, "param", 100)

    url = (
        "https://storage.googleapis.com/download.tensorflow.org/"
        "example_images/YellowLabradorLooking_new.jpg"
    )

    return url, iterations


# @pytest.mark.parametrize("deepdream_fixture", [2], indirect=True)
# def test_mock_deepdream(deepdream_fixture):
#     img_src, output_dir, iterations = deepdream_fixture

#     # Single Octave
#     deep_dream_simple(
#         image_path=img_src,
#         iterations=iterations,
#         learning_rate=0.1,
#         save_video=True,
#         output_path=f"deepdream_simple.png",
#     )


# @pytest.mark.parametrize("deepdream_fixture", [1], indirect=True)
# def test_mock_deepdream_octaved(deepdream_fixture):
#     img_src, iterations = deepdream_fixture

#     # Multi-Octave
#     deep_dream_octaved(
#         image_path=img_src,
#         iterations=iterations,
#         learning_rate=0.1,
#         save_video=True,
#         output_path=f"deepdream_octaved.png",
#     )



@pytest.mark.parametrize("deepdream_fixture", [1], indirect=True)
def test_mock_deepdream(deepdream_fixture):
    img_src, iterations = deepdream_fixture

    # Rolled
    deep_dream(
        image_path=img_src,
        iterations=iterations,
        learning_rate=0.1,
        save_video=True,
        output_path=f"deepdream.png",
    )
